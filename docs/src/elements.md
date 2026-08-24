```@meta
CurrentModule = FerriteOperators
```

# Writing elements

An element consists of an integrator (its setup-time description), a cache,
and request-typed kernels. The **residual kernel is mandatory** (validated at
setup); everything else is derived from it by ForwardDiff unless an analytic
kernel is declared.

```julia
struct MyIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

struct MyCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end

function FerriteOperators.setup_element_cache(m::MyIntegrator, sdh::SubDofHandler)
    qr = getquadraturerule(m.qrc, sdh)
    ip = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return MyCache(CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::MyCache) =
    MyCache(FerriteOperators.duplicate_for_device(device, c.cv))

FerriteOperators.reinit_values!(c::MyCache, cell) = reinit!(c.cv, cell)

function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::MyCache, args::CellArgs)
    (; cv) = cache
    uₑ = args.states.u
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # ... accumulate into req.r ...
    end
end
```

`setup_operator` wraps a cache lacking analytic coverage of a kind in
[`ADElementCache`](@ref) automatically, and the engine calls `assemble_cell!`
on the resolved cache unconditionally — it never forks between an analytic
kernel and the AD fallback itself.

The residual kernel must be eltype-generic in `eltype(args.states.*)`,
`eltype(args.p)`, and the context time — that is the entire AD contract.
Kernels never write global state; the geometry cache in `args.cell` is
read-only.

An element is a scheme-agnostic integrand: it reads slot *values* and never
encodes a time discretization. [The layer contract](devdocs/design.md) states
where each piece of information a kernel might want belongs.

## Storage classes for elements with local problems

A cache carries data of three different lifetimes, and `duplicate_for_device`
treats each one differently:

- **Immutable problem structure** — meshes, handlers, factorizable
  sparsity patterns — is built once in `setup_element_cache` from data the
  integrator carries. `duplicate_for_device` may deliberately ALIAS it across
  workers: sharing read-only state per worker is legal and intended, not an
  oversight.
- **Per-worker mutable solve workspace** (a scratch buffer, a nonlinear
  solver's iteration state) lives as ordinary cache fields and is duplicated —
  not aliased — per worker, so concurrent workers never race on it.
- **Per-item/per-QP state that must persist across sweeps** (a retained
  factorization, a converged local value used as the next sweep's guess)
  lives in an [`ItemStates`](@ref) cache field, which `duplicate_for_device`
  deliberately aliases too: entries are indexed by item, and the cell
  partition assigns each item to exactly one worker at a time, so the aliased
  slots a worker touches are disjoint from every other worker's.

`assemble_cell!` kernels are pure evaluations at fixed internal state. A local
nonlinear solve that EVOLVES state — anything whose result must survive to the
next sweep or the next step — belongs in [`condense_cell!`](@ref) only, never
inline inside a kernel.

An inline local solve of a *history-free* (stateless) implicit relation stays
legitimate inside a kernel: if its root is a pure function of the gathered
`args` alone, it is recomputable from `args` at any later time, including from
postprocessing ([`evaluate_quadrature!`](@ref) is the route to re-run it
outside a sweep). Differentiating such an inline iteration by AD is then the
kernel author's responsibility — accuracy is limited by the inner solve's own
tolerance; the clean upgrade once that limit matters is an analytic tangent
via the implicit function theorem, the same shape [`condense_cell!`](@ref)
uses for a corrector.

Inline STATE EVOLUTION has no error to catch it: a kernel receives `q` as an
already-gathered slot and never writes the global solution vector, so an
element that re-solves from committed history inside its `assemble_cell!`
kernel has nowhere to put the new state — it silently freezes the committed
state across steps instead of advancing it. State that must survive a step
goes through [`condense_cell!`](@ref). State kept in a private cache field
instead of an [`ItemStates`](@ref) store puts commit/rollback/freshness
entirely on the element author: [`rollback_state!`](@ref)/
[`commit_state!`](@ref) and the freshness guard in [`item_state`](@ref) cannot
see private storage.

## Analytic kernels

Analytic kernels are an opt-in, declared through a compile-time trait so no
`hasmethod` probe reaches the hot loop:

```julia
FerriteOperators.provides_analytic(::Type{<:MyCache}, ::FerriteOperators.JacobianKind) = true
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, cache::MyCache, args::CellArgs)
    # ... accumulate into req.K ...
end
```

There is exactly one root method for [`provides_analytic`](@ref), so a
specialization is always strictly more specific and a blanket declaration
cannot create an ambiguity. `setup_operator` checks trait against kernel per
element cache: a kind the trait claims without a matching `assemble_cell!`
method is a loud `ArgumentError` at setup, never a silent fallback to AD.

A cache lacking analytic coverage of some kind is wrapped in
[`ADElementCache`](@ref) at `setup_operator` time (`ad_backend =
ForwardDiffAD()` by default; `ad_backend = nothing` opts out). Per request the
decorator forwards to the inner's own kernel where declared and differentiates
the residual kernel otherwise — this is a construction-time, per-cache
resolution, not a per-cell branch: element files need not know decoration
exists. A composite wraps its non-analytic inners as ONE sub-composite rather
than each individually (naive per-inner wrapping costs one full seeding pass
per wrapped inner, worse than not wrapping at all once two or more inners need
it).

`provides_analytic` means exactly "there is a hand-written kernel", also on a
wrapped cache. Whether the RESOLVED cache serves a kind at all (hand kernel,
AD, or generic completion) is the internal `FerriteOperators.serves_kind`;
element authors only ever declare `provides_analytic`.

The available request types are [`ResidualRequest`](@ref),
[`JacobianRequest`](@ref), [`JacobianResidualRequest`](@ref) (the fused Newton
path), [`WeightedJacobianRequest`](@ref) (a scheme's combined matrix — see
[weighted Jacobians](operators.md#Weighted-Jacobians)), and the sensitivity
requests [`ParameterJacobianRequest`](@ref), [`ParameterVJPRequest`](@ref),
[`TimeSensitivityRequest`](@ref), [`StateJVPRequest`](@ref),
[`StateVJPRequest`](@ref).

## The `args` bundle

`args` is a [`CellArgs`](@ref) for cell kernels and a [`FacetArgs`](@ref) for
facet kernels — the same four fields, no supertype between the two.
Annotating the parameter (`args::CellArgs`) is permitted; kernels select on
the `(request, cache)` pair, never on `args`.

- `args.states` — NamedTuple of element-local state buffers, one per slot
  declared at setup (`setup_operator(...; slots = (:u, :uprev))`). A slot's
  *source* decides how it gathers: a plain vector reads `celldofs(cell)`,
  [`AffineRate`](@ref) reconstructs over that same field-space gather, and
  [`InternalSource`](@ref) restricts to a cell's condensed internal-dof range
  — the mechanism that makes a condensed element's internal state `q` an
  ordinary slot (see [Condensed elements](#Condensed-elements-(internal-variables))).
- `args.cell` — the geometry cache of the current item, read-only.
- `args.p` — the user parameter bag, produced by the overridable
  [`query_cell_parameters`](@ref) (facets get their own
  [`query_facet_parameters`](@ref) per facet). Configuration only: time lives
  in `ctx`, history in slots.
- `args.ctx` — the per-sweep solver scalars, i.e. the
  [`TimeIntegrationContext`](@ref) `(t, Δt, γ̃)` read through
  `evaluation_time(args.ctx)` and `stage_scaling(args.ctx)`, or `nothing` for
  stationary problems. This is the one open channel: a scheme with richer
  per-sweep scalars passes its own context type. `γ̃` is the *normalized*
  local stage interval of the element-local internal-variable problem — see
  its docstring for the exact contract and for why it is **not** a rate slope.

A derivative sweep rebuilds `args` with one field replaced instead of
re-deriving it from scratch: [`with_states`](@ref), [`with_parameters`](@ref)
and [`with_context`](@ref).

## Values and reinitialization

Elements own their values objects (`CellValues` etc.) and implement
[`reinit_values!`](@ref): the mandatory two-arg method reinitializes all of
them; specializing the kind-dispatched three-arg form reinitializes only what
that request needs — an element may carry several values objects, and not
every request needs all of them. The loop owns the geometry cache reinit only.

Kernels are pure evaluation: repeated kernel invocations within one sweep (AD
chunk passes, split Jacobian-then-residual fallbacks) do not reinitialize
again. Facet kernels reinitialize their own `FacetValues` per facet, since the
local facet index is theirs.

## Facets

The framework owns the facet loop: it walks each cell's facets, gates on
[`is_facet_in_cache`](@ref), queries facet parameters per facet, and hands the
sweep's request to the facet kernel.

```julia
function FerriteOperators.assemble_facet!(req::ResidualRequest, cache::MyFacetCache, args::FacetArgs, lfi::Int)
    reinit!(cache.fv, args.cell, lfi)
    # accumulate into req.r; args.p came from query_facet_parameters(cache, cell, lfi, p)
end
```

Facet contributions have no AD fallback in any sweep: a surface cache serves
the sweep's request analytically or not at all. A cache serving a fused
weighted sweep therefore implements `assemble_facet!` for
[`WeightedJacobianRequest`](@ref); per-slot facet kernels are not composed
behind the driver's back.

## Elements with global dofs

Some unknowns belong to no cell: the macroscopic strain of a stress-driven RVE,
a lumped chamber pressure coupling a whole surface, a Lagrange multiplier
enforcing an integral constraint. Ferrite gives them dofs through
`AlgebraicVariable`, numbered after the spatial dofs and never appearing in
`celldofs`. An element declares which of them its local system carries:

```julia
FerriteOperators.global_dofs(m::MyIntegrator, sdh::SubDofHandler) = algebraic_dofs(sdh.dh, :εbar)
```

The declaration lives on the **integrator**, one per subdomain, shared by the
volumetric and the boundary kernel of that subdomain, and is resolved once at
setup — before any cache is built. The local layout is then a contract:

```
[ celldofs(cell) ; the declared global dofs, in declaration order ]
```

so the tail occupies [`global_dof_range`](@ref). The framework passes no extra
channel and `CellArgs`/`FacetArgs` keep their four fields; an element cache
resolves its own range at setup and stores it:

```julia
struct MyCache{CV, AV} <: AbstractVolumetricElementCache
    cv::CV
    av::AV                    # Ferrite's AlgebraicValues for the variable
    range_u::UnitRange{Int}
    range_ε::UnitRange{Int}   # where the global dofs sit in the local system
end

function FerriteOperators.setup_element_cache(m::MyIntegrator, sdh::SubDofHandler)
    ip = Ferrite.getfieldinterpolation(sdh, m.field_name)
    cv = CellValues(getquadraturerule(m.qrc, sdh), ip, FerriteOperators.geometric_subdomain_interpolation(sdh))
    return MyCache(cv, AlgebraicValues(m.variable), dof_range(sdh, m.field_name), global_dof_range(m, sdh))
end

function FerriteOperators.assemble_cell!(req::ResidualRequest, c::MyCache, args::CellArgs)
    uₑ = args.states.u                                   # length ndofs_per_cell + 3
    ε̄  = sum(uₑ[J] * algebraic_basis_value(c.av, jε) for (jε, J) in pairs(c.range_ε))
    for qp in 1:getnquadpoints(c.cv)
        dΩ = getdetJdV(c.cv, qp)
        σ  = c.E ⊡ (ε̄ + function_symmetric_gradient(c.cv, qp, uₑ, c.range_u))
        for (iu, I) in pairs(c.range_u)
            req.r[I] += (shape_symmetric_gradient(c.cv, qp, iu) ⊡ σ) * dΩ
        end
        for (iε, I) in pairs(c.range_ε)
            Eᵢ = algebraic_basis_value(c.av, iε)
            req.r[I] += (Eᵢ ⊡ σ - c.σ̄ ⊡ Eᵢ) * dΩ
        end
    end
end
```

Everything the engine sizes follows the declaration: `Ke`, `re`, the slot
buffers, the sensitivity buffers, and the ForwardDiff seeds are padded by the
declared count, so an AD fallback differentiates the FULL augmented system —
the `ε̄`-`ε̄` block included. `allocate_element_matrix` and friends keep meaning
the FIELD space, so an element overriding them states `ndofs_per_cell` and
never the augmented size.

**The sparsity is the caller's declaration.** FerriteOperators does not infer
which entries the coupling creates from the dof declaration — whether every
cell couples to the variable, or only one facet set, is a modelling statement
only the caller can make. It travels as a Ferrite coupling descriptor on the
operator specification:

```julia
coupling = CellCoupling(1:getncells(grid); algebraic_coupling = ((:u, :εbar), (:εbar, :εbar)))
spec     = StandardOperatorSpecification(; algebraic_couplings = (coupling,))
strategy = AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), SequentialCPUDevice())
```

A missing descriptor is not silent: it surfaces as Ferrite's
missing-sparsity-entry error on the first assembly.

Setup raises the restrictions this layout implies. [`ColoredScheduling`](@ref)
is rejected — coloring works by giving no two items of a color a shared dof,
and a declared global dof is shared by *every* item of its subdomain, so no
coloring isolates it; the parallel route is the atomic scatter of
[`SequentialScheduling`](@ref) under a parallel device. The
[`ElementAssembly`](@ref) form is rejected too, its per-element dof maps being
built from `celldofs`, and so is patch assembly, whose patch-local dof map is
built the same way: the declared tail would have no patch-local number and be
dropped. A condensed element cache without an analytic `Consistent` Jacobian
kernel is rejected as well — the generic combination
`∂F/∂ū|_q + ∂F/∂q · dq/dū` reads a corrector block spanning the field space
while the AD partials span the augmented system. The declaration itself is
validated too: in bounds, without duplicates, and — sampled on the subdomain's
first cell — disjoint from `celldofs`, since a dof appearing in both head and
tail would receive every contribution twice.

!!! note "Requires Ferrite's mesh-free algebraic variables"
    See the canonical capability note under [Algebraic
    terms](#Algebraic-terms-(items-with-no-mesh-support)); the same Ferrite
    `AlgebraicVariable` vocabulary carries this declaration. [`global_dofs`](@ref) defaults to `()`, so an operator declaring
    none is unaffected.

## Facet items

The [facet loop](#Facets) is the **fused route**: the boundary term rides the
cell sweep, and every cell's every facet is tested against
[`is_facet_in_cache`](@ref). A term supported on a small fraction of the
boundary — a tying constraint on an endocardial surface, a contact patch — pays
that (cells × facets) rediscovery on every Newton iterate to find a set it
already knows. Such a term declares its facets instead, and gets its own
traversal:

```julia
FerriteOperators.facet_items(m::MyIntegrator, sdh::SubDofHandler) = getfacetset(get_grid(sdh.dh), "endocardium")
FerriteOperators.setup_facet_item_cache(m::MyIntegrator, sdh::SubDofHandler) = MyFacetCache(...)
```

**Same kernels, two routes.** [`setup_facet_item_cache`](@ref) returns the same
kind of object [`setup_boundary_cache`](@ref) does, and the driver calls the
same `assemble_facet!(req, cache, args::FacetArgs, lfi)` methods over the same
[`FacetArgs`](@ref). There is no second kernel entry point and no second args
record: moving a term between the two routes is a change of *declaration*, with
zero element edits. The two coexist — one operator can carry a whole-boundary
Neumann term on the fused route and a facet-set term as items.

Which route to use:

| | fused route ([`setup_boundary_cache`](@ref)) | facet items ([`facet_items`](@ref)) |
|---|---|---|
| term supported on most cells (whole-boundary Neumann) | fine | no gain |
| term supported on few facets of many cells | pays the per-facet gate | the declared set is the traversal |
| sensitivity coverage (∂F/∂θ, ∂F/∂t, state products) | **omitted** | **included** |
| scheduling | the cell sweep's | its own partition |

**The declared set IS the gate.** [`is_facet_in_cache`](@ref) is *not* consulted
on this route. That gate exists so the fused sweep can rediscover membership
while walking; here membership is the item list, so a cache whose gate and
declaration disagree contributes on what was declared. A facet whose cell is
not in `sdh.cellset`, a local facet index the cell does not have, and a facet
declared twice are all setup errors.

**One item is one owning cell with all of its declared facets** — never one
facet. Two facets of a cell therefore share one local system, are assembled
against one geometry cache and scatter once, and can never land on different
workers. Both partitions follow from the owning cells:
[`SequentialScheduling`](@ref) hands out one chunk and lets the atomic scatter
resolve the dofs neighbouring cells' facets share; [`ColoredScheduling`](@ref)
colors the owning cells with Ferrite's cell coloring restricted to that set,
which makes the barrier promise hold for facet items exactly as it holds for
cells.

The local system is **owning-cell-shaped**: `Ke`, `re` and the slot buffers are
sized like the cell family's, including the [`global_dofs`](@ref) padding. That
is what makes the tying shape work — a facet term whose local system is
`[celldofs(cell); one lumped pressure dof]` writes its coupling rows and
columns through the augmented tail, and the engine scatters through the
augmented dof vector:

```julia
FerriteOperators.global_dofs(m::MyTyingIntegrator, sdh::SubDofHandler) = algebraic_dofs(sdh.dh, :p)

function FerriteOperators.assemble_facet!(req::JacobianResidualRequest, c::MyTyingCache, args::FacetArgs, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    P = first(c.range_p)                       # `global_dof_range`, resolved at setup
    p = args.states.u[P]
    # ... req.r[I] += p * ∫Nᵢ ; req.K[I, P] += ∫Nᵢ ; req.K[P, I] += ∫Nᵢ ...
end
```

**Sensitivities.** A facet-item term *does* enter the sensitivity sweeps — the
fused route's omission (see [Sensitivities](operators.md#Sensitivities)) does
not apply here. The no-AD-fallback rule of [Facets](#Facets) still holds, so
the cache must implement `assemble_facet!` for the sensitivity request itself.
Declaring the kind (`setup_operator(...; requests = (ParameterJacobianKind,))`)
makes setup demand that kernel loudly, instead of letting a sweep reach a
missing method.

Two things a facet item deliberately does not do. It never calls
[`reinit_values!`](@ref) — a facet kernel reinitializes its own `FacetValues`
for the local facet index it was handed, on this route exactly as on the fused
one. And it contributes nothing to [`evaluate_functional`](@ref): a facet
functional is a surface integral over the declared set, which needs a facet
hook of its own next to [`evaluate_cell_functional`](@ref); the family has
none.

## Algebraic terms (items with no mesh support)

A term whose rows belong to no cell at all — a 0D circulation model's own
equations, a lumped balance, an `AlgebraicCoupling`-only block — is its own item
family. **An item of this family IS a set of global dofs and nothing else:** no
geometry cache, no values object, no quadrature. Two declarations on the
integrator introduce it:

```julia
FerriteOperators.algebraic_items(m::MyIntegrator, dh::DofHandler) =
    [[only(algebraic_dofs(dh, :p1)), only(algebraic_dofs(dh, :p2))] for _ in 1:nchambers]

FerriteOperators.setup_algebraic_cache(m::MyIntegrator, dh::DofHandler) = MyChamberCache(m.parameters)
```

[`algebraic_items`](@ref) lists one dof vector per item, in the order the local
system uses; [`setup_algebraic_cache`](@ref) builds **one** cache serving them
all — the analogue of one element cache per `SubDofHandler` serving all its
cells. It has no silent fallback: declaring items without it is a setup error.

!!! note "Requires Ferrite's mesh-free algebraic variables"
    The whole vocabulary these features are spelled in — `AlgebraicVariable`,
    `algebraic_dofs`, `AlgebraicValues`, `AlgebraicCoupling`, and the
    `algebraic_coupling`/`algebraic_couplings` keywords — is not in the
    registered Ferrite 1.6. There the declaration fails loudly at its own call
    site before any FerriteOperators surface is reached; every FO-side
    declaration defaults to `()`, so operators declaring none are unaffected.

Kernels dispatch through [`assemble_algebraic!`](@ref), the family's own entry
point next to `assemble_cell!` and `assemble_facet!`, and receive an
[`AlgebraicArgs`](@ref) — the same four fields as `CellArgs` with the
[`AlgebraicItem`](@ref) where the geometry cache would be:

```julia
function FerriteOperators.assemble_algebraic!(req::ResidualRequest, c::MyChamberCache, args::AlgebraicArgs)
    k = args.item.index                # which of the declared items this is
    Δ = args.states.u[1] - args.states.u[2]
    req.r[1] += c.conductances[k] * Δ - args.p * c.sources[k]
    req.r[2] -= c.conductances[k] * Δ - args.p * c.sources[k]
end
```

`args.item` carries `index` and `dofs`; the local buffers are `n × n` and `n`
for an item of `n` dofs, and `args.states` gathers through the item's dofs like
any other item's gather. The mandatory-residual rule and the analytic/AD
resolution are the cell family's, so ∂F/∂θ of a 0D model comes out of the same
ForwardDiff seeding.

Items of one declaration must be **uniformly sized**, which is what keeps a
worker's local buffers fixed-size; the check is a setup error naming the
offending item. Items usually *share* dofs (several rows on one lumped
unknown), and that is what fixes the scheduling: [`SequentialScheduling`](@ref)
puts the whole family in one chunk and lets the atomic scatter resolve the
collisions, while [`ColoredScheduling`](@ref) — whose promise is that no two
items of a barrier share a dof — can only run **one item per barrier**. The
partition is derived, not rejected.

The sparsity is the caller's here too. Entries between two algebraic variables
travel as an `AlgebraicCoupling`; diagonal entries are always allocated, so only
the off-diagonal ones need declaring:

```julia
spec = StandardOperatorSpecification(;
    algebraic_couplings = (CellCoupling(1:getncells(grid); algebraic_coupling = ((:u, :p1),)),
                           AlgebraicCoupling(; algebraic_coupling = ((:p1, :p2),))))
```

Two consequences of an item having no cell. An [`InternalSource`](@ref) slot
gathers the item's own [`internal_variable_range`](@ref) — EMPTY unless the
algebraic cache is itself condensed (below), which is what lets a condensed
cell element and a *stateless* algebraic term share one operator with the
algebraic kernel seeing a zero-length buffer for that slot. And a reduction
reaches the family but contributes nothing by default: a term with no mesh
support carries no volume, so [`evaluate_functional`](@ref) keeps summing the
cell contributions alone unless the cache implements
[`evaluate_algebraic_functional`](@ref).

### Condensed internal state on algebraic items

An algebraic item can carry its own condensed internal state `q` — a
circulation chamber's fast internal variables, eliminated item-locally exactly
like a condensed cell's per-quadrature-point state. Two declarations, the
algebraic-item analogues of [`get_number_of_internal_dofs_per_element`](@ref)
and [`condense_cell!`](@ref):

```julia
FerriteOperators.get_number_of_internal_dofs_per_algebraic_item(m::MyIntegrator, cache::MyChamberCache, items) =
    fill(1, length(items))   # uniform per provider, like the items' own dof count

function FerriteOperators.condense_algebraic!(cache::MyChamberCache, args::AlgebraicArgs, weights::NamedTuple)
    q = local_solve(args.states.u, args.states.qprev, stage_scaling(args.ctx))
    args.states.q[1] = q
    # ... store the corrector `dq/du` (and `dq/dθ`, if serving ∂F/∂θ) in an `ItemStates` field, keyed by `args.item.index` ...
    return CondensationReport(true, 1, 0, 0, -args.item.index, 0, 0.0, 1.0)
end
```

Both are queried/called only when [`has_internal_state`](@ref) holds for the
algebraic cache — a stateless cache changes nothing, and every existing
algebraic operator keeps behaving exactly as before. Internal dofs are
numbered into the SAME tail a condensed cell's are, cell block first:
`[ū | q_cells | q_items]` — cell ranges stay keyed by cellid
([`internal_variable_range`](@ref)`(ivh, cellid::Int)`), item ranges are a
SEPARATE method keyed by [`AlgebraicItem`](@ref)
(`internal_variable_range(ivh, args.item)`), deliberately not one `Int`-keyed
method shared by both — a cellid and an item index are unrelated integers, and
collapsing them onto one dispatch would silently return the wrong range
whenever they coincide. [`condense_internal!`](@ref)'s domain sweep reaches
both families unconditionally; `condense_algebraic!`'s report `worst_cell`
convention documents how a folded report tells a cell from an item apart.

**Analytic-first, no generic AD bootstrap.** A condensed CELL cache lacking
analytic coverage still gets generic AD paths (the
[`condensed_corrector`](@ref) and [`local_conditions!`](@ref) combinations
below). The algebraic-item family has none of them: an item's buffers are
sized from a dof count, so the framework builds no `:q` differentiation
configuration to combine against, and there is no cellid to key a corrector
store by either. A condensed algebraic cache is therefore admissible for a
`Consistent` sensitivity/Jacobian kind only by serving it analytically or by
declaring [`internal_state_insensitive`](@ref) — `setup_operator` rejects
anything else at setup, naming `assemble_algebraic!` and the remedy. That is
this family's standing limitation.

## The three item families side by side

| | cell items | facet items ([`facet_items`](@ref)) | algebraic items ([`algebraic_items`](@ref)) |
|---|---|---|---|
| an item is | one cell of the subdomain | one owning cell with all of its declared facets | one vector of global dofs, no mesh support |
| local system | `[celldofs(cell); global dofs]` | the same, owning-cell-shaped | `n × n` over the item's own `n` dofs |
| [`ColoredScheduling`](@ref) | Ferrite's cell coloring; rejected once [`global_dofs`](@ref) are declared | the owning cells' coloring, restricted to the declared set | derived as one item per barrier |
| sensitivities | analytic kernel or the [`ADElementCache`](@ref) fallback | included, but **analytic-only** | analytic kernel or the [`ADElementCache`](@ref) fallback |
| functional hook | [`evaluate_cell_functional`](@ref) | none — the family contributes nothing | [`evaluate_algebraic_functional`](@ref) |
| condensation | [`condense_cell!`](@ref) | not supported: `q` belongs to the cell family's item for that same cell | [`condense_algebraic!`](@ref) |

## Condensed elements (internal variables)

Elements with per-quadrature-point internal state append their unknowns after
the FE dofs (`u = [ū; q]`, managed by the [`InternalVariableHandler`](@ref)),
own their local stage problem, and are solved in two phases:

```julia
report = condense_internal!(op, weights, states, p, ctx)   # solves every q, stores correctors, writes the tail
update_linearization!(op, r, states, p, ctx)                # pure evaluation at frozen q
```

[`condense_internal!`](@ref) is the ONLY writer of `q`: it runs once over the
whole domain, solves each quadrature point's local problem in
[`condense_cell!`](@ref) — the one element hook allowed to evolve internal
state — writes the trial `q` into the
`[ū; q]` tail, and stores a corrector (an element-allocated
[`ItemStates`](@ref) cache field) that the `Consistent` correction mode reads.
Every evaluation sweep afterwards is a PURE function of `(ū, q, p, t)` at
frozen `q`; no sweep writes back. `q` is gathered through an
[`InternalSource`](@ref) slot like any other state — declared at setup
(`slots = (:u, :q, …)`) and sourced per call (`states = (u = u, q =
InternalSource(u), …)`).

Whether the corrector is stored at all is a construction-time election
([`CorrectorElection`](@ref)), because per-quadrature-point corrector storage
is the phase's binding cost at scale — gigabytes at 10⁶ cells for a tensor
slope:

```julia
SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q; corrector = Recompute())
```

[`Stored`](@ref) (the default) keeps the corrector per item;
[`Recompute`](@ref) keeps none and re-derives it from the converged `(u, q)`
where a kernel needs it, which is exact rather than approximate — the
corrector is a closed form in that pair. Elect `Recompute()` for memory-bound
ASSEMBLED sweeps and `Stored()` for action-style use, where every operator
application would otherwise re-derive the same corrector. Write the element so
the election is invisible to its kernels: read the corrector through ONE
access point that either reads the store or recomputes.

What the election costs beyond memory is the freshness guard and, possibly,
the kernel's input requirements. Under `Recompute()` there is no stamp, so
nothing detects a missing [`condense_internal!`](@ref) — the q-ordering
contract is unchanged, only its enforcement is gone. And a recomputing access
point reads whatever the closed form needs: a corrector re-derived from
`(u, q)` and the stage scaling makes `stage_scaling(args.ctx)` a requirement
of every Jacobian sweep, where the stored election reads the retained block and
needs no context at all — which is exactly what the shipped
`SimpleCondensedLinearViscoelasticity` does (see the
[example element reference](example-elements.md)).

A Jacobian-shaped kind's [`CorrectionMode`](@ref) (`Consistent`, the default,
or `FrozenQ`) selects the total `∂F/∂·|_q + ∂F/∂q · dq/d·` or the partial
`∂F/∂·|_q` alone; `FrozenQ` must always be spelled and is refused at
construction for the sensitivity kinds (a wrong gradient, unlike a wrong
iteration matrix, is never a legitimate election). Reading an uncondensed or
stale corrector throws, naming the cell; [`rollback_state!`](@ref) invalidates
every corrector the operator carries (a rejected trial's `q` is stale),
[`commit_state!`](@ref) does not (the committed point is the last condensed
point). [`condensed_update_linearization!`](@ref) is the fused convenience
entry point — condense, bail out on `!report.converged`, evaluate — that a
Newton loop calls once per trial point.

Declare [`has_internal_state`](@ref) for such caches — it governs the
sensitivity admissibility rules in
[Sensitivities](operators.md#Sensitivities): a kind with no `CorrectionMode`
is always the total, so a plain AD fallback — which differentiates a pure
kernel and therefore computes only the frozen-`q` partial — is missing the
correction unless the cache serves the kind analytically or declares it
[`internal_state_insensitive`](@ref).

A condensed cache still gets wrapped in [`ADElementCache`](@ref) when it lacks
some kind's analytic coverage, and the decorator's `Consistent` state
Jacobian/JacobianResidual then has a GENERIC path: it AD-differentiates the
pure residual seeding `ū` and `q` separately and combines them with the
`dq/dū` block, read through [`condensed_corrector`](@ref) — `Jₑ =
∂F/∂ū|_q + ∂F/∂q · dq/dū`. This is the getting-started path (bigger than the
compact Tier-1 corrector most analytic kernels read, since the framework needs
the completed `nq × ndofs` block); an element serving `Consistent` kinds
analytically never needs to implement it. `condensed_corrector` receives the
item's [`CellArgs`](@ref), so it serves either election — read the store, or
re-derive the block from `args.states`.

The parameter and time sensitivities have the same shape of generic path, out
of the element's LOCAL CONDITIONS rather than a stored block:

```julia
function FerriteOperators.local_conditions!(L, cache::MyCache, args)
    # L .= the residual form of the equations condense_cell! solved for q
end
```

Given [`local_conditions!`](@ref), the decorator differentiates it for
`∂L/∂q` — factorized once per item — plus `∂L/∂θ` and `∂L/∂t`, and closes the
implicit function theorem against the same `∂F/∂q` block: `dq/dθ =
−(∂L/∂q)⁻¹ ∂L/∂θ`, `dF/dθ = ∂F/∂θ|_q + ∂F/∂q · dq/dθ`. `L` is evaluated,
never solved, and must be eltype-generic — it is what gets differentiated, so
`q`, the parameters and the evaluation time all reach it Dual-valued (which
also means a parameter bag whose [`rebuild_parameters`](@ref) cannot return a
Dual-valued object rules the route out). Without the hook, and without an
analytic kernel or an [`internal_state_insensitive`](@ref) declaration, those
kinds keep the admissibility rejection.

!!! warning "Experimental surface"
    The local-model seam is a CANDIDATE contract: `local_conditions!`'s
    signature may change in a minor release. The assembled results of the
    kinds it admits are not affected.

The `∂F/∂q` block itself is available to a solver, not only to the decorator:
[`allocate_internal_jacobian`](@ref) builds the rectangular
`residual_size(op) × ndofs(ivh)` target and
[`update_internal_jacobian!`](@ref) fills it — the block a Schur-complement
consumer wants. Elements serve it through the analytic
`assemble_cell!(::JacobianRequest{:q}, …)` kernel or by ForwardDiff seeding of
the `:q` slot, which needs no admissibility guard: `q` is the seed itself, so
`Consistent` and `FrozenQ` coincide.

The algebraic-item family ([Algebraic terms](#Algebraic-terms-(items-with-no-mesh-support)))
condenses through the same [`condense_internal!`](@ref) sweep and the same
`[ū | q_cells | q_items]` tail, via its own hook
([`condense_algebraic!`](@ref)), and its items ride the same `∂F/∂q` target,
their column block sitting after the cell block. None of the generic routes of
this section reach it, for the reason that section states, and `JacobianKind{:q}`
is analytic-first there too.

## Composition

[`NonlinearCompositeIntegrator`](@ref) and its bilinear/linear siblings stack
several sub-integrators over one domain into a single element:

```julia
setup_operator(strategy, BilinearCompositeIntegrator(mass, diffusion), dh)
```

The request carries the buffers, so one generic fan-out serves every request
type, and each inner receives its own [`query_cell_parameters`](@ref) view.
Empty caches are dropped when the composition is built, so an all-empty
composition collapses to the empty cache and a single surviving cache is
returned unwrapped — the engine's empty-boundary fast path survives
composition. Composed inners must agree on their quadrature rule; a
`getnquadpoints` query on a disagreeing composite throws.

The scope bound is same-(context, sink) multiphysics on **one** domain: terms
evaluated at different contexts or scattered into different targets are
separate sweeps over separate integrators, and the type carries exactly one
field so there is nowhere to smuggle a per-inner context or weight. No values
objects are shared by construction — deliberate sharing stays an element-side
concern.

Construction is rejected loudly for an empty tuple, for a sub-integrator with
condensed internal state (composing condensed elements is not supported), and
for cross-sink mixes. A **bilinear** inner inside a nonlinear composite is
legitimate — the operator a bilinear form induces scatters into the same
matrix and residual — whereas an [`AbstractLinearIntegrator`](@ref) describes a
load form, whose sink is a vector alone, and never composes into a nonlinear or
bilinear operator. Nested composites are flattened at construction.

The inners share one local system, so they share its tail: a composite's
[`global_dofs`](@ref) are what its inners declare. Silent inners (the default
`()`) read the tail a declaring inner puts there; two inners declaring
*different* dofs is an `ArgumentError` at setup, there being no unambiguous
tail for the composed local system to have.

[`CompositeVolumetricElementCache`](@ref) and
[`CompositeSurfaceElementCache`](@ref) are the caches these integrators build,
and remain available for hand-built compositions.

Routing and composition compose in one order — a `*MultiDomainIntegrator`
whose values are composite integrators. A composite of routers is not
supported.

[`NonlinearMultiDomainIntegrator`](@ref) and its bilinear/linear siblings map
**volumetric cellset names** to integrators, so one operator can carry
different physics per subdomain. A name claims the subdomain whose cells lie
in that cellset, and it resolves that subdomain's element cache *and* its
boundary cache — facetset names take no part in routing. Resolution runs once
per operator setup; an unclaimed subdomain, an ambiguous claim, or a declared
name claiming nothing is an `ArgumentError` there, never a silently empty
contribution. It samples each subdomain's first cell, so the requirement that
a subdomain lie *entirely* within one declared cellset is an assumption in
production and a checked, cell-exact rejection under
[`FerriteOperators.debug_mode`](@ref).

## Functionals

```julia
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:energy}, cache::MyCache, args) =
    # return this cell's ∫ contribution (a Number or a Tensors tensor)

FerriteOperators.functional_value_type(::FunctionalKind{:energy}) = Float64

Φ = evaluate_functional(op, FunctionalKind(:energy), states, p, ctx)
```

Global reductions (energies for line searches, dissipation, quantities of
interest) are request kinds whose kernels *return* their cell contribution;
the engine sums per worker and reduces in a fixed order, so results are
deterministic for a fixed worker count. Cell items contribute through the hook
above and algebraic items through [`evaluate_algebraic_functional`](@ref);
facet items contribute nothing.

[`FerriteOperators.functional_value_type`](@ref) declares the type the
reduction accumulates in. It is **required under a parallel device** — the
per-worker partials are one typed array allocated before the batch runs, so an
undeclared kind evaluated on a `PolyesterDevice` is an `ArgumentError` naming
the trait. Sequentially it is optional: without it the first contributing cell
fixes the accumulator's type.

With the declaration each worker's fold starts at `zero(T)` — the reduction's
additive identity — and a kernel returning some other type fails loudly
instead of widening the accumulator. That identity is also what makes an empty
sum well-defined, which is why the two kinds of "nothing came back" resolve
differently:

| situation | when it is decided | result |
|---|---|---|
| the operator's partitions carry no items, or every subdomain's cache is an `EmptyVolumetricElementCache` | **structural**, checked before any cell runs | `ArgumentError` — misconfiguration, whatever the kind declares |
| the sweep runs and every kernel returns `nothing` | **data-dependent** | `zero(T)` when the value type is declared; `ArgumentError` when it is not |

## Unit-testing a kernel

Kernels are pure evaluation, so they can be called directly on a single cell
without an operator. Building the cell cache and the [`CellArgs`](@ref) by
hand is the supported testing seam:

```julia
cache = FerriteOperators.setup_element_cache(MyIntegrator(qrc, :u), sdh)

cc = Ferrite.CellCache(dh)
reinit!(cc, 1)                      # geometry for cell 1
reinit_values!(cache, cc)           # the element's own values objects

uₑ = rand(ndofs_per_cell(sdh))
rₑ = zeros(ndofs_per_cell(sdh))
args = CellArgs((u = uₑ,), cc, p, nothing)
assemble_cell!(ResidualRequest(rₑ), cache, args)
```

`CellArgs` is constructed positionally as `(states, cell, p, ctx)`; `ctx` is
whatever the kernel reads (`nothing` when it reads none). Pass further slots
as additional entries of the states NamedTuple.

## Example elements

Worked implementations of everything above live in
`FerriteOperatorsExampleElements`, a separate package under
`lib/FerriteOperatorsExampleElements` — one element per feature of the
contract: a bilinear form and its induced residual, a linear form, a nonlinear
element with analytic tangent, and condensed elements with per-quadrature-point
internal state — one with a linear local problem, one whose local problem is
nonlinear and communicates with the outer solver through the context and
element-cache channels. They are FerriteOperators' own test fixtures and are meant to
be read and copied. Add them to an environment with

```julia
Pkg.add(url = "https://github.com/termi-official/FerriteOperators.jl",
        subdir = "lib/FerriteOperatorsExampleElements")
```

Their docstrings are collected in the
[example element reference](example-elements.md).

The generic functions and types above are collected in the
[element API reference](element-api.md).
