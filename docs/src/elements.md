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

This alone buys the assembled Jacobian, the fused Newton path, and all
sensitivities through the AD fallback.

The residual kernel must be eltype-generic in `eltype(args.states.*)`,
`eltype(args.p)`, and the context time — that is the entire AD contract.
Kernels never write global state; the geometry cache in `args.cell` is
read-only. Kernels select on the `(request, cache)` pair alone, never on
`args`, so annotating the parameter (`args::CellArgs`) is permitted; leaving
it unannotated works exactly the same.

An element is a scheme-agnostic integrand: it reads slot *values* and never
encodes a time discretization. [The layer contract](design.md) states where
each piece of information a kernel might want belongs.

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
[`condense_cell!`](@ref) — the element hook that replaces the local solve
elements used to run inside every kernel — writes the trial `q` into the
`[ū; q]` tail, and stores a corrector (an element-allocated
[`ItemStates`](@ref) cache field) that the `Consistent` correction mode reads.
Every evaluation sweep afterwards is a PURE function of `(ū, q, p, t)` at
frozen `q`; no sweep writes back. `q` is gathered through an
[`InternalSource`](@ref) slot like any other state — declared at setup
(`slots = (:u, :q, …)`) and sourced per call (`states = (u = u, q =
InternalSource(u), …)`).

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
is always the total, so a plain AD fallback (which computes only the
frozen-`q` partial, now that the kernel is pure) is missing the correction
unless the cache serves the kind analytically or declares it
[`internal_state_insensitive`](@ref).

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
for cross-sink mixes. A *bilinear* inner inside a nonlinear composite is
legitimate; a *linear* (load) form has a different sink and never composes
into a nonlinear or bilinear operator. Nested composites are flattened at
construction.

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
deterministic for a fixed worker count. Volumetric contributions only.

[`FerriteOperators.functional_value_type`](@ref) declares the type the
reduction accumulates in. It is **required under a parallel device** — the
per-worker partials are one typed array allocated before the batch runs, so an
undeclared kind evaluated on a `PolyesterDevice` is an `ArgumentError` naming
the trait. Sequentially it is optional: without it the first contributing cell
fixes the accumulator's type.

With the declaration each worker's fold starts at `zero(T)` — the reduction's
additive identity, so a worker that sees no contribution adds nothing — and a
kernel returning some other type fails loudly instead of widening the
accumulator.

Two kinds of "nothing came back" are kept apart, and the difference decides
whether you get a value or an error:

| situation | when it is decided | result |
|---|---|---|
| the operator's partitions carry no items, or every subdomain's cache is an `EmptyVolumetricElementCache` | **structural**, checked before any cell runs | `ArgumentError` — misconfiguration, whatever the kind declares |
| the sweep runs and every kernel returns `nothing` | **data-dependent** | `zero(T)` when the value type is declared; `ArgumentError` when it is not |

The second row is the consistency rule: an all-quiet sweep is an empty sum, and
an empty sum only has a value once its type is known. Declaring
`functional_value_type` is what makes it well-defined.

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
