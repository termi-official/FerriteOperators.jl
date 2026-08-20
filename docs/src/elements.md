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

function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::MyCache, args)
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
read-only. The `args` parameter stays **unannotated**: kernels select on the
`(request, cache)` pair alone, and an open parameter lets the element serve
any operator family's args type (setup warns about a concrete annotation).

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
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, cache::MyCache, args)
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

## The kernel-args channel protocol

`args` is any object carrying the channels below; [`KernelArgs`](@ref) is what
this package's operators build, not the contract. The contract is structural
(a Tables.jl-style protocol): there is no supertype, and kernels never
dispatch on the args.

- `args.states` — NamedTuple of element-local state buffers, one per slot
  declared at setup (`setup_operator(...; slots = (:u, :uprev))`). Slots are
  gathered through the element-overridable `load_element_unknowns!`, so
  condensed elements receive their full `[ū; q]` local layout for every slot.
- `args.cell` — the geometry cache of the current item, read-only.
- `args.p` — the user parameter bag, produced by the overridable
  [`query_cell_parameters`](@ref) (facets get their own
  [`query_facet_parameters`](@ref) per facet). Configuration only: time lives
  in `ctx`, history in slots.
- `args.scratch` — per-worker scratch declared by the solver
  (`setup_operator(...; scratch = (name = () -> ...,))`) and/or the element
  ([`declare_scratch`](@ref)).
- `args.ctx` — the per-sweep solver scalars, i.e. the
  [`TimeIntegrationContext`](@ref) `(t, Δt, γ̃)` read through
  `evaluation_time(args.ctx)` and `stage_scaling(args.ctx)`, or `nothing` for
  stationary problems. `γ̃` is the *normalized* local stage interval of the
  element-local internal-variable problem — see its docstring for the exact
  contract and for why it is **not** a rate slope.

Per-slot metadata is reserved protocol vocabulary: an args family may carry a
per-slot property, and `KernelArgs` carries none.

An operator family may build its own args type; it then implements the three
rebuild seams the framework re-seeds channels through —
[`with_states`](@ref), [`with_parameters`](@ref) and [`with_context`](@ref) —
as plain methods on that type. There is no abstract fallback: a family missing
a seam gets a `MethodError` on the sweep that needs it, never a silently
unseeded derivative.

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
function FerriteOperators.assemble_facet!(req::ResidualRequest, cache::MyFacetCache, args, lfi::Int)
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
the FE dofs (`u = [ū; q]`, managed by the [`InternalVariableHandler`](@ref))
and own their local stage problem: the previous state arrives through a slot
(e.g. `uprev`), the local solve scales by `stage_scaling(args.ctx)`, and the trial result
is written into the element-local `u` buffer — the framework propagates it
into the global trial vector. That per-evaluation write-back *is* the
condensation contract: `q(ū)` is refreshed at every trial evaluation, line
search included.

Declare [`has_internal_state`](@ref) for such caches — it governs the
sensitivity admissibility rules in
[Sensitivities](operators.md#Sensitivities).

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
without an operator. Building the cell cache and the [`KernelArgs`](@ref) by
hand is the supported testing seam:

```julia
cache = FerriteOperators.setup_element_cache(MyIntegrator(qrc, :u), sdh)

cc = Ferrite.CellCache(dh)
reinit!(cc, 1)                      # geometry for cell 1
reinit_values!(cache, cc)           # the element's own values objects

uₑ = rand(ndofs_per_cell(sdh))
rₑ = zeros(ndofs_per_cell(sdh))
args = KernelArgs((u = uₑ,), cc, p, nothing, nothing)
assemble_cell!(ResidualRequest(rₑ), cache, args)
```

`KernelArgs` is constructed positionally as `(states, cell, p, scratch, ctx)`;
`scratch` and `ctx` are whatever the kernel reads (`nothing` when it reads
neither). Pass further slots as additional entries of the states NamedTuple.

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
