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
  `evaluation_time(args.ctx)` and `args.ctx.γ̃`, or `nothing` for stationary
  problems. `γ̃` is the *normalized* local stage interval of the
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
(e.g. `uprev`), the local solve scales by `args.ctx.γ̃`, and the trial result
is written into the element-local `u` buffer — the framework propagates it
into the global trial vector. That per-evaluation write-back *is* the
condensation contract: `q(ū)` is refreshed at every trial evaluation, line
search included.

Declare [`has_internal_state`](@ref) for such caches — it governs the
sensitivity admissibility rules in
[Sensitivities](operators.md#Sensitivities).

## Composition

[`CompositeVolumetricElementCache`](@ref) combines several elements over one
domain: the request carries the buffers, so one generic fan-out serves every
request type. Its scope is same-(context, sink) multiphysics on one domain —
terms evaluated at different contexts or scattered into different targets are
separate sweeps over separate integrators.

[`NonlinearMultiDomainIntegrator`](@ref) and its bilinear/linear siblings map
subdomains to integrators, so one operator can carry different physics per
`SubDofHandler`.

## Functionals

```julia
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:energy}, cache::MyCache, args) =
    # return this cell's ∫ contribution (a Number or a Tensors tensor)

Φ = evaluate_functional(op, FunctionalKind(:energy), states, p, ctx)
```

Global reductions (energies for line searches, dissipation, quantities of
interest) are request kinds whose kernels *return* their cell contribution;
the engine sums per worker and reduces in a fixed order, so results are
deterministic for a fixed worker count. Volumetric contributions only.

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

## Element API reference

```@autodocs
Modules = [FerriteOperators]
Pages = [
    "core/requests.jl",
    "core/element_interface.jl",
    "elements/composite_elements.jl",
    "elements/domain_elements.jl",
    "elements/simple_diffusion.jl",
    "elements/simple_mass.jl",
    "elements/simple_hyperelasticity.jl",
    "elements/simple_linear_viscoelasticity.jl",
]
```
