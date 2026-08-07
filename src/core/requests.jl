####################################
## Assembly requests (element interface v2)
####################################

"""
    TimeIntegrationContext(t, Δt, γ̃)

Solver-controlled scalars the framework must understand. `t` is the evaluation
time (Dual-typed during ∂F/∂t sweeps), `Δt` the physical step size for
reference, and `γ̃` is defined as the scalar in the canonical local-stage form

    q = q_ref + γ̃ · g(·, q)

of the element-local internal-variable problem (the reference state `q_ref`
is dof-shaped and flows through a slot, solver-folded for multistep schemes).

!!! warning
    `γ̃` is NOT the rate-reconstruction slope of any state slot. Under
    backward Euler the two happen to be reciprocals (`slope = 1/Δt`,
    `γ̃ = Δt`), so writing `1/γ̃` for a rate slope is accidentally right under
    BE and silently wrong under every other scheme (Newmark:
    `slope = γ/(βΔt)` while `γ̃ = Δt`). Rate slopes belong to the slot that
    carries the reconstruction, never to the context.
"""
struct TimeIntegrationContext{T}
    t::T
    Δt::T
    γ̃::T
end
function TimeIntegrationContext(t, Δt, γ̃)
    tp, Δtp, γ̃p = promote(t, Δt, γ̃)
    return TimeIntegrationContext{typeof(tp)}(tp, Δtp, γ̃p)
end
# Deliberately no 1-arg convenience constructor: a defaulted γ̃ (e.g. zero)
# would silently break local internal-variable stage problems that scale by
# it. All three scalars are the solver's explicit statement.

# Framework code touches contexts only through these accessors, so schemes
# with richer per-sweep scalars can pass their own context types.
"Evaluation time of the sweep this context belongs to."
evaluation_time(ctx::TimeIntegrationContext) = ctx.t
"Rebuild `ctx` with the evaluation time replaced by `t̃` (Dual-typed in ∂F/∂t sweeps)."
with_time(ctx::TimeIntegrationContext, t̃) = TimeIntegrationContext(t̃, oftype(t̃, ctx.Δt), oftype(t̃, ctx.γ̃))

"""
    AbstractAssemblyRequest

What a kernel is asked to compute for one cell. Kernels dispatch on the
request type, never on argument shapes, so kernel sets for different state
slots can coexist without ambiguity.
"""
abstract type AbstractAssemblyRequest end

"Accumulate the local residual into `r`."
struct ResidualRequest{V <: AbstractVector} <: AbstractAssemblyRequest
    r::V
end

"Accumulate ∂F/∂slot into `K`, slot ∈ (:u, :du, :v, :a, …)."
struct JacobianRequest{slot, M <: AbstractMatrix} <: AbstractAssemblyRequest
    K::M
end
JacobianRequest{slot}(K::M) where {slot, M <: AbstractMatrix} = JacobianRequest{slot, M}(K)

"Accumulate ∂F/∂u and the residual in one sweep (the Newton hot path)."
struct JacobianResidualRequest{M <: AbstractMatrix, V <: AbstractVector} <: AbstractAssemblyRequest
    K::M
    r::V
end

"Accumulate the dense local parameter Jacobian ∂Fₑ/∂θ into `B` (ndofsₑ × nθ)."
struct ParameterJacobianRequest{M <: AbstractMatrix} <: AbstractAssemblyRequest
    B::M
end

"Accumulate the local adjoint pullback (∂Fₑ/∂θ)ᵀλₑ into `g` (length nθ)."
struct ParameterVJPRequest{V <: AbstractVector, L <: AbstractVector} <: AbstractAssemblyRequest
    g::V
    λₑ::L
end

"Accumulate the explicit time sensitivity ∂Fₑ/∂t into `g`."
struct TimeSensitivityRequest{V <: AbstractVector} <: AbstractAssemblyRequest
    g::V
end

"""
    KernelArgs(states, cell, p, scratch, ctx)

The argument bundle passed to v2 element kernels. `states` is a `NamedTuple`
of cell-local slot buffers (e.g. `(u = uₑ,)`), `cell` the read-only geometry
cache, `p` the user parameter bag, `scratch` per-worker scratch space, and
`ctx` the [`TimeIntegrationContext`](@ref) (or `nothing` while the legacy
parameter channel still carries time).
"""
struct KernelArgs{States <: NamedTuple, Cell, P, Scratch, Ctx}
    states::States
    cell::Cell
    p::P
    scratch::Scratch
    ctx::Ctx
end

"""
    assemble_cell!(req::AbstractAssemblyRequest, cache, args::KernelArgs)

The v2 volumetric kernel entry point. Elements must at least provide the
[`ResidualRequest`](@ref) method (validated at setup); every other request falls back to automatic
differentiation of the residual kernel unless [`provides_analytic`](@ref)
declares an analytic method.
"""
function assemble_cell! end

####################################
## Query protocol
####################################

"""
    unwrap_parameters(p) -> p′

Solver-side wrapper trait: a solver that must wrap the user parameter bag
defines one unwrapping rule here instead of every element handling the
wrapper. Defaults to identity.
"""
unwrap_parameters(p) = p

"""
    query_cell_parameters(cache, cell, p)

Element-overridable query producing the element-local parameter view `pₑ`
handed to volumetric kernels. The default applies [`unwrap_parameters`](@ref)
and passes the bag through. Parameter layouts (parameter fields) gather their
per-element views through this seam.
"""
query_cell_parameters(cache, cell, p) = unwrap_parameters(p)

"""
    query_facet_parameters(cache, cell, local_facet_index, p)

Facet analogue of [`query_cell_parameters`](@ref) — boundary caches get their
own parameter query per facet instead of reusing the volumetric object.
"""
query_facet_parameters(cache, cell, local_facet_index, p) = unwrap_parameters(p)

"""
    provides_analytic(::Type{CacheType}, kind) -> Bool

`true` iff the v2 element cache implements `assemble_cell!` analytically for
the given request *kind* singleton (`JacobianKind()`, `ParameterJacobianKind()`,
…). Everything except the mandatory residual defaults to `false`, i.e.
AD-from-residual.

There is deliberately exactly ONE root method (with the residual default as a
runtime branch): specializations are therefore always strictly more specific,
so blanket declarations like
`provides_analytic(::Type{<:MyCache}, kind) = true` cannot create dispatch
ambiguities, and there is no request-type-parameter matching to get subtly
wrong. Kernel/trait consistency is validated once at operator setup
([`validate_element_cache`](@ref)).
"""
provides_analytic(::Type, kind) = kind isa ResidualKind

"""
    validate_element_cache(cache)

Setup-time consistency check for v2 element caches: a cache that opts into the
request protocol must implement the mandatory [`ResidualRequest`](@ref)
kernel. Runs once per subdomain at `setup_operator` time — a typo'd port fails
loudly here instead of silently assembling through the wrong path.
"""
function validate_element_cache(cache)
    T = typeof(cache)
    hasmethod(assemble_cell!, Tuple{ResidualRequest, T, KernelArgs}) || throw(ArgumentError(
        "$(T) implements no `assemble_cell!(::ResidualRequest, ::$(nameof(T)), ::KernelArgs)` " *
        "method. The residual kernel is mandatory: it is the basis for AD-derived Jacobians " *
        "and sensitivities."))
    return nothing
end

"""
    declare_scratch(cache) -> NamedTuple of nullary constructors

Element-side scratch declaration: each entry is instantiated once per worker
and reaches kernels via `args.scratch`. Solvers declare their own entries via
`setup_operator(...; scratch = (name = () -> ...,))`; both land merged in the
workspace. This replaces smuggling solver state through model trees or
parameter bags.
"""
declare_scratch(cache) = (;)

####################################
## Reserved item shapes (contract space only — see the plan §2.1b)
####################################

"""
    InterfaceKernelArgs

RESERVED: the N-participant argument bundle for interface/pair items (DG
facet pairs as the `N = 2` case, contact pairs, non-local one-to-many
couplings with an indexed collection of secondary sides). Each participant
carries its own states/geometry/restriction; `p`, `scratch`, and
per-participant contexts ride alongside. Implemented with the phase-4
interface work — the name and shape are reserved now so nothing forecloses
them.
"""
abstract type InterfaceKernelArgs end

####################################
## Differentiable parameter protocol
####################################

"""
    parameter_vector(p) -> AbstractVector

Flat vector view θ of the differentiable parameters in `p`. Together with
[`rebuild_parameters`](@ref) this is the seam through which parameter
sensitivities are seeded. Implement both for custom parameter types.
"""
parameter_vector(p::Real) = SVector(p)
parameter_vector(p::AbstractVector) = p
parameter_vector(::T) where T = throw(ArgumentError(
    "No `parameter_vector` method for parameter type $T. Implement " *
    "`parameter_vector`/`rebuild_parameters` to enable parameter sensitivities."))

"""
    rebuild_parameters(p, θ) -> typeof-compatible parameters

Reconstruct a parameter object structurally equal to `p` with the
differentiable entries replaced by `θ` (possibly Dual-valued).
"""
rebuild_parameters(::Real, θ) = θ[1]
rebuild_parameters(::AbstractVector, θ) = θ
rebuild_parameters(::T, θ) where T = throw(ArgumentError(
    "No `rebuild_parameters` method for parameter type $T."))
