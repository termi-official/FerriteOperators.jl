####################################
## Assembly requests (element interface v2)
####################################

"""
    TimeIntegrationContext(t, Δt, γ̃)

Solver-controlled scalars the framework must understand. `t` is the evaluation
time (Dual-typed during ∂F/∂t sweeps), `Δt` the physical step size for
reference, and `γ̃` the stage scaling of the local internal-variable problem.
`γ̃` is deliberately distinct from `Δt`: collapsing them makes rate-coupled
materials silently wrong under any scheme but backward Euler.
"""
struct TimeIntegrationContext{T}
    t::T
    Δt::T
    γ̃::T
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

The v2 volumetric kernel entry point. Elements opt in via
[`implements_v2_kernels`](@ref) and must at least provide the
[`ResidualRequest`](@ref) method; every other request falls back to automatic
differentiation of the residual kernel unless [`provides_analytic`](@ref)
declares an analytic method.
"""
function assemble_cell! end

"""
    implements_v2_kernels(::Type{CacheType}) -> Bool

Transitional trait: `true` iff the element cache implements the v2
[`assemble_cell!`](@ref) request protocol. Defaults to `false`, routing the
cache through the legacy arity-dispatched `assemble_element!` interface. Dies
with the legacy interface.
"""
implements_v2_kernels(::Type) = false

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
    implements_v2_kernels(T) || return nothing
    hasmethod(assemble_cell!, Tuple{ResidualRequest, T, KernelArgs}) || throw(ArgumentError(
        "$(T) opts into the v2 kernel protocol (`implements_v2_kernels`) but implements no " *
        "`assemble_cell!(::ResidualRequest, ::$(nameof(T)), ::KernelArgs)` method. The residual " *
        "kernel is mandatory: it is the basis for AD-derived Jacobians and sensitivities."))
    return nothing
end

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
