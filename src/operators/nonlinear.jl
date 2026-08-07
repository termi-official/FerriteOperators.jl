"""
    LinearizedFerriteOperator(J, caches)

A model for a function with its fully assembled linearization. Entry points
route request kinds through the shared assembly engine; element kernels are
the request-typed `assemble_cell!`/`assemble_facet!` methods.
"""
@concrete struct LinearizedFerriteOperator <: AbstractNonlinearOperator
    J
    engine
    integrator
end

# Interface. The states/ctx forms are canonical; the u-vector forms are
# conveniences for stationary problems (states = (u = u,), no context).
update_linearization!(op::LinearizedFerriteOperator, states::NamedTuple, p, ctx) =
    assemble_into!(JacobianKind(), (op.J,), op, states, p, ctx)
update_linearization!(op::LinearizedFerriteOperator, u::AbstractVector, p) =
    update_linearization!(op, (u = u,), p, nothing)
update_linearization!(op::LinearizedFerriteOperator, residual::AbstractVector, states::NamedTuple, p, ctx) =
    assemble_into!(JacobianResidualKind(), (op.J, residual), op, states, p, ctx)
update_linearization!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    update_linearization!(op, residual, (u = u,), p, nothing)
residual!(op::LinearizedFerriteOperator, residual::AbstractVector, states::NamedTuple, p, ctx) =
    assemble_into!(ResidualKind(), (residual,), op, states, p, ctx)
residual!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    residual!(op, residual, (u = u,), p, nothing)

# Call-time admissibility over all subdomain caches; the same per-cache check
# runs at setup for kinds declared via `setup_operator(...; requests)` — see
# `assert_sensitivity_admissible` for the rationale.
function _check_sensitivity_supported(op, kind)
    for sc in op.engine.subdomain_caches
        assert_sensitivity_admissible(typeof(sc.domain.element), kind)
    end
    return nothing
end

"""
    update_parameter_jacobian!(B, op, u, p)

Assemble the parameter Jacobian ∂F/∂θ into `B` (`residual_size(op) × nθ`),
evaluated at the trial state `u`. θ is the flat parameter view defined by
[`parameter_vector`](@ref)/[`rebuild_parameters`](@ref); elements provide
analytic [`ParameterJacobianRequest`](@ref) kernels or fall back to AD of
their residual. Never writes back into `u`.

!!! note
    The u-vector convenience forms evaluate at the stationary point
    `states = (u = u,)` with no time-integration context. Elements reading
    further slots (`uprev`, …) or `args.ctx` must use the states/ctx forms.
"""
function update_parameter_jacobian!(B::AbstractMatrix, op::LinearizedFerriteOperator, states::NamedTuple, p, ctx)
    _check_sensitivity_supported(op, ParameterJacobianKind())
    nθ = length(parameter_vector(p))
    size(B) == (residual_size(op), nθ) || throw(DimensionMismatch(
        "expected B of size $((residual_size(op), nθ)), got $(size(B))"))
    fill!(B, zero(eltype(B)))
    assembler = ParameterJacobianAssembler{eltype(B), typeof(B), dof_scatter_needs_atomic(op.engine.strategy)}(B)
    run_sweep!(ParameterJacobianKind(), assembler, op, states, p, ctx)
    return B
end
update_parameter_jacobian!(B::AbstractMatrix, op::LinearizedFerriteOperator, u::AbstractVector, p) =
    update_parameter_jacobian!(B, op, (u = u,), p, nothing)

"""
    parameter_vjp!(g, op, λ, u, p)

Accumulate the adjoint pullback `g = (∂F/∂θ)ᵀ λ` (length nθ) at the trial
state `u` without materializing ∂F/∂θ. Never writes back into `u`.
"""
function parameter_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, states::NamedTuple, p, ctx)
    _check_sensitivity_supported(op, ParameterVJPKind(λ))
    length(g) == length(parameter_vector(p)) || throw(DimensionMismatch(
        "expected g of length $(length(parameter_vector(p))), got $(length(g))"))
    length(λ) == residual_size(op) || throw(DimensionMismatch(
        "expected λ of length $(residual_size(op)), got $(length(λ))"))
    fill!(g, zero(eltype(g)))
    atomic = parameter_scatter_needs_atomic(op.engine.strategy)
    if atomic && op.engine.strategy.scheduling isa ColoredScheduling
        @warn "PerColorAssemblyStrategy provides no isolation for parameter-space " *
              "accumulation; the VJP scatter falls back to atomic adds." maxlog = 1
    end
    assembler = ParameterVJPAssembler{eltype(g), typeof(g), atomic}(g)
    run_sweep!(ParameterVJPKind(λ), assembler, op, states, p, ctx)
    return g
end
parameter_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, u::AbstractVector, p) =
    parameter_vjp!(g, op, λ, (u = u,), p, nothing)

"""
    state_jvp!(Jv, op, v, states, p, ctx)
    state_jvp!(Jv, op, v, u, p)

Matrix-free action of the state Jacobian: `Jv = (∂F/∂u)·v` at the trial
state, computed kernel-level (one directional-Dual sweep per cell for the AD
fallback; analytic [`StateJVPRequest`](@ref) kernels win per cache) — no
matrix is materialized anywhere. Never writes back into the caller's state.
Restricted to operators without condensed unknowns for now
(`unknown_size == residual_size`).
"""
function state_jvp!(Jv::AbstractVector, op::LinearizedFerriteOperator, v::AbstractVector, states::NamedTuple, p, ctx)
    unknown_size(op) == residual_size(op) || throw(ArgumentError(
        "state_jvp!/state_vjp! are not yet supported for operators with condensed unknowns."))
    _check_sensitivity_supported(op, StateJVPKind(v))
    length(Jv) == residual_size(op) || throw(DimensionMismatch(
        "expected Jv of length $(residual_size(op)), got $(length(Jv))"))
    length(v) == unknown_size(op) || throw(DimensionMismatch(
        "expected v of length $(unknown_size(op)), got $(length(v))"))
    assemble_into!(StateJVPKind(v), (Jv,), op, states, p, ctx)
    return Jv
end
state_jvp!(Jv::AbstractVector, op::LinearizedFerriteOperator, v::AbstractVector, u::AbstractVector, p) =
    state_jvp!(Jv, op, v, (u = u,), p, nothing)

"""
    state_vjp!(g, op, λ, states, p, ctx)
    state_vjp!(g, op, λ, u, p)

Matrix-free pullback of the state Jacobian: `g = (∂F/∂u)ᵀλ` at the trial
state — the action adjoint time stepping applies. Kernel-level (per-cell
gradient of `λₑ·rₑ` for the AD fallback; analytic [`StateVJPRequest`](@ref)
kernels win per cache). Never writes back into the caller's state. Restricted
to operators without condensed unknowns for now.
"""
function state_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, states::NamedTuple, p, ctx)
    unknown_size(op) == residual_size(op) || throw(ArgumentError(
        "state_jvp!/state_vjp! are not yet supported for operators with condensed unknowns."))
    _check_sensitivity_supported(op, StateVJPKind(λ))
    length(g) == unknown_size(op) || throw(DimensionMismatch(
        "expected g of length $(unknown_size(op)), got $(length(g))"))
    length(λ) == residual_size(op) || throw(DimensionMismatch(
        "expected λ of length $(residual_size(op)), got $(length(λ))"))
    assemble_into!(StateVJPKind(λ), (g,), op, states, p, ctx)
    return g
end
state_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, u::AbstractVector, p) =
    state_vjp!(g, op, λ, (u = u,), p, nothing)

"""
    ADSensitivity()

Default derivative method: ForwardDiff seeding through the residual kernel.
"""
struct ADSensitivity end

"""
    FiniteDifferenceSensitivity(h = cbrt(eps(Float64)))

Central-difference derivative method: evaluates the PRIMAL residual at
perturbed times on a protected copy of `u` (so trial write-back never leaks)
and forms `(F(t+h) − F(t−h)) / 2h`. Exact local solves, no Dual propagation —
admissible for condensed elements, where it yields the total t-derivative at
fixed `u` including the element-local state's response (as does AD, when
admissible). Accuracy O(h²).
"""
struct FiniteDifferenceSensitivity{T}
    h::T
end
FiniteDifferenceSensitivity() = FiniteDifferenceSensitivity(cbrt(eps(Float64)))

"""
    time_sensitivity!(g, op, states, t, ctx; method = ADSensitivity())
    time_sensitivity!(g, op, u, t; method = ADSensitivity())

Assemble the time sensitivity ∂F/∂t into `g` (`residual_size(op)`) at the
trial state, never writing back into the caller's state. Until the phase-2
context seeding lands, `t` doubles as the parameter object handed to the
elements (the bare-time convention).

Method hierarchy: with [`ADSensitivity`](@ref) (default), each element cache
that declares an analytic [`TimeSensitivityRequest`](@ref) kernel is used
directly, and every other cache falls back to ForwardDiff through its
residual kernel — analytic kernels always win per cache.
[`FiniteDifferenceSensitivity`](@ref) is an operator-level override that
differences primal residual evaluations and therefore BYPASSES analytic
sensitivity kernels; prefer it only where AD is inadmissible (condensed
internal state without analytic kernels or insensitivity declarations).
"""
function time_sensitivity!(g::AbstractVector, op::LinearizedFerriteOperator, states::NamedTuple, t, ctx; method = ADSensitivity())
    length(g) == residual_size(op) || throw(DimensionMismatch(
        "expected g of length $(residual_size(op)), got $(length(g))"))
    return _time_sensitivity!(method, g, op, states, t, ctx)
end
time_sensitivity!(g::AbstractVector, op::LinearizedFerriteOperator, u::AbstractVector, t; method = ADSensitivity()) =
    time_sensitivity!(g, op, (u = u,), t, nothing; method)

function _time_sensitivity!(::ADSensitivity, g, op, states, t, ctx)
    _check_sensitivity_supported(op, TimeSensitivityKind(t))
    assemble_into!(TimeSensitivityKind(t), (g,), op, states, t, ctx)
    return g
end

function _time_sensitivity!(method::FiniteDifferenceSensitivity, g, op, states, t, ctx)
    # Primal evaluations at perturbed times (bare-time p convention) — no
    # internal-state admissibility check needed: the local solves run exactly
    # as in a normal residual evaluation. The u slot is copied so the
    # condensation trial write-back of the perturbed evaluations never leaks
    # into the caller's state; ctx is held fixed.
    h  = method.h * max(one(t), abs(t))
    uw = copy(states.u)
    statesw = merge(states, (u = uw,))
    rp = similar(g); residual!(op, rp, statesw, t + h, ctx)
    copyto!(uw, states.u)
    rm = similar(g); residual!(op, rm, statesw, t - h, ctx)
    g .= (rp .- rm) ./ (2h)
    return g
end

"""
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear form to the vector `in`.
"""
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)
(op::LinearizedFerriteOperator)(residual, u, p) = residual!(op, residual, u, p)
Base.eltype(op::LinearizedFerriteOperator) = eltype(op.J)
Base.size(op::LinearizedFerriteOperator) = size(op.J)
Base.size(op::LinearizedFerriteOperator, axis) = size(op.J, axis)

residual_size(op::LinearizedFerriteOperator) = ndofs(op.engine.dh)
unknown_size(op::LinearizedFerriteOperator)  = ndofs(op.engine.dh) + ndofs(op.engine.ivh)
