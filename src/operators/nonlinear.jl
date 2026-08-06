"""
    LinearizedFerriteOperator(J, caches)

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_element! -> update jacobian/residual contribution with internal state variables
"""
@concrete struct LinearizedFerriteOperator <: AbstractNonlinearOperator
    J
    engine
    integrator
end

# Interface
update_linearization!(op::LinearizedFerriteOperator, u::AbstractVector, p) =
    assemble_into!(JacobianKind(), (op.J,), op, u, p)
update_linearization!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    assemble_into!(JacobianResidualKind(), (op.J, residual), op, u, p)
residual!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    assemble_into!(ResidualKind(), (residual,), op, u, p)

"""
    update_parameter_jacobian!(B, op, u, p)

Assemble the parameter Jacobian ∂F/∂θ into `B` (`residual_size(op) × nθ`),
evaluated at the trial state `u`. θ is the flat parameter view defined by
[`parameter_vector`](@ref)/[`rebuild_parameters`](@ref); elements provide
analytic [`ParameterJacobianRequest`](@ref) kernels or fall back to AD of
their residual. Never writes back into `u`.
"""
# Sensitivity sweeps differentiate the residual at fixed local state; a
# condensed element's residual contains an implicit function (the local
# solve), whose correct treatment needs implicit differentiation (see
# references/implicit-ad-plasti.jl). Reject loudly instead of returning a
# silently wrong adjoint.
function _check_sensitivity_supported(op)
    unknown_size(op) == residual_size(op) || throw(ArgumentError(
        "Sensitivity sweeps are not yet supported for operators with condensed internal " *
        "variables: the element-local solves require implicit differentiation, which is " *
        "not implemented. See the v2 plan and references/implicit-ad-plasti.jl."))
    return nothing
end

function update_parameter_jacobian!(B::AbstractMatrix, op::LinearizedFerriteOperator, u::AbstractVector, p)
    _check_sensitivity_supported(op)
    nθ = length(parameter_vector(p))
    size(B) == (residual_size(op), nθ) || throw(DimensionMismatch(
        "expected B of size $((residual_size(op), nθ)), got $(size(B))"))
    fill!(B, zero(eltype(B)))
    assembler = ParameterJacobianAssembler{eltype(B), typeof(B), strategy_needs_atomic(op.engine.strategy)}(B)
    task = AssemblyTask(ParameterJacobianKind(), assembler, u, p)
    execute_on_subdomains!(task, op.engine)
    return B
end

"""
    parameter_vjp!(g, op, λ, u, p)

Accumulate the adjoint pullback `g = (∂F/∂θ)ᵀ λ` (length nθ) at the trial
state `u` without materializing ∂F/∂θ. Never writes back into `u`.
"""
function parameter_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, u::AbstractVector, p)
    _check_sensitivity_supported(op)
    length(g) == length(parameter_vector(p)) || throw(DimensionMismatch(
        "expected g of length $(length(parameter_vector(p))), got $(length(g))"))
    length(λ) == residual_size(op) || throw(DimensionMismatch(
        "expected λ of length $(residual_size(op)), got $(length(λ))"))
    fill!(g, zero(eltype(g)))
    # The VJP target is indexed by parameter, not by dof: coloring provides no
    # isolation here, so any parallel device needs atomic accumulation.
    atomic = !(op.engine.strategy.device isa SequentialCPUDevice)
    if atomic && op.engine.strategy.scheduling isa ColoredScheduling
        @warn "PerColorAssemblyStrategy provides no isolation for parameter-space " *
              "accumulation; the VJP scatter falls back to atomic adds." maxlog = 1
    end
    assembler = ParameterVJPAssembler{eltype(g), typeof(g), atomic}(g)
    task = AssemblyTask(ParameterVJPKind(λ), assembler, u, p)
    execute_on_subdomains!(task, op.engine)
    return g
end

"""
    time_sensitivity!(g, op, u, t)

Assemble the explicit time sensitivity ∂F/∂t into `g` (`residual_size(op)`)
at the trial state `u` and evaluation time `t`. Until the phase-2 context API
lands, `t` doubles as the parameter object handed to the elements (the
bare-time convention). Never writes back into `u`.
"""
function time_sensitivity!(g::AbstractVector, op::LinearizedFerriteOperator, u::AbstractVector, t)
    _check_sensitivity_supported(op)
    length(g) == residual_size(op) || throw(DimensionMismatch(
        "expected g of length $(residual_size(op)), got $(length(g))"))
    assembler = start_assemble(op.engine.strategy, g)
    task = AssemblyTask(TimeSensitivityKind(t), assembler, u, t)
    execute_on_subdomains!(task, op.engine)
    finalize_assembly!(assembler)
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
unknown_size(op::LinearizedFerriteOperator)  = ndofs(op.engine.dh) + ndofs(op.engine.subdomain_caches[1].domain.ivh)
