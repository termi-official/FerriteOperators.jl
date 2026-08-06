"""
    LinearizedFerriteOperator(J, caches)

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_element! -> update jacobian/residual contribution with internal state variables
"""
@concrete struct LinearizedFerriteOperator <: AbstractNonlinearOperator
    J
    strategy
    subdomain_caches
    dh
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

residual_size(op::LinearizedFerriteOperator) = ndofs(op.subdomain_caches[1].domain.sdh.dh)
unknown_size(op::LinearizedFerriteOperator)  = ndofs(op.subdomain_caches[1].domain.sdh.dh) + ndofs(op.subdomain_caches[1].domain.ivh)
