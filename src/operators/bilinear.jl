@concrete struct BilinearFerriteOperator <: AbstractBilinearOperator
    A
    engine
    integrator
end

update_operator!(op::BilinearFerriteOperator, p) =
    assemble_into!(BilinearKind(), (op.A,), op, (;), p, nothing)

"""
    residual!(op::BilinearFerriteOperator, residual, states, p, ctx)
    residual!(op::BilinearFerriteOperator, residual, u, p)

Assemble `residual = A·u` from the element residual kernels — the action of
the operator induced by the bilinear form, evaluated without touching `op.A`.
The residual kernel is mandatory for every element cache (validated at
setup), so this entry point is available for every bilinear operator.
"""
residual!(op::BilinearFerriteOperator, residual::AbstractVector, states::NamedTuple, p, ctx) =
    assemble_into!(ResidualKind(), (residual,), op, states, p, ctx)
residual!(op::BilinearFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    residual!(op, residual, (u = u,), p, nothing)

mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::BilinearFerriteOperator) = eltype(op.A)
Base.size(op::BilinearFerriteOperator) = size(op.A)
Base.size(op::BilinearFerriteOperator, axis) = size(op.A, axis)
