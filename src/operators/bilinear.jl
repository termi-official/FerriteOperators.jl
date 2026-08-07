@concrete struct BilinearFerriteOperator <: AbstractBilinearOperator
    A
    engine
    integrator
end

update_operator!(op::BilinearFerriteOperator, p) =
    assemble_into!(BilinearKind(), (op.A,), op, (;), p, nothing)

mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::BilinearFerriteOperator) = eltype(op.A)
Base.size(op::BilinearFerriteOperator) = size(op.A)
Base.size(op::BilinearFerriteOperator, axis) = size(op.A, axis)
