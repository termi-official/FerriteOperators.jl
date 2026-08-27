"""
    BilinearFerriteOperator <: AbstractBilinearOperator

The operator [`setup_operator`](@ref) returns for an [`AbstractBilinearIntegrator`](@ref):
the assembled matrix `op.A` plus the [`AssemblyEngine`](@ref) and integrator
that built it. The matrix IS the Jacobian (see [`AbstractBilinearOperator`](@ref)).
"""
@concrete struct BilinearFerriteOperator <: AbstractBilinearOperator
    A
    engine
    integrator
end

"""
    update_operator!(op::BilinearFerriteOperator, p, ctx = nothing)

Assemble the operator's matrix `op.A` from the element kernels. `ctx` is the
sweep's context, read by kernels through [`evaluation_time`](@ref) and friends:
a coefficient like `ρ(x, t)` or `D(x, t)` needs one, a constant one does not.
"""
update_operator!(op::BilinearFerriteOperator, p, ctx = nothing) =
    assemble_into!(BilinearKind(), (op.A,), op, (;), p, ctx)

"""
    evaluate!(op::BilinearFerriteOperator, residual, states, p, ctx)
    evaluate!(op::BilinearFerriteOperator, residual, u, p)

Assemble `residual = A·u` from the element residual kernels — the action of the
operator induced by the bilinear form, evaluated without touching `op.A`. The
residual kernel is mandatory for every element cache (validated at setup), so
this entry point exists for every bilinear operator.
"""
evaluate!(op::BilinearFerriteOperator, residual::AbstractVector, states::NamedTuple, p, ctx) =
    assemble_into!(ResidualKind(), (residual,), op, states, p, ctx)
evaluate!(op::BilinearFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    evaluate!(op, residual, (u = u,), p, nothing)

mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::BilinearFerriteOperator) = eltype(op.A)
Base.size(op::BilinearFerriteOperator) = size(op.A)
Base.size(op::BilinearFerriteOperator, axis) = size(op.A, axis)
