"""
    LinearFerriteOperator <: AbstractLinearOperator

The operator [`setup_operator`](@ref) returns for an [`AbstractLinearIntegrator`](@ref):
the assembled load vector `op.b` plus the [`AssemblyEngine`](@ref) and
integrator that built it.
"""
@concrete struct LinearFerriteOperator <: AbstractLinearOperator
    b
    engine
    integrator
end

"""
    update_operator!(op::LinearFerriteOperator, p, ctx = nothing)

Assemble the operator's vector `op.b` from the element kernels. `ctx` is the
sweep's context, read by kernels through [`evaluation_time`](@ref) and friends:
a load like `f(x, t)` needs one, a constant one does not.
"""
update_operator!(op::LinearFerriteOperator, p, ctx = nothing) =
    assemble_into!(LinearKind(), (op.b,), op, (;), p, ctx)
