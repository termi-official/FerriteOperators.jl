@concrete struct LinearFerriteOperator <: AbstractLinearOperator
    b
    engine
    integrator
end

update_operator!(op::LinearFerriteOperator, p) =
    assemble_into!(LinearKind(), (op.b,), op, nothing, p)
