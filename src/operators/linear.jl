@concrete struct LinearFerriteOperator <: AbstractLinearOperator
    b
    strategy
    subdomain_caches
    dh
    integrator
end

update_operator!(op::LinearFerriteOperator, p) =
    assemble_into!(LinearKind(), (op.b,), op, nothing, p)
