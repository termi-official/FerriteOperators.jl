"""
    EvaluationFerriteOperator(engine, integrator)

The payload-free operator: an [`AssemblyEngine`](@ref) and the integrator that
built it, with no matrix and no vector. Built by
[`setup_evaluation_operator`](@ref).

It serves every entry point that reads `op.engine` and writes nowhere the
operator owns — [`evaluate_functional`](@ref) and [`evaluate_quadrature!`](@ref)
in particular, whose sinks are the caller's value and the caller's
[`QVector`](@ref). The assembly entry points (`update_operator!`,
`update_linearization!`, `evaluate!`) have no target here and say so instead of
allocating one.
"""
@concrete struct EvaluationFerriteOperator
    engine
    integrator
end

@noinline _reject_payload_entry(entry::Symbol) = throw(ArgumentError(
    "`$entry` assembles into an operator's matrix or vector, and an " *
    "`EvaluationFerriteOperator` holds neither: `setup_evaluation_operator` builds the element " *
    "caches and the engine alone, for the evaluation-only entry points (`evaluate_functional`, " *
    "`evaluate_quadrature!`). Set the term up with `setup_operator` to assemble it."))

update_operator!(::EvaluationFerriteOperator, p, ctx = nothing) = _reject_payload_entry(:update_operator!)
update_linearization!(::EvaluationFerriteOperator, args...) = _reject_payload_entry(:update_linearization!)
evaluate!(::EvaluationFerriteOperator, args...) = _reject_payload_entry(:evaluate!)
