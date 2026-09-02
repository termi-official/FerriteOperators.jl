####################################
## Quadrature evaluation as a request kind
####################################
#
# Per-quadrature-point evaluation runs through the SAME engine as assembly — no
# bespoke task system, no separate operator type, and the same `reinit_values!`
# hook as every other kind.

"""
    QuadratureEvaluationKind(f, q, set)

Evaluate `f(uₑ, qp, cell, element_cache, pₑ, ctx)` at every quadrature point of
every cell (optionally restricted to `set`), storing the returned values in
the [`QVector`](@ref) `q`. `q` is shared across workers — different cells own
disjoint slices, so no duplication is needed.

`ctx` is the sweep's context, the same channel the request-shaped kernels read
their time from ([`evaluation_time`](@ref)); `p` stays configuration.
"""
@concrete struct QuadratureEvaluationKind
    f
    q
    set
end

function execute_kind!(kind::QuadratureEvaluationKind, task, ws)
    kind.set !== nothing && cellid(ws.cell) ∉ kind.set && return
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    statesₑ = load_slots!(ws, task.states)
    # A mutable view into the shared `QVector`, so the loop below IS the write-back.
    qₑ = get_range_for_cell(kind.q, cellid(ws.cell))
    reinit_values!(ws.element, ws.cell, kind)
    for qp in 1:getnquadpoints(ws.element)
        qₑ[qp] = kind.f(statesₑ.u, qp, ws.cell, ws.element, pₑ, task.ctx)
    end
    return nothing
end

"""
    evaluate_quadrature!(q::QVector, op, u, p, f, [set = nothing]; ctx = nothing)

Evaluate `f(uₑ, qp, cell, element_cache, pₑ, ctx)` at every quadrature point and
store the returned values in `q`, using `op`'s assembly engine. With `set`
given, only its cells are evaluated and the remaining entries of `q` are left
untouched.

`ctx` reaches the kernel unchanged, so a per-quadrature-point evaluation reads
the time of a transient sweep exactly as an element kernel does — through
[`evaluation_time`](@ref) — and `nothing` (the default) is the stationary
evaluation.
"""
function evaluate_quadrature!(q::QVector, op, u, p, f, set = nothing; ctx = nothing)
    # The sink is the QVector inside the kind — no scatter assembler.
    run_sweep!(QuadratureEvaluationKind(f, q, set), nothing, op, (u = u,), p, ctx)
    return q
end
