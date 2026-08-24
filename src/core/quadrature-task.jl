####################################
## Quadrature evaluation as a request kind
####################################
#
# Per-quadrature-point evaluation runs through the SAME engine as assembly — no
# bespoke task system, no separate operator type, and the same `reinit_values!`
# hook as every other kind.

"""
    query_element_quadrature_data(element, cell, ivh, q::QVector)

Element-overridable query producing the mutable per-cell quadrature-data
slice the evaluation writes into. Defaults to the cell's slice of `q`.
"""
query_element_quadrature_data(element, cell, ivh, q::QVector) = get_range_for_cell(q, cellid(cell))

"""
    store_quadrature_data!(q::QVector, qe, cell, ivh, element)

Element-overridable write-back counterpart of
[`query_element_quadrature_data`](@ref), a no-op by default because the default
query hands out a mutable view.
"""
store_quadrature_data!(q::QVector, qe, cell, ivh, element) = nothing

"""
    QuadratureEvaluationKind(f, q, set)

Evaluate `f(uₑ, qp, cell, element_cache, pₑ)` at every quadrature point of
every cell (optionally restricted to `set`), storing the returned values in
the [`QVector`](@ref) `q`. `q` is shared across workers — different cells own
disjoint slices, so no duplication is needed.
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
    qₑ = query_element_quadrature_data(ws.element, ws.cell, ws.ivh, kind.q)
    reinit_values!(ws.element, ws.cell, kind)
    for qp in 1:getnquadpoints(ws.element)
        qₑ[qp] = kind.f(statesₑ.u, qp, ws.cell, ws.element, pₑ)
    end
    store_quadrature_data!(kind.q, qₑ, ws.cell, ws.ivh, ws.element)
    return nothing
end

"""
    evaluate_quadrature!(q::QVector, op, u, p, f, [set = nothing])

Evaluate `f(uₑ, qp, cell, element_cache, pₑ)` at every quadrature point and
store the returned values in `q`, using `op`'s assembly engine. With `set`
given, only its cells are evaluated and the remaining entries of `q` are left
untouched.
"""
function evaluate_quadrature!(q::QVector, op, u, p, f, set = nothing)
    # The sink is the QVector inside the kind — no scatter assembler.
    run_sweep!(QuadratureEvaluationKind(f, q, set), nothing, op, (u = u,), p, nothing)
    return q
end
