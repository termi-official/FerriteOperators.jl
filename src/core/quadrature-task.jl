"""
    query_element_quadrature_data(element_cache, cell, ivh, q::QVector)

Return the per-element quadrature data buffer for the current cell.

The default implementation returns a mutable view into `q` for `cellid(cell)`,
so the user function `f(qe, ue, cell, element_cache, pe)` writes directly to
the global [`QVector`](@ref) with no extra copy. Override this method together
with [`store_quadrature_data!`](@ref) if your element needs a local staging buffer.
"""
query_element_quadrature_data(element, cell, ivh, q::QVector) = get_data_for_index(q, cellid(cell))

"""
    store_quadrature_data!(q::QVector, qe, cell, ivh, element_cache)

Copy the element-local quadrature results `qe` back into the global [`QVector`](@ref).

The default implementation is a no-op because `qe` is already a view into `q`
(see [`query_element_quadrature_data`](@ref)). Override if your element uses a
local staging buffer.
"""
store_quadrature_data!(q::QVector, qe, cell, ivh, element) = nothing

"""
    QuadratureEvaluationTask

Task that evaluates a user function at every quadrature point of every cell and
stores the result in a [`QVector`](@ref).

Fields:
- `f`   – user function `f(qe, ue, cell, element_cache, pe)`, called once per cell.
          `qe` is a mutable view into `q` for that cell; `ue` is the element-local
          unknown vector; `pe` is the element-local parameter slice.
- `u`   – global solution vector (passed to [`load_element_unknowns!`](@ref))
- `p`   – global parameter object (passed to [`query_element_parameters`](@ref))
- `q`   – output [`QVector`](@ref); shared across all workers (write access is safe
          because different cells own disjoint slices)
"""
@concrete struct QuadratureEvaluationTask
    f
    u
    p
    q
end
# q is the shared output — all workers write to disjoint cell slices, so it is not duplicated.
duplicate_for_device(device, task::QuadratureEvaluationTask) =
    QuadratureEvaluationTask(task.f, task.u, task.p, task.q)

"""
    QuadratureEvaluationWorkspace <: AbstractWorkspace

Per-worker scratch data for [`QuadratureEvaluationTask`](@ref).
"""
@concrete struct QuadratureEvaluationWorkspace <: AbstractWorkspace
    ue
    element
    cell
    ivh
end

Ferrite.reinit!(ws::QuadratureEvaluationWorkspace, cellid) = reinit!(ws.cell, cellid)

function duplicate_for_device(device::AbstractCPUDevice, ws::QuadratureEvaluationWorkspace)
    return create_quadrature_evaluation_workspace(
        duplicate_for_device(device, ws.element),
        ws.cell.dh,
        duplicate_for_device(device, ws.ivh),
    )
end

function create_quadrature_evaluation_workspace(element, sdh, ivh)
    return QuadratureEvaluationWorkspace(
        allocate_element_unknown_vector(element, sdh),
        element,
        CellCache(sdh),
        ivh,
    )
end

function execute_single_task!(task::QuadratureEvaluationTask, ws::QuadratureEvaluationWorkspace)
    uₑ = query_element_unknown_buffer(ws.element, ws.ue)
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)
    qₑ = query_element_quadrature_data(ws.element, ws.cell, ws.ivh, task.q)

    load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    task.f(qₑ, uₑ, ws.cell, ws.element, pₑ)
    store_quadrature_data!(task.q, qₑ, ws.cell, ws.ivh, ws.element)
end

"""
    QuadratureFerriteOperator

An operator for evaluating user-defined functions at quadrature points and storing
the results in a [`QVector`](@ref).

Build with [`setup_quadrature_operator`](@ref) and execute with
[`evaluate_quadrature!`](@ref).
"""
@concrete struct QuadratureFerriteOperator
    strategy
    subdomain_caches
    dh
    integrator
end

"""
    setup_quadrature_operator(strategy, integrator, dh) -> QuadratureFerriteOperator

Set up a [`QuadratureFerriteOperator`](@ref) that can be used with
[`evaluate_quadrature!`](@ref) to evaluate a function at all quadrature points.
"""
function setup_quadrature_operator(strategy, integrator, dh::AbstractDofHandler)
    element_caches = setup_elements(integrator, dh)
    ivh    = setup_internal_variable_handler(integrator, element_caches, dh)
    device = strategy.device

    subdomain_caches = [begin
        partition = compute_partition(strategy, sdh)
        n  = n_workers(strategy, device, partition)
        ws = create_quadrature_evaluation_workspace(element_cache, sdh, ivh)
        dc = setup_device_instances(device, ws, n)
        SubdomainCache(AssemblyDomain(sdh, ivh, element_cache, EmptySurfaceElementCache()), dc, partition)
    end for (sdh, element_cache) in zip(dh.subdofhandlers, element_caches)]

    return QuadratureFerriteOperator(strategy, subdomain_caches, dh, integrator)
end

"""
    evaluate_quadrature!(op::QuadratureFerriteOperator, q::QVector, u, p, f)

Evaluate `f(qe, ue, cell, element_cache, pe)` at every quadrature point
and store results in `q`.

- `qe` — mutable view into `q` for the current cell (length = nqp for that cell)
- `ue` — element-local unknowns loaded from `u`
- `cell` — [`CellCache`](@ref) for the current cell
- `element_cache` — element cache (user-provided subtype of
  [`AbstractVolumetricElementCache`](@ref))
- `pe` — element-local parameters derived from `p`
"""
function evaluate_quadrature!(op::QuadratureFerriteOperator, q::QVector, u, p, f)
    task = QuadratureEvaluationTask(f, u, p, q)
    execute_on_subdomains!(task, op.strategy, op.subdomain_caches)
end
