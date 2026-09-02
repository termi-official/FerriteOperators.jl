module FerriteOperatorsPolyesterExt

using FerriteOperators, Polyester

# One iteration of the `@batch` loop per WORKER, and `workspaces` is the setup-time
# per-worker device cache, so `length(workspaces)` is the worker count both loops run
# with. A barrier's items are grouped into blocks of `min_items` and split into that
# many contiguous, block-aligned ranges: privacy is per worker index, never per thread,
# so the split is correct whatever Polyester schedules the iterations onto.
@inline function worker_items(w, num_workers, num_items, min_items)
    num_blocks  = cld(num_items, min_items)
    first_block = ((w - 1) * num_blocks) ÷ num_workers + 1
    last_block  = (w * num_blocks) ÷ num_workers
    return ((first_block - 1) * min_items + 1):min(num_items, last_block * min_items)
end

# Workers for one barrier: never more than the device cache holds, and never more than
# there are blocks to hand out.
@inline active_workers(workspaces, num_items, min_items) =
    min(length(workspaces), cld(num_items, min_items))

function FerriteOperators.execute_on_device!(task, device::FerriteOperators.PolyesterDevice, workspaces, items)
    (; min_items_per_worker) = device
    # The per-worker copy of the task's scatter target — the only part of a task that is
    # not shared read-only — is one duplicate per worker, so a sweep's cost here is the
    # worker count and not the item count.
    tasks = [FerriteOperators.duplicate_for_device(device, task) for _ in eachindex(workspaces)]

    for chunk in items
        num_items   = length(chunk)
        num_workers = active_workers(workspaces, num_items, min_items_per_worker)
        num_workers == 0 && continue
        @batch for w in 1:num_workers
            local_task = tasks[w]
            local_ws   = workspaces[w]

            for itemid in worker_items(w, num_workers, num_items, min_items_per_worker)
                cellid = chunk[itemid]
                FerriteOperators.reinit!(local_ws, cellid)
                FerriteOperators.execute_single_task!(local_task, local_ws)
            end
        end
    end
end

function FerriteOperators.reduce_on_device(task, device::FerriteOperators.PolyesterDevice, workspaces, items)
    (; min_items_per_worker) = device
    T = FerriteOperators.functional_value_type(task.kind)
    T === Nothing && throw(ArgumentError(
        "$(nameof(typeof(task.kind))) does not declare `functional_value_type`, which the " *
        "parallel route requires: the per-worker partials are one typed array allocated " *
        "before the batch runs, so the reduction's value type has to be known up front. " *
        "Declare it — `FerriteOperators.functional_value_type(::$(typeof(task.kind))) = Float64` " *
        "— or evaluate on a `SequentialCPUDevice`, whose fold takes the type from the first " *
        "contributing item."))
    num_workers_max = length(workspaces)

    tasks = [FerriteOperators.duplicate_for_device(device, task) for _ in eachindex(workspaces)]
    # One partial per worker, carried across the barriers so a worker's whole contribution
    # folds in one sequence. Seeded with the reduction's additive identity, so a worker
    # that contributes nothing hands back `zero(T)`.
    partials = zeros(T, num_workers_max)

    for chunk in items
        num_items   = length(chunk)
        num_workers = active_workers(workspaces, num_items, min_items_per_worker)
        num_workers == 0 && continue
        @batch for w in 1:num_workers
            local_task = tasks[w]
            local_ws   = workspaces[w]

            partials[w] = FerriteOperators.fold_items(
                local_task, local_ws, view(chunk, worker_items(w, num_workers, num_items, min_items_per_worker)),
                partials[w])
        end
    end

    # Fixed worker order, so the result is deterministic for a fixed worker count.
    total = zero(T)
    for w in 1:num_workers_max
        total += partials[w]
    end
    return total
end

end # module FerriteOperatorsPolyesterExt
