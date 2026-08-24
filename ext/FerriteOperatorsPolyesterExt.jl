module FerriteOperatorsPolyesterExt

using FerriteOperators, Polyester

function FerriteOperators.execute_on_device!(task, device::FerriteOperators.PolyesterDevice, workspaces, items)
    (; chunksize) = device
    num_items_max = maximum(length.(items))
    num_tasks_max = ceil(Int, num_items_max / chunksize)

    # TODO preallocate this
    tasks = [FerriteOperators.duplicate_for_device(device, task) for tid in 1:num_tasks_max]

    for chunk in items
        num_items   = length(chunk)
        num_tasks   = ceil(Int, num_items / chunksize)
        @batch for tasksetid in 1:num_tasks
            local_task = tasks[tasksetid]
            local_ws   = workspaces[tasksetid]

            first_itemid = (tasksetid-1)*chunksize+1
            last_itemid  = min(num_items, tasksetid*chunksize)

            for itemid in first_itemid:last_itemid
                cellid = chunk[itemid]
                FerriteOperators.reinit!(local_ws, cellid)
                FerriteOperators.execute_single_task!(local_task, local_ws)
            end
        end
    end
end

function FerriteOperators.reduce_on_device(task, device::FerriteOperators.PolyesterDevice, workspaces, items)
    (; chunksize) = device
    T = FerriteOperators.functional_value_type(task.kind)
    T === Nothing && throw(ArgumentError(
        "$(nameof(typeof(task.kind))) does not declare `functional_value_type`, which the " *
        "parallel route requires: the per-taskset partials are one typed array allocated " *
        "before the batch runs, so the reduction's value type has to be known up front. " *
        "Declare it — `FerriteOperators.functional_value_type(::$(typeof(task.kind))) = Float64` " *
        "— or evaluate on a `SequentialCPUDevice`, whose fold takes the type from the first " *
        "contributing item."))
    num_items_max = maximum(length.(items))
    num_tasks_max = ceil(Int, num_items_max / chunksize)

    # TODO preallocate this
    tasks = [FerriteOperators.duplicate_for_device(device, task) for tid in 1:num_tasks_max]
    # One partial per taskset, carried across the barriers so a taskset's whole
    # contribution folds in one sequence. Seeded with the reduction's additive
    # identity, so a taskset that contributes nothing hands back `zero(T)`.
    partials = zeros(T, num_tasks_max)

    for chunk in items
        num_items = length(chunk)
        num_tasks = ceil(Int, num_items / chunksize)
        @batch for tasksetid in 1:num_tasks
            local_task = tasks[tasksetid]
            local_ws   = workspaces[tasksetid]

            first_itemid = (tasksetid-1)*chunksize+1
            last_itemid  = min(num_items, tasksetid*chunksize)

            partials[tasksetid] = FerriteOperators.fold_items(
                local_task, local_ws, view(chunk, first_itemid:last_itemid), partials[tasksetid])
        end
    end

    # Fixed taskset order, so the result is deterministic for a fixed worker count.
    total = zero(T)
    for tasksetid in 1:num_tasks_max
        total += partials[tasksetid]
    end
    return total
end

end # module FerriteOperatorsPolyesterExt
