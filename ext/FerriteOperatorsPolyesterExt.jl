module FerriteOperatorsPolyesterExt

using FerriteOperators, Polyester

function FerriteOperators.execute_on_device!(task, device::FerriteOperators.PolyesterDevice, device_cache, items)
    (; chunksize) = device
    workspaces = device_cache
    num_items_max = maximum(length.(items))
    num_tasks_max = ceil(Int, num_items_max / chunksize)

    # TODO can we sneak this into the device cache?
    tasks = [FerriteOperators.duplicate_for_device(device, task) for tid in 1:num_tasks_max]

    for chunk in items
        num_items   = length(chunk)
        num_tasks   = ceil(Int, num_items / chunksize)
        @batch for tasksetid in 1:num_tasks
            # Query the local task and workspace
            local_task = tasks[tasksetid]
            local_ws   = workspaces[tasksetid]

            # Compute the range of tasks
            first_itemid = (tasksetid-1)*chunksize+1
            last_itemid  = min(num_items, tasksetid*chunksize)

            # These are the local tasks
            for itemid in first_itemid:last_itemid
                cellid = chunk[itemid]
                FerriteOperators.reinit!(local_ws, cellid)
                FerriteOperators.execute_single_task!(local_task, local_ws)
            end
        end
    end
end

end # module FerriteOperatorsPolyesterExt
