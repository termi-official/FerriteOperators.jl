abstract type AbstractDevice{ValueType, IndexType} end
abstract type AbstractCPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end
abstract type AbstractGPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end

value_type(::AbstractDevice{ValueType}) where ValueType = ValueType
index_type(::AbstractDevice{<:Any, IndexType}) where IndexType = IndexType


"""
    SequentialCPUDevice

Sequential algorithms on CPU.
"""
struct SequentialCPUDevice{ValueType, IndexType} <: AbstractCPUDevice{ValueType, IndexType}
end
SequentialCPUDevice() = SequentialCPUDevice{Float64, Int}()

function execute_on_device!(task, device::SequentialCPUDevice, device_cache, items)
    ws = _first_workspace(device_cache)
    for chunk in items
        for cellid in chunk
            reinit!(ws, cellid)
            execute_on_cell!(task, ws)
        end
    end
end


"""
    PolyesterDevice

Threaded algorithms via Polyester.jl .
"""
struct PolyesterDevice{ValueType, IndexType} <: AbstractCPUDevice{ValueType, IndexType}
    chunksize::IndexType
end
PolyesterDevice() = PolyesterDevice{Float64, Int}(32)
PolyesterDevice(i::Int) = PolyesterDevice{Float64, Int}(i)

function execute_on_device!(task, device::PolyesterDevice, device_cache, items)
    (; chunksize) = device
    workspaces = device_cache.task_local_caches
    num_items_max = maximum(length.(items))
    num_tasks_max = ceil(Int, num_items_max / chunksize)

    # TODO can we sneak this into the device cache?
    tasks = [duplicate_for_device(device, task) for tid in 1:num_tasks_max]

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
                reinit!(local_ws, cellid)
                execute_on_cell!(local_task, local_ws)
            end
        end
    end
end


"""
    CudaDevice

Please add CUDA.jl to your Project to make this device work.
"""
struct CudaDevice{ValueType, IndexType} <: AbstractGPUDevice{ValueType, IndexType}
    threads::Union{IndexType, Nothing}
    blocks::Union{IndexType, Nothing}
end

CudaDevice() = CudaDevice{Float32, Int32}(nothing, nothing)
CudaDevice(threads::IndexType, blocks::IndexType) where IndexType = CudaDevice{Float32, IndexType}(threads, blocks)

# KA compat
default_backend(::SequentialCPUDevice) = KA.CPU()
default_backend(::PolyesterDevice) = KA.CPU()
