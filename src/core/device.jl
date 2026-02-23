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

function execute_task_on_device!(task, ::SequentialCPUDevice, cache)
    task_buffer = get_task_buffer(task, cache, 1)
    for chunk in get_items(task, cache)
        for taskid in chunk
            reinit!(task_buffer, taskid)
            execute_task_on_single_cell!(task, task_buffer)
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

function execute_task_on_device!(task, device::PolyesterDevice, cache)
    (; chunksize) = device
    itemsets = get_items(task, cache)
    num_items_max = maximum(length.(itemsets))
    num_tasks_max = ceil(Int, num_items_max / chunksize)

    # TODO can we sneak this into the device cache?
    tasks = [duplicate_for_device(device, task) for tid in 1:num_tasks_max]

    for items in itemsets
        num_items   = length(items)
        num_tasks   = ceil(Int, num_items / chunksize)
        @batch for tasksetid in 1:num_tasks
            # Query the local task and buffer
            local_task = tasks[tasksetid]
            local_task_buffer = get_task_buffer(local_task, cache, tasksetid)

            # Compute the range of tasks
            first_itemid = (tasksetid-1)*chunksize+1
            last_itemid  = min(num_items, tasksetid*chunksize)

            # These are the local tasks
            for itemid in first_itemid:last_itemid
                taskid = items[itemid]
                reinit!(local_task_buffer, taskid)
                execute_task_on_single_cell!(local_task, local_task_buffer)
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

"""
    RocDevice

Please add AMDGPU.jl to your Project to make this device work.
"""
struct RocDevice{ValueType, IndexType} <: AbstractGPUDevice{ValueType, IndexType}
    threads::Union{IndexType, Nothing}
    blocks::Union{IndexType, Nothing}
end

RocDevice() = RocDevice{Float64, Int32}(nothing, nothing)
RocDevice(threads::IndexType, blocks::IndexType) where IndexType = RocDevice{Float64, IndexType}(threads, blocks)

# Fallback for GPU: sequential CPU cell loop (used for tasks without a GPU kernel dispatch)
function execute_task_on_device!(task, device::AbstractGPUDevice, cache)
    task_buffer = get_task_buffer(task, cache, 1)
    for chunk in get_items(task, cache)
        for taskid in chunk
            reinit!(task_buffer, taskid)
            execute_task_on_single_cell!(task, task_buffer)
        end
    end
end

# KA compat
KA.backend(::AbstractCPUDevice) = KA.CPU()
KA.backend(device::AbstractGPUDevice) = error("Load the GPU package associated with $(typeof(device)) (e.g. CUDA.jl for CudaDevice).")
KA.functional(::AbstractCPUDevice) = KA.functional(KA.CPU())
KA.functional(device::AbstractGPUDevice) = default_backend(device) |> KA.functional
#TODO: remove when AMDGPU.jl creates new release that includes (https://github.com/JuliaGPU/AMDGPU.jl/pull/884)
KA.functional(device::RocDevice) = true 
