abstract type AbstractDevice{ValueType, IndexType} end
abstract type AbstractCPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end
abstract type AbstractGPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end

value_type(::AbstractDevice{ValueType}) where ValueType = ValueType
index_type(::AbstractDevice{<:Any, IndexType}) where IndexType = IndexType


"""
    execute_on_device!(task, device, workspaces, items)

Execute a task on a device. Workspaces store the worker-specific
scratches for the tasks. The items is a simple nested list of task-specific
indices, where the tasks identified by the inner list are guaranteed to be
executable in parallel.
"""
function execute_on_device!(task, device::AbstractDevice, workspaces, items)
    throw(ArgumentError(
        "Task execution is not yet implemented for $(typeof(device)). " *
        "Implement execute_on_device! for this device type."
    ))
end

"""
    setup_device_instances(device, object, n_instances)

Create a device scratch by duplicating `object` for `n_instances` parallel workers.
For [`SequentialCPUDevice`](@ref), returns a 1-element tuple `(object,)`.
For threaded CPU devices, returns a `Vector` of `n_instances` independent copies
produced by [`duplicate_for_device`](@ref).
For GPU devices this should return a struct of arrays variant of `object`.
"""

function setup_device_instances(device::AbstractDevice, obj, n_instances)
    throw(ArgumentError(
        "Device cache setup is not yet implemented for $(typeof(device)). " *
        "Implement setup_device_instances for this device type."
    ))
end

function setup_device_instances(device::AbstractCPUDevice, obj, n_instances)
    return [duplicate_for_device(device, obj) for _ in 1:n_instances]
end



"""
    SequentialCPUDevice()

Sequential algorithms on CPU.
"""
struct SequentialCPUDevice{ValueType, IndexType} <: AbstractCPUDevice{ValueType, IndexType}
end
SequentialCPUDevice() = SequentialCPUDevice{Float64, Int}()

function execute_on_device!(task, device::SequentialCPUDevice, workspaces, items)
    workspace = workspaces[1]
    for chunk in items
        for cellid in chunk
            reinit!(workspace, cellid)
            execute_single_task!(task, workspace)
        end
    end
end

function setup_device_instances(::SequentialCPUDevice, obj, n_instances)
    return (obj,)
end



"""
    PolyesterDevice(chunksize)

Threaded algorithms via Polyester.jl. Load Polyester.jl to activate this device.
"""
struct PolyesterDevice{ValueType, IndexType} <: AbstractCPUDevice{ValueType, IndexType}
    chunksize::IndexType
end
PolyesterDevice() = PolyesterDevice{Float64, Int}(32)
PolyesterDevice(i::Int) = PolyesterDevice{Float64, Int}(i)



"""
    CudaDevice(threads, blocks)

GPU device using CUDA.jl. Load CUDA.jl to activate this device.
"""
struct CudaDevice{ValueType, IndexType} <: AbstractGPUDevice{ValueType, IndexType}
    threads::Union{IndexType, Nothing}
    blocks::Union{IndexType, Nothing}
end

CudaDevice() = CudaDevice{Float32, Int32}(nothing, nothing)
CudaDevice(threads::IndexType, blocks::IndexType) where IndexType = CudaDevice{Float32, IndexType}(threads, blocks)
