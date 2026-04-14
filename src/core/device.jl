abstract type AbstractDevice{ValueType, IndexType} end
abstract type AbstractCPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end
abstract type AbstractGPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end

value_type(::AbstractDevice{ValueType}) where ValueType = ValueType
index_type(::AbstractDevice{<:Any, IndexType}) where IndexType = IndexType

"""
    SequentialCPUDevice()

Sequential algorithms on CPU.
"""
struct SequentialCPUDevice{ValueType, IndexType} <: AbstractCPUDevice{ValueType, IndexType}
end
SequentialCPUDevice() = SequentialCPUDevice{Float64, Int}()

function execute_on_device!(task, device::SequentialCPUDevice, device_cache, items)
    ws = device_cache[1]
    for chunk in items
        for cellid in chunk
            reinit!(ws, cellid)
            execute_single_task!(task, ws)
        end
    end
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

function execute_on_device!(task, device::AbstractGPUDevice, device_cache, items)
    throw(ArgumentError(
        "GPU assembly is not yet implemented for $(typeof(device)). " *
        "Implement execute_on_device!, setup_device_cache, and n_workers for this device type."
    ))
end

"""
    CudaDevice(threads, blocks)

GPU device using CUDA.jl. Load CUDA.jl and implement the required device methods to use this.
"""
struct CudaDevice{ValueType, IndexType} <: AbstractGPUDevice{ValueType, IndexType}
    threads::Union{IndexType, Nothing}
    blocks::Union{IndexType, Nothing}
end

CudaDevice() = CudaDevice{Float32, Int32}(nothing, nothing)
CudaDevice(threads::IndexType, blocks::IndexType) where IndexType = CudaDevice{Float32, IndexType}(threads, blocks)
