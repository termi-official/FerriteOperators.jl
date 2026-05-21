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
    threads::IndexType
    blocks::IndexType
end

CudaDevice() = CudaDevice{Float32, Int32}(Int32(0), Int32(0))
CudaDevice(threads::IndexType, blocks::IndexType) where IndexType = CudaDevice{Float32, IndexType}(threads, blocks)

"""
    RocDevice

Please add AMDGPU.jl to your Project to make this device work.
"""
struct RocDevice{ValueType, IndexType} <: AbstractGPUDevice{ValueType, IndexType}
    threads::IndexType
    blocks::IndexType
end

RocDevice() = RocDevice{Float64, Int32}(Int32(0), Int32(0))
RocDevice(threads::IndexType, blocks::IndexType) where IndexType = RocDevice{Float64, IndexType}(threads, blocks)

# TODO: remove once [https://github.com/JuliaGPU/CUDA.jl/pull/3095] is merged and released.
struct DeviceReshapedView{T, Ti <: Integer, D <: AbstractVector{T}} <: AbstractMatrix{T}
    data::D
    offset::Ti
    nrows::Ti
    ncols::Ti
end

@inline Base.size(v::DeviceReshapedView) = (Int(v.nrows), Int(v.ncols))
@inline Base.size(v::DeviceReshapedView, d::Int) = d == 1 ? Int(v.nrows) : Int(v.ncols)
@inline function Base.getindex(v::DeviceReshapedView, i::Integer, j::Integer)
    @inbounds v.data[v.offset + (j - 1) * v.nrows + i - 1]
end
@inline function Base.setindex!(v::DeviceReshapedView, val, i::Integer, j::Integer)
    @inbounds v.data[v.offset + (j - 1) * v.nrows + i - 1] = val
end

#CPU -> use normal calls
@inline function device_reshape_view(data::Vector, offset, nrows, ncols)
    flat = @view data[offset:(offset + nrows * ncols - 1)]
    return reshape(flat, (nrows, ncols))
end
@inline function device_reshape_view(data::AbstractVector, offset, nrows, ncols)
    return DeviceReshapedView(data, offset, nrows, ncols)
end

struct DeviceStridedIterator{Ti <: Integer}
    tid::Ti # thread id
    range::StepRange{Ti, Ti} #  this in the loop give idx, which is the index of the item to process and starts from tid and get incremented by stride.
end

function DeviceStridedIterator(num_items::Ti, tid::Ti, stride::Ti) where {Ti <: Integer}
    DeviceStridedIterator(tid, tid:stride:num_items)
end

Base.iterate(l::DeviceStridedIterator) = iterate(l.range)
Base.iterate(l::DeviceStridedIterator, state) = iterate(l.range, state)

# KA compat
KA.backend(::AbstractCPUDevice) = KA.CPU()
KA.backend(device::AbstractGPUDevice) = error("Load the GPU package associated with $(typeof(device)) (e.g. CUDA.jl for CudaDevice).")
KA.functional(::AbstractCPUDevice) = KA.functional(KA.CPU())
KA.functional(device::AbstractGPUDevice) = KA.functional(KA.backend(device))

#TODO: revisit this section, needs further refinment.
const DEFAULT_GPU_THREADS_PER_BLOCK = 256
function _compute_groupsize(device::AbstractGPUDevice, ncells::Integer)
    Ti = index_type(device)
    max_tpb = iszero(device.threads) ? Ti(DEFAULT_GPU_THREADS_PER_BLOCK) : device.threads
    return convert(Ti, min(ncells, max_tpb))
end

function compute_total_nthreads(device::AbstractGPUDevice, ncells::Integer)
    Ti = index_type(device)
    threads_per_block = _compute_groupsize(device, ncells)
    nblocks           = convert(Ti, cld(ncells, threads_per_block))
    return convert(Ti, nblocks * threads_per_block)
end

total_nthreads(device::AbstractGPUDevice) = device.threads * device.blocks

# TODO: do we need `make_device`? 
make_device(::Type{D}, threads, blocks) where {D <: AbstractGPUDevice} = D(threads, blocks)
make_sequential_device(::Type{D}) where {V, I, D <: AbstractGPUDevice{V,I}} = make_device(D, one(I), one(I))

function resolve_device_config(device::D, dh::AbstractDofHandler) where {V, I, D <: AbstractGPUDevice{V,I}}
    iszero(device.threads) || iszero(device.blocks) || return device
    ncells  = getncells(get_grid(dh))
    tpb     = _compute_groupsize(device, ncells)
    nblocks = convert(I, cld(ncells, tpb))
    return make_device(D, tpb, nblocks)
end
