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

# GPU-safe 2D view into a flat vector — replaces @view + reshape which fails on GPU
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
    return reshape(flat, (Int(nrows), Int(ncols)))
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

# Logic-agnostic grid-stride kernel — calls work(idx, tid) for each item.
KA.@kernel function _device_strided_kernel!(work, num_items::Ti) where {Ti <: Integer}
    tid = convert(Ti, KA.@index(Global, Linear))
    if tid <= num_items
        iter = DeviceStridedIterator(num_items, tid, convert(Ti, prod(KA.@ndrange)))
        for idx in iter
            work(idx, iter.tid)
        end
    end
end

# AbstractDeviceWork interface — every work type must implement:
abstract type AbstractDeviceWork end
getdevice(::T) where {T <: AbstractDeviceWork} = error("getdevice not implemented for $T")
getnitems(::T) where {T <: AbstractDeviceWork} = error("getnitems not implemented for $T")

function launch!(work::AbstractDeviceWork)
    device = getdevice(work)
    backend = KA.backend(device)
    Ti = index_type(device)
    n = convert(Ti, getnitems(work))
    kernel = _device_strided_kernel!(backend, Int(device.threads))
    kernel(work, n; ndrange = Int(device.blocks * device.threads))
    KA.synchronize(backend)
end

@concrete struct DeviceTaskWork <: AbstractDeviceWork
    device
    task
    u
    p
    device_cache
    items
end

getdevice(w::DeviceTaskWork) = w.device
getnitems(w::DeviceTaskWork) = length(w.items)

# work being executed on GPU.
@inline function (w::DeviceTaskWork)(idx, tid)
    taskid = w.items[idx]
    task_buffer = get_task_buffer_for_device(w.task, w.u, w.p, w.device_cache, tid, taskid)
    execute_task_on_single_cell!(w.task, task_buffer)
end

function execute_task_on_device!(task, device::AbstractGPUDevice, cache)
    ##TODO: Revisit, because this might be suboptimal design-wise.
    (; u, p, subdomain) = cache
    (; strategy_cache) = subdomain
    (; device_cache) = strategy_cache
    for items in get_items(task, cache)
        DeviceTaskWork(device, task, u, p, device_cache, items) |> launch!
    end
end


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

function resolve_device_config(device::D, dh::AbstractDofHandler) where {V, I, D <: AbstractGPUDevice{V,I}}
    iszero(device.threads) || iszero(device.blocks) || return device
    ncells  = getncells(get_grid(dh))
    tpb     = _compute_groupsize(device, ncells)
    nblocks = convert(I, cld(ncells, tpb))
    return D(tpb, nblocks)
end
