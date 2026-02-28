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

CudaDevice() = CudaDevice{Float32, Int32}(Int32(256), Int32(20))
CudaDevice(threads::IndexType, blocks::IndexType) where IndexType = CudaDevice{Float32, IndexType}(threads, blocks)

"""
    RocDevice

Please add AMDGPU.jl to your Project to make this device work.
"""
struct RocDevice{ValueType, IndexType} <: AbstractGPUDevice{ValueType, IndexType}
    threads::IndexType
    blocks::IndexType
end

RocDevice() = RocDevice{Float64, Int32}(Int32(256), Int32(20))
RocDevice(threads::IndexType, blocks::IndexType) where IndexType = RocDevice{Float64, IndexType}(threads, blocks)

# Logic-agnostic grid-stride iterator
struct DeviceStridedIterator{Ti <: Integer}
    start::Ti
    stride::Ti
    stop::Ti
end

@inline function Base.iterate(iter::DeviceStridedIterator)
    iter.start > iter.stop ? nothing : (iter.start, iter.start)
end

@inline function Base.iterate(iter::DeviceStridedIterator, state)
    next = state + iter.stride
    next > iter.stop ? nothing : (next, next)
end

# GPU kernel: grid-stride loop, each thread processes one or more elements
KA.@kernel function _execute_task_kernel!(task, cache, @Const(items), num_items::Ti, nblocks::Ti, nthreads::Ti) where {Ti <: Integer}
    # KA.@index returns Int — convert to Ti for type-uniform DeviceStridedIterator
    thread_id = Ti(KA.@index(Global, Linear))
    local_tid = Ti(KA.@index(Local, Linear))
    groupsize = Ti(KA.@groupsize()[1])
    stride = nblocks * nthreads

    # Destructure the subdomain cache
    (; u, p, subdomain) = cache
    (; strategy_cache) = subdomain
    (; device_cache) = strategy_cache

    # Materialize local cache: Register → MArrays, Shared → @localmem views, Global → pool views
    local_cache = materialize(device_cache, thread_id, local_tid, groupsize)
    task_buffer = get_task_buffer_for_gpu(u, p, local_cache)

    for idx in DeviceStridedIterator(thread_id, stride, num_items)
        taskid = items[idx]
        reinit!(task_buffer, taskid)
        execute_task_on_single_cell!(task, task_buffer)
    end
end

function execute_task_on_device!(task, device::AbstractGPUDevice, cache)
    #Note: cache here encapsulates three things: 1) subdomain cache, 2) global params, 3) previous solution (case nonlinear)
    backend = default_backend(device)
    itemsets = get_items(task, cache) # aka what cells to iterate over

    Ti = index_type(device)
    for items in itemsets
        # Convert items to a plain GPU array (OrderedSet / Set not GPU-compatible)
        items_gpu = Adapt.adapt(backend, Ti.(collect(items)))

        num_items         = convert(Ti, length(items))
        threads_per_block = convert(Ti, min(num_items, device.threads))
        nblocks           = convert(Ti, cld(num_items, threads_per_block))

        # KA requires Int for groupsize/ndrange (not Int32)
        kernel = _execute_task_kernel!(backend, Int(threads_per_block))
        kernel(
            task,
            cache,
            items_gpu,
            num_items,
            nblocks,
            threads_per_block;
            ndrange = Int(nblocks * threads_per_block)
        )
        KA.synchronize(backend)
    end
end

# Adapt helper: CPU → identity, GPU → use backend adaptor (Array → ROCArray/CuArray)
_adapt_for_device(::AbstractCPUDevice, x) = x
_adapt_for_device(device::AbstractGPUDevice, x) = Adapt.adapt(default_backend(device), x)

# KA compat
default_backend(device::AbstractGPUDevice) = error("Load the GPU package associated with $(typeof(device)) (e.g. CUDA.jl for CudaDevice).")
KA.backend(::AbstractCPUDevice) = KA.CPU()
KA.backend(device::AbstractGPUDevice) = error("Load the GPU package associated with $(typeof(device)) (e.g. CUDA.jl for CudaDevice).")
KA.functional(::AbstractCPUDevice) = KA.functional(KA.CPU())
KA.functional(device::AbstractGPUDevice) = default_backend(device) |> KA.functional
#TODO: remove when AMDGPU.jl creates new release that includes (https://github.com/JuliaGPU/AMDGPU.jl/pull/884)
KA.functional(device::RocDevice) = true


## APIs for local buffer management ##
max_sharedmem_per_block(device::AbstractGPUDevice) = error("Load the GPU package associated with $(typeof(device)) (e.g. CUDA.jl for CudaDevice).")
max_registers_per_block(device::AbstractGPUDevice) = error("Load the GPU package associated with $(typeof(device)) (e.g. CUDA.jl for CudaDevice).")

# Compute threads per block — must match kernel launch in execute_task_on_device!
function _compute_groupsize(device::AbstractGPUDevice, ncells::Integer)
    Ti = index_type(device)
    return convert(Ti, min(ncells, device.threads))
end

# Compute total number of GPU threads — must match kernel launch in execute_task_on_device!
function compute_total_nthreads(device::AbstractGPUDevice, ncells::Integer)
    Ti = index_type(device)
    threads_per_block = _compute_groupsize(device, ncells)
    nblocks           = convert(Ti, cld(ncells, threads_per_block))
    return convert(Ti, nblocks * threads_per_block)
end

total_nthreads(device::AbstractGPUDevice) = device.threads * device.blocks

function resolve_launch_config(device::CudaDevice{V,I}, ncells::Integer) where {V,I}
    tpb     = _compute_groupsize(device, ncells)
    nblocks = convert(I, cld(ncells, tpb))
    return CudaDevice{V,I}(tpb, nblocks)
end

function resolve_launch_config(device::RocDevice{V,I}, ncells::Integer) where {V,I}
    tpb     = _compute_groupsize(device, ncells)
    nblocks = convert(I, cld(ncells, tpb))
    return RocDevice{V,I}(tpb, nblocks)
end
