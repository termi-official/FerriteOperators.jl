struct StandardOperatorSpecification
end

abstract type AbstractAssemblyStrategy end
# This one is the super type for strategies giving us a full matrix with indexing and stuff
abstract type AbstractFullAssemblyStrategy <: AbstractAssemblyStrategy end
# This one is the super type for strategies giving us an object which we ONLY can multiply a vector with
abstract type AbstractMatrixFreeStrategy <: AbstractAssemblyStrategy end

"""
    SequentialAssemblyStrategy()
"""
struct SequentialAssemblyStrategy{DeviceType} <: AbstractFullAssemblyStrategy
    device::DeviceType
    operator_specification
end
SequentialAssemblyStrategy(device) = SequentialAssemblyStrategy(device, StandardOperatorSpecification())

struct SequentialAssemblyStrategyCache{DeviceType, DeviceCacheType}
    device::DeviceType
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
end

setup_operator_strategy_cache(strategy, integrator, dh) = strategy

## 1. Trait: what does this task need? ##
abstract type AbstractTaskBufferRequirement end
struct BilinearBufferRequirement  <: AbstractTaskBufferRequirement end  # Ke + cell + element
struct NonlinearBufferRequirement <: AbstractTaskBufferRequirement end  # Ke + ue + re + cell + element
struct LinearBufferRequirement    <: AbstractTaskBufferRequirement end  # re + cell + element

# Integrator → BufferRequirement mapping defined after integrator types (see FerriteOperators.jl)
# Each task also declares its requirement (overrides in operators/*.jl)
buffer_requirement(task) = error("buffer_requirement not implemented for $(typeof(task))")

## 2. Memory type (decided at setup time based on hardware query) ##
abstract type AbstractMemoryType end
abstract type AbstractCPUMemoryType <: AbstractMemoryType end
abstract type AbstractGPUMemoryType <: AbstractMemoryType end

struct CPUMemoryType        <: AbstractCPUMemoryType end
# NOTE: Register & Shared need: 1) compile-time sizes (for stack/shared allocation) and 2) allocated in the kernel
struct GPURegisterMemoryType <: AbstractGPUMemoryType end  # MArray, compile-time sizes
struct GPUSharedMemoryType   <: AbstractGPUMemoryType end  # @localmem, per-block
struct GPUGlobalMemoryType   <: AbstractGPUMemoryType end  # CuArray, pre-allocated

## 3. GPU memory type selection ##

# How much memory does each buffer requirement need per element (in bytes)?
buffer_memory_per_element(::BilinearBufferRequirement, ndofs, ::Type{T}) where T  = ndofs * ndofs * sizeof(T)
buffer_memory_per_element(::NonlinearBufferRequirement, ndofs, ::Type{T}) where T = (ndofs * ndofs + 2 * ndofs) * sizeof(T)
buffer_memory_per_element(::LinearBufferRequirement, ndofs, ::Type{T}) where T    = ndofs * sizeof(T)

function select_memory_type(device::AbstractGPUDevice, req::AbstractTaskBufferRequirement, ndofs::Int)
    # T   = value_type(device)
    # mem = buffer_memory_per_element(req, ndofs, T)

    # # Heuristic: register budget ≈ max_registers * 4 bytes, use at most 25%
    # register_budget = max_registers_per_block(device) * 4 ÷ 4
    # if mem <= register_budget
    #     return GPURegisterMemoryType()
    # end

    # # Shared memory: budget per thread ≈ total shared / 256 threads (conservative)
    # shared_budget = max_sharedmem_per_block(device) ÷ 256
    # if mem <= shared_budget
    #     return GPUSharedMemoryType()
    # end
    # TODO: implement this efficiently.
    return GPUGlobalMemoryType()
end

## Local Cache Factory ##
# GPU allocation occurs in two steps: (1) factory creation on the CPU and (2) kernel-side materialization.# All factories support `materialize(factory, tid, local_tid, groupsize)` → local cache.
abstract type AbstractGPULocalCacheFactory end

# Register: stack-allocated MArrays (per-thread, in registers)
struct RegisterLocalCacheFactory{Req <: AbstractTaskBufferRequirement, N, T} <: AbstractGPULocalCacheFactory end

@inline materialize(::RegisterLocalCacheFactory{BilinearBufferRequirement, N, T}, tid, local_tid, groupsize) where {N, T} =
    BilinearLocalCache(MMatrix{N, N, T}(undef))
@inline materialize(::RegisterLocalCacheFactory{NonlinearBufferRequirement, N, T}, tid, local_tid, groupsize) where {N, T} =
    NonlinearLocalCache(MMatrix{N, N, T}(undef), MVector{N, T}(undef), MVector{N, T}(undef))
@inline materialize(::RegisterLocalCacheFactory{LinearBufferRequirement, N, T}, tid, local_tid, groupsize) where {N, T} =
    LinearLocalCache(MVector{N, T}(undef))

# Shared: @localmem allocates once per block, each thread indexes with local_tid
struct SharedLocalCacheFactory{Req <: AbstractTaskBufferRequirement, N, T} <: AbstractGPULocalCacheFactory end

@inline function materialize(::SharedLocalCacheFactory{BilinearBufferRequirement, N, T}, tid, local_tid, groupsize) where {N, T}
    shared = KA.@localmem T (N, N, groupsize)
    BilinearLocalCache(view(shared, :, :, local_tid))
end
@inline function materialize(::SharedLocalCacheFactory{NonlinearBufferRequirement, N, T}, tid, local_tid, groupsize) where {N, T}
    Ke_shared = KA.@localmem T (N, N, groupsize)
    ue_shared = KA.@localmem T (N, groupsize)
    re_shared = KA.@localmem T (N, groupsize)
    NonlinearLocalCache(view(Ke_shared, :, :, local_tid), view(ue_shared, :, local_tid), view(re_shared, :, local_tid))
end
@inline function materialize(::SharedLocalCacheFactory{LinearBufferRequirement, N, T}, tid, local_tid, groupsize) where {N, T}
    shared = KA.@localmem T (N, groupsize)
    LinearLocalCache(view(shared, :, local_tid))
end

# Global: pre-allocated GPU array pools, views created per-thread in kernel
abstract type AbstractGlobalLocalCacheFactory <: AbstractGPULocalCacheFactory end

@concrete struct BilinearGlobalLocalCacheFactory <: AbstractGlobalLocalCacheFactory
    Ke_pool  # GPU array (N, N, total_nthreads)
end
@inline materialize(f::BilinearGlobalLocalCacheFactory, tid, local_tid, groupsize) =
    BilinearLocalCache(view(f.Ke_pool, :, :, tid))

@concrete struct NonlinearGlobalLocalCacheFactory <: AbstractGlobalLocalCacheFactory
    Ke_pool  # GPU array (N, N, total_nthreads)
    ue_pool  # GPU array (N, total_nthreads)
    re_pool  # GPU array (N, total_nthreads)
end
@inline materialize(f::NonlinearGlobalLocalCacheFactory, tid, local_tid, groupsize) =
    NonlinearLocalCache(view(f.Ke_pool, :, :, tid), view(f.ue_pool, :, tid), view(f.re_pool, :, tid))

@concrete struct LinearGlobalLocalCacheFactory <: AbstractGlobalLocalCacheFactory
    re_pool  # GPU array (N, total_nthreads)
end
@inline materialize(f::LinearGlobalLocalCacheFactory, tid, local_tid, groupsize) =
    LinearLocalCache(view(f.re_pool, :, tid))

## Local caches: typed per buffer requirement ##
@concrete struct BilinearLocalCache
    Ke
end

@concrete struct NonlinearLocalCache
    Ke
    ue
    re
end

@concrete struct LinearLocalCache
    re
end


## Allocate local cache based on requirement ##
function allocate_local_cache(::BilinearBufferRequirement, ::CPUMemoryType, element_cache, sdh)
    BilinearLocalCache(allocate_element_matrix(element_cache, sdh))
end

function allocate_local_cache(::NonlinearBufferRequirement, ::CPUMemoryType, element_cache, sdh)
    NonlinearLocalCache(
        allocate_element_matrix(element_cache, sdh),
        allocate_element_unknown_vector(element_cache, sdh),
        allocate_element_residual_vector(element_cache, sdh),
    )
end

function allocate_local_cache(::LinearBufferRequirement, ::CPUMemoryType, element_cache, sdh)
    LinearLocalCache(allocate_element_residual_vector(element_cache, sdh))
end

## GPU: allocate local cache → returns an AbstractGPULocalCacheFactory ##
# Kernel calls `materialize(factory, tid, local_tid, groupsize)` to create the actual local cache.

function _gpu_zeros(device::AbstractGPUDevice, T, dims...)
    Adapt.adapt(default_backend(device), zeros(T, dims...))
end

# Register
function allocate_local_cache(req::AbstractTaskBufferRequirement, ::GPURegisterMemoryType, device::AbstractGPUDevice, sdh; total_nthreads)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    return RegisterLocalCacheFactory{typeof(req), N, T}()
end

# Shared
function allocate_local_cache(req::AbstractTaskBufferRequirement, ::GPUSharedMemoryType, device::AbstractGPUDevice, sdh; total_nthreads)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    return SharedLocalCacheFactory{typeof(req), N, T}()
end

# Global
function allocate_local_cache(::BilinearBufferRequirement, ::GPUGlobalMemoryType, device::AbstractGPUDevice, sdh; total_nthreads)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    Ke_pool = _gpu_zeros(device, T, N, N, total_nthreads)
    return BilinearGlobalLocalCacheFactory(Ke_pool)
end
function allocate_local_cache(::NonlinearBufferRequirement, ::GPUGlobalMemoryType, device::AbstractGPUDevice, sdh; total_nthreads)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    Ke_pool = _gpu_zeros(device, T, N, N, total_nthreads)
    ue_pool = _gpu_zeros(device, T, N, total_nthreads)
    re_pool = _gpu_zeros(device, T, N, total_nthreads)
    return NonlinearGlobalLocalCacheFactory(Ke_pool, ue_pool, re_pool)
end
function allocate_local_cache(::LinearBufferRequirement, ::GPUGlobalMemoryType, device::AbstractGPUDevice, sdh; total_nthreads)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    re_pool = _gpu_zeros(device, T, N, total_nthreads)
    return LinearGlobalLocalCacheFactory(re_pool)
end

@concrete struct SimpleAssemblyCache
    # NOTE: idea here is instead of allocating useless memory (also not expressive development-wise), we allocate only the needed local cache. 
    local_cache # (either BilinearLocalCache, NonlinearLocalCache, or LinearLocalCache)
    cell
    ivh
    element
end

function setup_element_strategy_cache(strategy::SequentialAssemblyStrategy, req::AbstractTaskBufferRequirement, element_cache, ivh, sdh)
    # NOTE: `CPUMemoryType()` is redundant here but needed to unify the api for both cpu and gpu
    local_cache = allocate_local_cache(req, CPUMemoryType(), element_cache, sdh)
    return SequentialAssemblyStrategyCache(strategy.device, SimpleAssemblyCache(local_cache, CellCache(sdh), ivh, element_cache))
end

"""
    PerColorAssemblyStrategy(chunksize, coloralg)
"""
struct PerColorAssemblyStrategy{DeviceType} <: AbstractFullAssemblyStrategy
    device::DeviceType
    coloralg
    operator_specification
end
PerColorAssemblyStrategy(device, alg = ColoringAlgorithm.WorkStream) = PerColorAssemblyStrategy(device, alg, StandardOperatorSpecification())

@concrete struct PerColorAssemblyStrategyCache
    device
    # Scratch for the device to store its data
    device_cache
    # Everything related to the coloring is stored here
    colors
end

@concrete struct ThreadedAssemblyCache
    task_local_caches
end

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:SequentialCPUDevice}, req::AbstractTaskBufferRequirement, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, req, element_cache, ivh, sdh, 1)
end

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:PolyesterDevice}, req::AbstractTaskBufferRequirement, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, req, element_cache, ivh, sdh, strategy.device.chunksize)
end

function _setup_element_strategy_cache_cpu(strategy::PerColorAssemblyStrategy, req::AbstractTaskBufferRequirement, element_cache, ivh, sdh, chunksize)
    (; device) = strategy
    (; dh)     = sdh
    grid       = get_grid(dh)

    colors = Ferrite.create_coloring(grid, sdh.cellset; alg=strategy.coloralg)

    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    task_local_caches = [
        SimpleAssemblyCache(
            # NOTE: `CPUMemoryType()` is redundant here but needed to unify the api for both cpu and gpu
            allocate_local_cache(req, CPUMemoryType(), element_cache, sdh),
            CellCache(sdh),
            duplicate_for_device(device, ivh),
            duplicate_for_device(device, element_cache),
        )
    for tid in 1:nchunksmax]
    return PerColorAssemblyStrategyCache(strategy.device, ThreadedAssemblyCache(task_local_caches), colors)
end

"""
    ElementAssemblyStrategy
"""
struct ElementAssemblyStrategy{DeviceType} <: AbstractMatrixFreeStrategy
    device::DeviceType
end

@concrete struct ElementAssemblyOperatorStrategy
    device
    eadata
end

function setup_operator_strategy_cache(strategy::ElementAssemblyStrategy{<:AbstractCPUDevice}, integrator, dh)
    (;device) = strategy
    eadata = Adapt.adapt(device, EAVector(dh)) # will only adapt if device <: AbstractGPUDevice
    return ElementAssemblyOperatorStrategy(device, eadata)
end

function setup_operator_strategy_cache(strategy::ElementAssemblyStrategy{<:AbstractGPUDevice}, integrator, dh)
    (;device) = strategy
    eadata = Adapt.adapt(device, EAVector(dh))
    return ElementAssemblyOperatorStrategy(device, eadata)
end

@concrete struct ElementAssemblyStrategyCache
    device
    # Scratch for the device to store its data
    device_cache
end

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:SequentialCPUDevice}, req::AbstractTaskBufferRequirement, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, req, element_cache, ivh, sdh, getncells(get_grid(sdh.dh)))
end

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:PolyesterDevice}, req::AbstractTaskBufferRequirement, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, req, element_cache, ivh, sdh, strategy.device.chunksize)
end

function _setup_element_strategy_cache_cpu(strategy::ElementAssemblyOperatorStrategy, req::AbstractTaskBufferRequirement, element_cache, ivh, sdh, chunksize)
    (; device) = strategy
    (; dh)     = sdh
    grid       = get_grid(dh)

    ncellsmax  = getncells(grid)
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    task_local_caches = [
        SimpleAssemblyCache(
            # NOTE: `CPUMemoryType()` is redundant here but needed to unify the api for both cpu and gpu
            allocate_local_cache(req, CPUMemoryType(), element_cache, sdh),
            CellCache(sdh),
            duplicate_for_device(device, ivh),
            duplicate_for_device(device, element_cache),
        )
    for tid in 1:nchunksmax]
    return ElementAssemblyStrategyCache(strategy.device, ThreadedAssemblyCache(task_local_caches))
end

matrix_type(strategy::AbstractAssemblyStrategy) = matrix_type(strategy.device, strategy.operator_specification)
matrix_type(device::AbstractDevice, ::StandardOperatorSpecification) = SparseMatrixCSC{value_type(device), index_type(device)}


## GPU ##

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:AbstractGPUDevice}, req::AbstractTaskBufferRequirement, element_cache, ivh, sdh)
    (; device) = strategy
    (; dh)     = sdh
    ncells     = getncells(get_grid(dh))
    N          = ndofs_per_cell(sdh)
    mem_type   = select_memory_type(device, req, N)
    # Compute total threads consistently with kernel launch (see execute_task_on_device!)
    total_nthreads = compute_total_nthreads(device, ncells)
    # allocate_local_cache returns an AbstractGPULocalCacheFactory:
    # - RegisterLocalCacheFactory / SharedLocalCacheFactory (zero-size isbitstype)
    # - BilinearGlobalLocalCacheFactory / NonlinearGlobalLocalCacheFactory / LinearGlobalLocalCacheFactory (GPU array pools)
    # All support `materialize(factory, tid, local_tid, groupsize)` in the kernel.
    device_cache = allocate_local_cache(req, mem_type, device, sdh; total_nthreads)
    # TODO: adapt cell, ivh, element for GPU
    return ElementAssemblyStrategyCache(device, device_cache)
end
