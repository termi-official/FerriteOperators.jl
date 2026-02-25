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

## 4. GPU local cache config (compile-time sizes for register/shared types) ##
# For GPURegisterMemoryType and GPUSharedMemoryType the kernel creates buffers on the fly.
# We store the sizes as type parameters so the compiler knows them at compile time.
struct GPULocalCacheConfig{MemType <: AbstractGPUMemoryType, Req <: AbstractTaskBufferRequirement, N, T} end

GPULocalCacheConfig(::MemType, ::Req, ::Val{N}, ::Type{T}) where {MemType <: AbstractGPUMemoryType, Req <: AbstractTaskBufferRequirement, N, T} =
    GPULocalCacheConfig{MemType, Req, N, T}()

# Kernel-side: create actual local cache from compile-time config
@inline function create_local_cache(::GPULocalCacheConfig{GPURegisterMemoryType, BilinearBufferRequirement, N, T}) where {N, T}
    BilinearLocalCache(MMatrix{N, N, T}(undef))
end
@inline function create_local_cache(::GPULocalCacheConfig{GPURegisterMemoryType, NonlinearBufferRequirement, N, T}) where {N, T}
    NonlinearLocalCache(MMatrix{N, N, T}(undef), MVector{N, T}(undef), MVector{N, T}(undef))
end
@inline function create_local_cache(::GPULocalCacheConfig{GPURegisterMemoryType, LinearBufferRequirement, N, T}) where {N, T}
    LinearLocalCache(MVector{N, T}(undef))
end

# GPUSharedMemoryType: @localmem must be called in kernel body, so we cannot dispatch here.
# The kernel itself will handle shared memory allocation via @localmem.
# TODO: implement GPUSharedMemoryType create_local_cache when needed

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

## GPU: allocate local cache based on memory type ##
# RegisterType / SharedType: return compile-time config (kernel creates buffers)
function allocate_local_cache(req::AbstractTaskBufferRequirement, ::Union{GPURegisterMemoryType, GPUSharedMemoryType}, device::AbstractGPUDevice, sdh)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    return GPULocalCacheConfig(GPURegisterMemoryType(), req, Val(N), T)
end

# GlobalType: pre-allocate device arrays (similar to CPU but with device value type)
function allocate_local_cache(::BilinearBufferRequirement, ::GPUGlobalMemoryType, device::AbstractGPUDevice, sdh)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    BilinearLocalCache(zeros(T, N, N))
end
function allocate_local_cache(::NonlinearBufferRequirement, ::GPUGlobalMemoryType, device::AbstractGPUDevice, sdh)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    NonlinearLocalCache(zeros(T, N, N), zeros(T, N), zeros(T, N))
end
function allocate_local_cache(::LinearBufferRequirement, ::GPUGlobalMemoryType, device::AbstractGPUDevice, sdh)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    LinearLocalCache(zeros(T, N))
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
    error("GPU setup_element_strategy_cache not yet implemented")
end
