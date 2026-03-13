struct StandardOperatorSpecification
end

abstract type AbstractAssemblyStrategy end
# This one is the super type for strategies giving us a full matrix with indexing and stuff
abstract type AbstractFullAssemblyStrategy <: AbstractAssemblyStrategy end
# This one is the super type for strategies giving us an object which we ONLY can multiply a vector with
abstract type AbstractMatrixFreeStrategy <: AbstractAssemblyStrategy end

# NOTE: problem here on GPU in particular is that `nblocks` and `nthreads` might be zeros or way more that ncells.
# Updates the strategy with the resolved device config (e.g. launch config for GPU).
resolve_device_config(strategy::AbstractAssemblyStrategy,dh::AbstractDofHandler) = strategy

"""
    SequentialAssemblyStrategy()
"""
struct SequentialAssemblyStrategy{DeviceType} <: AbstractFullAssemblyStrategy
    device::DeviceType
    operator_specification
end
SequentialAssemblyStrategy(device) = SequentialAssemblyStrategy(device, StandardOperatorSpecification())
resolve_device_config(strategy::SequentialAssemblyStrategy{<:AbstractGPUDevice}, dh::AbstractDofHandler) = SequentialAssemblyStrategy(resolve_device_config(strategy.device, dh), strategy.operator_specification)

struct SequentialAssemblyStrategyCache{DeviceType, DeviceCacheType}
    device::DeviceType
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
end

setup_operator_strategy_cache(strategy, integrator, dh) = strategy

## Local Cache Container ##
# GPU allocation occurs in two steps: (1) container creation on the CPU and (2) kernel-side indexing.
# All containers support `container[tid]` → local cache for that thread (consistent with Ferrite's CellValuesContainer).
abstract type AbstractLocalCacheContainer end

@concrete struct BilinearLocalCacheContainer <: AbstractLocalCacheContainer
    Ke_pool  # GPU array (N, N, total_nthreads)
end
@inline Base.getindex(c::BilinearLocalCacheContainer, tid) =
    BilinearLocalCache(view(c.Ke_pool, :, :, tid))

@concrete struct NonlinearLocalCacheContainer <: AbstractLocalCacheContainer
    Ke_pool  # GPU array (N, N, total_nthreads)
    ue_pool  # GPU array (N, total_nthreads)
    re_pool  # GPU array (N, total_nthreads)
end
@inline Base.getindex(c::NonlinearLocalCacheContainer, tid) =
    NonlinearLocalCache(view(c.Ke_pool, :, :, tid), view(c.ue_pool, :, tid), view(c.re_pool, :, tid))

@concrete struct LinearLocalCacheContainer <: AbstractLocalCacheContainer
    re_pool  # GPU array (N, total_nthreads)
end
@inline Base.getindex(c::LinearLocalCacheContainer, tid) =
    LinearLocalCache(view(c.re_pool, :, tid))

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


#TODO: Unify `allocate_local_cache` signature across devices.
## CPU local cache ##
function allocate_local_cache(::AbstractBilinearIntegrator, element_cache, sdh)
    BilinearLocalCache(allocate_element_matrix(element_cache, sdh))
end

function allocate_local_cache(::AbstractNonlinearIntegrator, element_cache, sdh)
    NonlinearLocalCache(
        allocate_element_matrix(element_cache, sdh),
        allocate_element_unknown_vector(element_cache, sdh),
        allocate_element_residual_vector(element_cache, sdh),
    )
end

function allocate_local_cache(::AbstractLinearIntegrator, element_cache, sdh)
    LinearLocalCache(allocate_element_residual_vector(element_cache, sdh))
end

## GPU local cache ##
function allocate_local_cache(::AbstractBilinearIntegrator, device::AbstractGPUDevice, sdh)
    backend = default_backend(device)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    nt = total_nthreads(device)
    Ke_pool = KA.zeros(backend, T, N, N, nt)
    return BilinearLocalCacheContainer(Ke_pool)
end
function allocate_local_cache(::AbstractNonlinearIntegrator, device::AbstractGPUDevice, sdh)
    backend = default_backend(device)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    nt = total_nthreads(device)
    Ke_pool = KA.zeros(backend, T, N, N, nt)
    ue_pool = KA.zeros(backend, T, N, nt)
    re_pool = KA.zeros(backend, T, N, nt)
    return NonlinearLocalCacheContainer(Ke_pool, ue_pool, re_pool)
end
function allocate_local_cache(::AbstractLinearIntegrator, device::AbstractGPUDevice, sdh)
    backend = default_backend(device)
    N = ndofs_per_cell(sdh)
    T = value_type(device)
    nt = total_nthreads(device)
    re_pool = KA.zeros(backend, T, N, nt)
    return LinearLocalCacheContainer(re_pool)
end

@concrete struct SimpleAssemblyCache
    # NOTE: idea here is instead of allocating useless memory (also not expressive development-wise), we allocate only the needed local cache.
    local_cache # (either BilinearLocalCache, NonlinearLocalCache, or LinearLocalCache)
    cell
    ivh
    element
end

# GPU analog of SimpleAssemblyCache — holds containers instead of concrete caches.
# Each container's extract/indexing creates per-thread cache in the kernel.
@concrete struct GPUAssemblyCache
    local_cache_container    # AbstractLocalCacheContainer (Ke/ue/re pools)
    cell_cache_container     # Ferrite.CellCacheContainer (batched ImmutableCellCache)
    ivh                      # InternalVariableHandler (GPU-adapted, read-only shared)
    element_cache_container  # e.g. SimpleBilinearDiffusionElementCache{<:CellValuesContainer}
end


function setup_element_strategy_cache(strategy::SequentialAssemblyStrategy, integrator, element_cache, ivh, sdh)
    local_cache = allocate_local_cache(integrator, element_cache, sdh)
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
resolve_device_config(strategy::PerColorAssemblyStrategy{<:AbstractGPUDevice}, dh::AbstractDofHandler) = PerColorAssemblyStrategy(resolve_device_config(strategy.device, dh), strategy.coloralg, strategy.operator_specification)

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

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:SequentialCPUDevice}, integrator, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, integrator, element_cache, ivh, sdh, 1)
end

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:PolyesterDevice}, integrator, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, integrator, element_cache, ivh, sdh, strategy.device.chunksize)
end

function _setup_element_strategy_cache_cpu(strategy::PerColorAssemblyStrategy, integrator, element_cache, ivh, sdh, chunksize)
    (; device) = strategy
    (; dh)     = sdh
    grid       = get_grid(dh)

    colors = Ferrite.create_coloring(grid, sdh.cellset; alg=strategy.coloralg)

    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    task_local_caches = [
        SimpleAssemblyCache(
            allocate_local_cache(integrator, element_cache, sdh),
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
resolve_device_config(strategy::ElementAssemblyStrategy{<:AbstractGPUDevice}, dh::AbstractDofHandler) = ElementAssemblyStrategy(resolve_device_config(strategy.device, dh))

@concrete struct ElementAssemblyOperatorStrategy
    device
    eadata
end

function setup_operator_strategy_cache(strategy::ElementAssemblyStrategy, integrator, dh)
    (;device) = strategy
    # CPU -> as is, GPU → adapt for GPU
    eadata = _adapt_for_device(device, EAVector(dh)) 
    return ElementAssemblyOperatorStrategy(device, eadata)
end

@concrete struct ElementAssemblyStrategyCache
    device
    # Scratch for the device to store its data
    device_cache
end

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:SequentialCPUDevice}, integrator, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, integrator, element_cache, ivh, sdh, getncells(get_grid(sdh.dh)))
end

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:PolyesterDevice}, integrator, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, integrator, element_cache, ivh, sdh, strategy.device.chunksize)
end

function _setup_element_strategy_cache_cpu(strategy::ElementAssemblyOperatorStrategy, integrator, element_cache, ivh, sdh, chunksize)
    (; device) = strategy
    (; dh)     = sdh
    grid       = get_grid(dh)

    ncellsmax  = getncells(grid)
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    task_local_caches = [
        SimpleAssemblyCache(
            allocate_local_cache(integrator, element_cache, sdh),
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

# GPU setup: takes an adapted GPU SubDofHandler — grid/cell_dofs shared across subdomains.
function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:AbstractGPUDevice}, integrator, element_cache, ivh, sdh, device_sdh, ncells::Integer)
    
    (; device) = strategy
    backend    = default_backend(device)
    nt         = total_nthreads(device)

    local_cache_container    = allocate_local_cache(integrator, device, sdh)
    cell_cache_container     = Ferrite.CellCacheContainer(backend, nt, device_sdh)
    gpu_ivh                  = duplicate_for_device(device, ivh)
    element_cache_container  = duplicate_for_device(device, element_cache)

    device_cache = GPUAssemblyCache(local_cache_container, cell_cache_container, gpu_ivh, element_cache_container)
    return ElementAssemblyStrategyCache(device, device_cache)
end
