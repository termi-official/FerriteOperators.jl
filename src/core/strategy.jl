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
# Sequential on GPU: force 1 thread, 1 block — no race conditions, no coloring needed.
function resolve_device_config(strategy::SequentialAssemblyStrategy{D}, dh::AbstractDofHandler) where {V, I, D <: AbstractGPUDevice{V,I}}
    device = D(one(I), one(I))
    return SequentialAssemblyStrategy(device, strategy.operator_specification)
end

struct SequentialAssemblyStrategyCache{DeviceType, DeviceCacheType}
    device::DeviceType
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
end

setup_operator_strategy_cache(strategy, integrator, dh) = strategy

## Assembly caches ##
@concrete struct BilinearAssemblyCache
    Ke
end

@concrete struct NonlinearAssemblyCache
    Ke
    ue
    re
end

@concrete struct LinearAssemblyCache
    re
end

## GPU pool containers ##
@concrete struct BilinearAssemblyCacheContainer
    Ke_pool  # GPU array (N, N, total_nthreads)
end
@inline Base.getindex(c::BilinearAssemblyCacheContainer, tid) =
    BilinearAssemblyCache(view(c.Ke_pool, :, :, tid))

@concrete struct NonlinearAssemblyCacheContainer
    Ke_pool  # GPU array (N, N, total_nthreads)
    ue_pool  # GPU array (N, total_nthreads)
    re_pool  # GPU array (N, total_nthreads)
end
@inline Base.getindex(c::NonlinearAssemblyCacheContainer, tid) =
    NonlinearAssemblyCache(view(c.Ke_pool, :, :, tid), view(c.ue_pool, :, tid), view(c.re_pool, :, tid))

@concrete struct LinearAssemblyCacheContainer
    re_pool  # GPU array (N, total_nthreads)
end
@inline Base.getindex(c::LinearAssemblyCacheContainer, tid) =
    LinearAssemblyCache(view(c.re_pool, :, tid))

## Local cache allocation ##
function allocate_assembly_cache(::AbstractBilinearIntegrator, element_cache, device::AbstractCPUDevice, sdh)
    BilinearAssemblyCache(allocate_element_matrix(device, element_cache, sdh))
end
function allocate_assembly_cache(::AbstractNonlinearIntegrator, element_cache, device::AbstractCPUDevice, sdh)
    NonlinearAssemblyCache(
        allocate_element_matrix(device, element_cache, sdh),
        allocate_element_unknown_vector(device, element_cache, sdh),
        allocate_element_residual_vector(device, element_cache, sdh),
    )
end
function allocate_assembly_cache(::AbstractLinearIntegrator, element_cache, device::AbstractCPUDevice, sdh)
    LinearAssemblyCache(allocate_element_residual_vector(device, element_cache, sdh))
end

#TODO: should we use same inner functions like `allocate_element_matrix` for GPU, eventhough it dosn't actually allocate element matrix but a pool?
function allocate_assembly_cache(::AbstractBilinearIntegrator, element_cache, device::AbstractGPUDevice, sdh)
    BilinearAssemblyCacheContainer(allocate_element_matrix(device, element_cache, sdh))
end
function allocate_assembly_cache(::AbstractNonlinearIntegrator, element_cache, device::AbstractGPUDevice, sdh)
    NonlinearAssemblyCacheContainer(
        allocate_element_matrix(device, element_cache, sdh),
        allocate_element_unknown_vector(device, element_cache, sdh),
        allocate_element_residual_vector(device, element_cache, sdh),
    )
end
function allocate_assembly_cache(::AbstractLinearIntegrator, element_cache, device::AbstractGPUDevice, sdh)
    LinearAssemblyCacheContainer(allocate_element_residual_vector(device, element_cache, sdh))
end

@concrete struct SimpleAssemblyCache
    # NOTE: On CPU, assembly_cache holds a concrete BilinearAssemblyCache/NonlinearAssemblyCache/LinearAssemblyCache.
    # On GPU, it holds the corresponding *Container (e.g. BilinearAssemblyCacheContainer) — a pool indexed by thread id.
    # Same for cell (CellCache on CPU, CellCacheContainer on GPU) and element (concrete vs container).
    assembly_cache
    cell
    ivh
    element
end


function setup_element_strategy_cache(strategy::SequentialAssemblyStrategy{<:AbstractCPUDevice}, integrator, element_cache, ivh, sdh)
    local_cache = allocate_assembly_cache(integrator, element_cache, strategy.device, sdh)
    return SequentialAssemblyStrategyCache(strategy.device, SimpleAssemblyCache(local_cache, CellCache(sdh), ivh, element_cache))
end

function setup_element_strategy_cache(strategy::SequentialAssemblyStrategy{<:AbstractGPUDevice}, integrator, element_cache, ivh, sdh)
    device_cache = _setup_gpu_assembly_cache(strategy.device, integrator, element_cache, ivh, sdh)
    return SequentialAssemblyStrategyCache(strategy.device, device_cache)
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
            allocate_assembly_cache(integrator, element_cache, device, sdh),
            CellCache(sdh),
            duplicate_for_device(device, ivh),
            duplicate_for_device(device, element_cache),
        )
    for tid in 1:nchunksmax]
    return PerColorAssemblyStrategyCache(strategy.device, ThreadedAssemblyCache(task_local_caches), colors)
end

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:AbstractGPUDevice}, integrator, element_cache, ivh, sdh, gpu_colors)
    device_cache = _setup_gpu_assembly_cache(strategy.device, integrator, element_cache, ivh, sdh)
    return PerColorAssemblyStrategyCache(strategy.device, device_cache, gpu_colors)
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
    eadata = Adapt.adapt(KA.backend(device), EAVector(dh))
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
            allocate_assembly_cache(integrator, element_cache, device, sdh),
            CellCache(sdh),
            duplicate_for_device(device, ivh),
            duplicate_for_device(device, element_cache),
        )
    for tid in 1:nchunksmax]
    return ElementAssemblyStrategyCache(strategy.device, ThreadedAssemblyCache(task_local_caches))
end

matrix_type(strategy::AbstractAssemblyStrategy) = matrix_type(strategy.device, strategy.operator_specification)
matrix_type(device::AbstractCPUDevice, ::StandardOperatorSpecification) = SparseMatrixCSC{value_type(device), index_type(device)}

vector_type(strategy::AbstractAssemblyStrategy) = vector_type(strategy.device)
vector_type(device::AbstractCPUDevice) = Vector{value_type(device)}

function _setup_gpu_assembly_cache(device, integrator, element_cache, ivh, sdh)
    backend = KA.backend(device)
    nt      = total_nthreads(device)
    return SimpleAssemblyCache(
        allocate_assembly_cache(integrator, element_cache, device, sdh),
        Ferrite.CellCacheContainer(backend, nt, sdh),
        duplicate_for_device(device, ivh),
        duplicate_for_device(device, element_cache),
    )
end

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:AbstractGPUDevice}, integrator, element_cache, ivh, sdh)
    device_cache = _setup_gpu_assembly_cache(strategy.device, integrator, element_cache, ivh, sdh)
    return ElementAssemblyStrategyCache(strategy.device, device_cache)
end
