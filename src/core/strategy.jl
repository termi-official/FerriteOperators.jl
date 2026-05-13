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
# Sequential on GPU: force 1 thread, 1 block.
function resolve_device_config(strategy::SequentialAssemblyStrategy{D}, ::AbstractDofHandler) where {D <: AbstractGPUDevice}
    return SequentialAssemblyStrategy(make_sequential_device(D), strategy.operator_specification)
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

## Assembly cache allocation ##
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

@concrete struct ElementAssemblyOperatorStrategy <: AbstractMatrixFreeStrategy
    device
    eadata
end

function setup_operator_strategy_cache(strategy::ElementAssemblyStrategy, integrator, dh)
    (;device) = strategy
    # CPU -> as is, GPU → adapt for GPU
    eadata = Adapt.adapt(KA.backend(device), EAVector(dh))
    return ElementAssemblyOperatorStrategy(device, eadata)
end

setup_operator_strategy_cache(strategy, integrator, dh) = strategy


####################################
## Workspace                      ##
####################################

"""
    AbstractWorkspace

Abstract supertype for all per-worker workspace types used by the task/device system.

Every concrete workspace must implement:
- `Ferrite.reinit!(ws, cellid)` — reinitialise geometry and element caches for the given cell
- `duplicate_for_device(device::AbstractCPUDevice, ws)` — create an independent copy for a parallel worker

New device backends must allocate and manage workspaces of a concrete subtype.
"""
abstract type AbstractWorkspace end

"""
    AssemblyWorkspace

Per-worker workspace for square operator assembly (bilinear, nonlinear, linear).
Holds pre-allocated element-local buffers and caches that are reused across cells.

Fields:
- `Ke`: element stiffness matrix
- `ue`: element unknown vector
- `re`: element residual vector
- `cell`: geometry cache ([`CellCache`](@ref))
- `ivh`: internal variable handler
- `element`: element cache (user-defined, subtype of [`AbstractVolumetricElementCache`](@ref))
"""
@concrete struct AssemblyWorkspace <: AbstractWorkspace
    Ke
    ue
    re
    cell
    ivh
    element
    boundary_element
end

Ferrite.reinit!(ws::AssemblyWorkspace, cellid) = reinit!(ws.cell, cellid)

function duplicate_for_device(device::AbstractCPUDevice, ws::AssemblyWorkspace)
    return create_assembly_workspace(
        duplicate_for_device(device, ws.element),
        duplicate_for_device(device, ws.boundary_element),
        ws.cell.dh,
        duplicate_for_device(device, ws.ivh),
    )
end

"""
    create_assembly_workspace(element, boundary_element, sdh, ivh)

Create a single [`AssemblyWorkspace`](@ref) with freshly allocated element-local buffers.
"""
function create_assembly_workspace(element, boundary_element, sdh, ivh)
    return AssemblyWorkspace(
        allocate_element_matrix(element, sdh),
        allocate_element_unknown_vector(element, sdh),
        allocate_element_residual_vector(element, sdh),
        CellCache(sdh),
        ivh,
        element,
        boundary_element,
    )
end

####################################
## Partition                      ##
####################################

"""
    compute_partition(strategy, sdh)

Compute the work partition for the given strategy and SubDofHandler.
Returns an iterable of iterables: the outer level represents synchronization barriers
(e.g. colors), the inner level represents parallelizable work items (cell IDs).
"""
compute_partition(::SequentialAssemblyStrategy, sdh) = (sdh.cellset,)
compute_partition(::ElementAssemblyOperatorStrategy, sdh) = (sdh.cellset,)

function compute_partition(strategy::PerColorAssemblyStrategy, sdh)
    return Ferrite.create_coloring(get_grid(sdh.dh), sdh.cellset; alg=strategy.coloralg)
end

"""
    n_workers(strategy, device, partition) -> Int

Compute the number of parallel workers needed for the given strategy, device, and partition.
"""
n_workers(strategy, ::SequentialCPUDevice, partition) = 1
function n_workers(strategy, device::PolyesterDevice, partition)
    ncellsmax = maximum(length, partition)
    return ceil(Int, ncellsmax / device.chunksize)
end

function n_workers(strategy, device::AbstractGPUDevice, partition)
    throw(ArgumentError(
        "GPU assembly is not yet implemented for $(typeof(device)). " *
        "Implement n_workers for this device type."
    ))
end


####################################
## Matrix/Vector type             ##
####################################

matrix_type(strategy::AbstractAssemblyStrategy) = matrix_type(strategy.device, strategy.operator_specification)
matrix_type(device::AbstractCPUDevice, ::StandardOperatorSpecification) = SparseMatrixCSC{value_type(device), index_type(device)}
vector_type(strategy::AbstractAssemblyStrategy) = vector_type(strategy.device)
vector_type(device::AbstractDevice) = Vector{value_type(device)}

function Adapt.adapt_structure(::AbstractAssemblyStrategy, dh::DofHandler)
    error("Device specific implementation for `adapt_structure(::AbstractAssemblyStrategy,dh::DofHandler)` is not implemented yet")
end

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
