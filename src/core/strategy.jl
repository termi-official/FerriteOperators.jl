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
@concrete struct AssemblyWorkspace
    Ke
    ue
    re
    cell
    ivh
    element
end
function setup_element_strategy_cache(strategy::SequentialAssemblyStrategy, element_cache, ivh, sdh)
    Ke = allocate_element_matrix(element_cache, sdh)
    ue = allocate_element_unknown_vector(element_cache, sdh)
    re = allocate_element_residual_vector(element_cache, sdh)
    return SequentialAssemblyStrategyCache(strategy.device, AssemblyWorkspace(Ke, ue, re, CellCache(sdh), ivh, element_cache))
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

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:SequentialCPUDevice}, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, element_cache, ivh, sdh, 1)
end

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:PolyesterDevice}, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, element_cache, ivh, sdh, strategy.device.chunksize)
end

function _setup_element_strategy_cache_cpu(strategy::PerColorAssemblyStrategy, element_cache, ivh, sdh, chunksize)
    (; device) = strategy
    (; dh)     = sdh
    grid       = get_grid(dh)

    colors = Ferrite.create_coloring(grid, sdh.cellset; alg=strategy.coloralg)

    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    task_local_caches = [
        AssemblyWorkspace(
            allocate_element_matrix(element_cache, sdh),
            allocate_element_unknown_vector(element_cache, sdh),
            allocate_element_residual_vector(element_cache, sdh),
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
    return ElementAssemblyOperatorStrategy(strategy.device, EAVector(dh))
end

@concrete struct ElementAssemblyStrategyCache
    device
    # Scratch for the device to store its data
    device_cache
end

function Adapt.adapt_structure(::AbstractAssemblyStrategy, dh::DofHandler)
    error("Device specific implementation for `adapt_structure(::AbstractAssemblyStrategy,dh::DofHandler)` is not implemented yet")
end

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:SequentialCPUDevice}, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, element_cache, ivh, sdh, getncells(get_grid(sdh.dh)))
end

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:PolyesterDevice}, element_cache, ivh, sdh)
    return _setup_element_strategy_cache_cpu(strategy, element_cache, ivh, sdh, strategy.device.chunksize)
end

function _setup_element_strategy_cache_cpu(strategy::ElementAssemblyOperatorStrategy, element_cache, ivh, sdh, chunksize)
    (; device) = strategy
    (; dh)     = sdh
    grid       = get_grid(dh)

    ncellsmax  = getncells(grid)
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    task_local_caches = [
        AssemblyWorkspace(
            allocate_element_matrix(element_cache, sdh),
            allocate_element_unknown_vector(element_cache, sdh),
            allocate_element_residual_vector(element_cache, sdh),
            CellCache(sdh),
            duplicate_for_device(device, ivh),
            duplicate_for_device(device, element_cache),
        )
    for tid in 1:nchunksmax]
    return ElementAssemblyStrategyCache(strategy.device, ThreadedAssemblyCache(task_local_caches))
end

matrix_type(strategy::AbstractAssemblyStrategy) = matrix_type(strategy.device, strategy.operator_specification)
matrix_type(device::AbstractDevice, ::StandardOperatorSpecification) = SparseMatrixCSC{value_type(device), index_type(device)}

# Workspace protocol
Ferrite.reinit!(ws::AssemblyWorkspace, cellid) = reinit!(ws.cell, cellid)

# Extract the first usable workspace from a device cache
_first_workspace(ws) = ws
_first_workspace(ws::ThreadedAssemblyCache) = first(ws.task_local_caches)
