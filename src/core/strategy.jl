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

# TODO
setup_operator_strategy_cache(strategy, integrator, dh) = strategy
setup_element_strategy_cache(strategy::SequentialAssemblyStrategy, element_cache, sdh) = SequentialAssemblyStrategyCache(strategy.device, nothing)

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
    tlds
    Aes
    ues
    res
end

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:SequentialCPUDevice}, element_cache, sdh)
    return _setup_element_strategy_cache_cpu(strategy, element_cache, sdh, 1)
end

function setup_element_strategy_cache(strategy::PerColorAssemblyStrategy{<:PolyesterDevice}, element_cache, sdh)
    return _setup_element_strategy_cache_cpu(strategy, element_cache, sdh, strategy.device.chunksize)
end

function _setup_element_strategy_cache_cpu(strategy::PerColorAssemblyStrategy, element_cache, sdh, chunksize)
    (; device) = strategy
    (; dh)     = sdh
    grid       = get_grid(dh)

    colors = Ferrite.create_coloring(grid, sdh.cellset; alg=strategy.coloralg)

    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunksmax]
    Aes  = [allocate_element_matrix(element_cache, sdh) for tid in 1:nchunksmax]
    ues  = [allocate_element_unknown_vector(element_cache, sdh) for tid in 1:nchunksmax]
    res  = [allocate_element_residual_vector(element_cache, sdh) for tid in 1:nchunksmax]
    PerColorAssemblyStrategyCache(strategy.device, ThreadedAssemblyCache(tlds, Aes, ues, res), colors)
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

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:SequentialCPUDevice}, element_cache, sdh)
    return _setup_element_strategy_cache_cpu(strategy, element_cache, sdh, getncells(get_grid(sdh.dh)))
end

function setup_element_strategy_cache(strategy::ElementAssemblyOperatorStrategy{<:PolyesterDevice}, element_cache, sdh)
    return _setup_element_strategy_cache_cpu(strategy, element_cache, sdh, strategy.device.chunksize)
end

function _setup_element_strategy_cache_cpu(strategy::ElementAssemblyOperatorStrategy, element_cache, sdh, chunksize)
    (; device) = strategy
    (; dh)     = sdh
    grid       = get_grid(dh)

    ncellsmax  = getncells(grid)
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunksmax]
    Aes  = [allocate_element_matrix(element_cache, sdh) for tid in 1:nchunksmax]
    ues  = [allocate_element_unknown_vector(element_cache, sdh) for tid in 1:nchunksmax]
    res  = [allocate_element_residual_vector(element_cache, sdh) for tid in 1:nchunksmax]
    ElementAssemblyStrategyCache(strategy.device, ThreadedAssemblyCache(tlds, Aes, ues, res))
end

matrix_type(strategy::AbstractAssemblyStrategy) = matrix_type(strategy.device, strategy.operator_specification)
matrix_type(device::AbstractDevice, ::StandardOperatorSpecification) = SparseMatrixCSC{value_type(device), index_type(device)}
