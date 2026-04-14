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

"""
    PerColorAssemblyStrategy(chunksize, coloralg)
"""
struct PerColorAssemblyStrategy{DeviceType} <: AbstractFullAssemblyStrategy
    device::DeviceType
    coloralg
    operator_specification
end
PerColorAssemblyStrategy(device, alg = ColoringAlgorithm.WorkStream) = PerColorAssemblyStrategy(device, alg, StandardOperatorSpecification())

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

setup_operator_strategy_cache(strategy, integrator, dh) = strategy


####################################
## Workspace                      ##
####################################

"""
    AbstractWorkspace

Abstract supertype for all per-worker workspace types used by the task/device system.

Every concrete workspace must implement:
- `Ferrite.reinit!(ws, cellid)` — reinitialise geometry and element caches for the given cell
- `execute_single_task!(task, ws)` — execute one task unit using the workspace state

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
end

Ferrite.reinit!(ws::AssemblyWorkspace, cellid) = reinit!(ws.cell, cellid)

"""
    create_assembly_workspace(element, sdh, ivh)

Create a single [`AssemblyWorkspace`](@ref) with freshly allocated element-local buffers.
"""
function create_assembly_workspace(element, sdh, ivh)
    return AssemblyWorkspace(
        allocate_element_matrix(element, sdh),
        allocate_element_unknown_vector(element, sdh),
        allocate_element_residual_vector(element, sdh),
        CellCache(sdh),
        ivh,
        element,
    )
end


####################################
## Device cache                   ##
####################################

@concrete struct ThreadedAssemblyCache
    task_local_caches
end

"""
    setup_device_cache(device, ws_builder, n_workers)

Create a device cache containing workspace(s) for `n_workers` parallel workers.
`ws_builder()` is called once per workspace to create each instance.
For [`SequentialCPUDevice`](@ref), returns the single workspace directly.
For threaded devices, returns a [`ThreadedAssemblyCache`](@ref) wrapping `n_workers` workspaces.
"""
function setup_device_cache(::SequentialCPUDevice, ws_builder, n_workers)
    return ws_builder()
end

function setup_device_cache(::AbstractCPUDevice, ws_builder, n_workers)
    return ThreadedAssemblyCache([ws_builder() for _ in 1:n_workers])
end

function setup_device_cache(device::AbstractGPUDevice, ws_builder, n_workers)
    throw(ArgumentError(
        "GPU assembly is not yet implemented for $(typeof(device)). " *
        "Implement setup_device_cache for this device type."
    ))
end

# Extract the first usable workspace from a device cache
_first_workspace(ws) = ws
_first_workspace(ws::ThreadedAssemblyCache) = first(ws.task_local_caches)


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
## Matrix type                    ##
####################################

matrix_type(strategy::AbstractAssemblyStrategy) = matrix_type(strategy.device, strategy.operator_specification)
matrix_type(device::AbstractDevice, ::StandardOperatorSpecification) = SparseMatrixCSC{value_type(device), index_type(device)}

function Adapt.adapt_structure(::AbstractAssemblyStrategy, dh::DofHandler)
    error("Device specific implementation for `adapt_structure(::AbstractAssemblyStrategy,dh::DofHandler)` is not implemented yet")
end
