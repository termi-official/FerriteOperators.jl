struct StandardOperatorSpecification
end

abstract type AbstractAssemblyStrategy end

####################################
## Operator form (the MFEM assembly level)
####################################

"""
Which representation of the operator is produced — the MFEM assembly level.
Orthogonal to how the work is scheduled and to the device it runs on.
"""
abstract type AbstractAssemblyForm end

"FULL level: assemble into a global sparse matrix / global vector."
struct FullAssembly{Spec} <: AbstractAssemblyForm
    operator_specification::Spec
end
FullAssembly() = FullAssembly(StandardOperatorSpecification())

"ELEMENT level: store per-element matrices; the operator acts matrix-free."
struct ElementAssembly <: AbstractAssemblyForm end

"ELEMENT level after setup: carries the per-element storage layout."
@concrete struct ElementAssemblyData <: AbstractAssemblyForm
    eadata
end

####################################
## Scheduling policy
####################################

"""
How parallel work is made safe — the second strategy axis. Sequential
scheduling relies on atomic scatter under parallel devices; colored
scheduling provides race freedom without atomics (required for eltypes
without atomic support, and for run-to-run reproducibility).
"""
abstract type AbstractSchedulingPolicy end
struct SequentialScheduling <: AbstractSchedulingPolicy end
struct ColoredScheduling{Alg} <: AbstractSchedulingPolicy
    alg::Alg
end
ColoredScheduling() = ColoredScheduling(ColoringAlgorithm.WorkStream)

####################################
## The composed strategy
####################################

"""
    AssemblyStrategy(form, scheduling, device)

An assembly strategy is the composition of three orthogonal axes: the operator
form ([`AbstractAssemblyForm`](@ref) — what is produced), the scheduling
policy ([`AbstractSchedulingPolicy`](@ref) — how parallel work is made safe),
and the device (where it runs). The historical strategy names remain as
convenience constructors for the common compositions.
"""
struct AssemblyStrategy{F <: AbstractAssemblyForm, S <: AbstractSchedulingPolicy, D} <: AbstractAssemblyStrategy
    form::F
    scheduling::S
    device::D
end

SequentialAssemblyStrategy(device) = AssemblyStrategy(FullAssembly(), SequentialScheduling(), device)
PerColorAssemblyStrategy(device, alg = ColoringAlgorithm.WorkStream) = AssemblyStrategy(FullAssembly(), ColoredScheduling(alg), device)
ElementAssemblyStrategy(device) = AssemblyStrategy(ElementAssembly(), SequentialScheduling(), device)

function setup_operator_strategy_cache(strategy::AssemblyStrategy{ElementAssembly, <:AbstractSchedulingPolicy, <:AbstractCPUDevice}, integrator, dh)
    return AssemblyStrategy(ElementAssemblyData(EAVector(dh)), strategy.scheduling, strategy.device)
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
- `slot_buffers`: NamedTuple of element-local state buffers, one per declared slot
- `re`: element residual vector
- `cell`: geometry cache ([`CellCache`](@ref))
- `ivh`: internal variable handler
- `element`: element cache (user-defined, subtype of [`AbstractVolumetricElementCache`](@ref))
- `boundary_element`: surface cache walked by the facet driver
- `scratch`/`scratch_decls`: per-worker scratch instances and their constructors
- `ad`: per-worker derivative-sweep buffers and ForwardDiff configs ([`ADWorkspace`](@ref))
"""
@concrete struct AssemblyWorkspace <: AbstractWorkspace
    Ke
    slot_buffers   # NamedTuple of element-local state buffers keyed by slot name
    re
    cell
    ivh
    element
    boundary_element
    scratch        # per-worker scratch instances: solver-declared ∪ element-declared
    scratch_decls  # the nullary constructors, kept for per-worker re-instantiation
    ad             # per-worker derivative-sweep buffers + ForwardDiff configs
end

Ferrite.reinit!(ws::AssemblyWorkspace, cellid) = reinit!(ws.cell, cellid)

function duplicate_for_device(device::AbstractCPUDevice, ws::AssemblyWorkspace)
    return create_assembly_workspace(
        duplicate_for_device(device, ws.element),
        duplicate_for_device(device, ws.boundary_element),
        ws.cell.dh,
        duplicate_for_device(device, ws.ivh),
        keys(ws.slot_buffers),
        ws.scratch_decls,
    )
end

"""
    create_assembly_workspace(element, boundary_element, sdh, ivh, slots)

Create a single [`AssemblyWorkspace`](@ref) with freshly allocated
element-local buffers, one state buffer per declared slot name. Slot buffers
are sized by `allocate_element_unknown_vector`, so condensed elements get
their full `[ū; q]`-sized local vectors for every slot.
"""
function create_assembly_workspace(element, boundary_element, sdh, ivh, slots::NTuple{N, Symbol} = (:u,), scratch_decls::NamedTuple = (;)) where {N}
    slot_buffers = NamedTuple{slots}(ntuple(_ -> allocate_element_unknown_vector(element, sdh), N))
    return AssemblyWorkspace(
        allocate_element_matrix(element, sdh),
        slot_buffers,
        allocate_element_residual_vector(element, sdh),
        CellCache(sdh),
        ivh,
        element,
        boundary_element,
        map(f -> f(), scratch_decls),
        scratch_decls,
        create_ad_workspace(element, sdh),
    )
end

####################################
## Partition                      ##
####################################

"""
    CellItems(sdh)

The default work-item provider: the cells of one `SubDofHandler`. Item
providers are what `compute_partition` consumes — future item families
(interface pairs, contact pairs, patches, local BVPs) are further provider
types, not `SubDofHandler`s.
"""
struct CellItems{SDH <: SubDofHandler}
    sdh::SDH
end

"""
    compute_partition(strategy, provider)

Compute the work partition for the given strategy and item provider.
Returns an iterable of iterables: the outer level represents synchronization barriers
(e.g. colors), the inner level represents parallelizable work items (cell IDs).
"""
compute_partition(strategy::AssemblyStrategy, sdh::SubDofHandler) = compute_partition(strategy.scheduling, CellItems(sdh))
compute_partition(strategy::AssemblyStrategy, provider) = compute_partition(strategy.scheduling, provider)
compute_partition(::SequentialScheduling, provider::CellItems) = (collect(provider.sdh.cellset),)

function compute_partition(scheduling::ColoredScheduling, provider::CellItems)
    return Ferrite.create_coloring(get_grid(provider.sdh.dh), collect(provider.sdh.cellset); alg=scheduling.alg)
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

matrix_type(strategy::AssemblyStrategy) = matrix_type(strategy.device, strategy.form.operator_specification)
matrix_type(device::AbstractDevice, ::StandardOperatorSpecification) = SparseMatrixCSC{value_type(device), index_type(device)}
vector_type(strategy::AbstractAssemblyStrategy) = vector_type(strategy.device)
vector_type(device::AbstractDevice) = Vector{value_type(device)}

function Adapt.adapt_structure(::AbstractAssemblyStrategy, dh::DofHandler)
    error("Device specific implementation for `adapt_structure(::AbstractAssemblyStrategy,dh::DofHandler)` is not implemented yet")
end
