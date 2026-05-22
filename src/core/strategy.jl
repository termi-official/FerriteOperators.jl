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

setup_operator_strategy_cache(strategy, integrator, dh) = strategy

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
    AbstractAssemblyCache

Supertype for element-local scratch caches. Concrete variants carry only the
buffers their element actually needs (`Ke` for a pure bilinear element,
`re` for a pure linear element, all three for a nonlinear element, …). The
element picks the variant via [`allocate_assembly_cache`](@ref); tasks read
their buffers directly off `ws.cache` (e.g. `ws.cache.Ke`).

Field types differ between CPU and GPU:
- **CPU:** matrices/vectors per worker.
- **GPU pool:** 3D/2D pools indexed by worker id.
- **GPU per-thread:** views over the pool (`SubArray`s).
"""
abstract type AbstractAssemblyCache end

"""
    BilinearAssemblyCache(Ke)

Cache for a pure bilinear element — only the local stiffness matrix `Ke`.
"""
@concrete struct BilinearAssemblyCache <: AbstractAssemblyCache
    Ke
end
Adapt.@adapt_structure BilinearAssemblyCache

"""
    LinearAssemblyCache(re)

Cache for a pure linear element — only the local residual vector `re`.
"""
@concrete struct LinearAssemblyCache <: AbstractAssemblyCache
    re
end
Adapt.@adapt_structure LinearAssemblyCache

"""
    NonlinearAssemblyCache(Ke, ue, re)

Cache for a nonlinear element — local jacobian `Ke`, unknown buffer `ue`,
and residual `re`. Covers `AssembleLinearizationJ`, `AssembleLinearizationR`,
and `AssembleLinearizationJR`.
"""
@concrete struct NonlinearAssemblyCache <: AbstractAssemblyCache
    Ke
    ue
    re
end
Adapt.@adapt_structure NonlinearAssemblyCache

# Per-thread slicing of pool buffers — invoked by `workspace_for` on GPU.
# Each variant slices exactly the fields it carries.
@inline per_thread(c::BilinearAssemblyCache, tid) =
    BilinearAssemblyCache(view(c.Ke, :, :, tid))
@inline per_thread(c::LinearAssemblyCache, tid) =
    LinearAssemblyCache(view(c.re, :, tid))
@inline per_thread(c::NonlinearAssemblyCache, tid) =
    NonlinearAssemblyCache(view(c.Ke, :, :, tid), view(c.ue, :, tid), view(c.re, :, tid))

"""
    AssemblyWorkspace

Per-worker scratch for assembling local element contributions. The same struct
is used for both CPU and GPU; the field types differ:

- **CPU (per worker):** `setup_device_instances` produces a `Vector` of these,
  one per worker. `cache` holds plain element-local buffers, `cell` is a
  `Ferrite.CellCache` (mutated via `reinit!`).
- **GPU (pool):** `setup_device_instances` produces a *single* workspace whose
  `cache` holds SOA pools indexed by the global thread id, and `cell` is a
  `Ferrite.CellCacheContainer`. Inside the kernel, `workspace_for(ws, tid, cellid)`
  returns a *per-thread* `AssemblyWorkspace` whose buffers/caches are views and
  functors over the pool.

Fields:
- `cache`: [`AssemblyCache`](@ref) — element-local buffers (`Ke`, `ue`, `re`)
- `cell`: geometry cache — [`CellCache`](@ref) on CPU, `CellCacheContainer` on
  the GPU pool, `ImmutableCellCache` per GPU thread
- `ivh`: internal variable handler
- `element`: element cache (subtype of [`AbstractVolumetricElementCache`](@ref))
- `boundary_element`: surface/boundary element cache
"""
@concrete struct AssemblyWorkspace <: AbstractWorkspace
    cache
    cell
    ivh
    element
    boundary_element
end
Adapt.@adapt_structure AssemblyWorkspace

Ferrite.reinit!(ws::AssemblyWorkspace, cellid) = reinit!(ws.cell, cellid)

function duplicate_for_device(device::AbstractCPUDevice, ws::AssemblyWorkspace)
    element_dup = duplicate_for_device(device, ws.element)
    sdh         = ws.cell.dh
    return AssemblyWorkspace(
        # Re-allocate same cache kind on target device using ws.cache as the prototype.
        allocate_assembly_cache(device, ws.cache, element_dup, sdh),
        CellCache(sdh),
        duplicate_for_device(device, ws.ivh),
        element_dup,
        duplicate_for_device(device, ws.boundary_element),
    )
end

"""
    create_assembly_workspace(integrator, element, boundary_element, sdh, ivh)

Create a single [`AssemblyWorkspace`](@ref) with freshly allocated CPU element-local
buffers. The cache variant is selected by `integrator`'s supertype via
`allocate_assembly_cache` (bilinear/linear/nonlinear → matching cache shape).
This is always a CPU prototype; GPU pooling is handled separately by
`setup_device_instances`.
"""
function create_assembly_workspace(integrator, element, boundary_element, sdh, ivh)
    cpu = SequentialCPUDevice()
    return AssemblyWorkspace(
        allocate_assembly_cache(cpu, integrator, element, sdh),
        CellCache(sdh),
        ivh,
        element,
        boundary_element,
    )
end


# extract one ws per thread from the pool.
@inline function workspace_for(ws::AssemblyWorkspace, tid, cellid)
    return AssemblyWorkspace(
        per_thread(ws.cache, tid),
        ws.cell[tid](Int(cellid)),
        ws.ivh,
        per_thread(ws.element, tid),
        per_thread(ws.boundary_element, tid),
    )
end

# TODO: upstream this to Ferrite ?
function Adapt.adapt_structure(backend::KA.Backend, sdh::SubDofHandler)
    KAExt = Base.get_extension(Ferrite, :FerriteKAExt)
    dh = sdh.dh
    gpu_grid      = KAExt.DeviceGrid(backend, dh.grid)
    dof_ranges    = Tuple(Ferrite.dof_range(sdh, i) for i in 1:length(sdh.field_names))
    field_indices = NamedTuple{ntuple(i -> dh.field_names[i], length(dh.field_names))}(collect(1:length(dh.field_names)))
    return KAExt.DeviceSubDofHandler(
        Adapt.adapt(backend, collect(Int, sdh.cellset)),
        Adapt.adapt(backend, dh.cell_dofs),
        Adapt.adapt(backend, dh.cell_dofs_offset),
        sdh.ndofs_per_cell,
        Ferrite.nnodes_per_cell(dh.grid, first(sdh.cellset)),
        field_indices,
        dof_ranges,
        gpu_grid,
    )
end

function setup_device_instances(device::AbstractGPUDevice, ws::AssemblyWorkspace, n_instances)
    backend    = KA.backend(device)
    cpu_sdh    = ws.cell.dh
    device_sdh = Adapt.adapt(backend, cpu_sdh)
    return AssemblyWorkspace(
        allocate_assembly_cache(device, ws.cache, ws.element, cpu_sdh),
        Ferrite.CellCacheContainer(backend, n_instances, device_sdh),
        duplicate_for_device(device, ws.ivh),
        duplicate_for_device(device, ws.element),
        #FIXME: this is broken upstream.
        duplicate_for_device(device, ws.boundary_element),
    )
end

# FIXME: this should be put in device.jl but cannot due to the depedency on woorkspaces.
KA.@kernel function _gpu_execute_kernel!(task, ws, @Const(cellids), num_items::Ti) where {Ti <: Integer}
    tid = convert(Ti, KA.@index(Global, Linear))
    if tid <= num_items
        stride = convert(Ti, prod(KA.@ndrange))
        idx = tid
        while idx <= num_items
            execute_single_task!(task, workspace_for(ws, tid, cellids[idx]))
            idx += stride
        end
    end
end

function execute_on_device!(task, device::AbstractGPUDevice, workspaces, items)
    backend = KA.backend(device)
    Ti = index_type(device)
    # `items` is already device-resident (adapted once in compute_partition) — no per-launch adapt.
    for chunck in items
        n = convert(Ti, length(chunck))
        kernel = _gpu_execute_kernel!(backend, Int(device.threads))
        kernel(task, workspaces, chunck, n; ndrange = Int(device.threads * device.blocks))
        KA.synchronize(backend)
    end
end

####################################
## Partition                      ##
####################################

# Move the partition's cell ids onto the device once, at setup time.
# CPU: no-op (keep the host OrderedSet / color vectors).
# GPU: collect each chunk into a device array so the kernel launch never re-adapts.
adapt_partition(::AbstractCPUDevice, partition) = partition
adapt_partition(device::AbstractGPUDevice, partition) =
    map(chunk -> Adapt.adapt(KA.backend(device), collect(chunk)), partition)

"""
    compute_partition(strategy, sdh)

Compute the work partition for the given strategy and SubDofHandler.
Returns an iterable of iterables: the outer level represents synchronization barriers
(e.g. colors), the inner level represents parallelizable work items (cell IDs).
The cell ids are adapted to the strategy's device once here (no-op on CPU).
"""
compute_partition(strategy::SequentialAssemblyStrategy, sdh) = adapt_partition(strategy.device, (sdh.cellset,))
compute_partition(strategy::ElementAssemblyOperatorStrategy, sdh) = adapt_partition(strategy.device, (sdh.cellset,))

function compute_partition(strategy::PerColorAssemblyStrategy, sdh)
    colors = Ferrite.create_coloring(get_grid(sdh.dh), sdh.cellset; alg=strategy.coloralg)
    return adapt_partition(strategy.device, colors)
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

# GPU: one worker per launched thread (threads × blocks). The device must be
# resolved (non-zero threads/blocks) by `resolve_device_config` before this point.
n_workers(strategy, device::AbstractGPUDevice, partition) = total_nthreads(device)


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

