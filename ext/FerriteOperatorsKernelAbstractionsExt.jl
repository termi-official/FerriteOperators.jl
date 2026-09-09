module FerriteOperatorsKernelAbstractionsExt

using FerriteOperators, Ferrite

import Adapt: Adapt, adapt
import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize, @uniform, @groupsize

import FerriteOperators: KernelAbstractionsDevice, AssemblyWorkspace, AssemblyTask, VectorAssembler
import FerriteOperators: CooperativeElement, WorkerPerElement, QVector, MatrixFreeActionKind
import FerriteOperators: device_worker_view, launch_geometry, n_workers, value_type
import FerriteOperators: cooperative_group_size, cooperative_scratch_shape,
    cooperative_load!, cooperative_stage!, cooperative_store!
import FerriteOperators: element_value_type, item_dofs, query_cell_parameters
import FerriteOperators: assembly_iterator, item_update_flags, position_item

# Ferrite's own KA extension supplies `distribute_to_workers`, the device
# `CellCache` and every `Adapt` rule this one builds on, and it needs four
# packages loaded, not one. Without it `adapt(backend, dh)` silently returns the
# HOST handler (Adapt's fallback is the identity), which is the difference
# between a loud setup error and a kernel reading host memory.
function _assert_ferrite_ka_loaded()
    Base.get_extension(Ferrite, :FerriteKAExt) === nothing && throw(ArgumentError(
        "FerriteKAExt is not loaded, so Ferrite has no device handler, geometry cache or " *
        "struct-of-arrays layout. It is triggered by Adapt, GPUArrays, GPUArraysCore and " *
        "KernelAbstractions together — `using CUDA` loads all four; otherwise load them " *
        "explicitly."))
    return nothing
end

####################################
## Device transfer
####################################

Adapt.@adapt_structure AssemblyWorkspace
Adapt.@adapt_structure AssemblyTask
# The flat per-cell quadrature store: its `data` is what moves, the offset layout
# being an isbits range wherever the point count is uniform.
Adapt.@adapt_structure QVector

# The `atomic` parameter is a compile-time constant, not a field, so the
# generated rule's positional constructor would drop it.
Adapt.adapt_structure(to, a::VectorAssembler{<:Any, <:Any, atomic}) where {atomic} =
    _adapt_vector_assembler(adapt(to, a.f), Val(atomic))
_adapt_vector_assembler(f::VT, ::Val{atomic}) where {VT, atomic} =
    VectorAssembler{eltype(VT), VT, atomic}(f)

# Field-wise adaptation for element caches, so a cache author writes the
# batching (`setup_device_instances`) and the slicing (`device_worker_view`) and
# no third rule: the fields crossing the kernel boundary are the batched ones,
# and each already carries its own. A cache whose type parameters are not
# determined by its fields needs its own `Adapt.adapt_structure` method.
Adapt.adapt_structure(to, cache::FerriteOperators.AbstractVolumetricElementCache) =
    _adapt_fields(to, cache)
_adapt_fields(to, x::T) where {T} =
    Base.typename(T).wrapper(ntuple(i -> adapt(to, getfield(x, i)), Val(fieldcount(T)))...)

####################################
## Device item iteration
####################################

"""
    DeviceCellDofs(cell_dofs, base, n) <: AbstractVector{Int}

One cell's dof range as a VIEW into the device handler's flat `cell_dofs`: three
registers, no storage and no copy. It is what [`DeviceCellCursor`](@ref) answers
`celldofs` with, and the reason a device action sweep neither stages a dof row
nor reads one back.
"""
struct DeviceCellDofs{I <: Integer, V <: AbstractVector{I}} <: AbstractVector{I}
    cell_dofs::V
    base::Int
    n::Int
end
Base.size(d::DeviceCellDofs) = (d.n,)
Base.IndexStyle(::Type{<:DeviceCellDofs}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(d::DeviceCellDofs, i::Int) = d.cell_dofs[d.base + i]

"""
    DeviceCellCursor

The device counterpart of Ferrite's `CellCache` for a sweep positioned by
CONSTRUCTION: an immutable value the kernel holds in registers, carrying the
item's id and the offset of its dof range in the handler's own `cell_dofs`.
Positioning it is one indexed read of `cell_dofs_offset` — the item index is
already in a register — where a `CellCache` would copy the cell's dof row into a
per-worker slab that the gather and the scatter then read back out of global
memory.

`coords` is the per-worker coordinate slab, the one member a positioning may
still stage, and only for a sweep whose [`item_update_flags`](@ref) ask for it:
the two stored levels of the matrix-free action read the cell id alone, while
`Recompute()` re-derives the geometry per quadrature point and the
quadrature-data fill forms it once. The node ids are answered from the grid
instead of staged, there being no sweep that reads them per point.

`stride` is the subdomain's constant per-cell dof count where
[`with_uniform_dof_stride`](@ref) found `cell_dofs_offset` affine in the cell
id (`Int`), letting a positioning compute `dofbase` by arithmetic instead of
reading that array — or `Nothing`, the array-read fallback every other
subdomain shape takes. It is a TYPE parameter rather than a runtime branch, so
the two `_reposition` methods below compile to two different kernels and
neither pays for the other's check.

!!! warning "Experimental surface"
    Internal to the device sweep; it may change in a minor release.
"""
struct DeviceCellCursor{SDH, X, S}
    sdh::SDH
    coords::X
    cellid::Int
    dofbase::Int
    stride::S
end

Adapt.@adapt_structure DeviceCellCursor

Ferrite.cellid(c::DeviceCellCursor) = c.cellid
Ferrite.celldofs(c::DeviceCellCursor) =
    DeviceCellDofs(c.sdh.cell_dofs, c.dofbase, Ferrite.ndofs_per_cell(c.sdh))
Ferrite.getcoordinates(c::DeviceCellCursor) = c.coords
Ferrite.getnodes(c::DeviceCellCursor) =
    Ferrite.get_node_ids(Ferrite.getcells(Ferrite.get_grid(c.sdh), c.cellid))
Ferrite.reinit!(cv::Ferrite.AbstractCellValues, c::DeviceCellCursor) = Ferrite.reinit!(
    cv, Ferrite.reinit_needs_cell(cv) ? Ferrite.getcells(Ferrite.get_grid(c.sdh), c.cellid) : nothing,
    c.coords)

# The scatter addresses the dof view directly: every assembler in this package
# reads `celldofs` off the address it is handed, and the cursor's is already the
# handler's own range.
@inline FerriteOperators._scatter_address(::Nothing, c::DeviceCellCursor) = Ferrite.celldofs(c)

@inline function FerriteOperators._position_cell(ws::AssemblyWorkspace, cell::DeviceCellCursor, item, kind)
    positioned = _reposition(cell, item, item_update_flags(kind, ws.element))
    FerriteOperators._refresh_dof_head!(ws.dofs, positioned)
    return FerriteOperators._with_cell(ws, positioned)
end

@inline function _reposition(c::DeviceCellCursor{<:Any, <:Any, Int}, item, flags::Ferrite.UpdateFlags)
    i = Int(item)
    flags.coords && Ferrite.getcoordinates!(c.coords, Ferrite.get_grid(c.sdh), i)
    return DeviceCellCursor(c.sdh, c.coords, i, (i - 1) * c.stride, c.stride)
end
@inline function _reposition(c::DeviceCellCursor{<:Any, <:Any, Nothing}, item, flags::Ferrite.UpdateFlags)
    i = Int(item)
    flags.coords && Ferrite.getcoordinates!(c.coords, Ferrite.get_grid(c.sdh), i)
    return DeviceCellCursor(c.sdh, c.coords, i, Int(@inbounds c.sdh.cell_dofs_offset[i]) - 1, nothing)
end

"""
    with_uniform_dof_stride(c::DeviceCellCursor, sdh::Ferrite.SubDofHandler)

Where `sdh`'s flat `cell_dofs_offset` is affine in the cell id — every cell up
to and including this subdomain's shares one dof count — carries that count on
the cursor so `_reposition` computes `dofbase` by arithmetic. `nothing`
otherwise (a grid mixing element types/orders before this subdomain's cells),
which keeps the array-read `_reposition` method.
"""
FerriteOperators.with_uniform_dof_stride(c::DeviceCellCursor, sdh::Ferrite.SubDofHandler) =
    DeviceCellCursor(c.sdh, c.coords, c.cellid, c.dofbase, _uniform_dof_stride(sdh))

function _uniform_dof_stride(sdh::Ferrite.SubDofHandler)
    stride = Ferrite.ndofs_per_cell(sdh)
    offsets = sdh.dh.cell_dofs_offset
    for cid in sdh.cellset
        offsets[cid] == (cid - 1) * stride + 1 || return nothing
    end
    return stride
end

"""
    assembly_iterator(::MatrixFreeActionKind, element_cache, sdh)

The matrix-free action's iterator: Ferrite's `CellCache` over a HOST
`SubDofHandler` — the CPU sweeps are unchanged — and a [`DeviceCellCursor`](@ref)
over a device handler, which is where staging a dof row per item is the
dominant cost of the sweep.
"""
FerriteOperators.assembly_iterator(::MatrixFreeActionKind, element_cache, sdh) = _action_iterator(sdh)
_action_iterator(sdh::Ferrite.SubDofHandler) = Ferrite.CellCache(sdh)
_action_iterator(sdh) = DeviceCellCursor(sdh, nothing, -1, 0, nothing)

# The coordinate slab is the cursor's only batched member; the handler behind it
# is shared read-only, and the position is per item rather than per worker.
FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice, c::DeviceCellCursor, n_instances::Int) =
    DeviceCellCursor(c.sdh,
        KA.zeros(device.backend, Ferrite.get_coordinate_type(Ferrite.get_grid(c.sdh)),
                 n_instances, Ferrite.nnodes_per_cell(c.sdh)),
        c.cellid, c.dofbase, c.stride)

device_worker_view(c::DeviceCellCursor, worker) =
    DeviceCellCursor(c.sdh, device_worker_view(c.coords, worker), c.cellid, c.dofbase, c.stride)

####################################
## Setup
####################################

function FerriteOperators.setup_device_handler(device::KernelAbstractionsDevice, dh::Ferrite.AbstractDofHandler)
    _assert_ferrite_ka_loaded()
    return adapt(device.backend, dh)
end

FerriteOperators.adapt_partition(device::KernelAbstractionsDevice, partition) =
    [_adapt_chunk(device, color) for color in partition]
# A `UnitRange` chunk (`compute_partition`'s contiguous-cellset case) needs no
# device copy: it is already isbits, and the kernel indexes it by arithmetic.
_adapt_chunk(device::KernelAbstractionsDevice, color::AbstractUnitRange{Int}) = color
_adapt_chunk(device::KernelAbstractionsDevice, color) = adapt(device.backend, collect(Int, color))

"""
    adapt_shared(device::KernelAbstractionsDevice, x)

Shared read-only cache data, moved into device memory once at setup — a plain
`adapt` onto the backend, which is what makes an element cache's
quadrature-data store reach the kernel without the cache naming a backend
itself.
"""
FerriteOperators.adapt_shared(device::KernelAbstractionsDevice, x) = adapt(device.backend, x)

FerriteOperators.allocate_vector(device::KernelAbstractionsDevice, dh) =
    KA.zeros(device.backend, value_type(device), ndofs(dh))

FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice, cv::CellValues, n_instances::Int) =
    Ferrite.distribute_to_workers(device.backend, cv, n_instances)

"""
    setup_device_instances(device::KernelAbstractionsDevice, a::AbstractArray, n)

A plain per-worker scratch buffer, batched with the WORKER as the leading
(stride-1) index — the forward of [`device_worker_view`](@ref)'s existing
`AbstractArray` method, so a cache whose scratch is an ordinary array reaches
the device through the same pair every other cache uses. The batch is zeroed:
scratch carries nothing between items.
"""
FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice, a::AbstractArray, n_instances::Int) =
    KA.zeros(device.backend, eltype(a), n_instances, size(a)...)

"""
    setup_device_instances(device::KernelAbstractionsDevice, ws::AssemblyWorkspace, n, iterator)

The batched workspace `n` GPU workers share: the element buffers become one
array each with the WORKER as the leading (stride-1) index, so consecutive
workers touch adjacent addresses, and the item iterator and element cache
recurse into their own struct-of-arrays layouts.
[`device_worker_view`](@ref) is the inverse.

Each batch carries the ELTYPE of the host buffer it replaces, so the element's
own precision follows onto the device rather than the device's `value_type`
overriding it.

`iterator` is the subdomain's DEVICE item iterator
([`assembly_iterator`](@ref) over the device handler) — the host
`SubDofHandler` the workspace's own iterator carries would give a cache over the
host grid, which `adapt` returns unchanged and no error reports.
"""
function FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice,
        ws::AssemblyWorkspace, n_instances::Int, iterator)
    iterator === nothing && throw(ArgumentError(
        "$(nameof(typeof(device))) needs the subdomain's device item iterator to build a device " *
        "workspace. Workspaces reach it through the four-argument `setup_device_instances`, " *
        "which `setup_subdomain_caches` calls."))
    backend = device.backend
    ndofs_local = size(ws.Ke, 1)
    return AssemblyWorkspace(
        KA.zeros(backend, eltype(ws.Ke), n_instances, ndofs_local, ndofs_local),
        # Batched for layout symmetry; a device sweep never gathers into them
        # (`load_slots!` resizes, which a worker view cannot do — see
        # `execute_on_device!`).
        map(buffer -> KA.zeros(backend, eltype(buffer), n_instances, length(buffer)), ws.slot_buffers),
        KA.zeros(backend, eltype(ws.re), n_instances, length(ws.re)),
        _batch_iterator(device, iterator, n_instances),
        FerriteOperators.DeviceInternalVariableHandler(),
        FerriteOperators.setup_device_instances(device, ws.element, n_instances),
        ws.sensitivity,
        ws.dofs,
    )
end

# Ferrite's own geometry cache batches through its struct-of-arrays route; a
# cursor batches only what it stages.
_batch_iterator(device, cc::CellCache, n) = Ferrite.distribute_to_workers(device.backend, cc, n)
_batch_iterator(device, c::DeviceCellCursor, n) = FerriteOperators.setup_device_instances(device, c, n)
# A cooperative sweep positions its item in ONE segment of a barrier-split kernel
# and reads it in the next, so its geometry cache has to be the MUTABLE one every
# segment recovers for itself — a cursor is positioned by construction, and the
# value would not survive the barrier.
_batch_iterator(device::KernelAbstractionsDevice{<:Any, <:Any, <:Any, CooperativeElement},
        c::DeviceCellCursor, n) = Ferrite.distribute_to_workers(device.backend, CellCache(c.sdh), n)

####################################
## Execution
####################################

# One work item per grid-stride step, following Ferrite's GPU assembly how-to:
# the per-worker scratch is sliced once, and the loop body is the same
# `reinit!` + driver-body pair every device runs. It reaches the driver through
# `execute_kind!` rather than `execute_single_task!`, whose `@timeit_debug`
# frame GPUCompiler rejects.
@kernel function _cell_sweep_kernel!(task, workspaces, @Const(items))
    worker = @index(Global, Linear)
    stride = prod(KA.@ndrange())
    local_task = AssemblyTask(task.kind, _worker_assembler(task.inner_assembler, worker),
                              task.states, task.p, task.ctx)
    base = device_worker_view(workspaces, worker)
    n_items = length(items)
    for i in worker:stride:n_items
        # `position_item` rather than `reinit!`: a device iterator positions by
        # CONSTRUCTION, so only the workspace it returns is on the item, and it
        # stages what this KIND reads and nothing else.
        ws = position_item(base, @inbounds(items[i]), local_task.kind)
        FerriteOperators.execute_kind!(local_task.kind, local_task, ws)
    end
end

# Ferrite's `distribute_to_workers` covers its own assemblers — a shared handle
# on the GPU, a real per-worker copy on the CPU backend, whose CSC assembler
# owns permutation buffers. This package's `VectorAssembler` owns none and is
# shared, which is what the `get_substruct` method beside it says.
_worker_assemblers(backend, assembler::VectorAssembler, n) = Ferrite.SoAContainer(assembler, n)
_worker_assemblers(backend, assembler, n) = Ferrite.distribute_to_workers(backend, assembler, n)
# A sweep that scatters nothing carries no assembler (`QuadratureDataKind` writes
# into the element cache's own store), and a worker's share of none is none.
_worker_assemblers(backend, ::Nothing, n) = nothing
@inline _worker_assembler(assemblers, worker) = assemblers[worker]
@inline _worker_assembler(::Nothing, worker) = nothing

FerriteOperators.execute_on_device!(task, device::KernelAbstractionsDevice, workspaces, items) =
    _grid_stride_sweep!(task, device, workspaces, items)

# One work item per grid-stride step, the mapping every device kind but the
# cooperative one runs in.
function _grid_stride_sweep!(task, device::KernelAbstractionsDevice, workspaces, items)
    # The built-in primal driver gathers the global vectors into the per-worker
    # slot buffers through `load_slots!`, and that gather RESIZES them — which a
    # worker's view of a shared batch cannot do. A kind carrying its own driver
    # answers for its own gather; `MatrixFreeActionKind`'s is fixed-width.
    (task.kind isa FerriteOperators.PrimalKind && FerriteOperators.depends_on_unknowns(task.kind)) && throw(ArgumentError(
        "$(nameof(typeof(device))) does not support $(nameof(typeof(task.kind))) sweeps: they " *
        "gather the state slots per item, and the gather resizes a per-worker buffer that is a " *
        "view into a shared device batch. Assemble state-dependent kinds on a CPU device."))
    backend = device.backend
    # The same count `setup_device_instances` sized the caches for; the kernel
    # indexes them unchecked, and `launch_geometry` is monotone in the item
    # count, so no barrier launches more workers than there are caches.
    n = n_workers(device, items)
    device_task = adapt(backend, AssemblyTask(
        task.kind, _worker_assemblers(backend, task.inner_assembler, n),
        task.states, task.p, task.ctx))

    for chunk in items
        isempty(chunk) && continue
        workgroup, blocks = launch_geometry(device, length(chunk))
        kernel = _cell_sweep_kernel!(backend, workgroup)
        kernel(device_task, workspaces, chunk; ndrange = workgroup * blocks)
        # The barriers are what makes the colored scatter race-free: the next
        # color must not start while workers of this one are still accumulating.
        KA.synchronize(backend)
    end
    return nothing
end

####################################
## Cooperative execution: one workgroup per element
####################################

# Each pipeline step is its own function because a KernelAbstractions
# `@synchronize` is a LEXICAL split of the kernel body: the statements between
# two barriers become their own workitem loop on the CPU backend, so nothing
# assigned in one segment survives into the next and every segment recovers its
# worker view itself.
#
# A backend's index intrinsics are not all `Int` — CUDA's are `Int32` — so every
# step normalizes them once and the element entries see the `Int` pair their
# contract names.
@inline function _coop_prepare!(workspaces, items, group, lane, nlanes)
    lane, nlanes = Int(lane), Int(nlanes)
    ws = device_worker_view(workspaces, group)
    # One lane positions the group's geometry cache; the residual buffer needs
    # no cell to be zeroed.
    lane == 1 && Ferrite.reinit!(ws, items[group])
    for i in lane:nlanes:length(ws.re)
        @inbounds ws.re[i] = zero(eltype(ws.re))
    end
    return nothing
end

# The gather is its own barrier-separated step because the dofs it reads are
# what `reinit!` above wrote, and the lattice load below reads slots other lanes
# gathered.
@inline function _coop_gather!(task, workspaces, group, lane, nlanes)
    lane, nlanes = Int(lane), Int(nlanes)
    ws = device_worker_view(workspaces, group)
    dofs = item_dofs(ws)
    buffer = ws.slot_buffers.u
    for i in lane:nlanes:length(dofs)
        @inbounds buffer[i] = task.states.u[dofs[i]]
    end
    return nothing
end

@inline function _coop_load!(workspaces, group, lane, nlanes, scratch)
    lane, nlanes = Int(lane), Int(nlanes)
    ws = device_worker_view(workspaces, group)
    cooperative_load!(scratch, ws.element, ws.slot_buffers.u, lane, nlanes)
    return nothing
end

@inline function _coop_stage!(task, workspaces, group, lane, nlanes, scratch, stage)
    lane, nlanes = Int(lane), Int(nlanes)
    ws = device_worker_view(workspaces, group)
    args = CellArgs((u = ws.slot_buffers.u,), ws.cell,
                    query_cell_parameters(ws.element, ws.cell, task.p), task.ctx)
    cooperative_stage!(scratch, ws.element, args, stage, lane, nlanes)
    return nothing
end

@inline function _coop_store!(workspaces, group, lane, nlanes, scratch)
    lane, nlanes = Int(lane), Int(nlanes)
    ws = device_worker_view(workspaces, group)
    cooperative_store!(ws.re, scratch, ws.element, lane, nlanes)
    return nothing
end

@inline function _coop_scatter!(task, workspaces, group, lane, nlanes)
    lane, nlanes = Int(lane), Int(nlanes)
    ws = device_worker_view(workspaces, group)
    dofs = item_dofs(ws)
    slice = lane:nlanes:length(dofs)
    Ferrite.assemble!(task.inner_assembler[group], view(dofs, slice), view(ws.re, slice))
    return nothing
end

@kernel function _cooperative_sweep_2d!(task, workspaces, @Const(items),
        ::Val{T}, ::Val{LEN}, ::Val{NBOX}) where {T, LEN, NBOX}
    group = @index(Group, Linear)
    lane  = @index(Local, Linear)
    @uniform nlanes = prod(@groupsize())
    scratch = @localmem T (LEN, NBOX)
    _coop_prepare!(workspaces, items, group, lane, nlanes)
    @synchronize
    _coop_gather!(task, workspaces, group, lane, nlanes)
    @synchronize
    _coop_load!(workspaces, group, lane, nlanes, scratch)
    @synchronize
    _coop_stage!(task, workspaces, group, lane, nlanes, scratch, 1)
    @synchronize
    _coop_stage!(task, workspaces, group, lane, nlanes, scratch, 2)
    @synchronize
    _coop_stage!(task, workspaces, group, lane, nlanes, scratch, 3)
    @synchronize
    _coop_store!(workspaces, group, lane, nlanes, scratch)
    @synchronize
    _coop_scatter!(task, workspaces, group, lane, nlanes)
end

@kernel function _cooperative_sweep_3d!(task, workspaces, @Const(items),
        ::Val{T}, ::Val{LEN}, ::Val{NBOX}) where {T, LEN, NBOX}
    group = @index(Group, Linear)
    lane  = @index(Local, Linear)
    @uniform nlanes = prod(@groupsize())
    scratch = @localmem T (LEN, NBOX)
    _coop_prepare!(workspaces, items, group, lane, nlanes)
    @synchronize
    _coop_gather!(task, workspaces, group, lane, nlanes)
    @synchronize
    _coop_load!(workspaces, group, lane, nlanes, scratch)
    @synchronize
    _coop_stage!(task, workspaces, group, lane, nlanes, scratch, 1)
    @synchronize
    _coop_stage!(task, workspaces, group, lane, nlanes, scratch, 2)
    @synchronize
    _coop_stage!(task, workspaces, group, lane, nlanes, scratch, 3)
    @synchronize
    _coop_stage!(task, workspaces, group, lane, nlanes, scratch, 4)
    @synchronize
    _coop_stage!(task, workspaces, group, lane, nlanes, scratch, 5)
    @synchronize
    _coop_store!(workspaces, group, lane, nlanes, scratch)
    @synchronize
    _coop_scatter!(task, workspaces, group, lane, nlanes)
end

"""
    execute_on_device!(task, device::KernelAbstractionsDevice{…, CooperativeElement}, workspaces, items)

The intra-element mapping: ONE WORKGROUP per item, sized by the element
(`cooperative_group_size`), with the element state and every contraction
intermediate in group-local memory and a barrier between pipeline steps.

It consumes neither `execute_single_task!` nor the whole-element kernel — the
element's cooperative entries are a different decomposition of the same math —
and it launches one group per item rather than grid-striding, because a barrier
cannot live inside a loop the CPU backend has to split. That is why
[`n_workers`](@ref) is the largest barrier here: every item of it holds a
workspace slice concurrently.

The PARTIAL-assembly fill a cooperative operator also runs
([`QuadratureDataKind`](@ref)) has no lattice pipeline — one cell fills its own
per-quadrature-point slice — and takes the grid-stride mapping instead.
"""
function FerriteOperators.execute_on_device!(task,
        device::KernelAbstractionsDevice{<:Any, <:Any, <:Any, CooperativeElement}, workspaces, items)
    task.kind isa FerriteOperators.QuadratureDataKind &&
        return _grid_stride_sweep!(task, device, workspaces, items)
    task.kind isa FerriteOperators.MatrixFreeActionKind || throw(ArgumentError(
        "`CooperativeElement` executes the matrix-free action and nothing else (got " *
        "$(nameof(typeof(task.kind)))): its kernel is the element's own lattice pipeline, not " *
        "the generic per-item driver. Elect `element_mapping = WorkerPerElement()` for every " *
        "other sweep."))
    backend = device.backend
    cache = workspaces.element
    dim = FerriteOperators.cooperative_lattice_dim(cache)
    dim in (2, 3) || throw(ArgumentError(
        "The cooperative kernel is written for 2D and 3D lattices; $(nameof(typeof(cache))) " *
        "declares `cooperative_lattice_dim` $dim. The pipeline length is `2·dim - 1` and every " *
        "barrier is a statement in the kernel body, so a new lattice dimension is a new kernel."))
    workgroup = cooperative_group_size(cache)
    boxes, columns = cooperative_scratch_shape(cache)
    T = Val(element_value_type(cache))
    device_task = adapt(backend, AssemblyTask(
        task.kind, _worker_assemblers(backend, task.inner_assembler, n_workers(device, items)),
        task.states, task.p, task.ctx))

    for chunk in items
        isempty(chunk) && continue
        ndrange = workgroup * length(chunk)
        if dim == 3
            _cooperative_sweep_3d!(backend, workgroup)(device_task, workspaces, chunk, T, boxes, columns; ndrange)
        else
            _cooperative_sweep_2d!(backend, workgroup)(device_task, workspaces, chunk, T, boxes, columns; ndrange)
        end
        KA.synchronize(backend)
    end
    return nothing
end

function FerriteOperators.reduce_on_device(task, device::KernelAbstractionsDevice, workspaces, items)
    throw(ArgumentError(
        "$(nameof(typeof(device))) does not support value-returning sweeps (functionals, " *
        "quadrature evaluation): the per-worker partials and their reduction have no device " *
        "path yet. Evaluate the functional on a CPU device."))
end

end # module FerriteOperatorsKernelAbstractionsExt
