module FerriteOperatorsKernelAbstractionsExt

using FerriteOperators, Ferrite

import Adapt: Adapt, adapt
import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize, @uniform, @groupsize

import FerriteOperators: KernelAbstractionsDevice, AssemblyWorkspace, AssemblyTask, VectorAssembler
import FerriteOperators: CooperativeElement, WorkerPerElement, LanesPerElement, QVector,
    MatrixFreeActionKind
import FerriteOperators: device_worker_view, launch_geometry, lane_launch_geometry, n_workers,
    value_type
import FerriteOperators: cooperative_group_size, cooperative_scratch_shape,
    cooperative_load!, cooperative_stage!, cooperative_store!
import FerriteOperators: element_action_row, element_local_length, ElementUnknownWindow
import FerriteOperators: element_value_type, item_dofs, query_cell_parameters
import FerriteOperators: default_assembly_iterator, decorate_device_iterator,
    item_update_flags, position_item
import FerriteOperators: position_iterator, iterator_dofs

# Without FerriteKAExt, `adapt(backend, dh)` silently returns the HOST handler
# (Adapt's fallback is the identity) and the kernel reads host memory.
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
Adapt.@adapt_structure QVector

# The `atomic` parameter is a compile-time constant, not a field, so the
# generated rule's positional constructor would drop it.
Adapt.adapt_structure(to, a::VectorAssembler{<:Any, <:Any, atomic}) where {atomic} =
    _adapt_vector_assembler(adapt(to, a.f), Val(atomic))
_adapt_vector_assembler(f::VT, ::Val{atomic}) where {VT, atomic} =
    VectorAssembler{eltype(VT), VT, atomic}(f)

# Field-wise, so a cache author writes only the batching/slicing pair. A cache
# whose type parameters are not determined by its fields needs its own method.
Adapt.adapt_structure(to, cache::FerriteOperators.AbstractVolumetricElementCache) =
    _adapt_fields(to, cache)
_adapt_fields(to, x::T) where {T} =
    Base.typename(T).wrapper(ntuple(i -> adapt(to, getfield(x, i)), Val(fieldcount(T)))...)

####################################
## Device item iteration
####################################

# One cell's dof range as a VIEW into the device handler's flat `cell_dofs`, so
# a device action sweep never stages a dof row.
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

The device counterpart of Ferrite's `CellCache`, positioned by CONSTRUCTION: an
immutable value the kernel holds in registers.

`coords` is the only member a positioning may stage, and only where
[`item_update_flags`](@ref) ask for it. `stride` is the subdomain's constant
per-cell dof count, or `Nothing` for the array-read fallback; it is a TYPE
parameter, so the two [`position_iterator`](@ref) methods below compile to two
kernels and neither pays for the other's check.

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

FerriteOperators.iterator_dofs(c::DeviceCellCursor) = Ferrite.celldofs(c)

@inline function FerriteOperators.position_iterator(c::DeviceCellCursor{<:Any, <:Any, Int}, item, flags::Ferrite.UpdateFlags)
    i = Int(item)
    flags.coords && Ferrite.getcoordinates!(c.coords, Ferrite.get_grid(c.sdh), i)
    return DeviceCellCursor(c.sdh, c.coords, i, (i - 1) * c.stride, c.stride)
end
@inline function FerriteOperators.position_iterator(c::DeviceCellCursor{<:Any, <:Any, Nothing}, item, flags::Ferrite.UpdateFlags)
    i = Int(item)
    flags.coords && Ferrite.getcoordinates!(c.coords, Ferrite.get_grid(c.sdh), i)
    return DeviceCellCursor(c.sdh, c.coords, i, Int(@inbounds c.sdh.cell_dofs_offset[i]) - 1, nothing)
end

FerriteOperators.default_assembly_iterator(::MatrixFreeActionKind, sdh) = _action_iterator(sdh)
_action_iterator(sdh::Ferrite.SubDofHandler) = Ferrite.CellCache(sdh)
_action_iterator(sdh) = DeviceCellCursor(sdh, nothing, -1, 0, nothing)

# The stride exists only where `cell_dofs_offset` is affine in the cell id over
# GLOBAL cell numbering, which is what `_uniform_dof_stride` checks on the HOST.
FerriteOperators.decorate_device_iterator(c::DeviceCellCursor, sdh::Ferrite.SubDofHandler) =
    DeviceCellCursor(c.sdh, c.coords, c.cellid, c.dofbase, _uniform_dof_stride(sdh))

function _uniform_dof_stride(sdh::Ferrite.SubDofHandler)
    stride = Ferrite.ndofs_per_cell(sdh)
    offsets = sdh.dh.cell_dofs_offset
    for cid in sdh.cellset
        offsets[cid] == (cid - 1) * stride + 1 || return nothing
    end
    return stride
end

# `coords` is the only batched member; the handler is shared read-only.
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
# A `UnitRange` chunk is already isbits and the kernel indexes it by arithmetic.
_adapt_chunk(device::KernelAbstractionsDevice, color::AbstractUnitRange{Int}) = color
_adapt_chunk(device::KernelAbstractionsDevice, color) = adapt(device.backend, collect(Int, color))

FerriteOperators.adapt_shared(device::KernelAbstractionsDevice, x) = adapt(device.backend, x)

FerriteOperators.allocate_vector(device::KernelAbstractionsDevice, dh) =
    KA.zeros(device.backend, value_type(device), ndofs(dh))

FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice, cv::CellValues, n_instances::Int) =
    Ferrite.distribute_to_workers(device.backend, cv, n_instances)

# The WORKER is the leading (stride-1) index, the inverse of
# `device_worker_view`'s `AbstractArray` method.
FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice, a::AbstractArray, n_instances::Int) =
    KA.zeros(device.backend, eltype(a), n_instances, size(a)...)

# Each batch carries the ELTYPE of the host buffer it replaces, so the element's
# own precision follows onto the device. `iterator` must be the subdomain's
# DEVICE iterator: the workspace's own carries the HOST handler, which `adapt`
# returns unchanged and no error reports.
function FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice,
        ws::AssemblyWorkspace, n_instances::Int, iterator)
    iterator === nothing && throw(ArgumentError(
        "$(nameof(typeof(device))) needs the subdomain's device item iterator to build a device " *
        "workspace. Workspaces reach it through the four-argument `setup_device_instances`, " *
        "which the cell family's `setup_family_caches` method calls."))
    backend = device.backend
    ndofs_local = size(ws.Ke, 1)
    return AssemblyWorkspace(
        KA.zeros(backend, eltype(ws.Ke), n_instances, ndofs_local, ndofs_local),
        # Batched for layout symmetry; a device sweep never gathers into them.
        map(buffer -> KA.zeros(backend, eltype(buffer), n_instances, length(buffer)), ws.slot_buffers),
        KA.zeros(backend, eltype(ws.re), n_instances, length(ws.re)),
        _batch_iterator(device, iterator, n_instances),
        FerriteOperators.DeviceInternalVariableHandler(),
        FerriteOperators.setup_device_instances(device, ws.element, n_instances),
        ws.sensitivity,
        ws.dofs,
    )
end

# The generic default is `setup_device_instances` itself, so a downstream device
# iterator implements only that 3-arg hook and needs no method here.
_batch_iterator(device, it, n) = FerriteOperators.setup_device_instances(device, it, n)

# Ferrite's geometry cache has no generic `setup_device_instances` method.
_batch_iterator(device, cc::CellCache, n) = Ferrite.distribute_to_workers(device.backend, cc, n)

# A cooperative sweep positions its item in ONE segment of a barrier-split
# kernel and reads it in the next, so its geometry cache has to be the MUTABLE
# one every segment recovers for itself: a cursor is positioned by construction
# and the value would not survive the barrier.
_batch_iterator(device::KernelAbstractionsDevice{<:Any, <:Any, <:Any, CooperativeElement},
        c::DeviceCellCursor, n) = Ferrite.distribute_to_workers(device.backend, CellCache(c.sdh), n)

####################################
## Execution
####################################

# Reaches the driver through `execute_kind!` rather than `execute_single_task!`,
# whose `@timeit_debug` frame GPUCompiler rejects.
@kernel function _cell_sweep_kernel!(task, workspaces, @Const(items))
    worker = @index(Global, Linear)
    stride = prod(KA.@ndrange())
    local_task = AssemblyTask(task.kind, _worker_assembler(task.inner_assembler, worker),
                              task.states, task.p, task.ctx)
    base = device_worker_view(workspaces, worker)
    n_items = length(items)
    for i in worker:stride:n_items
        # A device iterator positions by CONSTRUCTION, so only the workspace
        # `position_item` returns is on the item.
        ws = position_item(base, @inbounds(items[i]), local_task.kind)
        FerriteOperators.execute_kind!(local_task.kind, local_task, ws)
    end
end

# `VectorAssembler` owns no per-worker scratch and is shared across workers.
_worker_assemblers(backend, assembler::VectorAssembler, n) = Ferrite.SoAContainer(assembler, n)
_worker_assemblers(backend, assembler, n) = Ferrite.distribute_to_workers(backend, assembler, n)
_worker_assemblers(backend, ::Nothing, n) = nothing
@inline _worker_assembler(assemblers, worker) = assemblers[worker]
@inline _worker_assembler(::Nothing, worker) = nothing

FerriteOperators.execute_on_device!(task, device::KernelAbstractionsDevice, workspaces, items) =
    _grid_stride_sweep!(task, device, workspaces, items)

function _grid_stride_sweep!(task, device::KernelAbstractionsDevice, workspaces, items)
    # `load_slots!` RESIZES the per-worker slot buffers, which a worker's view of
    # a shared batch cannot do.
    (task.kind isa FerriteOperators.PrimalKind && FerriteOperators.depends_on_unknowns(task.kind)) && throw(ArgumentError(
        "$(nameof(typeof(device))) does not support $(nameof(typeof(task.kind))) sweeps: they " *
        "gather the state slots per item, and the gather resizes a per-worker buffer that is a " *
        "view into a shared device batch. Assemble state-dependent kinds on a CPU device."))
    backend = device.backend
    # The kernel indexes the per-worker caches unchecked, and `launch_geometry`
    # is monotone in the item count, so no barrier outruns this count.
    n = n_workers(device, items)
    device_task = adapt(backend, AssemblyTask(
        task.kind, _worker_assemblers(backend, task.inner_assembler, n),
        task.states, task.p, task.ctx))

    for chunk in items
        isempty(chunk) && continue
        workgroup, blocks = launch_geometry(device, length(chunk))
        kernel = _cell_sweep_kernel!(backend, workgroup)
        kernel(device_task, workspaces, chunk; ndrange = workgroup * blocks)
        # What makes the colored scatter race-free: the next color must not start
        # while workers of this one are still accumulating.
        KA.synchronize(backend)
    end
    return nothing
end

####################################
## Lane execution: one block of lanes per element
####################################

# Consecutive threads are consecutive SLOTS at the same lane, not the lanes of
# one element: `K`'s SLOT index is stride-1, so a warp of consecutive slots at
# one `(i, j)` reads adjacent addresses.
@inline function _lane_position(thread::Int, ::Val{NLANES}, ::Val{PER}) where {NLANES, PER}
    group, local_index = divrem(thread - 1, NLANES * PER)
    lane, block = divrem(local_index, PER)
    return group * PER + block + 1, lane + 1
end

# One element slot per grid-stride step, one ROW BLOCK per lane. No barrier and
# no group-local memory.
@kernel function _lane_sweep_kernel!(task, workspaces, @Const(items),
        ::Val{NLANES}, ::Val{PER}, ::Val{ND}, n_slots) where {NLANES, PER, ND}
    # `@index` must sit at the top of the body: on the CPU backend it expands
    # against a loop variable the kernel transform injects there, and inside a
    # call argument there is none.
    thread = @index(Global, Linear)
    slot, lane = _lane_position(Int(thread), Val(NLANES), Val(PER))
    # A workgroup carries whole blocks, so the last group may hold slots the
    # caches were not sized for.
    if slot ≤ n_slots
        local_task = AssemblyTask(task.kind, _worker_assembler(task.inner_assembler, slot),
                                  task.states, task.p, task.ctx)
        base = device_worker_view(workspaces, slot)
        n_items = length(items)
        for i in slot:n_slots:n_items
            ws = position_item(base, @inbounds(items[i]), local_task.kind)
            _lane_action!(local_task, ws, lane, Val(NLANES), Val(ND))
        end
    end
end

# The lanes of one element share the slot's positioned workspace READ-ONLY,
# which is why this mapping serves the ELEMENT level alone. `query_cell_parameters`
# runs once PER LANE on that shared cache, so a parameter query that GATHERS into
# the cache rather than returning a value would race; such a cache must serve
# `WorkerPerElement()` instead.
@inline function _lane_action!(task, ws, lane::Int, ::Val{NLANES}, ::Val{ND}) where {NLANES, ND}
    dofs = item_dofs(ws)
    uₑ = ElementUnknownWindow{eltype(ws.re)}(task.states.u, dofs)
    args = CellArgs((u = uₑ,), ws.cell, query_cell_parameters(ws.element, ws.cell, task.p), task.ctx)
    for i in lane:NLANES:ND
        yᵢ = element_action_row(ws.element, uₑ, args, i)
        Ferrite.assemble!(task.inner_assembler, (@inbounds dofs[i]), yᵢ)
    end
    return nothing
end

# The ELEMENT-level fill has no rows to split and takes the grid-stride mapping.
function FerriteOperators.execute_on_device!(task,
        device::KernelAbstractionsDevice{<:Any, <:Any, <:Any, LanesPerElement}, workspaces, items)
    task.kind isa FerriteOperators.QuadratureDataKind &&
        return _grid_stride_sweep!(task, device, workspaces, items)
    task.kind isa FerriteOperators.MatrixFreeActionKind || throw(ArgumentError(
        "`LanesPerElement` executes the matrix-free action and nothing else (got " *
        "$(nameof(typeof(task.kind)))): its kernel gives one lane one row of the element's " *
        "action, which is not a shape the generic per-item driver has. Elect " *
        "`element_mapping = WorkerPerElement()` for every other sweep."))
    backend = device.backend
    nd = element_local_length(workspaces.element)
    nlanes = _lane_count(device.element_mapping, nd, device.max_workgroup_size)
    device_task = adapt(backend, AssemblyTask(
        task.kind, _worker_assemblers(backend, task.inner_assembler, n_workers(device, items)),
        task.states, task.p, task.ctx))

    for chunk in items
        isempty(chunk) && continue
        workgroup, blocks, n_slots = lane_launch_geometry(device, nlanes, length(chunk))
        _lane_sweep_kernel!(backend, workgroup)(
            device_task, workspaces, chunk, Val(nlanes), Val(workgroup ÷ nlanes), nd, n_slots;
            ndrange = workgroup * blocks)
        KA.synchronize(backend)
    end
    return nothing
end

# `with_element_mapping` already checked an explicit `lanes` against the group size.
_lane_count(mapping::LanesPerElement, ::Val{ND}, max_workgroup_size::Int) where {ND} =
    mapping.lanes === nothing ? min(ND, max_workgroup_size) : mapping.lanes

####################################
## Cooperative execution: one workgroup per element
####################################

# Each pipeline step is its own function because a KernelAbstractions
# `@synchronize` LEXICALLY splits the kernel body: nothing assigned in one
# segment survives into the next, so every segment recovers its worker view
# itself. Index intrinsics are not `Int` on every backend (CUDA's are `Int32`),
# so each step normalizes them once.
@inline function _coop_prepare!(workspaces, items, group, lane, nlanes)
    lane, nlanes = Int(lane), Int(nlanes)
    ws = device_worker_view(workspaces, group)
    lane == 1 && Ferrite.reinit!(ws, items[group])
    for i in lane:nlanes:length(ws.re)
        @inbounds ws.re[i] = zero(eltype(ws.re))
    end
    return nothing
end

# Its own barrier-separated step: the dofs it reads are what `reinit!` above
# wrote, and the lattice load below reads slots other lanes gathered.
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

# `group`/`lane` MUST be re-derived at the top of the loop body: a `for` loop
# containing a `@synchronize` is transformed as its own independent split,
# starting from a fresh per-segment state. `nlanes` needs no re-derivation, a
# `@uniform` computed before any barrier being an ordinary closure variable.
@kernel function _cooperative_sweep!(task, workspaces, @Const(items),
        ::Val{T}, ::Val{LEN}, ::Val{NBOX}, ::Val{DIM}) where {T, LEN, NBOX, DIM}
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
    for stage in 1:(2DIM - 1)
        group = @index(Group, Linear)
        lane  = @index(Local, Linear)
        _coop_stage!(task, workspaces, group, lane, nlanes, scratch, stage)
        @synchronize
    end
    _coop_store!(workspaces, group, lane, nlanes, scratch)
    @synchronize
    _coop_scatter!(task, workspaces, group, lane, nlanes)
end

# One group per item rather than a grid stride, because a barrier cannot live
# inside a loop the CPU backend has to split — which is why `n_workers` is the
# largest barrier here. The fill has no lattice pipeline and takes the
# grid-stride mapping.
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
    workgroup = cooperative_group_size(cache)
    boxes, columns = cooperative_scratch_shape(cache)
    T = Val(element_value_type(cache))
    device_task = adapt(backend, AssemblyTask(
        task.kind, _worker_assemblers(backend, task.inner_assembler, n_workers(device, items)),
        task.states, task.p, task.ctx))

    for chunk in items
        isempty(chunk) && continue
        ndrange = workgroup * length(chunk)
        _cooperative_sweep!(backend, workgroup)(device_task, workspaces, chunk, T, boxes, columns, Val(dim); ndrange)
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
