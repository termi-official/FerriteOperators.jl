module FerriteOperatorsKernelAbstractionsExt

using FerriteOperators, Ferrite

import Adapt: Adapt, adapt
import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index, @Const

import FerriteOperators: KernelAbstractionsDevice, AssemblyWorkspace, AssemblyTask, VectorAssembler
import FerriteOperators: device_worker_view, launch_geometry, n_workers, value_type

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
## Setup
####################################

function FerriteOperators.setup_device_handler(device::KernelAbstractionsDevice, dh::Ferrite.AbstractDofHandler)
    _assert_ferrite_ka_loaded()
    return adapt(device.backend, dh)
end

FerriteOperators.adapt_partition(device::KernelAbstractionsDevice, partition) =
    [adapt(device.backend, collect(Int, color)) for color in partition]

FerriteOperators.allocate_vector(device::KernelAbstractionsDevice, dh) =
    KA.zeros(device.backend, value_type(device), ndofs(dh))

FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice, cv::CellValues, n_instances::Int) =
    Ferrite.distribute_to_workers(device.backend, cv, n_instances)

"""
    setup_device_instances(device::KernelAbstractionsDevice, ws::AssemblyWorkspace, n, device_sdh)

The batched workspace `n` GPU workers share: the element buffers become one
array each with the WORKER as the leading (stride-1) index, so consecutive
workers touch adjacent addresses, and the geometry cache and element cache
recurse into their own struct-of-arrays layouts.
[`device_worker_view`](@ref) is the inverse.

Each batch carries the ELTYPE of the host buffer it replaces, so the element's
own precision follows onto the device rather than the device's `value_type`
overriding it.

The geometry cache is built from `device_sdh`, the subdomain's
`DeviceSubDofHandler` — the host `SubDofHandler` the workspace carries would
give a cache over the host grid, which `adapt` returns unchanged and no error
reports.
"""
function FerriteOperators.setup_device_instances(device::KernelAbstractionsDevice,
        ws::AssemblyWorkspace, n_instances::Int, device_sdh)
    device_sdh === nothing && throw(ArgumentError(
        "$(nameof(typeof(device))) needs the subdomain's device handler to build a device " *
        "geometry cache. Workspaces reach it through the four-argument " *
        "`setup_device_instances`, which `setup_subdomain_caches` calls."))
    backend = device.backend
    ndofs_local = size(ws.Ke, 1)
    return AssemblyWorkspace(
        KA.zeros(backend, eltype(ws.Ke), n_instances, ndofs_local, ndofs_local),
        # Batched for layout symmetry; a device sweep never gathers into them
        # (`load_slots!` resizes, which a worker view cannot do — see
        # `execute_on_device!`).
        map(buffer -> KA.zeros(backend, eltype(buffer), n_instances, length(buffer)), ws.slot_buffers),
        KA.zeros(backend, eltype(ws.re), n_instances, length(ws.re)),
        Ferrite.distribute_to_workers(backend, CellCache(device_sdh), n_instances),
        FerriteOperators.DeviceInternalVariableHandler(),
        FerriteOperators.setup_device_instances(device, ws.element, n_instances),
        ws.sensitivity,
        ws.dofs,
    )
end

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
    local_task = AssemblyTask(task.kind, task.inner_assembler[worker], task.states, task.p, task.ctx)
    ws = device_worker_view(workspaces, worker)
    for i in worker:stride:length(items)
        Ferrite.reinit!(ws, items[i])
        FerriteOperators.execute_kind!(local_task.kind, local_task, ws)
    end
end

# Ferrite's `distribute_to_workers` covers its own assemblers — a shared handle
# on the GPU, a real per-worker copy on the CPU backend, whose CSC assembler
# owns permutation buffers. This package's `VectorAssembler` owns none and is
# shared, which is what the `get_substruct` method beside it says.
_worker_assemblers(backend, assembler::VectorAssembler, n) = Ferrite.SoAContainer(assembler, n)
_worker_assemblers(backend, assembler, n) = Ferrite.distribute_to_workers(backend, assembler, n)

function FerriteOperators.execute_on_device!(task, device::KernelAbstractionsDevice, workspaces, items)
    # State-dependent kinds gather the global vectors into the per-worker slot
    # buffers, and that gather RESIZES them — which a worker's view of a shared
    # batch cannot do. `update_operator!` on a bilinear or linear operator, the
    # state-independent pair, is what this device serves.
    FerriteOperators.depends_on_unknowns(task.kind) && throw(ArgumentError(
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

function FerriteOperators.reduce_on_device(task, device::KernelAbstractionsDevice, workspaces, items)
    throw(ArgumentError(
        "$(nameof(typeof(device))) does not support value-returning sweeps (functionals, " *
        "quadrature evaluation): the per-worker partials and their reduction have no device " *
        "path yet. Evaluate the functional on a CPU device."))
end

end # module FerriteOperatorsKernelAbstractionsExt
