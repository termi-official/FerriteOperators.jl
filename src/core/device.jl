abstract type AbstractDevice{ValueType, IndexType} end

"""
    AbstractCPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType}

Dispatch anchor for CPU devices ([`SequentialCPUDevice`](@ref), [`PolyesterDevice`](@ref)):
`duplicate_for_device`'s per-worker copies are plain Julia objects, as opposed
to a GPU device's struct-of-arrays layout.
"""
abstract type AbstractCPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end

"""
    AbstractGPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType}

Dispatch anchor for GPU devices: a downstream device implementation subtypes
this to pick up the GPU-shaped [`setup_device_instances`](@ref)/
[`duplicate_for_device`](@ref) contract instead of the CPU one.
"""
abstract type AbstractGPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end

"""
    value_type(device) -> Type

The scalar element type `device` assembles with (the `ValueType` type
parameter of its `AbstractDevice`).
"""
value_type(::AbstractDevice{ValueType}) where ValueType = ValueType
index_type(::AbstractDevice{<:Any, IndexType}) where IndexType = IndexType

"""
    duplicate_for_device(device, x)

`x`'s per-worker counterpart for `device`: an independent copy of mutable
per-worker scratch, the same object shared read-only where sharing is safe, or
(GPU) a device-resident layout.

The generic fallback shares an `isbits` `x` — there is nothing mutable to alias
— and throws a `MethodError` for anything else. So a cache or workspace type
reachable from a [`setup_operator`](@ref) call that owns mutable state needs a
method, and a missing one surfaces at setup rather than as an aliasing bug at
assembly time.
"""
function duplicate_for_device end


"""
    execute_on_device!(task, device, workspaces, items)

Execute a task on a device. `workspaces` holds the per-worker scratch; `items`
is a nested list of task-specific indices whose inner lists are guaranteed to
be executable in parallel.
"""
function execute_on_device!(task, device::AbstractDevice, workspaces, items)
    throw(ArgumentError(
        "Task execution is not yet implemented for $(typeof(device)). " *
        "Implement execute_on_device! for this device type."
    ))
end

"""
    reduce_on_device(task, device, workspaces, items) -> value

The VALUE-RETURNING counterpart of [`execute_on_device!`](@ref), for tasks
whose per-item kernel returns its contribution instead of scattering it —
nothing is written into the workspaces, so this is the shape a scalar or tensor
evaluation runs in. Values fold per worker in item order through
[`fold_items`](@ref) and the partials reduce in worker order, so the result is
deterministic for a fixed worker count. `nothing` means nothing contributed.
"""
function reduce_on_device(task, device::AbstractDevice, workspaces, items)
    throw(ArgumentError(
        "Value-returning task execution is not yet implemented for $(typeof(device)). " *
        "Implement reduce_on_device for this device type."
    ))
end

"""
    initial_partial(kind) -> zero(T) or `nothing`

The seed of a worker's fold: `zero(T)` for a kind declaring
[`functional_value_type`](@ref) `T`, `nothing` for one that does not, whose
accumulator type the first contributing item fixes instead. The additive
identity doubles as "has seen nothing yet" — a worker whose items all return
`nothing` hands back `zero(T)`, a no-op in the host reduction. Every consumer
relies on that, so a non-additive combiner would need its own neutral element.
"""
initial_partial(kind) = _initial_partial(functional_value_type(kind))
_initial_partial(::Type{Nothing}) = nothing
_initial_partial(::Type{T}) where {T} = zero(T)

"""
    fold_items(task, ws, items, acc = initial_partial(task.kind)) -> value

One worker's partial: run `task` over `items` in order on the workspace `ws`
and sum what the per-item kernels return into `acc`, a `nothing` return
contributing nothing. Pass the previous partial back in to continue the fold
over the next barrier of the worker's partition.

With a declared [`functional_value_type`](@ref) the fold returns `T`; without
one it returns `Union{Nothing, T}`, the first non-`nothing` value fixing the
accumulator's type through the function barrier below — which keeps the loop
doing the work concretely typed and dispatch-free either way.
"""
fold_items(task, ws, items) = fold_items(task, ws, items, initial_partial(task.kind))

# Declared value type: `acc` arrives typed, so there is nothing to scan for.
fold_items(task, ws, items, acc::T) where {T} =
    _fold_items_from(task, ws, items, firstindex(items) - 1, acc, T)

# Undeclared: scan for the first value, which fixes the accumulator's type.
function fold_items(task, ws, items, ::Nothing)
    for i in eachindex(items)
        reinit!(ws, items[i])
        val = execute_single_task!(task, ws)
        val === nothing || return _fold_items_from(task, ws, items, i, val, Nothing)
    end
    return nothing
end

function _fold_items_from(task, ws, items, start, acc, ::Type{T}) where {T}
    for i in (start + 1):lastindex(items)
        reinit!(ws, items[i])
        val = execute_single_task!(task, ws)
        val === nothing || (acc += _checked_contribution(task.kind, T, val))
    end
    return acc
end

# The declared value type is a contract with the kernels: unchecked, a
# disagreeing kernel would either widen the reduction silently or fail deep
# inside the accumulation. `Nothing` marks undeclared, and both branches fold
# away against a concrete `val`.
@inline _checked_contribution(kind, ::Type{Nothing}, val) = val
@inline function _checked_contribution(kind, ::Type{T}, val) where {T}
    val isa T && return val
    throw(ArgumentError(
        "$(nameof(typeof(kind))) declares `functional_value_type` $(T), but a kernel returned " *
        "$(typeof(val)). Make the kernel return $(T), or fix the declaration."))
end

# Fixed-order host-side reduction of two partials, either of which may be the
# "contributed nothing" marker.
_reduce_partials(::Nothing, ::Nothing) = nothing
_reduce_partials(::Nothing, b) = b
_reduce_partials(a, ::Nothing) = a
_reduce_partials(a, b) = a + b

"""
    setup_device_instances(device, object, n_instances)
    setup_device_instances(device, object, n_instances, device_handler)

Device scratch: `object` duplicated for `n_instances` parallel workers.
[`SequentialCPUDevice`](@ref) returns the 1-element tuple `(object,)`, a
threaded CPU device a `Vector` of `n_instances` independent
`duplicate_for_device` copies, and a GPU device a struct-of-arrays variant
whose per-worker slice is [`device_worker_view`](@ref).

The 4-argument form is what the engine calls on a subdomain's WORKSPACE, and
`device_handler` is that subdomain's device-resident `SubDofHandler`
([`setup_device_handler`](@ref)) — the only handler a device `Ferrite.CellCache`
may be built from. It defaults to the 3-argument form, which every nested call
(element caches, values objects) uses.
"""
function setup_device_instances(device::AbstractDevice, obj, n_instances)
    throw(ArgumentError(
        "Device cache setup is not yet implemented for $(typeof(device)). " *
        "Implement setup_device_instances for this device type."
    ))
end

function setup_device_instances(device::AbstractCPUDevice, obj, n_instances)
    return [duplicate_for_device(device, obj) for _ in 1:n_instances]
end

setup_device_instances(device::AbstractDevice, obj, n_instances, device_handler) =
    setup_device_instances(device, obj, n_instances)

# The author-facing half of the contract: what is missing on a GPU device is
# almost always the CACHE's pair of methods, not the device's, so the error says
# which object has no layout instead of blaming the device.
function setup_device_instances(device::AbstractGPUDevice, obj, n_instances)
    throw(ArgumentError(
        "$(nameof(typeof(obj))) has no struct-of-arrays layout for $(nameof(typeof(device))). " *
        "A cache reaching a GPU device declares which of its fields are batched per worker and " *
        "which are shared, through a pair of methods:\n" *
        "  FerriteOperators.setup_device_instances(dev::AbstractGPUDevice, c::$(nameof(typeof(obj))), n)\n" *
        "  FerriteOperators.device_worker_view(c::$(nameof(typeof(obj))), worker)\n" *
        "recursing into the batched fields and passing the shared ones through unchanged."))
end

"""
    device_worker_view(batched, worker)

Worker `worker`'s slice of a struct-of-arrays object
[`setup_device_instances`](@ref) built — the counterpart of indexing a CPU
device's per-worker `Vector`, and called INSIDE the assembly kernel.

There is deliberately no generic fallback. A cache author writes one method per
batched cache, slicing the fields that were batched and passing the shared ones
through unchanged:

    setup_device_instances(dev::AbstractGPUDevice, c::MyCache, n) =
        MyCache(c.D, setup_device_instances(dev, c.cellvalues, n))
    device_worker_view(c::MyCache, w) = MyCache(c.D, device_worker_view(c.cellvalues, w))

so a field that was forgotten surfaces as a `MethodError` rather than as
silently aliased scratch.
"""
function device_worker_view end

# The two shapes `setup_device_instances` returns for a GPU device: a batched
# array whose LEADING index is the worker (stride-1, so consecutive workers
# touch adjacent addresses), and Ferrite's struct-of-arrays container.
device_worker_view(a::AbstractArray, worker) = Ferrite.view_from_shared(a, worker)
device_worker_view(c::Ferrite.SoAContainer, worker) = c[worker]

"""
    setup_device_handler(device, dh) -> handler

The device-resident counterpart of `dh` a device's caches are built against,
resolved once per engine and split per subdomain by
[`device_subdomain_handler`](@ref). `nothing` for a CPU device, which reads the
host handler directly.
"""
setup_device_handler(::AbstractDevice, dh) = nothing

"""
    device_subdomain_handler(device_handler, subdomain_index)

Subdomain `subdomain_index`'s entry of what [`setup_device_handler`](@ref)
returned, `nothing` where there is no device handler.
"""
device_subdomain_handler(::Nothing, index) = nothing
device_subdomain_handler(device_handler, index) = device_handler.subdofhandlers[index]

"""
    adapt_partition(device, partition)

The partition as the device's kernels consume it. The identity on a CPU device;
a GPU device moves each barrier's item list into device memory here, once at
setup, so a sweep transfers nothing.
"""
adapt_partition(::AbstractDevice, partition) = partition



"""
    SequentialCPUDevice()

Sequential algorithms on CPU.
"""
struct SequentialCPUDevice{ValueType, IndexType} <: AbstractCPUDevice{ValueType, IndexType}
end
SequentialCPUDevice() = SequentialCPUDevice{Float64, Int}()

function execute_on_device!(task, device::SequentialCPUDevice, workspaces, items)
    workspace = workspaces[1]
    for chunk in items
        for cellid in chunk
            reinit!(workspace, cellid)
            execute_single_task!(task, workspace)
        end
    end
end

function reduce_on_device(task, device::SequentialCPUDevice, workspaces, items)
    workspace = workspaces[1]
    total = initial_partial(task.kind)
    for chunk in items
        total = fold_items(task, workspace, chunk, total)
    end
    return total
end

function setup_device_instances(::SequentialCPUDevice, obj, n_instances)
    return (obj,)
end



"""
    PolyesterDevice(; min_items_per_worker = 32)

Threaded algorithms via Polyester.jl. Load Polyester.jl to activate this device.

`min_items_per_worker` is the smallest share of a barrier's items a worker is
given: a barrier of `n` items runs on `min(Threads.nthreads(), cld(n,
min_items_per_worker))` workers, so a LOWER value means MORE parallelism on a
small barrier. Keyword-only — the number is a threshold, not a worker count.
"""
struct PolyesterDevice{ValueType, IndexType} <: AbstractCPUDevice{ValueType, IndexType}
    min_items_per_worker::Int
end
PolyesterDevice(; min_items_per_worker::Int = 32) = PolyesterDevice{Float64, Int}(min_items_per_worker)



"""
    KernelAbstractionsDevice(backend; value_type = Float64, index_type = Int,
                             items_per_worker = 2, max_workgroup_size = 64)

Assembly on a KernelAbstractions.jl `backend` — `CUDABackend()` for CUDA,
`KA.CPU()` to run the same kernels on the host. Load KernelAbstractions.jl and
Adapt.jl to activate this device; the vendor package (and the device matrix type
it names, see [`StandardOperatorSpecification`](@ref)) is the caller's, this
package depends on none.

`value_type`/`index_type` are the scalar and matrix-index types the operator
assembles with — `Float32`/`Int32` is the usual GPU election, and the element
caches are built for `value_type` through the three-argument
[`setup_element_cache`](@ref).

`items_per_worker` and `max_workgroup_size` are the LAUNCH POLICY, and
[`n_workers`](@ref) derives the per-worker cache size from them, so a sweep
launches exactly the geometry setup allocated for. A barrier of `n` items runs
`ceil(n / items_per_worker)` workers in groups of at most `max_workgroup_size`,
so a SMALLER group size means MORE groups over the same workers — which is what
spreads a colour across the device's multiprocessors, and what the default
favours. `items_per_worker` trades the per-worker scratch (element buffers,
geometry cache and values objects, all sized by the worker count) against that
parallelism.

REQUIRES [`ColoredScheduling`](@ref) and covers CELL items only. What is
rejected at setup, each with a message naming the limitation:
[`SequentialScheduling`](@ref), facet items, algebraic items, patch and transfer
operators, condensed internal state, nonlinear (AD-decorated) integrators, a
[`BlockedOperatorSpecification`](@ref), constraints declared on the operator
specification, [`global_dofs`](@ref) declarations, and a matrix type Ferrite has
no assembler for. Value-returning sweeps (functionals, quadrature evaluation)
are rejected when they run.
"""
struct KernelAbstractionsDevice{Backend, ValueType, IndexType} <: AbstractGPUDevice{ValueType, IndexType}
    backend::Backend
    items_per_worker::Int
    max_workgroup_size::Int
end
KernelAbstractionsDevice(backend; value_type::Type = Float64, index_type::Type = Int,
        items_per_worker::Int = 2, max_workgroup_size::Int = 64) =
    KernelAbstractionsDevice{typeof(backend), value_type, index_type}(
        backend, items_per_worker, max_workgroup_size)

"""
    launch_geometry(device::KernelAbstractionsDevice, n_items) -> (workgroup_size, n_workgroups)

The launch geometry of one barrier of `n_items` items: `items_per_worker` items
per worker, at most `max_workgroup_size` workers per group. `workgroup_size *
n_workgroups` is the worker count of THAT barrier, and it is monotone in
`n_items`, so it never exceeds the [`n_workers`](@ref) the caches were sized for.
"""
function launch_geometry(device::KernelAbstractionsDevice, n_items::Integer)
    n_items ≤ 0 && return (1, 0)
    per_worker  = min(device.items_per_worker, n_items)
    n_effective = cld(n_items, per_worker)
    workgroup   = min(device.max_workgroup_size, n_effective)
    return workgroup, cld(n_items, per_worker * workgroup)
end
