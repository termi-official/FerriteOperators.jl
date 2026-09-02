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

Device scratch: `object` duplicated for `n_instances` parallel workers.
[`SequentialCPUDevice`](@ref) returns the 1-element tuple `(object,)`, a
threaded CPU device a `Vector` of `n_instances` independent
`duplicate_for_device` copies, and a GPU device should return a
struct-of-arrays variant.
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
