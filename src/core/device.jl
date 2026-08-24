abstract type AbstractDevice{ValueType, IndexType} end
abstract type AbstractCPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end
abstract type AbstractGPUDevice{ValueType, IndexType} <: AbstractDevice{ValueType, IndexType} end

value_type(::AbstractDevice{ValueType}) where ValueType = ValueType
index_type(::AbstractDevice{<:Any, IndexType}) where IndexType = IndexType


"""
    execute_on_device!(task, device, workspaces, items)

Execute a task on a device. Workspaces store the worker-specific
scratches for the tasks. The items is a simple nested list of task-specific
indices, where the tasks identified by the inner list are guaranteed to be
executable in parallel.
"""
function execute_on_device!(task, device::AbstractDevice, workspaces, items)
    throw(ArgumentError(
        "Task execution is not yet implemented for $(typeof(device)). " *
        "Implement execute_on_device! for this device type."
    ))
end

"""
    reduce_on_device(task, device, workspaces, items) -> value

The VALUE-RETURNING execution shape, the counterpart of
[`execute_on_device!`](@ref) for tasks whose per-item kernel returns its
contribution instead of scattering it. Nothing is written into the workspaces,
so this is the shape a scalar or tensor evaluation runs in.

Per worker the item values fold in item order through [`fold_items`](@ref);
across workers the partials reduce in worker order. Both orders are fixed, so
the result is deterministic for a fixed worker count. `nothing` means nothing
contributed.
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
[`functional_value_type`](@ref) `T`, and `nothing` for one that does not,
whose accumulator type is fixed by the first item that contributes instead.

`zero(T)` is the additive identity of the reduction, which is what lets a
worker seeded with it stand for "has seen nothing yet": a worker whose items
all return `nothing` returns `zero(T)`, and the host reduction adding it in is
a no-op. Every consumer of this seed relies on that, so a reduction over a
non-additive combiner would need its own neutral element.
"""
initial_partial(kind) = _initial_partial(functional_value_type(kind))
_initial_partial(::Type{Nothing}) = nothing
_initial_partial(::Type{T}) where {T} = zero(T)

"""
    fold_items(task, ws, items, acc = initial_partial(task.kind)) -> value

One worker's partial: run `task` over `items` in order on the workspace `ws`
and sum what the per-item kernels return into `acc`, `nothing` from a kernel
meaning "contributed nothing". Pass the previous partial back in to continue a
worker's fold over the next barrier of its partition, so a worker's whole
contribution accumulates in one sequence whatever the partition's shape.

With a declared [`functional_value_type`](@ref) the accumulator is `T`-typed
from the first item and the fold returns `T`. Without one it returns
`Union{Nothing, T}`: the first non-`nothing` value fixes the accumulator's
type through the function barrier below. Either way the loop doing the work
carries a concrete accumulator and dispatches nothing per item.
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

# The declared value type is a contract with the kernels: without this a
# disagreeing kernel would either widen the reduction silently or fail deep
# inside the accumulation. `Nothing` is the undeclared marker — nothing to
# disagree with — and both branches fold away against a concrete `val`.
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

Create a device scratch by duplicating `object` for `n_instances` parallel workers.
For [`SequentialCPUDevice`](@ref), returns a 1-element tuple `(object,)`.
For threaded CPU devices, returns a `Vector` of `n_instances` independent copies
produced by `duplicate_for_device`.
For GPU devices this should return a struct of arrays variant of `object`.
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
    PolyesterDevice(chunksize)

Threaded algorithms via Polyester.jl. Load Polyester.jl to activate this device.
"""
struct PolyesterDevice{ValueType, IndexType} <: AbstractCPUDevice{ValueType, IndexType}
    chunksize::IndexType
end
PolyesterDevice() = PolyesterDevice{Float64, Int}(32)
PolyesterDevice(i::Int) = PolyesterDevice{Float64, Int}(i)



"""
    CudaDevice(threads, blocks)

Placeholder for a future GPU device axis. Not implemented: no CUDA extension
is declared, and every entry point throws `ArgumentError`.
"""
struct CudaDevice{ValueType, IndexType} <: AbstractGPUDevice{ValueType, IndexType}
    threads::Union{IndexType, Nothing}
    blocks::Union{IndexType, Nothing}
end

CudaDevice() = CudaDevice{Float32, Int32}(nothing, nothing)
CudaDevice(threads::IndexType, blocks::IndexType) where IndexType = CudaDevice{Float32, IndexType}(threads, blocks)
