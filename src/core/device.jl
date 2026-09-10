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

The scalar element type of the GLOBAL system `device` assembles into (the
`ValueType` type parameter of its `AbstractDevice`): the system matrix and
vector it allocates. The ELEMENT-local scalar is the integrator's own election
([`element_value_type`](@ref)) and need not match — the scatter converts.
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
    setup_device_instances(device, object, n_instances, iterator)

Device scratch: `object` duplicated for `n_instances` parallel workers.
[`SequentialCPUDevice`](@ref) returns the 1-element tuple `(object,)`, a
threaded CPU device a `Vector` of `n_instances` independent
`duplicate_for_device` copies, and a GPU device a struct-of-arrays variant
whose per-worker slice is [`device_worker_view`](@ref).

The 4-argument form is what the engine calls on a subdomain's WORKSPACE, and
`iterator` is that subdomain's DEVICE-resident item iterator
([`assembly_iterator`](@ref) over the device handler
[`setup_device_handler`](@ref) resolved) — the workspace the host allocated
carries a host geometry cache, which `adapt` returns unchanged and no error
reports. It defaults to the 3-argument form, which every nested call (element
caches, values objects) uses.
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

setup_device_instances(device::AbstractDevice, obj, n_instances, iterator) =
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
# touch adjacent addresses), and Ferrite's struct-of-arrays container. A field
# an iterator does not stage is batched as nothing and sliced as nothing.
device_worker_view(a::AbstractArray, worker) = Ferrite.view_from_shared(a, worker)
device_worker_view(c::Ferrite.SoAContainer, worker) = c[worker]
device_worker_view(::Nothing, worker) = nothing

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
    adapt_shared(device, x) -> x

Read-only data every worker reads UNCHANGED, as the device's kernels consume
it: the identity on a CPU device, moved into device memory once at setup on a
GPU one. The counterpart of [`setup_device_instances`](@ref) for what is not per
worker — an element cache's quadrature-data store, a coefficient table — so a
cache's device layout says which of its fields are batched and which are shared
without naming a backend.

!!! warning "Experimental surface"
    This hook may change in a minor release.
"""
adapt_shared(::AbstractDevice, x) = x

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
    AbstractElementMapping

How ONE item's work maps onto a device's workers. The election lives on the
operator's FORM ([`MatrixFreeAction`](@ref)) and is realized on the device
through [`with_element_mapping`](@ref), because the seams that change shape with
it — [`n_workers`](@ref), [`setup_device_instances`](@ref),
[`execute_on_device!`](@ref) — all read the device.

[`WorkerPerElement`](@ref) is what every device does today; assembling forms
know no other.

!!! warning "Experimental surface"
    This mapping family may change in a minor release.
"""
abstract type AbstractElementMapping end

"""
    WorkerPerElement()

One worker owns one item from gather to scatter — the classical mapping, and
the only one an assembling form takes.
"""
struct WorkerPerElement <: AbstractElementMapping end

"""
    CooperativeElement()

One WORKGROUP owns one item: the group's workers split the element's lattice
between them, stage the element state and the intermediates in group-local
memory and synchronize between contraction stages. Needs group-local memory and
barriers, so it is a [`KernelAbstractionsDevice`](@ref) mapping only, and an
element serves it only where it implements the cooperative kernel entries
([`cooperative_stage!`](@ref)).
"""
struct CooperativeElement <: AbstractElementMapping end

"""
    LanesPerElement(; lanes = nothing)

One element's OUTPUT ROWS split across a block of lanes: lane `l` of a block of
`nlanes` owns rows `l:nlanes:ndofs_per_cell` of `yₑ`, reads all of `uₑ` through
the item's dof window, keeps its row's accumulator in a REGISTER and scatters
that row itself. The lanes of one element exchange no value, so this mapping
needs neither group-local memory nor a barrier — it is a grid-stride kernel over
`(element, lane)` pairs — and it is a [`KernelAbstractionsDevice`](@ref) mapping
only, a CPU device sweeping one item per worker from gather to scatter.

`lanes` is a LAUNCH POLICY, not element math: `nothing` takes the block from the
element's own extent ([`element_local_length`](@ref)) capped by the device's
`max_workgroup_size`, and an explicit count overrides that. A block shorter than
the row count gives each lane several rows, which is the same kernel. The
workgroup hosts `max_workgroup_size ÷ nlanes` ELEMENTS, so a small element
leaves no lane of a group idle.

It serves the [`ElementAssembly`](@ref) storage level alone, through
[`element_action_row`](@ref): the two per-quadrature-point levels re-derive an
element's values objects per cell into per-worker state that the lanes of one
element would race on. Its scatter addresses one dof per owned row, so an item
family whose scatter address is not a dof vector is outside it. Both are setup
errors naming the alternative.

The contrast with [`CooperativeElement`](@ref) is what makes it a separate
mapping: that one exists to split an element's LATTICE and needs the barriers
that go with it, while this one splits the element's dense product's ROWS, which
are independent by construction.

!!! warning "Experimental surface"
    This mapping and the element entry it calls may change in a minor release.
"""
struct LanesPerElement <: AbstractElementMapping
    lanes::Union{Int, Nothing}
end
LanesPerElement(; lanes::Union{Integer, Nothing} = nothing) =
    LanesPerElement(lanes === nothing ? nothing : Int(lanes))

"""
    with_element_mapping(device, mapping) -> device

The device that executes `mapping` ([`AbstractElementMapping`](@ref)), which a
matrix-free operator resolves once at [`setup_operator`](@ref): the mapping is
elected on the FORM, and the two setup seams that must change shape with it —
[`n_workers`](@ref) and [`setup_device_instances`](@ref) — read the device and
nothing else.

[`WorkerPerElement`](@ref) is every device's own execution and returns it
unchanged. A mapping a device has no kernel for is rejected here, at setup,
rather than as a missing method inside the first sweep.
"""
with_element_mapping(device::AbstractDevice, ::WorkerPerElement) = device

"""
    element_mapping(device) -> AbstractElementMapping

The mapping `device` executes, [`WorkerPerElement`](@ref) unless
[`with_element_mapping`](@ref) put another one there. This is how a CACHE
answers the mapping at setup — its per-worker scratch is not needed under
[`CooperativeElement`](@ref), whose kernel stages the same buffers in
group-local memory instead.
"""
element_mapping(::AbstractDevice) = WorkerPerElement()
with_element_mapping(device::AbstractDevice, mapping::AbstractElementMapping) = throw(ArgumentError(
    "$(nameof(typeof(device))) cannot execute $(nameof(typeof(mapping))): mapping one element " *
    "onto a cooperating group of workers needs group-local memory and barriers, which only a " *
    "KernelAbstractions backend exposes. Use `element_mapping = WorkerPerElement()`, or a " *
    "`KernelAbstractionsDevice`."))
with_element_mapping(device::AbstractDevice, ::LanesPerElement) = throw(ArgumentError(
    "$(nameof(typeof(device))) cannot execute LanesPerElement: splitting one element's output " *
    "rows across a block of lanes is a KernelAbstractions LAUNCH GEOMETRY, and a CPU device " *
    "sweeps one item per worker from gather to scatter. Use " *
    "`element_mapping = WorkerPerElement()`, or a `KernelAbstractionsDevice`."))

"""
    KernelAbstractionsDevice(backend; value_type = Float64, index_type = Int,
                             items_per_worker = 2, max_workgroup_size = 64)

Assembly on a KernelAbstractions.jl `backend` — `CUDABackend()` for CUDA,
`KA.CPU()` to run the same kernels on the host. Load KernelAbstractions.jl and
Adapt.jl to activate this device; the vendor package (and the device matrix type
it names, see [`StandardOperatorSpecification`](@ref)) is the caller's, this
package depends on none.

`value_type`/`index_type` are the scalar and matrix-index types of the GLOBAL
system — `Float32`/`Int32` is the usual GPU election. The precision the element
caches evaluate in is the integrator's, elected through its quadrature
collection ([`element_value_type`](@ref)), and the two are free to differ.

`items_per_worker` and `max_workgroup_size` are the LAUNCH POLICY, and
[`n_workers`](@ref) derives the per-worker cache size from them, so a sweep
launches exactly the geometry setup allocated for. A barrier of `n` items runs
`ceil(n / items_per_worker)` workers in groups of at most `max_workgroup_size`,
so a SMALLER group size means MORE groups over the same workers — which is what
spreads a colour across the device's multiprocessors, and what the default
favours. `items_per_worker` trades the per-worker scratch (element buffers,
geometry cache and values objects, all sized by the worker count) against that
parallelism.

`element_mapping` is resolved from the operator's form
([`with_element_mapping`](@ref)) and is not a constructor argument: a
[`FullAssembly`](@ref) operator is always [`WorkerPerElement`](@ref), and a
[`MatrixFreeAction`](@ref) one elects between that and
[`CooperativeElement`](@ref).

An ASSEMBLING sweep requires [`ColoredScheduling`](@ref) and covers CELL items
only. What is rejected at setup, each with a message naming the limitation:
[`SequentialScheduling`](@ref) under [`FullAssembly`](@ref), facet items,
algebraic items, patch and transfer operators, condensed internal state,
nonlinear (AD-decorated) integrators, a
[`BlockedOperatorSpecification`](@ref), constraints declared on the operator
specification, [`global_dofs`](@ref) declarations, and a matrix type Ferrite has
no assembler for. Value-returning sweeps (functionals, quadrature evaluation)
are rejected when they run.
"""
struct KernelAbstractionsDevice{Backend, ValueType, IndexType, Mapping} <: AbstractGPUDevice{ValueType, IndexType}
    backend::Backend
    items_per_worker::Int
    max_workgroup_size::Int
    element_mapping::Mapping
end
KernelAbstractionsDevice(backend; value_type::Type = Float64, index_type::Type = Int,
        items_per_worker::Int = 2, max_workgroup_size::Int = 64) =
    KernelAbstractionsDevice{typeof(backend), value_type, index_type, WorkerPerElement}(
        backend, items_per_worker, max_workgroup_size, WorkerPerElement())

element_mapping(device::KernelAbstractionsDevice) = device.element_mapping

for Mapping in (:WorkerPerElement, :CooperativeElement, :LanesPerElement)
    @eval with_element_mapping(device::KernelAbstractionsDevice{B, V, I}, mapping::$Mapping) where {B, V, I} =
        KernelAbstractionsDevice{B, V, I, $Mapping}(
            device.backend, device.items_per_worker, device.max_workgroup_size,
            _validated_mapping(device, mapping))
end

# The one mapping carrying a launch policy of its own: a lane block wider than
# the group it has to fit in cannot be launched, and saying so here is saying it
# at setup.
_validated_mapping(::KernelAbstractionsDevice, mapping::AbstractElementMapping) = mapping
function _validated_mapping(device::KernelAbstractionsDevice, mapping::LanesPerElement)
    mapping.lanes === nothing && return mapping
    1 ≤ mapping.lanes ≤ device.max_workgroup_size || throw(ArgumentError(
        "`LanesPerElement(; lanes = $(mapping.lanes))` does not fit this device's launch policy: " *
        "the lanes of one element are a block WITHIN a workgroup, so the count is between 1 and " *
        "the device's `max_workgroup_size` ($(device.max_workgroup_size)). Raise " *
        "`max_workgroup_size`, or drop `lanes` and let the element's own extent set it."))
    return mapping
end

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

"""
    lane_launch_geometry(device, nlanes, n_items) -> (workgroup_size, n_workgroups, n_slots)

The launch geometry of one barrier under [`LanesPerElement`](@ref): `n_slots`
ELEMENT slots, each a block of `nlanes` consecutive lanes, packed
`max_workgroup_size ÷ nlanes` blocks to a workgroup so a short block leaves no
lane of a group idle.

`n_slots` is [`launch_geometry`](@ref)'s own worker count for the same barrier —
the lane block stages nothing per lane, so the per-worker caches this mapping
needs are the grid-stride mapping's and [`n_workers`](@ref) sizes them
unchanged. Whole blocks are what a workgroup holds, so the last group may carry
slots beyond `n_slots`; the kernel drops those rather than the geometry rounding
the cache size up.
"""
function lane_launch_geometry(device::KernelAbstractionsDevice, nlanes::Integer, n_items::Integer)
    n_slots = prod(launch_geometry(device, n_items))
    n_slots ≤ 0 && return (Int(nlanes), 0, 0)
    per_group = clamp(device.max_workgroup_size ÷ nlanes, 1, n_slots)
    return Int(nlanes) * per_group, cld(n_slots, per_group), n_slots
end
