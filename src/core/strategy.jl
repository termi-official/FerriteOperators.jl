"""
    StandardOperatorSpecification(; algebraic_couplings = (), constraint_handler = nothing)

The operator's global matrix as a monolithic
`SparseMatrixCSC{value_type(device), index_type(device)}`, over the pattern
[`create_system_matrix`](@ref) builds from two declarations, both of which
[`BlockedOperatorSpecification`](@ref) shares:

- `algebraic_couplings` — Ferrite coupling descriptors (`CellCoupling`,
  `FacetCoupling`, `AlgebraicCoupling`) for the entries an element's
  [`global_dofs`](@ref) couple into. WHICH items couple is the caller's
  statement, never inferred from the dof declaration; a missing descriptor
  surfaces as Ferrite's missing-sparsity-entry error on the first assembly.
- `constraint_handler` — the constraint entries (`add_constraint_entries!`),
  so condensation has room to write. Sparsity ONLY: applying the constraints
  stays the caller's, through Ferrite's `apply!`/`apply_assemble!`.

The zero-argument form is the default of [`FullAssembly`](@ref).
"""
struct StandardOperatorSpecification{C, CH}
    algebraic_couplings::C
    constraint_handler::CH
end
StandardOperatorSpecification(; algebraic_couplings = (), constraint_handler = nothing) =
    StandardOperatorSpecification(algebraic_couplings, constraint_handler)

"""
    BlockedOperatorSpecification(block_sizes, matrix_type; algebraic_couplings = (), constraint_handler = nothing)

The operator's global matrix as a `BlockMatrix` over the row/column split
`block_sizes`, allocated from a `BlockSparsityPattern` and from the same two
declarations as [`StandardOperatorSpecification`](@ref). `matrix_type` is
REQUIRED and is the caller's — this package depends on neither BlockArrays nor
SparseMatricesCSR, weakly or otherwise, so the user loads them and names the
type (`BlockMatrix{Float64, Matrix{SparseMatrixCSR{1, Float64, Int}}}`).

The residual stays a plain `Vector` — `create_system_vector` is unchanged, and
Ferrite's `BlockAssembler` takes a non-blocked `f`. A LINEAR operator holds no
matrix at all, so a blocked specification on one is rejected at setup rather
than silently dropped.
"""
struct BlockedOperatorSpecification{B, MT, C, CH}
    block_sizes::B
    matrix_type::MT
    algebraic_couplings::C
    constraint_handler::CH
end
BlockedOperatorSpecification(block_sizes, matrix_type::Type;
        algebraic_couplings = (), constraint_handler = nothing) =
    BlockedOperatorSpecification(block_sizes, matrix_type, algebraic_couplings, constraint_handler)

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
"""
    SequentialScheduling()

Race freedom by atomic scatter: every worker assembles into the same global
target and the assembler resolves collisions atomically. Requires an eltype
with atomic support.
"""
struct SequentialScheduling <: AbstractSchedulingPolicy end

"""
    ColoredScheduling(alg = ColoringAlgorithm.WorkStream)

Race freedom by coloring: the partition is a sequence of colors, and no two
items of one color share a dof, so the scatter needs no atomics. Required for
eltypes without atomic support, and what makes a run reproducible
element-by-element.
"""
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
and the device (where it runs). The named constructors below build the
common compositions.
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
- `Ferrite.reinit!(ws, item)` — position the workspace on the item its family is indexed by (a cell id, a patch index, an algebraic item index)
- `duplicate_for_device(device::AbstractCPUDevice, ws)` — create an independent copy for a parallel worker

New device backends must allocate and manage workspaces of a concrete subtype.
"""
abstract type AbstractWorkspace end

"""
    SensitivityBuffers

Per-worker OUTPUT buffers the five sensitivity requests
([`ParameterJacobianRequest`](@ref), [`ParameterVJPRequest`](@ref),
[`TimeSensitivityRequest`](@ref), [`StateJVPRequest`](@ref),
[`StateVJPRequest`](@ref)) accumulate into and the engine scatters, plus the
rectangular ∂F/∂q block ([`update_internal_jacobian!`](@ref)), which is the
same shape of thing: a local block whose column space is not the field space
— the "outputs and payload gathers" half of the AD decorator's buffer split
(see [`ADElementCache`](@ref) for the other half, the seeds/configs).

Element-sized members (`λₑ`, `vₑ`, `Jvₑ`, `gu`, `gₜ`) are eager. The
rectangular members are (re)allocated on first use, their column count being
call-time or per-item knowledge rather than setup knowledge: `θ`/`Bₑ`/`gθ`
through [`parameter_sweep_buffers!`](@ref) because nθ arrives with `p`, and
`Kqₑ` through [`internal_sweep_buffers!`](@ref) because an item declares its
own internal dof count.
"""
@concrete mutable struct SensitivityBuffers
    λₑ        # residual-sized adjoint gather
    vₑ        # unknown-sized JVP direction gather
    Jvₑ       # residual-sized JVP output
    gu        # unknown-sized state-VJP output
    gₜ        # residual-sized time-sensitivity output
    θ         # flat primal parameter copy (nθ)
    Bₑ        # local parameter Jacobian block (residual × nθ)
    gθ        # parameter pullback output (nθ)
    Kqₑ       # local ∂F/∂q block (residual × the item's condensed internal dof count)
end

function create_sensitivity_buffers(element, sdh, n_global_dofs::Int = 0)
    vₑ  = pad_element_vector(allocate_element_unknown_vector(element, sdh), n_global_dofs)
    gu  = pad_element_vector(allocate_element_unknown_vector(element, sdh), n_global_dofs)
    λₑ  = pad_element_vector(allocate_element_residual_vector(element, sdh), n_global_dofs)
    Jvₑ = pad_element_vector(allocate_element_residual_vector(element, sdh), n_global_dofs)
    gₜ  = pad_element_vector(allocate_element_residual_vector(element, sdh), n_global_dofs)
    T   = eltype(Jvₑ)
    return SensitivityBuffers(λₑ, vₑ, Jvₑ, gu, gₜ, Vector{T}(), Matrix{T}(undef, length(Jvₑ), 0),
                              Vector{T}(), Matrix{T}(undef, length(Jvₑ), 0))
end

# The size-based path: an item family whose local system is described by a dof
# count alone (an algebraic item) has no `SubDofHandler` to allocate against.
create_sensitivity_buffers(n::Int, ::Type{T}) where {T} = SensitivityBuffers(
    zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
    Vector{T}(), Matrix{T}(undef, n, 0), Vector{T}(), Matrix{T}(undef, n, 0))

duplicate_for_device(device::AbstractCPUDevice, s::SensitivityBuffers) =
    SensitivityBuffers(copy(s.λₑ), copy(s.vₑ), copy(s.Jvₑ), copy(s.gu), copy(s.gₜ), copy(s.θ),
                       copy(s.Bₑ), copy(s.gθ), copy(s.Kqₑ))

"""
    parameter_sweep_buffers!(s::SensitivityBuffers, nθ) -> SensitivityBuffers

Size the parameter-sweep members (`θ`, `Bₑ`, `gθ`) for `nθ` flat parameters,
reallocating only when nθ changed since the last sweep on this worker.
"""
function parameter_sweep_buffers!(s::SensitivityBuffers, nθ::Int)
    if length(s.θ) != nθ
        T = eltype(s.θ)
        s.θ  = Vector{T}(undef, nθ)
        s.gθ = Vector{T}(undef, nθ)
        s.Bₑ = Matrix{T}(undef, size(s.Bₑ, 1), nθ)
    end
    return s
end

"""
    internal_sweep_buffers!(s::SensitivityBuffers, nq) -> SensitivityBuffers

Size the ∂F/∂q block (`Kqₑ`) for an item owning `nq` condensed internal dofs,
reallocating only when that count changed since the last item on this worker.
The count is per ITEM rather than per subdomain — a cell's internal dof count
is its own declaration — so this is the block's sizer, not a per-sweep one
([`update_internal_jacobian!`](@ref)).
"""
function internal_sweep_buffers!(s::SensitivityBuffers, nq::Int)
    size(s.Kqₑ, 2) == nq || (s.Kqₑ = Matrix{eltype(s.Kqₑ)}(undef, size(s.Kqₑ, 1), nq))
    return s
end

"""
    AssemblyWorkspace

Per-worker workspace for square operator assembly (bilinear, nonlinear, linear):
a fixed CORE of pre-allocated element-local buffers and caches reused across
cells, plus the [`SensitivityBuffers`](@ref) a nonlinear operator's sensitivity
entry points need.

IMMUTABLE: every field is bound at construction, and a sweep works by filling
the buffers those fields point at. No sweep ever rebinds a field, which is what
lets the workspace cross a device boundary — a scalar or tensor evaluation
RETURNS its value instead of parking it in a slot here (see
[`evaluate_functional`](@ref)).

Core fields:
- `Ke`: element stiffness matrix
- `slot_buffers`: NamedTuple of element-local state buffers, one per declared slot
- `re`: element residual vector
- `cell`: geometry cache (`Ferrite.CellCache`)
- `ivh`: internal variable handler
- `element`: element cache (user-defined, subtype of [`AbstractVolumetricElementCache`](@ref))
- `boundary_element`: surface cache walked by the facet driver
- `sensitivity`: [`SensitivityBuffers`](@ref), or `nothing` for an operator
  family that never issues a sensitivity kind (bilinear, linear)
- `dofs`: the augmented dof vector `[celldofs(cell); the declared global dofs]`
  (see [`global_dofs`](@ref)), or `nothing` where the integrator declares none.
  The tail is written once at construction, the head refreshed by
  `Ferrite.reinit!`; the `nothing` type is what makes the un-augmented path
  return `celldofs(ws.cell)` directly, with neither copy nor run-time branch.
"""
@concrete struct AssemblyWorkspace <: AbstractWorkspace
    Ke
    slot_buffers   # NamedTuple of element-local state buffers keyed by slot name
    re
    cell
    ivh
    element
    boundary_element
    sensitivity    # SensitivityBuffers, or `nothing`
    dofs           # augmented dof vector, or `nothing`
end

function Ferrite.reinit!(ws::AssemblyWorkspace, cellid)
    reinit!(ws.cell, cellid)
    _refresh_dof_head!(ws.dofs, ws.cell)
    return ws
end
@inline _refresh_dof_head!(::Nothing, cell) = nothing
@inline _refresh_dof_head!(dofs, cell) = copyto!(dofs, celldofs(cell))

# The tail as declared, recovered from the augmented vector so a per-worker
# duplicate rebuilds the same layout without carrying the declaration along.
_declared_global_dofs(ws::AssemblyWorkspace) = _declared_global_dofs(ws.dofs, ws.cell.dh)
_declared_global_dofs(::Nothing, sdh) = ()
_declared_global_dofs(dofs, sdh) = @view dofs[(ndofs_per_cell(sdh) + 1):end]

function duplicate_for_device(device::AbstractCPUDevice, ws::AssemblyWorkspace)
    return create_assembly_workspace(
        duplicate_for_device(device, ws.element),
        duplicate_for_device(device, ws.boundary_element),
        ws.cell.dh,
        duplicate_for_device(device, ws.ivh),
        keys(ws.slot_buffers);
        needs_sensitivity = ws.sensitivity !== nothing,
        global_dofs = _declared_global_dofs(ws),
    )
end

"""
    create_assembly_workspace(element, boundary_element, sdh, ivh, slots;
                              needs_sensitivity = true, global_dofs = ())

Create a single [`AssemblyWorkspace`](@ref) with freshly allocated
element-local buffers, one state buffer per declared slot name. Slot buffers
are sized by `allocate_element_unknown_vector` at construction, matching
`ndofs_per_cell`; a slot gathered through [`InternalSource`](@ref) (a
condensed element's `q`) is resized to fit the cell's internal-dof range on
every gather instead, since that count is generally different from — and can
vary per cell independently of — the field dof count.

`needs_sensitivity` selects whether [`SensitivityBuffers`](@ref) is built —
STRUCTURAL, decided by the integrator family (see
[`needs_ad_decoration`](@ref)): a bilinear or linear operator never issues a
sensitivity kind, so it carries none of this machinery.

`global_dofs` is the subdomain's [`global_dofs`](@ref) declaration. Every
element-local buffer is padded by its length, and the workspace carries the
augmented dof vector every gather and scatter of the sweep addresses.
"""
function create_assembly_workspace(element, boundary_element, sdh, ivh, slots::NTuple{N, Symbol} = (:u,);
        needs_sensitivity::Bool = true, global_dofs = ()) where {N}
    n = length(global_dofs)
    slot_buffers = NamedTuple{slots}(ntuple(_ -> pad_element_vector(allocate_element_unknown_vector(element, sdh), n), N))
    return AssemblyWorkspace(
        pad_element_matrix(allocate_element_matrix(element, sdh), n),
        slot_buffers,
        pad_element_vector(allocate_element_residual_vector(element, sdh), n),
        CellCache(sdh),
        ivh,
        element,
        boundary_element,
        needs_sensitivity ? create_sensitivity_buffers(element, sdh, n) : nothing,
        _augmented_dof_vector(sdh, global_dofs),
    )
end

function _augmented_dof_vector(sdh, global_dofs)
    n = length(global_dofs)
    n == 0 && return nothing
    nc = ndofs_per_cell(sdh)
    dofs = Vector{Int}(undef, nc + n)
    dofs[(nc + 1):end] .= global_dofs
    return dofs
end

####################################
## Partition                      ##
####################################

"""
    CellItems(sdh)

The default work-item provider: the cells of one `SubDofHandler`. Item
providers are what `compute_partition` consumes — every other item family
brings its own provider type rather than a `SubDofHandler`
([`AlgebraicItems`](@ref), [`PatchItems`](@ref)).
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
# The blocked spec names its own type: the block and the entry storage are the
# user's choice, and this package carries neither dependency.
matrix_type(::AbstractDevice, spec::BlockedOperatorSpecification) = spec.matrix_type
vector_type(strategy::AbstractAssemblyStrategy) = vector_type(strategy.device)
vector_type(device::AbstractDevice) = Vector{value_type(device)}

function Adapt.adapt_structure(::AbstractAssemblyStrategy, dh::DofHandler)
    error("Device specific implementation for `adapt_structure(::AbstractAssemblyStrategy,dh::DofHandler)` is not implemented yet")
end
