"""
    StandardOperatorSpecification(; algebraic_couplings = (), constraint_handler = nothing,
                                    matrix_type = nothing, sparsity_entries = nothing)

The operator's global matrix as a monolithic sparse matrix over the pattern
[`create_system_matrix`](@ref) builds from three declarations
[`BlockedOperatorSpecification`](@ref) shares (the zero-argument form is
[`FullAssembly`](@ref)'s default):

- `algebraic_couplings` — Ferrite coupling descriptors (`CellCoupling`,
  `FacetCoupling`, `AlgebraicCoupling`) for the entries a
  [`global_dofs`](@ref) or [`facet_item_global_dofs`](@ref) declaration couples
  into, never inferred from the dof declaration.
  A missing descriptor surfaces as Ferrite's missing-sparsity-entry error on
  the first assembly.
- `constraint_handler` — sparsity room for the constraint entries
  (`add_constraint_entries!`). Applying the constraints stays the caller's,
  through Ferrite's `apply!`/`apply_assemble!`.
- `sparsity_entries` — a callable `f(sp, dh)` run on the freshly built pattern
  after the entries above, for the coupling an ITEM FAMILY introduces and the
  `DofHandler`'s cell pattern does not carry: an item spanning two cells (an
  interface/DG traversal, [`item_provider`](@ref)) scatters a block indexed by
  the dofs of BOTH. `nothing` (the default) adds none. The framework never
  infers them — which dofs an item couples is the provider's own adjacency — the
  same doctrine [`global_dofs`](@ref) states for its tail:

      sparsity_entries = (sp, dh) -> Ferrite.add_interface_entries!(
          sp, dh, nothing; topology = ExclusiveTopology(get_grid(dh)))

  It is re-run for every matrix allocated under this specification (the
  operator's own, and the ones [`allocate_components`](@ref) and
  [`StageBlockOperator`](@ref) build beside it), so it must be a pure
  entry-adding function and not a pre-built pattern to hand out.
- `matrix_type` — the concrete matrix type to allocate, `nothing` (the default)
  meaning `SparseMatrixCSC{value_type(device), index_type(device)}`. Naming one
  is what a DEVICE matrix takes: this package depends on no GPU vendor package,
  so the user loads it and names the type
  (`StandardOperatorSpecification(; matrix_type = CuSparseMatrixCSC{Float32, Int32})`),
  exactly as [`BlockedOperatorSpecification`](@ref) does for its block storage.
  The pattern stays this package's. Its element type must be the device's
  `value_type`, and Ferrite must have a `start_assemble` method for it — both
  checked at setup.
"""
struct StandardOperatorSpecification{C, CH, MT, SE}
    algebraic_couplings::C
    constraint_handler::CH
    matrix_type::MT
    sparsity_entries::SE
end
StandardOperatorSpecification(; algebraic_couplings = (), constraint_handler = nothing,
        matrix_type = nothing, sparsity_entries = nothing) =
    StandardOperatorSpecification(algebraic_couplings, constraint_handler, matrix_type, sparsity_entries)

"""
    BlockedOperatorSpecification(block_sizes, matrix_type; algebraic_couplings = (),
                                 constraint_handler = nothing, sparsity_entries = nothing)

The operator's global matrix as a `BlockMatrix` over the row/column split
`block_sizes`, allocated from a `BlockSparsityPattern` and the same three
declarations as [`StandardOperatorSpecification`](@ref). `matrix_type` is
REQUIRED: this package depends on neither BlockArrays nor SparseMatricesCSR, so
the user loads them and names the type
(`BlockMatrix{Float64, Matrix{SparseMatrixCSR{1, Float64, Int}}}`).

The residual stays a plain `Vector` — Ferrite's `BlockAssembler` takes a
non-blocked `f`. A LINEAR operator holds no matrix, so a blocked specification
on one is rejected at setup.
"""
struct BlockedOperatorSpecification{B, MT, C, CH, SE}
    block_sizes::B
    matrix_type::MT
    algebraic_couplings::C
    constraint_handler::CH
    sparsity_entries::SE
end
BlockedOperatorSpecification(block_sizes, matrix_type::Type;
        algebraic_couplings = (), constraint_handler = nothing, sparsity_entries = nothing) =
    BlockedOperatorSpecification(block_sizes, matrix_type, algebraic_couplings, constraint_handler,
                                 sparsity_entries)

"""
    AbstractAssemblyStrategy

Dispatch anchor for an operator's assembly strategy; [`AssemblyStrategy`](@ref)
is the sole concrete composition of the form/scheduling/device axes below.
"""
abstract type AbstractAssemblyStrategy end

####################################
## Operator form (the MFEM assembly level)
####################################

####################################
## Construction-time storage elections
####################################

"""
    StorageElection
    Stored <: StorageElection
    Recompute <: StorageElection
    ElementAssembly <: StorageElection

WHAT a sweep keeps between evaluations — a construction-time election trading
memory against flops. Two consumers: the `storage` field of
[`MatrixFreeAction`](@ref), where the three members are MFEM's PARTIAL, NONE and
ELEMENT assembly levels, and [`corrector_election`](@ref).

- [`Recompute`](@ref) — keep nothing.
- [`Stored`](@ref) — keep the per-quadrature-point quantity.
- [`ElementAssembly`](@ref) — keep the dense element MATRICES. A
  matrix-free-action election only; a consumer that does not implement it says
  so ([`corrector_election_error`](@ref)).

`CorrectorElection` is an alias for this supertype.

!!! warning "Experimental surface"
    This election family may change in a minor release.
"""
abstract type StorageElection end

"""
    Stored()

Keep the per-quadrature-point quantity: a matrix-free element's geometric
factors ([`fill_quadrature_data!`](@ref) — MFEM's PARTIAL level), or a
condensed element's stored corrector ([`corrector_election`](@ref)).
"""
struct Stored <: StorageElection end

"""
    Recompute()

Keep nothing and re-derive at every point of use: a matrix-free element's
geometry at the quadrature point that consumes it (MFEM's NONE level), or a
condensed element's corrector from the item's current `(u, q)`
([`corrector_election`](@ref)). Recomputation is EXACT, not approximate.
"""
struct Recompute <: StorageElection end

"""
    ElementAssembly()

Keep the dense element MATRICES — MFEM's ELEMENT level, the third member of
[`MatrixFreeAction`](@ref)'s `storage` ladder. Every `mul!` is a gather, a dense
`yₑ = Kₑ·uₑ` and a scatter; no quadrature point is visited.

The matrices are formed at [`setup_operator`](@ref) and refilled by
[`update_operator!`](@ref), through the element's own element-matrix kernel
where it declares one ([`provides_analytic`](@ref) for `JacobianKind{:u}`) and
through `ndofs_per_cell` applications of [`apply_element_action!`](@ref) to the
unit vectors otherwise. An element serving neither is refused at setup.

It costs `ndofs_per_cell²` scalars per cell, so it is the LOW-order election.
[`WorkerPerElement`](@ref) or [`LanesPerElement`](@ref), but never
[`CooperativeElement`](@ref), a dense product having no lattice to split.

Bytes/cell against the same form's assembled matrix, on a tensor-product
hexahedral mesh with one `Float32` scalar field (`p` = polynomial order); the
`packed` column is [`SymmetricElementMatrix`](@ref)'s election
([`element_matrix_symmetry`](@ref)):

| `p` | `ElementAssembly` (`ndofs_per_cell² · 4`) | packed (`ndofs_per_cell·(ndofs_per_cell+1)/2 · 4`) | assembled (global CSR, amortized) |
|---|---|---|---|
| 1 | 256 | **144** | 220 |
| 2 | 2916 | **1512** | 4157 |
| 3 | 16384 | **8320** | 27318 |

`ndofs_per_cell²` overtakes the assembled matrix's per-cell share between
`p = 1` and `p = 2` and keeps widening, so this is the cheapest MATRIX-FREE
storage at `p = 1`–`2` and [`Stored`](@ref)/[`Recompute`](@ref) take over above
that. The fill is paid at [`setup_operator`](@ref) and on every
[`update_operator!`](@ref), never on a bare `mul!`.

An element whose bilinear form is symmetric may additionally elect
[`SymmetricElementMatrix`](@ref) ([`element_matrix_symmetry`](@ref)) — the
`packed` column — at the price of a per-worker `ndofs_per_cell²` scratch buffer
at FILL time only.
"""
struct ElementAssembly <: StorageElection end

@doc (@doc StorageElection) const CorrectorElection = StorageElection

####################################
## The ELEMENT level's symmetry election
####################################

"""
    element_matrix_symmetry(cache) -> GeneralElementMatrix()
                                    -> SymmetricElementMatrix()

Whether `cache`'s bilinear form is symmetric. Consumed by
[`ElementAssemblyCache`](@ref) ONLY, read once off the wrapped cache at
construction, so the action has no runtime branch.

`GeneralElementMatrix()` (the default) makes no claim and `Kₑ` is stored dense.
[`SymmetricElementMatrix`](@ref) is TRUSTED UNCHECKED: a wrong declaration
silently reads only the upper triangle, SYMMETRIZING the assembled operator
rather than erroring. Declare it only where the FORM is provably symmetric for
every value the cache's coefficients can take; where the coefficient is not
symmetric BY TYPE, check its value at construction (as the
`FerriteOperatorsTensorProduct` diffusion cache does).

!!! warning "Experimental surface"
    This election and the two singletons below may change in a minor release.
"""
element_matrix_symmetry(cache) = GeneralElementMatrix()

"""
    GeneralElementMatrix()

The default [`element_matrix_symmetry`](@ref) election: no symmetry claim,
`Kₑ` stored dense.
"""
struct GeneralElementMatrix end

"""
    SymmetricElementMatrix()

Declares `Kₑ = Kₑᵀ` for every cell of the subdomain — see
[`element_matrix_symmetry`](@ref) for the trust contract and the hazard of a
wrong declaration.
"""
struct SymmetricElementMatrix end

"""
Which representation of the operator is produced — the MFEM assembly level.
Orthogonal to how the work is scheduled and to the device it runs on.
[`FullAssembly`](@ref) and [`MatrixFreeAction`](@ref) are the members; the axis
and the `form` keyword of [`AssemblyStrategy`](@ref) are the extension point a
further assembly level is added at.
"""
abstract type AbstractAssemblyForm end

"FULL level: assemble into a global sparse matrix / global vector."
struct FullAssembly{Spec} <: AbstractAssemblyForm
    operator_specification::Spec
end
FullAssembly() = FullAssembly(StandardOperatorSpecification())

"""
    MatrixFreeAction(; element_mapping = WorkerPerElement(), storage = Stored())

ELEMENT/PARTIAL/NONE level: the operator stores no GLOBAL matrix and evaluates
its action `y = A·u` element by element on every `mul!`
([`MatrixFreeFerriteOperator`](@ref)). `setup_operator` returns that operator
for an [`AbstractBilinearIntegrator`](@ref) whose caches serve the elected
storage level.

`element_mapping` selects how one element's action maps onto the device's
workers ([`AbstractElementMapping`](@ref)) — the same element definition under
each. It is resolved onto the device at setup
([`with_element_mapping`](@ref)), so a mapping the device has no kernel for, or
a cache that does not serve it, is a setup error.

`storage` is WHAT the operator keeps between actions
([`StorageElection`](@ref)), and separates the three MFEM levels this form
spans:

- [`Stored`](@ref) (the default) is PARTIAL. Each element precomputes its
  per-quadrature-point factors into its own store
  ([`fill_quadrature_data!`](@ref)) and every action is then contractions and
  reads.
- [`Recompute`](@ref) is NONE. Nothing is kept and every action re-derives the
  geometry at the quadrature point that consumes it — the election for a mesh
  whose factors do not fit, or a coefficient that changes faster than a store
  can be refilled.
- [`ElementAssembly`](@ref) is ELEMENT. The dense element matrices are kept and
  every action is a gather, a dense product and a scatter — no quadrature point
  is visited. `ndofs_per_cell²` scalars per cell, so it is the LOW-order
  election, and [`WorkerPerElement`](@ref) or [`LanesPerElement`](@ref).

The first two are the element's own storage and reach its cache through
[`with_action_storage`](@ref); a cache that keeps nothing serves both
identically. The third is the framework's ([`ElementAssemblyCache`](@ref)) and
serves any bilinear cache. Both are filled at setup and refilled by
[`update_operator!`](@ref).

!!! warning "Experimental surface"
    The matrix-free form, its operator type and the element entry points it
    calls may change in a minor release.
"""
struct MatrixFreeAction{M <: AbstractElementMapping, S} <: AbstractAssemblyForm
    element_mapping::M
    storage::S
end
function MatrixFreeAction(; element_mapping::AbstractElementMapping = WorkerPerElement(), storage = Stored())
    storage isa StorageElection || throw(ArgumentError(
        "`MatrixFreeAction`'s `storage` election is `Stored()` (the element's " *
        "per-quadrature-point factors, the PARTIAL level), `Recompute()` (re-derive them per " *
        "action, the NONE level) or `ElementAssembly()` (the dense element matrices, the " *
        "ELEMENT level), got $(storage)."))
    return MatrixFreeAction(element_mapping, storage)
end

"""
    operator_specification(form) -> spec or `nothing`

The global-storage declaration a form carries, `nothing` for a form that
allocates no global array ([`MatrixFreeAction`](@ref)). The setup-time device
walls read the specification through this, so a storage-free form skips the
storage checks instead of naming a field it does not have.
"""
operator_specification(form::FullAssembly) = form.operator_specification
operator_specification(::MatrixFreeAction) = nothing

####################################
## Scheduling policy
####################################

"""
How parallel work is made safe — the second strategy axis. See
[`SequentialScheduling`](@ref) and [`ColoredScheduling`](@ref).
"""
abstract type AbstractSchedulingPolicy end
"""
    SequentialScheduling()

Race freedom by atomic scatter: every worker assembles into the same global
target, collisions resolved atomically. Requires an eltype with atomic support.
"""
struct SequentialScheduling <: AbstractSchedulingPolicy end

"""
    ColoredScheduling(alg = ColoringAlgorithm.WorkStream)

Race freedom by coloring: the partition is a sequence of colors, no two items
of one color sharing a dof, so the scatter needs no atomics. Required for
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

The composition of three orthogonal axes: the operator form
([`AbstractAssemblyForm`](@ref) — what is produced), the scheduling policy
([`AbstractSchedulingPolicy`](@ref) — how parallel work is made safe), and the
device. The named constructors below build the common compositions.
"""
struct AssemblyStrategy{F <: AbstractAssemblyForm, S <: AbstractSchedulingPolicy, D <: AbstractDevice} <: AbstractAssemblyStrategy
    form::F
    scheduling::S
    device::D
end

"""
    AssemblyStrategy(device; form = FullAssembly(), scheduling = SequentialScheduling())

Keyword convenience over the 3-arg constructor: the form/scheduling/device
axes are the model, and the defaults here cover the common composition
([`FullAssembly`](@ref) under [`SequentialScheduling`](@ref), i.e. one chunk —
the only composition that admits the global-dof declarations
([`global_dofs`](@ref), [`facet_item_global_dofs`](@ref)) and the algebraic
item family). Pass `scheduling = ColoredScheduling()` for a parallel device
without atomic support.
"""
AssemblyStrategy(device::AbstractDevice; form = FullAssembly(), scheduling = SequentialScheduling()) =
    AssemblyStrategy(form, scheduling, device)

"""
    default_strategy()

The recommended shared-memory [`AssemblyStrategy`](@ref) for whatever is currently loaded:
[`AssemblyStrategy`](@ref) over a [`PolyesterDevice`](@ref) once Polyester.jl is loaded
(`using Polyester` activates `FerriteOperatorsPolyesterExt`, which is what gives `PolyesterDevice` its
`execute_on_device!`/`reduce_on_device` methods — without the extension the struct exists but has no
execution route), and over a [`SequentialCPUDevice`](@ref) otherwise. A setup-time call: the extension
check is a runtime lookup, not a compile-time constant, so it reflects whatever is loaded when it runs.

Load Polyester to get the parallel default. What this returns MAY CHANGE between minor versions as
better defaults become available — a caller that needs a strategy that does not move under it should
construct one explicitly instead.
"""
default_strategy() = Base.get_extension(FerriteOperators, :FerriteOperatorsPolyesterExt) === nothing ?
    AssemblyStrategy(SequentialCPUDevice()) :
    AssemblyStrategy(PolyesterDevice())


####################################
## Workspace                      ##
####################################

"""
    AbstractWorkspace

Supertype of the per-worker workspaces a device backend allocates and manages.
Every concrete workspace must implement:
- `Ferrite.reinit!(ws, item)` — position it on the item its family is indexed by (a cell id, a patch index, an algebraic item index)
- `duplicate_for_device(device::AbstractCPUDevice, ws)` — an independent copy for a parallel worker
"""
abstract type AbstractWorkspace end

"""
    SensitivityBuffers

Per-worker OUTPUT buffers the five sensitivity requests
([`ParameterJacobianRequest`](@ref), [`ParameterVJPRequest`](@ref),
[`TimeSensitivityRequest`](@ref), [`StateJVPRequest`](@ref),
[`StateVJPRequest`](@ref)) accumulate into and the engine scatters, plus the
rectangular ∂F/∂q block ([`update_internal_jacobian!`](@ref)), also a local
block whose column space is not the field space. The outputs half of the AD
decorator's buffer split — [`ADElementCache`](@ref) holds the seeds and configs.

Element-sized members (`λₑ`, `vₑ`, `Jvₑ`, `gu`, `gₜ`) are eager; the rectangular
ones are (re)allocated once their column count is known — `θ`/`Bₑ`/`gθ` by
[`parameter_sweep_buffers!`](@ref), `Kqₑ` by [`internal_sweep_buffers!`](@ref).
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

Size the parameter-sweep members (`θ`, `Bₑ`, `gθ`) for the `nθ` flat parameters
arriving with `p`, reallocating only when nθ changed on this worker.
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
reallocating only when that count changed on this worker. The count is per
ITEM, not per subdomain, so this runs per item rather than per sweep
([`update_internal_jacobian!`](@ref)).
"""
function internal_sweep_buffers!(s::SensitivityBuffers, nq::Int)
    size(s.Kqₑ, 2) == nq || (s.Kqₑ = Matrix{eltype(s.Kqₑ)}(undef, size(s.Kqₑ, 1), nq))
    return s
end

"""
    AssemblyWorkspace

Per-worker workspace for square operator assembly (bilinear, nonlinear,
linear): element-local buffers and caches reused across cells, plus the
[`SensitivityBuffers`](@ref) a nonlinear operator's sensitivity entry points
need.

IMMUTABLE: every field is bound at construction, a sweep only filling the
buffers they point at. That is what lets the workspace cross a device boundary
— a scalar or tensor evaluation RETURNS its value instead of parking it in a
slot here (see [`evaluate_functional`](@ref)).

Core fields:
- `Ke`, `re`, `cell`, `ivh`: element matrix, element residual, `Ferrite.CellCache`, internal variable handler
- `slot_buffers`: NamedTuple of element-local state buffers, one per declared slot
- `element`: the volumetric element cache ([`AbstractVolumetricElementCache`](@ref))
- `sensitivity`: [`SensitivityBuffers`](@ref), or `nothing` for a family that never issues a sensitivity kind (bilinear, linear)
- `dofs`: augmented dof vector `[celldofs(cell); the declared global dofs]` ([`global_dofs`](@ref)), or `nothing`
  where the integrator declares none. The tail is written once at construction, the head refreshed by
  `Ferrite.reinit!`; the `nothing` type is what makes the un-augmented path return `celldofs(ws.cell)`
  directly, with neither copy nor run-time branch.
"""
@concrete struct AssemblyWorkspace <: AbstractWorkspace
    Ke
    slot_buffers
    re
    cell
    ivh
    element
    sensitivity
    dofs
end

Ferrite.reinit!(ws::AssemblyWorkspace, cellid) = _position_cell(ws, ws.cell, cellid, nothing)
@inline position_item(ws::AssemblyWorkspace, item, kind) = _position_cell(ws, ws.cell, item, kind)

# `position_iterator` makes the in-place-vs-by-construction distinction, so this
# never has to. `ws` is immutable, hence the rebuild.
@inline function _position_cell(ws::AssemblyWorkspace, cell, item, kind)
    positioned = position_iterator(cell, item, item_update_flags(kind, ws.element))
    _refresh_dof_head!(ws.dofs, positioned)
    return _with_cell(ws, positioned)
end

@inline _with_cell(ws::AssemblyWorkspace, cell) = AssemblyWorkspace(
    ws.Ke, ws.slot_buffers, ws.re, cell, ws.ivh, ws.element, ws.sensitivity, ws.dofs)

@inline _refresh_dof_head!(::Nothing, cell) = nothing
@inline _refresh_dof_head!(dofs, cell) = copyto!(dofs, celldofs(cell))

# The tail as declared, recovered from the augmented vector so a per-worker
# duplicate rebuilds the same layout without carrying the declaration along.
_declared_global_dofs(ws::AssemblyWorkspace) = _declared_global_dofs(ws.dofs, iterator_handler(ws.cell))
_declared_global_dofs(::Nothing, sdh) = ()
_declared_global_dofs(dofs, sdh) = @view dofs[(ndofs_per_cell(sdh) + 1):end]

function duplicate_for_device(device::AbstractCPUDevice, ws::AssemblyWorkspace)
    return create_assembly_workspace(
        duplicate_for_device(device, ws.element),
        iterator_handler(ws.cell),
        duplicate_for_device(device, ws.ivh),
        keys(ws.slot_buffers);
        needs_sensitivity = ws.sensitivity !== nothing,
        global_dofs = _declared_global_dofs(ws),
        iterator = duplicate_for_device(device, ws.cell),
    )
end

# A worker's own geometry cache, over the same handler and refreshing the same
# members, so its staging is private.
duplicate_for_device(::AbstractCPUDevice, cc::CellCache) = CellCache(cc.dh, cc.flags)

"""
    create_assembly_workspace(element, sdh, ivh, slots; needs_sensitivity = true,
                              global_dofs = (), iterator = assembly_iterator(nothing, element, sdh))

Create one [`AssemblyWorkspace`](@ref) with freshly allocated element-local
buffers, one state buffer per declared slot name and sized to `ndofs_per_cell`;
a slot gathered through [`InternalSource`](@ref) (a condensed element's `q`) is
resized to the cell's internal-dof range on every gather instead, that count
being unrelated to the field dof count and free to vary per cell.

`needs_sensitivity` selects whether [`SensitivityBuffers`](@ref) is built —
STRUCTURAL, decided by the integrator family ([`needs_ad_decoration`](@ref)).

`global_dofs` is the subdomain's [`global_dofs`](@ref) declaration: every
element-local buffer is padded by its length, and the workspace carries the
augmented dof vector the sweep's gathers and scatters address.

`iterator` is what positions the workspace on an item and what rides
`args.cell`, [`assembly_iterator`](@ref)'s answer for this subdomain.

The buffers carry whatever scalar the `allocate_element_*` hooks return — the
ELEMENT's precision ([`element_value_type`](@ref)), which need not be the
device's.
"""
function create_assembly_workspace(element, sdh, ivh, slots::NTuple{N, Symbol} = (:u,);
        needs_sensitivity::Bool = true, global_dofs = (),
        iterator = assembly_iterator(nothing, element, sdh)) where {N}
    n = length(global_dofs)
    slot_buffers = NamedTuple{slots}(ntuple(_ -> pad_element_vector(allocate_element_unknown_vector(element, sdh), n), N))
    return AssemblyWorkspace(
        pad_element_matrix(allocate_element_matrix(element, sdh), n),
        slot_buffers,
        pad_element_vector(allocate_element_residual_vector(element, sdh), n),
        iterator,
        ivh,
        element,
        needs_sensitivity ? create_sensitivity_buffers(element, sdh, n) : nothing,
        _augmented_dof_vector(sdh, global_dofs),
    )
end

"""
    device_worker_view(ws::AssemblyWorkspace, worker)

Worker `worker`'s slice of the batched workspace a GPU device's
[`setup_device_instances`](@ref) built. The internal-variable handler is shared
read-only; everything else slices.
"""
device_worker_view(ws::AssemblyWorkspace, worker) = AssemblyWorkspace(
    device_worker_view(ws.Ke, worker),
    map(b -> device_worker_view(b, worker), ws.slot_buffers),
    device_worker_view(ws.re, worker),
    device_worker_view(ws.cell, worker),
    ws.ivh,
    device_worker_view(ws.element, worker),
    ws.sensitivity,
    ws.dofs,
)

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
brings its own provider type ([`FacetItems`](@ref), [`AlgebraicItems`](@ref),
[`PatchItems`](@ref)).
"""
struct CellItems{SDH <: SubDofHandler}
    sdh::SDH
end

"""
    item_provider(kind, element_cache, sdh) -> provider

WHAT the items of a sweep of `kind` over `element_cache` on subdomain `sdh`
ARE. The second half of the iteration seam, beside
[`assembly_iterator`](@ref), which says how to POSITION on one of them.
Resolved ONCE per (sweep kind, element cache, subdomain) at
[`setup_operator`](@ref); its answer is what [`compute_partition`](@ref)
consumes, and the partition is what a sweep walks.

The default is [`CellItems`](@ref)`(sdh)` — the cells of the subdomain.

Two seams and not one because the iterator and the item set vary
INDEPENDENTLY: the facet family runs a custom provider ([`FacetItems`](@ref))
over the STOCK cell iterator, the matrix-free action a custom iterator over the
STOCK provider.

**The hazard.** Overloading one seam and not the other is not an error: the
other answers with its default, and an interface iterator left with the default
provider is positioned on CELL ids — succeeding, and assembling the wrong
operator. No framework check sees this; the item COUNT a sweep visits does.

A provider carries its family's whole partition safety argument — see
[`compute_partition`](@ref).
"""
item_provider(kind, element_cache, sdh) = CellItems(sdh)

"""
    compute_partition(strategy, provider)
    compute_partition(scheduling, provider)

The work partition for a scheduling policy and an item provider
([`item_provider`](@ref)): an iterable of iterables, the outer level
synchronization barriers (e.g. colors), the inner level work items that may run
concurrently (cell ids for [`CellItems`](@ref), item indices for every other
family).

**Partition obligations.** A partition is SAFE GIVEN A VALID PARTITION, never
*thread-safe*. What makes a parallel sweep race-free is a promise about the
provider's own item adjacency, which the framework cannot check.

- Under [`ColoredScheduling`](@ref) the provider promises that **no two items of
  one inner chunk share a scatter dof**. That promise is what lets the scatter
  run without atomics (`dof_scatter_needs_atomic`), so a chunk that breaks it is
  a silent race, not an error.
- Under [`SequentialScheduling`](@ref) the provider promises nothing: the atomic
  scatter resolves the collisions.

The three shipped providers each argue the promise in their own terms:
[`FacetItems`](@ref) colors the OWNING CELLS; [`AlgebraicItems`](@ref) has an
unknown sharing pattern and puts one item per barrier; [`PatchItems`](@ref)
REFUSES to color, not carrying the adjacency the promise would rest on.
"""
compute_partition(strategy::AssemblyStrategy, sdh::SubDofHandler) = compute_partition(strategy.scheduling, CellItems(sdh))
compute_partition(strategy::AssemblyStrategy, provider) = compute_partition(strategy.scheduling, provider)
compute_partition(::SequentialScheduling, provider::CellItems) = (_cell_chunk(provider.sdh.cellset),)

# The one chunk `SequentialScheduling` hands out, as a `UnitRange` where the
# cellset is contiguous — the common single-subdomain-over-the-whole-grid case —
# so a device kernel's item id is arithmetic rather than an indexed load, and
# `adapt_partition` moves nothing into device memory. Checked once at setup; a
# non-contiguous cellset keeps the `Vector`.
function _cell_chunk(cellset)
    cells = collect(cellset)
    if !isempty(cells) && issorted(cells) && cells[end] - cells[1] == length(cells) - 1
        return cells[1]:cells[end]
    end
    return cells
end

function compute_partition(scheduling::ColoredScheduling, provider::CellItems)
    return Ferrite.create_coloring(get_grid(provider.sdh.dh), collect(provider.sdh.cellset); alg=scheduling.alg)
end

"""
    n_workers(device, partition) -> Int

Number of parallel workers for this device and partition — the size of the
per-worker device cache [`setup_device_instances`](@ref) builds.

A [`PolyesterDevice`](@ref) can occupy at most `Threads.nthreads()` of them at
once, and `min_items_per_worker` is the smallest share a worker is given, so
the count is the smaller of the thread count and the number of such shares the
largest barrier of `partition` holds. Workspaces are therefore per WORKER, not
per share: a worker walks the items it was given with the one workspace it owns.

A [`KernelAbstractionsDevice`](@ref) sizes them from its launch policy
([`launch_geometry`](@ref)) over the largest barrier, the kernel indexing the
per-worker caches unchecked. Under [`CooperativeElement`](@ref) the worker IS
the workgroup and the count is the largest barrier itself.
[`LanesPerElement`](@ref) stages nothing per LANE, so this method serves it
unchanged.
"""
n_workers(::SequentialCPUDevice, partition) = 1
function n_workers(device::PolyesterDevice, partition)
    ncellsmax = maximum(length, partition)
    return min(Threads.nthreads(), cld(ncellsmax, device.min_items_per_worker))
end
n_workers(device::KernelAbstractionsDevice, partition) =
    max(1, prod(launch_geometry(device, maximum(length, partition; init = 0))))
n_workers(::KernelAbstractionsDevice{<:Any, <:Any, <:Any, CooperativeElement}, partition) =
    max(1, maximum(length, partition; init = 0))


####################################
## Matrix/Vector type             ##
####################################

matrix_type(strategy::AssemblyStrategy) = matrix_type(strategy.device, strategy.form.operator_specification)
matrix_type(device::AbstractDevice, ::StandardOperatorSpecification{<:Any, <:Any, Nothing}) =
    SparseMatrixCSC{value_type(device), index_type(device)}
# A spec that names its own type wins: block and entry storage (blocked) and
# device residency (standard) are the user's choice, and this package carries
# neither dependency.
matrix_type(::AbstractDevice, spec::StandardOperatorSpecification) = spec.matrix_type
matrix_type(::AbstractDevice, spec::BlockedOperatorSpecification) = spec.matrix_type
vector_type(strategy::AbstractAssemblyStrategy) = vector_type(strategy.device)
vector_type(device::AbstractDevice) = Vector{value_type(device)}
