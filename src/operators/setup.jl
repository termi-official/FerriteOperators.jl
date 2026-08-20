####################################
## Scheme protocols — the setup-time declarations
####################################

"""
    AbstractSchemeProtocol

A typed, DECLARATIONS-ONLY description of what a scheme asks of an operator,
passed positionally: `setup_operator(strategy, problem, dh, protocol)`.

A protocol declares exactly what setup consumes: [`get_declared_slots`](@ref)
(the state slot names sweeps may carry) and [`get_declared_kinds`](@ref) (the
request kinds the operator will be asked to compute). Those declarations size
the per-worker slot buffers, select which sweep-state families are built
([`sweep_state`](@ref)), and drive the setup-time trait ↔ kernel and
internal-state admissibility checks.

Protocols carry NO coefficients — γ, tableaus, and weights are per-evaluation
solver data, not declarations — and nothing term-shaped: no per-term contexts,
no domain algebra, no weight symbolism. Term-shaped data belongs to
integrators; anything needing its own context or sink is its own sweep.

Declaring is a hint, not a capability restriction: an operator always builds
what its own family issues ([`mandatory_kinds`](@ref)), so an undeclared kind
stays usable and a declaration only ever adds eager machinery and eager
checks.
"""
abstract type AbstractSchemeProtocol end

"""
    get_declared_slots(protocol) -> NTuple{N, Symbol}

The state slot names sweeps of this protocol may carry; the engine allocates
one per-worker slot buffer per name. Slot type tags are reserved vocabulary —
names are the whole declaration today.
"""
function get_declared_slots end

"""
    get_declared_kinds(protocol) -> Tuple of request-kind types

The request kinds this protocol declares, as their UnionAll bases
(`JacobianKind`, `ParameterVJPKind`, …). Declared kinds run their trait ↔
kernel and admissibility checks at setup instead of on first use, and select
the per-worker sweep-state families built eagerly.
"""
function get_declared_kinds end

"""
    DefaultProtocol(; slots = (:u,), requests = ())

The protocol the keyword form of [`setup_operator`](@ref) lowers to — its
constructor arguments ARE those keywords, so
`setup_operator(strategy, integrator, dh; slots, requests)` and
`setup_operator(strategy, integrator, dh, DefaultProtocol(; …))` build the
same operator. Declares no context type and no slot tags: the default world is
`integrator + dh`, and a scheme that needs more declares its own protocol.
"""
struct DefaultProtocol{slots, K <: Tuple} <: AbstractSchemeProtocol
    kinds::K       # a tuple of request-kind TYPES cannot ride in a type parameter
end
function DefaultProtocol(; slots = (:u,), requests::Tuple = ())
    kinds = map(_kind_type, requests)
    return DefaultProtocol{Tuple(slots), typeof(kinds)}(kinds)
end

# Kind types or instances normalize to their UnionAll base, so a payload type
# parameter (`ParameterVJPKind{Vector{Float64}}`) never makes a declaration
# silently miss its validation entry or its sweep-state family.
_kind_type(r) = Base.typename(r isa Type ? r : typeof(r)).wrapper

get_declared_slots(::DefaultProtocol{slots}) where {slots} = slots
get_declared_kinds(protocol::DefaultProtocol) = protocol.kinds

"""
    mandatory_kinds(integrator) -> Tuple of request-kind types

The kinds an operator of this integrator's family ALWAYS issues, independent
of the protocol. Unioned with [`get_declared_kinds`](@ref) when the engine selects
per-worker sweep-state families, which is what keeps a declaration additive:
the family's own kinds are served whether or not a protocol names them. The
default is the nonlinear set, so an unknown integrator type is served
conservatively.
"""
mandatory_kinds(integrator) = (JacobianResidualKind, JacobianKind, ResidualKind)
mandatory_kinds(::AbstractBilinearIntegrator) = (BilinearKind, ResidualKind)
mandatory_kinds(::AbstractLinearIntegrator) = (LinearKind, ResidualKind)

# A sensitivity sweep runs the VOLUMETRIC kernel only, so a boundary term that
# depends on the seeded quantity contributes nothing to the result. That is
# correct for the common case (parameter- and time-independent tractions) and
# silently incomplete otherwise, which is a warning rather than an error. Once
# per (declared kind set × boundary cache type) — the operator is what carries
# the combination, not the individual sweep.
function _warn_boundary_sensitivity(requests::Tuple, boundary_caches)
    any(K -> K <: SensitivityKind, requests) || return nothing
    surface = findfirst(c -> !(c isa EmptySurfaceElementCache), boundary_caches)
    surface === nothing && return nothing
    B = typeof(boundary_caches[surface])
    kinds = join(map(nameof, filter(K -> K <: SensitivityKind, collect(requests))), ", ")
    @warn "Sensitivity sweeps run the volumetric kernel only: boundary contributions are NOT " *
          "included in ∂F/∂θ, ∂F/∂t, or the matrix-free state products. This operator declares " *
          "$(kinds) and carries a `$(nameof(B))` boundary cache, so its sensitivities are " *
          "correct only if the boundary terms are independent of the seeded quantity — θ for " *
          "the parameter kinds, t for the time sensitivity, u for the state products. " *
          "`check_derivatives` detects the dependent case: its finite-difference referee " *
          "evaluates the FULL residual including boundary terms, so a failing parameter, time, " *
          "or state-product check on this operator is the signature of this omission." _id =
          Symbol(:boundary_sensitivity_, B, :_, hash(requests)) maxlog = 1
    return nothing
end

create_system_matrix(strategy, dh) = allocate_matrix(matrix_type(strategy), dh)
create_system_vector(strategy, dh) = allocate_vector(vector_type(strategy), dh)

function setup_elements(integrator, dh)
    return [setup_element_cache(integrator, sdh) for sdh in dh.subdofhandlers]
end

function setup_boundaries(integrator, dh)
    return [setup_boundary_cache(integrator, sdh) for sdh in dh.subdofhandlers]
end

function setup_internal_variable_handler(integrator::AbstractCondensedNonlinearIntegrator, element_caches, dh)
    num_dofs_per_element = zeros(Int, getncells(get_grid(dh))+1)
    for (sdh, cache) in zip(dh.subdofhandlers, element_caches)
        for (cellid, nidofs) in zip(sdh.cellset, get_number_of_internal_dofs_per_element(integrator, cache, sdh))
            num_dofs_per_element[1+cellid] = nidofs
        end
    end
    @assert all(num_dofs_per_element .≥ 0) "Number of internal dofs must be non-negative!"
    # `num_dofs_per_element` is padded with a leading zero, so the cumulative sum is exactly the
    # `ncells+1` block relative offsets the handler expects.
    offsets = cumsum(num_dofs_per_element)
    return InternalVariableHandler(offsets, ndofs(dh), offsets[end])
end

function setup_internal_variable_handler(integrator, element_caches, dh)
    return InternalVariableHandler(nothing, 0, 0)
end

function setup_subdomain_caches(strategy, element_caches, boundary_caches, ivh, dh;
        slots::NTuple{<:Any, Symbol}, derivative_family::Bool)
    device = strategy.device
    return [begin
        partition = compute_partition(strategy, sdh)
        n = n_workers(strategy, device, partition)
        ws = create_assembly_workspace(element_cache, boundary_cache, sdh, ivh, slots; derivative_family)
        dc = setup_device_instances(device, ws, n)
        SubdomainCache(AssemblyDomain(sdh, ivh, element_cache, boundary_cache), dc, partition)
    end for (sdh, element_cache, boundary_cache) in zip(dh.subdofhandlers, element_caches, boundary_caches)]
end

"""
    setup_engine(strategy, integrator, dh, protocol::AbstractSchemeProtocol)

Build the [`AssemblyEngine`](@ref) shared by all operator kinds from the
protocol's declarations: slot names size the per-worker slot buffers, and
declared kinds run their trait ↔ kernel and internal-state admissibility
checks eagerly (instead of on first use) and — unioned with the integrator
family's [`mandatory_kinds`](@ref) — select which per-worker sweep-state
families are built.
"""
function setup_engine(strategy::AbstractAssemblyStrategy, integrator, dh::AbstractDofHandler, protocol::AbstractSchemeProtocol)
    requests          = get_declared_kinds(protocol)
    operator_strategy = setup_operator_strategy_cache(strategy, integrator, dh)
    element_caches    = setup_elements(integrator, dh)
    foreach(cache -> validate_element_cache(cache, requests), element_caches)
    boundary_caches   = setup_boundaries(integrator, dh)
    ivh               = setup_internal_variable_handler(integrator, element_caches, dh)
    _warn_boundary_sensitivity(requests, boundary_caches)
    kinds             = (requests..., mandatory_kinds(integrator)...)
    subdomain_caches  = setup_subdomain_caches(operator_strategy, element_caches, boundary_caches, ivh, dh;
                                               slots = get_declared_slots(protocol),
                                               derivative_family = needs_derivative_family(kinds))
    return AssemblyEngine(operator_strategy, subdomain_caches, dh, ivh, protocol)
end

"""
    setup_operator(strategy, problem, dh, protocol::AbstractSchemeProtocol)
    setup_operator(strategy, problem, dh; slots = (:u,), requests = ())

Build the operator for `problem` (an integrator) over `dh`. The positional
form takes the scheme's declarations as a protocol
([`AbstractSchemeProtocol`](@ref)); the keyword form is sugar whose keywords
are the [`DefaultProtocol`](@ref) constructor arguments, and it lowers to the
positional form here.

Transfer and patch operators keep their own constructors
([`setup_transfer_operator`](@ref), [`assemble_patches!`](@ref)).
"""
function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler, protocol::AbstractSchemeProtocol)
    engine = setup_engine(strategy, integrator, dh, protocol)
    A      = create_system_matrix(engine.strategy, dh)
    return BilinearFerriteOperator(A, engine, integrator)
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractNonlinearIntegrator, dh::AbstractDofHandler, protocol::AbstractSchemeProtocol)
    engine = setup_engine(strategy, integrator, dh, protocol)
    J      = create_system_matrix(engine.strategy, dh)
    return LinearizedFerriteOperator(J, engine, integrator)
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractLinearIntegrator, dh::AbstractDofHandler, protocol::AbstractSchemeProtocol)
    engine = setup_engine(strategy, integrator, dh, protocol)
    b      = create_system_vector(engine.strategy, dh)
    return LinearFerriteOperator(b, engine, integrator)
end

setup_operator(strategy::AbstractAssemblyStrategy, integrator, dh::AbstractDofHandler;
        slots = (:u,), requests::Tuple = ()) =
    setup_operator(strategy, integrator, dh, DefaultProtocol(; slots, requests))

"""
    init_transfer_sparsity_pattern(dh_row::DofHandler, dh_col::DofHandler)

Build a `Ferrite.SparsityPattern` of size `(ndofs(dh_row) × ndofs(dh_col))` covering all
DoF pairs `(rdof, cdof)` that share a cell.  Both DofHandlers must live on the same grid
and have the same number of subdomains.
"""
function init_transfer_sparsity_pattern(dh_row::DofHandler, dh_col::DofHandler)
    nrdofs = ndofs(dh_row)
    ncdofs = ndofs(dh_col)
    nnz_per_row = ndofs_per_cell(dh_col.subdofhandlers[1])
    sp = SparsityPattern(nrdofs, ncdofs; nnz_per_row)
    rdofs_buf = Int[]
    cdofs_buf = Int[]
    for (sdh_row, sdh_col) in zip(dh_row.subdofhandlers, dh_col.subdofhandlers)
        resize!(rdofs_buf, ndofs_per_cell(sdh_row))
        resize!(cdofs_buf, ndofs_per_cell(sdh_col))
        for cellid in sdh_row.cellset
            celldofs!(rdofs_buf, dh_row, cellid)
            celldofs!(cdofs_buf, dh_col, cellid)
            for rdof in rdofs_buf
                for cdof in cdofs_buf
                    Ferrite.add_entry!(sp, rdof, cdof)
                end
            end
        end
    end
    return sp
end

"""
    setup_transfer_operator(strategy, integrator, dh_row, dh_col)

Set up a [`TransferFerriteOperator`](@ref) for assembling a rectangular sparse matrix of
size `(ndofs(dh_row) × ndofs(dh_col))`.

`dh_row` and `dh_col` must live on the **same** grid and their subdomain lists must
correspond 1-to-1 (same length, same cellsets at each index).

`integrator` must be an [`AbstractTransferIntegrator`](@ref); its
`setup_transfer_element_cache(integrator, sdh_row, sdh_col)` method is called once per
subdomain pair.

!!! warning "Experimental surface"
    Transfer operators are scheduled to be folded into the unified assembly
    engine. The constructors and the operator types may change in a minor
    release; the assembled matrix and its sparsity are not affected.
"""
function setup_transfer_operator(
        strategy::AbstractAssemblyStrategy,
        integrator::AbstractTransferIntegrator,
        dh_row::DofHandler,
        dh_col::DofHandler,
    )
    (strategy isa AssemblyStrategy && strategy.form isa FullAssembly && strategy.scheduling isa SequentialScheduling) || throw(ArgumentError("Transfer operators currently only support sequential full-assembly strategies (got $(typeof(strategy)))"))
    strategy.device isa SequentialCPUDevice || throw(ArgumentError("Transfer operators currently only support SequentialCPUDevice (got $(typeof(strategy.device)))"))
    @assert get_grid(dh_row) === get_grid(dh_col) "Both DofHandlers must share the same grid"
    @assert length(dh_row.subdofhandlers) == length(dh_col.subdofhandlers) "Mismatch in number of subdomains"

    # Build rectangular sparse matrix with the correct sparsity pattern
    Tv = value_type(strategy.device)
    sp = init_transfer_sparsity_pattern(dh_row, dh_col)
    P  = allocate_matrix(SparseMatrixCSC{Tv, Int}, sp)

    # Build per-subdomain caches (one per SubDofHandler pair)
    device = strategy.device
    subdomain_caches = SubdomainCache[]
    for (sdh_row, sdh_col) in zip(dh_row.subdofhandlers, dh_col.subdofhandlers)
        element = setup_transfer_element_cache(integrator, sdh_row, sdh_col)
        partition = compute_partition(strategy, sdh_row)
        n = n_workers(strategy, device, partition)
        tc = SameGridCellCache(sdh_row, sdh_col)
        ws = TransferWorkspace(element, allocate_transfer_element_matrix(element, sdh_row, sdh_col), tc)
        dc = setup_device_instances(device, ws, n)
        push!(subdomain_caches, SubdomainCache(TransferDomain(sdh_row, sdh_col), dc, partition))
    end

    return TransferFerriteOperator(P, strategy, subdomain_caches, dh_row, dh_col, integrator)
end

"""
    init_nested_transfer_sparsity_pattern(dh_fine, dh_coarse, fine2coarse)

Build a `Ferrite.SparsityPattern` of size `(ndofs(dh_fine) × ndofs(dh_coarse))` for a
nested-grid transfer operator.  The sparsity is determined by the `fine2coarse` mapping:
a (rdof, cdof) entry is added whenever `rdof` belongs to fine cell `i` and `cdof` belongs
to its parent coarse cell `fine2coarse[i]`.
"""
function init_nested_transfer_sparsity_pattern(
        dh_fine::DofHandler,
        dh_coarse::DofHandler,
        fine2coarse::AbstractVector{Int},
    )
    nrdofs   = ndofs(dh_fine)
    ncdofs   = ndofs(dh_coarse)
    nnz_hint = maximum(sdh -> ndofs_per_cell(sdh), dh_coarse.subdofhandlers; init = 1)
    sp       = SparsityPattern(nrdofs, ncdofs; nnz_per_row = nnz_hint)
    rdofs_buf = Int[]
    cdofs_buf = Int[]
    for fine_id in 1:getncells(get_grid(dh_fine))
        coarse_id = fine2coarse[fine_id]
        resize!(rdofs_buf, ndofs_per_cell(dh_fine,   fine_id))
        resize!(cdofs_buf, ndofs_per_cell(dh_coarse, coarse_id))
        celldofs!(rdofs_buf, dh_fine,   fine_id)
        celldofs!(cdofs_buf, dh_coarse, coarse_id)
        for rdof in rdofs_buf, cdof in cdofs_buf
            Ferrite.add_entry!(sp, rdof, cdof)
        end
    end
    return sp
end

"""
    setup_nested_transfer_operator(strategy, integrator, dh_fine, dh_coarse, fine2coarse, child_ref_coords)

Set up a [`NestedTransferFerriteOperator`](@ref) for assembling a rectangular sparse
matrix of size `(ndofs(dh_fine) × ndofs(dh_coarse))`.

`dh_fine` and `dh_coarse` must live on **different** grids where every fine cell is a
child of exactly one coarse cell, as encoded by `fine2coarse` and `child_ref_coords`.

!!! warning "Experimental surface"
    Transfer operators are scheduled to be folded into the unified assembly
    engine. The constructors and the operator types may change in a minor
    release; the assembled matrix and its sparsity are not affected.
"""
function setup_nested_transfer_operator(
        strategy::AbstractAssemblyStrategy,
        integrator::AbstractTransferIntegrator,
        dh_fine::DofHandler,
        dh_coarse::DofHandler,
        fine2coarse::AbstractVector{Int},
        child_ref_coords::AbstractVector,
    )
    (strategy isa AssemblyStrategy && strategy.form isa FullAssembly && strategy.scheduling isa SequentialScheduling) || throw(ArgumentError("Nested transfer operators currently only support sequential full-assembly strategies (got $(typeof(strategy)))"))
    strategy.device isa SequentialCPUDevice || throw(ArgumentError("Nested transfer operators currently only support SequentialCPUDevice (got $(typeof(strategy.device)))"))
    Tv  = value_type(strategy.device)
    sp  = init_nested_transfer_sparsity_pattern(dh_fine, dh_coarse, fine2coarse)
    P   = allocate_matrix(SparseMatrixCSC{Tv, Int}, sp)

    device = strategy.device
    subdomain_caches = SubdomainCache[]
    for (sdh_fine, sdh_coarse) in zip(dh_fine.subdofhandlers, dh_coarse.subdofhandlers)
        element = setup_transfer_element_cache(integrator, sdh_fine, sdh_coarse)
        partition = compute_partition(strategy, sdh_fine)
        n = n_workers(strategy, device, partition)
        tc = NestedGridCellCache(sdh_fine, sdh_coarse, fine2coarse, child_ref_coords)
        ws = TransferWorkspace(element, allocate_transfer_element_matrix(element, sdh_fine, sdh_coarse), tc)
        dc = setup_device_instances(device, ws, n)
        push!(subdomain_caches, SubdomainCache(TransferDomain(sdh_fine, sdh_coarse), dc, partition))
    end

    return NestedTransferFerriteOperator(P, strategy, subdomain_caches, dh_fine, dh_coarse, integrator)
end
