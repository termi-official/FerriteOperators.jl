####################################
## Scheme protocols — the setup-time declarations
####################################

"""
    AbstractSchemeProtocol

A typed, DECLARATIONS-ONLY description of what a scheme asks of an operator,
passed positionally: `setup_operator(strategy, problem, dh, protocol)`. It
declares [`get_declared_slots`](@ref) and [`get_declared_kinds`](@ref), which
size the per-worker slot buffers and pull the trait ↔ kernel and
internal-state admissibility checks forward to setup time.
[`ADElementCache`](@ref) decoration is decided separately and structurally by
[`needs_ad_decoration`](@ref).

Protocols carry NO coefficients (γ, tableaus and weights are per-evaluation
solver data) and nothing term-shaped — that belongs to integrators, and
anything needing its own context or sink is its own sweep.

Declaring is a hint, not a capability restriction: an undeclared kind stays
usable and runs its checks at the call-time entry points instead.
"""
abstract type AbstractSchemeProtocol end

"""
    get_declared_slots(protocol) -> NTuple{N, Symbol}

The state slot names sweeps of this protocol may carry; the engine allocates
one per-worker slot buffer per name. Slot type tags are reserved vocabulary —
names are the whole declaration.
"""
function get_declared_slots end

"""
    get_declared_kinds(protocol) -> Tuple of request-kind types

The request kinds this protocol declares, as their UnionAll bases
(`JacobianKind`, `ParameterVJPKind`, …). Declaring one moves its trait ↔ kernel
and admissibility checks to setup and builds its per-worker sweep-state family
eagerly.
"""
function get_declared_kinds end

"""
    DefaultProtocol(; slots = (:u,), requests = ())

The protocol the keyword form of [`setup_operator`](@ref) lowers to — its
constructor arguments ARE those keywords, so the two forms build the same
operator. Declares no context type and no slot tags: the default world is
`integrator + dh`, and a scheme needing more declares its own protocol.
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

# The omission the message names belongs to the FUSED route alone, and it is a
# warning rather than an error because the common case (parameter- and
# time-independent tractions) is correct. Keyed once per (declared kind set ×
# boundary cache type): the operator carries that combination, not the sweep.
function _warn_boundary_sensitivity(requests::Tuple, boundary_caches)
    any(K -> K <: SensitivityKind, requests) || return nothing
    surface = findfirst(c -> !(c isa EmptySurfaceElementCache), boundary_caches)
    surface === nothing && return nothing
    B = typeof(boundary_caches[surface])
    kinds = join(map(nameof, filter(K -> K <: SensitivityKind, collect(requests))), ", ")
    @warn "Sensitivity sweeps run the volumetric kernel only where a boundary term rides the " *
          "cell sweep: fused-route boundary contributions are NOT included in ∂F/∂θ, ∂F/∂t, or " *
          "the matrix-free state products — a term declared through `facet_items` is its own " *
          "traversal and DOES enter them. This operator declares " *
          "$(kinds) and carries a `$(nameof(B))` boundary cache, so its sensitivities are " *
          "correct only if the boundary terms are independent of the seeded quantity — θ for " *
          "the parameter kinds, t for the time sensitivity, u for the state products. " *
          "`check_derivatives` detects the dependent case: its finite-difference referee " *
          "evaluates the FULL residual including boundary terms, so a failing parameter, time, " *
          "or state-product check on this operator is the signature of this omission." _id =
          Symbol(:boundary_sensitivity_, B, :_, hash(requests)) maxlog = 1
    return nothing
end

"""
    create_system_matrix(strategy, dh)

Allocate the operator's global matrix over the sparsity pattern the strategy's
operator specification declares ([`StandardOperatorSpecification`](@ref) →
`SparsityPattern`, [`BlockedOperatorSpecification`](@ref) →
`BlockSparsityPattern`), with entries added by Ferrite's
`add_sparsity_entries!` from the specification's `algebraic_couplings` and
`constraint_handler`, allocated as `matrix_type(strategy)`.

The constraint handler contributes SPARSITY ENTRIES only; applying the
constraints to the assembled system stays the caller's, through Ferrite's
`apply!`/`apply_assemble!`.
"""
create_system_matrix(strategy, dh) = _create_system_matrix(strategy, strategy.form.operator_specification, dh)
create_system_vector(strategy, dh) = allocate_vector(vector_type(strategy), dh)

function _create_system_matrix(strategy, spec, dh)
    sp = init_operator_sparsity_pattern(spec, dh)
    couplings = spec.algebraic_couplings
    # The `algebraic_couplings` keyword exists only on Ferrite versions with
    # mesh-free algebraic variables, so it is passed only when non-empty.
    if isempty(couplings)
        add_sparsity_entries!(sp, dh, spec.constraint_handler)
    else
        add_sparsity_entries!(sp, dh, spec.constraint_handler; algebraic_couplings = couplings)
    end
    return allocate_matrix(matrix_type(strategy), sp)
end

init_operator_sparsity_pattern(::StandardOperatorSpecification, dh) = Ferrite.init_sparsity_pattern(dh)
init_operator_sparsity_pattern(spec::BlockedOperatorSpecification, dh) = BlockSparsityPattern(spec.block_sizes)

function setup_elements(integrator, dh, ad_backend, n_global_dofs)
    needs_ad_decoration(integrator) || return [setup_element_cache(integrator, sdh) for sdh in dh.subdofhandlers]
    return [decorate_element_cache(setup_element_cache(integrator, sdh), sdh, ad_backend, n)
            for (sdh, n) in zip(dh.subdofhandlers, n_global_dofs)]
end

function setup_boundaries(integrator, dh)
    return [setup_boundary_cache(integrator, sdh) for sdh in dh.subdofhandlers]
end

function _cell_internal_offsets(integrator, element_caches, dh)
    num_dofs_per_element = zeros(Int, getncells(get_grid(dh))+1)
    for (sdh, cache) in zip(dh.subdofhandlers, element_caches)
        for (cellid, nidofs) in zip(sdh.cellset, get_number_of_internal_dofs_per_element(integrator, cache, sdh))
            num_dofs_per_element[1+cellid] = nidofs
        end
    end
    @assert all(num_dofs_per_element .≥ 0) "Number of internal dofs must be non-negative!"
    # The leading zero pad makes the cumulative sum exactly the `ncells+1`
    # block-relative offsets the handler expects.
    return cumsum(num_dofs_per_element)
end

# `has_internal_state` alone is not a safe signal that a cache needs a REAL
# internal-dof block: it may be declared purely to opt into the
# sensitivity-admissibility rules (`internal_state_insensitive`) and never
# condense anything, without implementing the count hook at all.
#
# The probe subject is the `unwrap` fixpoint, so a decorator's forwarding
# method answers for its inner, and the ARGUMENT types must be the concrete
# ones the later call passes — an author-annotated method is not matched by an
# `Any` probe, and missing it hands the operator a placeholder handler that
# fails inside `condense_cell!` instead.
_declares_internal_dofs(hook, integrator, cache, arg) =
    has_internal_state(typeof(cache)) &&
    hasmethod(hook, Tuple{typeof(integrator), typeof(unwrap(cache)), typeof(arg)})

"""
    setup_internal_variable_handler(integrator, element_caches, algebraic_domain, dh)

Build the [`InternalVariableHandler`](@ref) from what the resolved caches
declare; `algebraic_domain` is `resolve_algebraic_domain`'s result, `nothing`
where the integrator declares no algebraic items.

Whether a block is REAL (an offsets array) or the placeholder (`nothing`) is
decided per block by the cache — [`has_internal_state`](@ref) AND an
implemented dof-count hook
([`get_number_of_internal_dofs_per_element`](@ref) /
[`get_number_of_internal_dofs_per_algebraic_item`](@ref)) — not by the
integrator's type: the cell and item blocks are independent, and either or
both can be real at once (the layout-collision case).
"""
function setup_internal_variable_handler(integrator, element_caches, algebraic_domain, dh)
    needs_cells = any(zip(dh.subdofhandlers, element_caches)) do (sdh, cache)
        _declares_internal_dofs(get_number_of_internal_dofs_per_element, integrator, cache, sdh)
    end
    needs_items = algebraic_domain !== nothing && _declares_internal_dofs(
        get_number_of_internal_dofs_per_algebraic_item, integrator, algebraic_domain[1], algebraic_domain[2])
    (needs_cells || needs_items) && return _build_internal_variable_handler(
        integrator, element_caches, algebraic_domain, dh, needs_cells, needs_items)
    return InternalVariableHandler(nothing, nothing, 0, 0)
end

function _build_internal_variable_handler(integrator, element_caches, algebraic_domain, dh, needs_cells, needs_items)
    cell_offsets = needs_cells ? _cell_internal_offsets(integrator, element_caches, dh) : nothing
    item_offsets = needs_items ? _algebraic_item_offsets(integrator, algebraic_domain[1], algebraic_domain[2]) : nothing
    cell_len = cell_offsets === nothing ? 0 : cell_offsets[end]
    item_len = item_offsets === nothing ? 0 : item_offsets[end]
    return InternalVariableHandler(cell_offsets, item_offsets, ndofs(dh), cell_len + item_len)
end

function setup_subdomain_caches(strategy, element_caches, boundary_caches, ivh, dh;
        slots::NTuple{<:Any, Symbol}, needs_sensitivity::Bool, global_dof_sets)
    device = strategy.device
    return [begin
        partition = compute_partition(strategy, sdh)
        n = n_workers(strategy, device, partition)
        ws = create_assembly_workspace(element_cache, boundary_cache, sdh, ivh, slots;
                                       needs_sensitivity, global_dofs = gdofs)
        dc = setup_device_instances(device, ws, n)
        SubdomainCache(AssemblyDomain(sdh, ivh, element_cache, boundary_cache), dc, partition)
    end for (sdh, element_cache, boundary_cache, gdofs) in
        zip(dh.subdofhandlers, element_caches, boundary_caches, global_dof_sets)]
end

# The `global_dofs` declaration is resolved once per subdomain, before any
# cache exists, and validated here rather than surfacing later as an
# out-of-bounds scatter or a doubly assembled entry.
function resolve_global_dof_sets(strategy, integrator, dh)
    sets = [global_dofs(integrator, sdh) for sdh in dh.subdofhandlers]
    all(isempty, sets) && return sets
    _reject_unsupported_global_dof_strategy(strategy)
    for (index, (sdh, gdofs)) in enumerate(zip(dh.subdofhandlers, sets))
        _validate_global_dofs(index, sdh, gdofs, ndofs(dh))
    end
    return sets
end

function _reject_unsupported_global_dof_strategy(strategy::AssemblyStrategy)
    strategy.scheduling isa ColoredScheduling && throw(ArgumentError(
        "An element declaring `global_dofs` cannot be assembled under `ColoredScheduling`: " *
        "coloring makes a scatter race-free by giving no two items of a color a shared dof, " *
        "and a declared global dof is shared by every item of its subdomain, so no coloring " *
        "isolates it. Use `SequentialScheduling`, whose parallel route is the atomic scatter."))
    strategy.form isa Union{ElementAssembly, ElementAssemblyData} && throw(ArgumentError(
        "An element declaring `global_dofs` cannot be assembled in the `ElementAssembly` " *
        "form: its per-element storage and dof maps are built from `celldofs`, which by " *
        "construction never contains a global dof. Use `FullAssembly`."))
    return nothing
end

function _validate_global_dofs(index, sdh, gdofs, ndofs_total)
    for d in gdofs
        1 <= d <= ndofs_total || throw(ArgumentError(
            "Subdomain $index declares the global dof $d, which is out of bounds for a " *
            "DofHandler with $ndofs_total dofs."))
    end
    allunique(gdofs) || throw(ArgumentError(
        "Subdomain $index declares the global dofs $(collect(gdofs)), which are not unique. " *
        "The declaration is the ordered tail of the element-local system, so a repeated dof " *
        "would receive the same contribution twice."))
    # Cheap sample: the first cell witnesses a head/tail overlap for the
    # uniform-field case this covers.
    isempty(sdh.cellset) && return nothing
    cdofs = celldofs(sdh.dh, first(sdh.cellset))
    for d in gdofs
        d in cdofs && throw(ArgumentError(
            "Subdomain $index declares the global dof $d, which is also a cell dof (found on " *
            "cell $(first(sdh.cellset))). The local system is `[celldofs(cell); global dofs]`, " *
            "so such a dof would receive every contribution twice. Only the first cell of the " *
            "subdomain is sampled."))
    end
    return nothing
end

"""
    setup_engine(strategy, integrator, dh, protocol::AbstractSchemeProtocol; ad_backend = ForwardDiffAD())

Build the [`AssemblyEngine`](@ref) shared by all operator kinds from the
protocol's declarations ([`AbstractSchemeProtocol`](@ref)).

Element caches lacking analytic coverage of some AD-decorator kind are wrapped
in [`ADElementCache`](@ref) at construction, for every kind the integrator
might issue and not only the declared ones, decided STRUCTURALLY by
[`needs_ad_decoration`](@ref). `ad_backend = nothing` opts out of wrapping.

Facet item ([`facet_items`](@ref)) and then algebraic item
([`algebraic_items`](@ref)) caches are appended after the cell subdomains, so
traversal order follows the declarations rather than which families are
present. The algebraic domain is resolved BEFORE the
[`InternalVariableHandler`](@ref) is built, since a condensed algebraic cache's
item block sizes itself from the resolved items and cache, and decorated
afterwards alongside the cell caches.
"""
function setup_engine(strategy::AbstractAssemblyStrategy, integrator, dh::AbstractDofHandler, protocol::AbstractSchemeProtocol;
        ad_backend = ForwardDiffAD())
    requests          = get_declared_kinds(protocol)
    global_dof_sets   = resolve_global_dof_sets(strategy, integrator, dh)
    operator_strategy = setup_operator_strategy_cache(strategy, integrator, dh)
    element_caches    = setup_elements(integrator, dh, ad_backend, map(length, global_dof_sets))
    foreach(cache -> validate_element_cache(cache, requests), element_caches)
    boundary_caches   = setup_boundaries(integrator, dh)
    foreach(cache -> validate_boundary_cache(cache, requests), boundary_caches)
    algebraic_domain  = resolve_algebraic_domain(integrator, dh, protocol)
    ivh               = setup_internal_variable_handler(integrator, element_caches, algebraic_domain, dh)
    _warn_boundary_sensitivity(requests, boundary_caches)
    needs_sensitivity = needs_ad_decoration(integrator)
    cell_caches       = setup_subdomain_caches(operator_strategy, element_caches, boundary_caches, ivh, dh;
                                               slots = get_declared_slots(protocol),
                                               needs_sensitivity,
                                               global_dof_sets)
    facet_caches      = setup_facet_item_caches(operator_strategy, integrator, dh, protocol, ivh;
                                                slots = get_declared_slots(protocol),
                                                needs_sensitivity,
                                                global_dof_sets)
    algebraic_caches  = setup_algebraic_caches(operator_strategy, algebraic_domain, protocol, ad_backend,
                                               needs_sensitivity, ivh)
    # The families carry different domain types; widening only where something
    # is declared keeps a cells-only operator's element type concrete.
    subdomain_caches  = (isempty(facet_caches) && isempty(algebraic_caches)) ? cell_caches :
        vcat(Vector{SubdomainCache}(cell_caches), facet_caches, algebraic_caches)
    return AssemblyEngine(operator_strategy, subdomain_caches, dh, ivh, protocol)
end

"""
    setup_operator(strategy, problem, dh, protocol::AbstractSchemeProtocol; ad_backend = ForwardDiffAD())
    setup_operator(strategy, problem, dh; slots = (:u,), requests = (), ad_backend = ForwardDiffAD())

Build the operator for `problem` (an integrator) over `dh`. The positional
form takes the scheme's declarations as an [`AbstractSchemeProtocol`](@ref);
the keyword form is sugar for [`DefaultProtocol`](@ref) and lowers to the
positional one. `ad_backend` selects the [`ADElementCache`](@ref) backend
wrapping caches that lack analytic coverage (`nothing` opts out).

Transfer and patch operators keep their own constructors
([`setup_transfer_operator`](@ref), [`assemble_patches!`](@ref)).
"""
function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler, protocol::AbstractSchemeProtocol; ad_backend = ForwardDiffAD())
    engine = setup_engine(strategy, integrator, dh, protocol; ad_backend)
    A      = create_system_matrix(engine.strategy, dh)
    return BilinearFerriteOperator(A, engine, integrator)
end

# A matrix specification on an operator that holds no matrix is a
# misconfiguration, not a degraded mode: the layout would be silently dropped.
function _reject_blocked_specification(strategy::AssemblyStrategy{<:FullAssembly})
    strategy.form.operator_specification isa BlockedOperatorSpecification || return nothing
    throw(ArgumentError(
        "A linear operator assembles a vector and holds no matrix, so a " *
        "`BlockedOperatorSpecification` has nothing to lay out. Use a " *
        "`StandardOperatorSpecification`, or build the blocked matrix on the bilinear or " *
        "nonlinear operator it belongs to."))
end
_reject_blocked_specification(strategy) = nothing

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractNonlinearIntegrator, dh::AbstractDofHandler, protocol::AbstractSchemeProtocol; ad_backend = ForwardDiffAD())
    engine = setup_engine(strategy, integrator, dh, protocol; ad_backend)
    J      = create_system_matrix(engine.strategy, dh)
    return LinearizedFerriteOperator(J, engine, integrator)
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractLinearIntegrator, dh::AbstractDofHandler, protocol::AbstractSchemeProtocol; ad_backend = ForwardDiffAD())
    _reject_blocked_specification(strategy)
    engine = setup_engine(strategy, integrator, dh, protocol; ad_backend)
    b      = create_system_vector(engine.strategy, dh)
    return LinearFerriteOperator(b, engine, integrator)
end

setup_operator(strategy::AbstractAssemblyStrategy, integrator, dh::AbstractDofHandler;
        slots = (:u,), requests::Tuple = (), ad_backend = ForwardDiffAD()) =
    setup_operator(strategy, integrator, dh, DefaultProtocol(; slots, requests); ad_backend)

"""
    init_transfer_sparsity_pattern(dh_row::DofHandler, dh_col::DofHandler)

Build a `Ferrite.SparsityPattern` of size `(ndofs(dh_row) × ndofs(dh_col))` covering all
DoF pairs `(rdof, cdof)` that share a cell. Both DofHandlers must live on the same grid
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

# Shared by `setup_transfer_operator`/`setup_nested_transfer_operator`: both
# restrict to sequential full assembly, only the error label differs.
function _validate_transfer_strategy(strategy, label)
    (strategy isa AssemblyStrategy && strategy.form isa FullAssembly && strategy.scheduling isa SequentialScheduling) ||
        throw(ArgumentError("$label currently only support sequential full-assembly strategies (got $(typeof(strategy)))"))
    strategy.device isa SequentialCPUDevice ||
        throw(ArgumentError("$label currently only support SequentialCPUDevice (got $(typeof(strategy.device)))"))
    return nothing
end

# One `SubdomainCache` per `(sdh_a, sdh_b)` pair; `tc_builder` is the one thing
# a same-grid vs. nested-grid transfer operator disagrees on.
function _build_transfer_subdomain_caches(strategy, integrator, pairs, tc_builder)
    device = strategy.device
    subdomain_caches = SubdomainCache[]
    for (sdh_a, sdh_b) in pairs
        element = setup_transfer_element_cache(integrator, sdh_a, sdh_b)
        partition = compute_partition(strategy, sdh_a)
        n = n_workers(strategy, device, partition)
        tc = tc_builder(sdh_a, sdh_b)
        ws = TransferWorkspace(element, allocate_transfer_element_matrix(element, sdh_a, sdh_b), tc)
        dc = setup_device_instances(device, ws, n)
        push!(subdomain_caches, SubdomainCache(TransferDomain(sdh_a, sdh_b), dc, partition))
    end
    return subdomain_caches
end

"""
    setup_transfer_operator(strategy, integrator, dh_row, dh_col)

Set up a [`TransferFerriteOperator`](@ref) assembling a rectangular sparse matrix of
size `(ndofs(dh_row) × ndofs(dh_col))`. `dh_row` and `dh_col` must live on the **same**
grid with 1-to-1 subdomain lists (same length, same cellsets at each index).

`integrator` must be an [`AbstractTransferIntegrator`](@ref); its
`setup_transfer_element_cache(integrator, sdh_row, sdh_col)` runs once per subdomain
pair.

!!! warning "Experimental surface"
    The transfer constructors and operator types may change in a minor release;
    the assembled matrix and its sparsity are not affected.
"""
function setup_transfer_operator(
        strategy::AbstractAssemblyStrategy,
        integrator::AbstractTransferIntegrator,
        dh_row::DofHandler,
        dh_col::DofHandler,
    )
    _validate_transfer_strategy(strategy, "Transfer operators")
    @assert get_grid(dh_row) === get_grid(dh_col) "Both DofHandlers must share the same grid"
    @assert length(dh_row.subdofhandlers) == length(dh_col.subdofhandlers) "Mismatch in number of subdomains"

    Tv = value_type(strategy.device)
    sp = init_transfer_sparsity_pattern(dh_row, dh_col)
    P  = allocate_matrix(SparseMatrixCSC{Tv, Int}, sp)

    subdomain_caches = _build_transfer_subdomain_caches(
        strategy, integrator, zip(dh_row.subdofhandlers, dh_col.subdofhandlers), SameGridCellCache)

    return TransferFerriteOperator(P, strategy, subdomain_caches, dh_row, dh_col, integrator)
end

"""
    init_nested_transfer_sparsity_pattern(dh_fine, dh_coarse, fine2coarse)

Build a `Ferrite.SparsityPattern` of size `(ndofs(dh_fine) × ndofs(dh_coarse))` for a
nested-grid transfer operator: entry `(rdof, cdof)` is added whenever `rdof` belongs to
fine cell `i` and `cdof` to its parent coarse cell `fine2coarse[i]`.
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

Set up a [`NestedTransferFerriteOperator`](@ref) assembling a rectangular sparse matrix
of size `(ndofs(dh_fine) × ndofs(dh_coarse))`. `dh_fine` and `dh_coarse` must live on
**different** grids where every fine cell is a child of exactly one coarse cell, as
encoded by `fine2coarse` and `child_ref_coords`.

!!! warning "Experimental surface"
    The transfer constructors and operator types may change in a minor release;
    the assembled matrix and its sparsity are not affected.
"""
function setup_nested_transfer_operator(
        strategy::AbstractAssemblyStrategy,
        integrator::AbstractTransferIntegrator,
        dh_fine::DofHandler,
        dh_coarse::DofHandler,
        fine2coarse::AbstractVector{Int},
        child_ref_coords::AbstractVector,
    )
    _validate_transfer_strategy(strategy, "Nested transfer operators")
    Tv  = value_type(strategy.device)
    sp  = init_nested_transfer_sparsity_pattern(dh_fine, dh_coarse, fine2coarse)
    P   = allocate_matrix(SparseMatrixCSC{Tv, Int}, sp)

    subdomain_caches = _build_transfer_subdomain_caches(
        strategy, integrator, zip(dh_fine.subdofhandlers, dh_coarse.subdofhandlers),
        (sdh_fine, sdh_coarse) -> NestedGridCellCache(sdh_fine, sdh_coarse, fine2coarse, child_ref_coords))

    return NestedTransferFerriteOperator(P, strategy, subdomain_caches, dh_fine, dh_coarse, integrator)
end
