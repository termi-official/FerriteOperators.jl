####################################
## Algebraic items — the engine side of terms with no mesh support
####################################
#
# An item of this family IS a set of global dofs and nothing else: a 0D model's
# own rows, an `AlgebraicCoupling`-only block, a lumped balance equation. No
# geometry cache, no values object, no quadrature — the drivers below are the
# cell family's with the geometry taken out, and everything else (gathers,
# request materialization, scatter, AD decorator) is machinery the cell family
# already uses.
#
# The element-facing half of the family (`algebraic_items`,
# `setup_algebraic_cache`, `assemble_algebraic!`, `AlgebraicArgs`) lives with the
# other element contracts, in core/requests.jl and core/element_interface.jl.

####################################
## Provider and partition
####################################

"""
    AlgebraicItems(items)

Work-item provider of the algebraic family: the declared dof vectors, addressed
by index. Items of one provider generally SHARE dofs (several 0D rows on one
lumped unknown), and that is what fixes both partitions.
[`SequentialScheduling`](@ref) hands out one chunk and lets the atomic scatter
resolve the collisions — the route a [`global_dofs`](@ref) declaration takes
too. [`ColoredScheduling`](@ref) promises that no two items of a barrier share
a dof, which for an unknown sharing pattern leaves one item per barrier.
"""
struct AlgebraicItems{I}
    items::I
end

compute_partition(::SequentialScheduling, provider::AlgebraicItems) = (collect(eachindex(provider.items)),)
compute_partition(::ColoredScheduling, provider::AlgebraicItems) = [[i] for i in eachindex(provider.items)]

####################################
## Workspace
####################################

"""
    AlgebraicWorkspace

Per-worker workspace of the algebraic item family: the item-sized local system
(`Ke`, `re`, one slot buffer per declared slot), the algebraic cache, the
[`SensitivityBuffers`](@ref) the sensitivity entry points need, and the item
declaration this worker resolves indices against.
`Ferrite.reinit!(ws, index)` positions it on an item.

Every buffer is FIXED-size: one provider's items are uniformly sized (validated
at setup), so no sweep ever resizes anything here.

`current` is a `Ref{Int}` for the same reason [`PatchAssemblyWorkspace`](@ref)
carries one. `ivh` mirrors [`AssemblyWorkspace`](@ref)'s field of the same
name: the shared [`InternalVariableHandler`](@ref) an `:q` slot's
[`InternalSource`](@ref) gather resolves the current item's range against.
"""
@concrete struct AlgebraicWorkspace <: AbstractWorkspace
    Ke
    slot_buffers   # NamedTuple of item-local state buffers keyed by slot name
    re
    element        # the algebraic cache, possibly AD-decorated
    sensitivity    # SensitivityBuffers, or `nothing`
    items          # the declared dof vectors; shared, read-only during a sweep
    current        # Ref{Int}: the index of the item being processed
    ivh            # shared InternalVariableHandler
end

Ferrite.reinit!(ws::AlgebraicWorkspace, index::Int) = (ws.current[] = index; ws)

@inline item_dofs(ws::AlgebraicWorkspace) = ws.items[ws.current[]]
@inline scatter_address(ws::AlgebraicWorkspace) = item_dofs(ws)

"The [`AlgebraicItem`](@ref) `ws` is positioned on, set by `Ferrite.reinit!(ws, index)`."
@inline current_item(ws::AlgebraicWorkspace) = AlgebraicItem(ws.current[], item_dofs(ws))

# Restricts the source to the item's condensed internal-dof range, exactly like
# the cell family's `InternalSource` gather. A stateless algebraic cache has an
# empty range (`1:0` from the placeholder handler), so the gather is a no-op.
function load_slot!(buf, src::InternalSource, ws::AlgebraicWorkspace)
    range = internal_variable_range(ws.ivh, current_item(ws))
    resize!(buf, length(range))
    buf .= @view src.u[range]
    return buf
end

function duplicate_for_device(device::AbstractCPUDevice, ws::AlgebraicWorkspace)
    return AlgebraicWorkspace(
        copy(ws.Ke),
        map(copy, ws.slot_buffers),
        copy(ws.re),
        duplicate_for_device(device, ws.element),
        duplicate_for_device(device, ws.sensitivity),
        ws.items,                                    # shared: read-only during a sweep
        Ref(ws.current[]),
        duplicate_for_device(device, ws.ivh),
    )
end

"""
    create_algebraic_workspace(cache, items, slots, ::Type{T}, ivh; needs_sensitivity = true)

Create a single [`AlgebraicWorkspace`](@ref) whose buffers are sized by the
common dof count of `items` — an item's local system spans its own dofs and
nothing else, so there is no padding to apply.

`needs_sensitivity` selects whether [`SensitivityBuffers`](@ref) is built, the
same STRUCTURAL decision the cell workspace makes (see
[`needs_ad_decoration`](@ref)).
"""
function create_algebraic_workspace(cache, items, slots::NTuple{N, Symbol}, ::Type{T}, ivh;
        needs_sensitivity::Bool = true) where {N, T}
    n = length(first(items))
    slot_buffers = NamedTuple{slots}(ntuple(_ -> zeros(T, n), N))
    return AlgebraicWorkspace(
        zeros(T, n, n),
        slot_buffers,
        zeros(T, n),
        cache,
        needs_sensitivity ? create_sensitivity_buffers(n, T) : nothing,
        items,
        Ref(0),
        ivh,
    )
end

####################################
## Drivers
####################################

execute_single_task!(task::AssemblyTask, ws::AlgebraicWorkspace) = execute_kind!(task.kind, task, ws)

# The single AlgebraicArgs construction seam, mirroring `_cell_args`.
_algebraic_args(ws, statesₑ, pₑ, ctx) = AlgebraicArgs(statesₑ, current_item(ws), pₑ, ctx)

execute_kind!(kind::PrimalKind, task, ws::AlgebraicWorkspace) = primal_algebraic_sweep!(kind, task, ws)

"""
    primal_algebraic_sweep!(kind, task, ws)

The primal driver body of the algebraic family: [`primal_cell_sweep!`](@ref)
with the geometry taken out. No `reinit_values!` (an item carries no values
objects) and no facet walk (it carries no facets); the buffer zeroing, the
parameter query, the [`depends_on_unknowns`](@ref)-gated slot gather and the
[`scatter_local!`](@ref) scatter are the cell family's.
"""
function primal_algebraic_sweep!(kind, task, ws)
    assembles_matrix(kind) && fill!(ws.Ke, zero(eltype(ws.Ke)))
    assembles_vector(kind) && fill!(ws.re, zero(eltype(ws.re)))
    pₑ = query_cell_parameters(ws.element, current_item(ws), task.p)
    if depends_on_unknowns(kind)
        statesₑ = load_slots!(ws, task.states)
        @timeit_debug "assemble algebraic item" assemble_algebraic!(
            materialize_request(kind, ws), ws.element, _algebraic_args(ws, statesₑ, pₑ, task.ctx))
    else
        @timeit_debug "assemble algebraic item" assemble_algebraic!(
            materialize_request(kind, ws), ws.element, _algebraic_args(ws, (;), pₑ, task.ctx))
    end
    scatter_local!(kind, task.inner_assembler, ws)
end

execute_kind!(kind::SensitivityKind, task, ws::AlgebraicWorkspace) = sensitivity_algebraic_sweep!(kind, task, ws)

"""
    sensitivity_algebraic_sweep!(kind, task, ws)

The [`sensitivity_cell_sweep!`](@ref) counterpart of the algebraic family: the
request is bound over [`SensitivityBuffers`](@ref), issued against the resolved
cache and scattered through [`scatter_request!`](@ref). Nothing is written back
into `u`.
"""
function sensitivity_algebraic_sweep!(kind, task, ws)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, current_item(ws), task.p)
    args = _algebraic_args(ws, statesₑ, pₑ, task.ctx)
    req = materialize_request(kind, ws, task)
    @timeit_debug "assemble algebraic sensitivity" assemble_algebraic!(req, ws.element, args)
    scatter_request!(req, task.inner_assembler, scatter_address(ws))
    return nothing
end

_item_family(ws::AlgebraicWorkspace) = :algebraic
_family_reduction_sweep(kind, task, ws::AlgebraicWorkspace) = functional_algebraic_sweep(kind, task, ws)

"""
    functional_algebraic_sweep(kind, task, ws) -> value

The [`functional_cell_sweep`](@ref) counterpart of the algebraic family:
gathers the state slots without writing anything back and returns what
[`evaluate_algebraic_functional`](@ref) gives for the item (`nothing` by
default).
"""
function functional_algebraic_sweep(kind, task, ws)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, current_item(ws), task.p)
    return evaluate_algebraic_functional(kind, ws.element, _algebraic_args(ws, statesₑ, pₑ, task.ctx))
end

# Trait-gated: a stateless algebraic cache (the common case) owns no internal
# dofs, so condensation contributes nothing and its slots are never gathered.
execute_kind!(kind::CondensationKind, task, ws::AlgebraicWorkspace) =
    has_internal_state(typeof(ws.element)) ? condensation_algebraic_sweep!(kind, task, ws) : nothing

"""
    condensation_algebraic_sweep!(kind::CondensationKind, task, ws) -> CondensationReport

The [`condensation_cell_sweep!`](@ref) counterpart of the algebraic family:
hands the gathered slots to [`condense_algebraic!`](@ref) and copies the
item-local `q` buffer it filled into the `:q` slot's global vector over the
item's [`internal_variable_range`](@ref) — the item block of the
`[ū | q_cells | q_items]` tail. Gated on [`has_internal_state`](@ref) above.
"""
function condensation_algebraic_sweep!(kind::CondensationKind, task, ws)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, current_item(ws), task.p)
    args = _algebraic_args(ws, statesₑ, pₑ, task.ctx)
    report = condense_algebraic!(ws.element, args, kind.weights)
    qsrc = _q_source(task.states)
    range = internal_variable_range(ws.ivh, current_item(ws))
    qsrc.u[range] .= statesₑ.q
    return report
end

execute_kind!(kind::JacobianKind{:q}, task, ws::AlgebraicWorkspace) =
    internal_jacobian_algebraic_sweep!(kind, task, ws)

"""
    internal_jacobian_algebraic_sweep!(kind::JacobianKind{:q}, task, ws)

The [`internal_jacobian_cell_sweep!`](@ref) counterpart of the algebraic
family: the item's ∂F/∂q block, its rows the item's own dofs and its columns
the item's [`internal_variable_range`](@ref). A stateless algebraic cache owns
no internal dofs, so its range is empty and it never reaches a kernel.
"""
function internal_jacobian_algebraic_sweep!(kind::JacobianKind{:q, C}, task, ws) where {C}
    range = internal_variable_range(ws.ivh, current_item(ws))
    isempty(range) && return nothing
    Kqₑ = internal_sweep_buffers!(ws.sensitivity, length(range)).Kqₑ
    fill!(Kqₑ, zero(eltype(Kqₑ)))
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, current_item(ws), task.p)
    @timeit_debug "assemble internal jacobian" assemble_algebraic!(
        JacobianRequest{:q, C}(Kqₑ), ws.element, _algebraic_args(ws, statesₑ, pₑ, task.ctx))
    assemble!(task.inner_assembler, item_dofs(ws), _internal_columns(ws.ivh, range), Kqₑ)
    return nothing
end

# Quadrature evaluation writes the per-quadrature-point data of a cell; an
# algebraic item has neither cell nor quadrature points.
execute_kind!(::QuadratureEvaluationKind, task, ws::AlgebraicWorkspace) = nothing

####################################
## Setup
####################################

"""
    validate_algebraic_cache(cache, declared_requests = ())

Setup-time consistency check for algebraic caches, the
[`validate_element_cache`](@ref) counterpart: the mandatory
[`ResidualRequest`](@ref) kernel must exist, every kind
[`provides_analytic`](@ref) claims must have a matching
[`assemble_algebraic!`](@ref) method, and — when the cache declares
[`has_internal_state`](@ref) — every primal kind
[`requires_admissibility_check`](@ref) names must be served analytically or
declared [`internal_state_insensitive`](@ref), exactly like a condensed CELL
cache. Those two are the only escapes here: an algebraic item has no cellid to
key a corrector store by, so there is no generic AD `Consistent` bootstrap (see
[`condense_algebraic!`](@ref)) and never `ADElementCache`'s
`condensed_corrector` combination.

Runs on the RAW cache, before [`decorate_algebraic_cache`](@ref): the two
subjects the [`AbstractElementCacheDecorator`](@ref) convention distinguishes
coincide here, since the generic routes a decorator would add are
`CellArgs`-shaped and this family has none. The kernel probe still takes the
[`unwrap`](@ref) fixpoint, so a hand-decorated cache is probed on its
author-written method set.
"""
function validate_algebraic_cache(cache, declared_requests::Tuple = ())
    T = typeof(cache)
    A = unwrap(T)
    hasmethod(assemble_algebraic!, Tuple{ResidualRequest, A, AlgebraicArgs}) || throw(ArgumentError(
        "$(A) implements no `assemble_algebraic!(::ResidualRequest, ::$(nameof(A)), " *
        "::AlgebraicArgs)` method. The residual kernel is mandatory: it is the basis for " *
        "AD-derived Jacobians and sensitivities."))
    for kind in _primal_validatable_kinds()
        _assert_trait_backed(T, kind, assemble_algebraic!, AlgebraicArgs)
        requires_admissibility_check(kind) &&
            assert_sensitivity_admissible(T, kind, assemble_algebraic!, AlgebraicArgs)
    end
    for K in declared_requests
        has_cell_request(K) || continue
        kind = validation_instance(K)
        _assert_trait_backed(T, kind, assemble_algebraic!, AlgebraicArgs)
        requires_admissibility_check(kind) &&
            assert_sensitivity_admissible(T, kind, assemble_algebraic!, AlgebraicArgs)
    end
    return nothing
end

"""
    decorate_algebraic_cache(cache, ndofs, ad_backend, ::Type{T})

The [`decorate_element_cache`](@ref) counterpart for the algebraic family:
resolve `cache` into the form the driver calls unconditionally, with
[`ADElementCache`](@ref) sized by the item's dof count rather than by a
`SubDofHandler`. `ad_backend === nothing` opts out of the AD step.
"""
function decorate_algebraic_cache(cache, ndofs::Int, ad_backend, ::Type{T}) where {T}
    fused = _maybe_fuse_split(cache)
    ad_backend === nothing && return fused
    return fully_analytic(typeof(fused)) ? fused :
        ADElementCache(fused, ndofs, T; backend = ad_backend)
end

"""
    resolve_algebraic_items(declared, ndofs_total) -> Vector{Vector{Int}}

The [`algebraic_items`](@ref) declaration resolved once, before any cache
exists: uniform item size (the fixed-size local buffers depend on it),
in-bounds dofs, and no dof repeated WITHIN an item, which would receive the
same contribution twice. Dofs shared BETWEEN items are the normal case, and are
what the partition derivation accounts for.
"""
function resolve_algebraic_items(declared, ndofs_total::Int)
    items = [collect(Int, item) for item in declared]
    n = length(first(items))
    n == 0 && throw(ArgumentError(
        "Algebraic item 1 carries no dofs. An item of this family IS its dof set, so an empty " *
        "one has no local system to assemble."))
    for (index, dofs) in pairs(items)
        length(dofs) == n || throw(ArgumentError(
            "Algebraic item $index carries $(length(dofs)) dofs while item 1 carries $n. The " *
            "items of one declaration are uniformly sized — that is what keeps a worker's local " *
            "buffers fixed-size. Split heterogeneous algebraic terms so that one " *
            "`algebraic_items` declaration carries only items of equal size."))
        allunique(dofs) || throw(ArgumentError(
            "Algebraic item $index declares the dofs $dofs, which are not unique. The " *
            "declaration IS the item's local system, so a repeated dof would receive the same " *
            "contribution twice."))
        for d in dofs
            1 <= d <= ndofs_total || throw(ArgumentError(
                "Algebraic item $index declares the dof $d, which is out of bounds for a " *
                "DofHandler with $ndofs_total dofs."))
        end
    end
    return items
end

# Uniform-count validation + the `nitems+1` cumsum, mirroring
# `_cell_internal_offsets`'s shape but over the item declaration.
function _algebraic_item_offsets(integrator, cache, items)
    counts = collect(Int, get_number_of_internal_dofs_per_algebraic_item(integrator, cache, items))
    length(counts) == length(items) || throw(ArgumentError(
        "get_number_of_internal_dofs_per_algebraic_item returned $(length(counts)) counts for " *
        "$(length(items)) items."))
    @assert all(counts .≥ 0) "Number of internal dofs must be non-negative!"
    n = isempty(counts) ? 0 : first(counts)
    all(==(n), counts) || throw(ArgumentError(
        "Algebraic item internal dof counts $(counts) are not uniform. Condensed internal " *
        "state on the algebraic-item family follows the same uniform-item-size rule as the " *
        "items' own dof count (see `resolve_algebraic_items`): one `algebraic_items` " *
        "declaration's local buffers are fixed-size, so every item must own the same number " *
        "of internal dofs."))
    return cumsum(vcat(0, counts))
end

"""
    resolve_algebraic_domain(integrator, dh, protocol) -> (cache, items) or `nothing`

The [`algebraic_items`](@ref) declaration resolved and validated once, before
[`setup_internal_variable_handler`](@ref) or any decoration: `nothing` where
the integrator declares no items, otherwise the raw cache
[`setup_algebraic_cache`](@ref) built and the resolved item dof vectors — what
[`setup_internal_variable_handler`](@ref) needs to size a condensed algebraic
cache's item block, and what [`setup_algebraic_caches`](@ref) needs to finish
the domain.
"""
function resolve_algebraic_domain(integrator, dh, protocol)
    declared = algebraic_items(integrator, dh)
    isempty(declared) && return nothing
    items = resolve_algebraic_items(declared, ndofs(dh))
    cache = setup_algebraic_cache(integrator, dh)
    validate_algebraic_cache(cache, get_declared_kinds(protocol))
    return (cache, items)
end

"""
    setup_algebraic_caches(strategy, algebraic_domain, protocol, ad_backend, needs_sensitivity, ivh)

The `SubdomainCache`s of the algebraic family, finishing what
[`resolve_algebraic_domain`](@ref) resolved: decorate the cache, derive the
partition, and build the shared [`AlgebraicWorkspace`](@ref) against `ivh`.
`algebraic_domain === nothing` (nothing declared) returns no caches.
[`setup_engine`](@ref) appends the result after the cell subdomains, so a
sweep's traversal order stays deterministic.
"""
function setup_algebraic_caches(strategy, algebraic_domain, protocol, ad_backend, needs_sensitivity::Bool, ivh)
    algebraic_domain === nothing && return SubdomainCache[]
    cache, items = algebraic_domain
    device = strategy.device
    T      = value_type(device)
    resolved  = needs_sensitivity ? decorate_algebraic_cache(cache, length(first(items)), ad_backend, T) : cache
    partition = compute_partition(strategy, AlgebraicItems(items))
    ws = create_algebraic_workspace(resolved, items, get_declared_slots(protocol), T, ivh; needs_sensitivity)
    dc = setup_device_instances(device, ws, n_workers(strategy, device, partition))
    return [SubdomainCache(AlgebraicDomain(resolved, items), dc, partition)]
end
