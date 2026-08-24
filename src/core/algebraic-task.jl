####################################
## Algebraic items — the engine side of terms with no mesh support
####################################
#
# An item of this family IS a set of global dofs and nothing else: a 0D model's
# own rows, an `AlgebraicCoupling`-only block, a lumped balance equation. There
# is no geometry cache, no values object and no quadrature, so the driver below
# is `primal_cell_sweep!` with the geometry taken out. Everything else — the
# gathers, the request materialization, the scatter, the AD decorator — is the
# machinery the cell family already uses.
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
[`SensitivityBuffers`](@ref) a nonlinear operator's sensitivity entry points
need, and the item declaration this worker resolves indices against.
`Ferrite.reinit!(ws, index)` positions it on an item.

Every buffer is FIXED-size: one provider's items are uniformly sized (validated
at setup), so no sweep ever resizes anything here.

`current` is a `Ref{Int}` for the same reason [`PatchAssemblyWorkspace`](@ref)
carries one — the item is resolved through the declaration between `reinit!`
and the kernel rather than handed to the kernel directly, which makes this a
CPU-scoped positioning mechanism.
"""
@concrete struct AlgebraicWorkspace <: AbstractWorkspace
    Ke
    slot_buffers   # NamedTuple of item-local state buffers keyed by slot name
    re
    element        # the algebraic cache, possibly AD-decorated
    sensitivity    # SensitivityBuffers, or `nothing`
    items          # the declared dof vectors; shared, read-only during a sweep
    current        # Ref{Int}: the index of the item being processed
end

Ferrite.reinit!(ws::AlgebraicWorkspace, index::Int) = (ws.current[] = index; ws)

@inline item_dofs(ws::AlgebraicWorkspace) = ws.items[ws.current[]]
@inline scatter_address(ws::AlgebraicWorkspace) = item_dofs(ws)

"The [`AlgebraicItem`](@ref) `ws` is positioned on, set by `Ferrite.reinit!(ws, index)`."
@inline current_item(ws::AlgebraicWorkspace) = AlgebraicItem(ws.current[], item_dofs(ws))

function duplicate_for_device(device::AbstractCPUDevice, ws::AlgebraicWorkspace)
    return AlgebraicWorkspace(
        copy(ws.Ke),
        map(copy, ws.slot_buffers),
        copy(ws.re),
        duplicate_for_device(device, ws.element),
        duplicate_for_device(device, ws.sensitivity),
        ws.items,                                    # shared: read-only during a sweep
        Ref(ws.current[]),
    )
end

"""
    create_algebraic_workspace(cache, items, slots, ::Type{T}; needs_sensitivity = true)

Create a single [`AlgebraicWorkspace`](@ref) whose buffers are sized by the
common dof count of `items` — an item's local system spans its own dofs and
nothing else, so there is no field space to allocate against and no padding to
apply.

`needs_sensitivity` selects whether [`SensitivityBuffers`](@ref) is built, the
same STRUCTURAL decision the cell workspace makes (see
[`needs_ad_decoration`](@ref)).
"""
function create_algebraic_workspace(cache, items, slots::NTuple{N, Symbol}, ::Type{T};
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
with the geometry taken out. There is no `reinit_values!` (an item carries no
values objects) and no facet walk (it carries no facets); everything else is
the same — zero the buffers [`assembles_matrix`](@ref)/
[`assembles_vector`](@ref) name, query the item parameters, gather the state
slots iff [`depends_on_unknowns`](@ref), run [`assemble_algebraic!`](@ref), and
scatter through [`scatter_local!`](@ref).
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

The [`sensitivity_cell_sweep!`](@ref) counterpart of the algebraic family:
gather the trial state, bind the request over [`SensitivityBuffers`](@ref),
issue it against the resolved cache, and scatter through
[`scatter_request!`](@ref). Nothing is written back into `u`.
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

execute_kind!(kind::FunctionalKind, task, ws::AlgebraicWorkspace) = functional_algebraic_sweep(kind, task, ws)

"""
    functional_algebraic_sweep(kind, task, ws) -> value

The [`functional_cell_sweep`](@ref) counterpart of the algebraic family:
gathers the state slots without writing anything back and returns what
[`evaluate_algebraic_functional`](@ref) gives for the item, `nothing` by
default.
"""
function functional_algebraic_sweep(kind, task, ws)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, current_item(ws), task.p)
    return evaluate_algebraic_functional(kind, ws.element, _algebraic_args(ws, statesₑ, pₑ, task.ctx))
end

# Condensation is per cell: an algebraic item owns no internal dofs, so it
# contributes nothing to the report and its slots are never gathered.
execute_kind!(::CondensationKind, task, ws::AlgebraicWorkspace) = nothing

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
[`assemble_algebraic!`](@ref) method, and the cache must not carry condensed
internal state — condensation is per cell, and an algebraic item has no place
in the [`InternalVariableHandler`](@ref).
"""
function validate_algebraic_cache(cache, declared_requests::Tuple = ())
    T = typeof(cache)
    hasmethod(assemble_algebraic!, Tuple{ResidualRequest, T, AlgebraicArgs}) || throw(ArgumentError(
        "$(T) implements no `assemble_algebraic!(::ResidualRequest, ::$(nameof(T)), " *
        "::AlgebraicArgs)` method. The residual kernel is mandatory: it is the basis for " *
        "AD-derived Jacobians and sensitivities."))
    has_internal_state(T) && throw(ArgumentError(
        "$(nameof(_display_cache_type(T))) declares `has_internal_state`, but an algebraic item " *
        "has no cell and therefore no internal-variable range to condense into. Condensed " *
        "internal state belongs to a volumetric element cache."))
    for kind in _primal_validatable_kinds()
        _assert_trait_backed(T, kind, assemble_algebraic!, AlgebraicArgs)
    end
    for K in declared_requests
        has_cell_request(K) || continue
        _assert_trait_backed(T, validation_instance(K), assemble_algebraic!, AlgebraicArgs)
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

"""
    setup_algebraic_caches(strategy, integrator, dh, protocol, ad_backend, needs_sensitivity)

The `SubdomainCache`s of the algebraic family — one per
[`algebraic_items`](@ref) declaration, and none where nothing is declared.
[`setup_engine`](@ref) appends them after the cell subdomains, so a sweep's
traversal order stays deterministic.
"""
function setup_algebraic_caches(strategy, integrator, dh, protocol, ad_backend, needs_sensitivity::Bool)
    declared = algebraic_items(integrator, dh)
    isempty(declared) && return SubdomainCache[]
    items  = resolve_algebraic_items(declared, ndofs(dh))
    cache  = setup_algebraic_cache(integrator, dh)
    validate_algebraic_cache(cache, get_declared_kinds(protocol))
    device = strategy.device
    T      = value_type(device)
    resolved  = needs_sensitivity ? decorate_algebraic_cache(cache, length(first(items)), ad_backend, T) : cache
    partition = compute_partition(strategy, AlgebraicItems(items))
    ws = create_algebraic_workspace(resolved, items, get_declared_slots(protocol), T; needs_sensitivity)
    dc = setup_device_instances(device, ws, n_workers(strategy, device, partition))
    return [SubdomainCache(AlgebraicDomain(resolved, items), dc, partition)]
end
