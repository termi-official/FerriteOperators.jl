####################################
## Facet items — boundary terms as their own traversal
####################################
#
# A boundary term is its own item family: one work item is one owning cell with
# all of that cell's declared facets, and the declared set IS the traversal.
# This is the ONE route a surface term takes — the cell sweep runs the
# volumetric kernel and nothing else.
#
# The element-facing half of the family (`facet_items`,
# `setup_facet_item_cache`) lives with the other element contracts, in
# core/element_interface.jl.

####################################
## Items, provider and partition
####################################

"""
    FacetItem(cellid, local_facets)

One work item of the facet family: the owning cell and the local indices of
its declared facets, ascending. Facets of one cell are never split across
items — that keeps facets sharing the owning cell's dofs on one worker, and
makes a facet item's local system owning-cell-shaped.
"""
struct FacetItem
    cellid::Int
    local_facets::Vector{Int}
end

"""
    FacetItems(sdh, items)

Work-item provider of the facet family: the resolved [`FacetItem`](@ref)s of
one `SubDofHandler`, addressed by index.

Both partitions are DERIVED from the owning cells. [`SequentialScheduling`](@ref)
hands out one chunk and lets the atomic scatter resolve the dofs neighbouring
cells' facets share (`dof_scatter_needs_atomic`). [`ColoredScheduling`](@ref)
colors the OWNING CELLS with Ferrite's cell coloring restricted to that set:
items of one color share no dofs precisely because their owning cells do not,
so the policy's barrier promise holds for facet items as it does for cells.
"""
struct FacetItems{SDH <: SubDofHandler, I}
    sdh::SDH
    items::I
end

compute_partition(::SequentialScheduling, provider::FacetItems) = (collect(eachindex(provider.items)),)

function compute_partition(scheduling::ColoredScheduling, provider::FacetItems)
    cells = [item.cellid for item in provider.items]
    index_of = Dict(cell => index for (index, cell) in pairs(cells))
    colors = Ferrite.create_coloring(get_grid(provider.sdh.dh), cells; alg = scheduling.alg)
    return [[index_of[cell] for cell in color] for color in colors]
end

####################################
## Workspace
####################################

"""
    FacetItemWorkspace

Per-worker workspace of the facet item family: [`AssemblyWorkspace`](@ref)'s
buffers with the surface cache in the element slot and the item declaration
added. A facet local system is OWNING-CELL-shaped, so `Ke`/`re`/the slot
buffers are sized like the cell family's — padded by this family's own
[`facet_item_global_dofs`](@ref) declaration, which lets a facet term couple to
a dof no cell owns (the tying shape) without augmenting the subdomain's cell
sweep.

`Ferrite.reinit!(ws, index)` positions it on an item: the geometry cache is
reinitialized on the owning cell and the augmented dof vector's head refreshed
from it, so [`item_dofs`](@ref)/[`scatter_address`](@ref) and every slot gather
resolve as on the cell route.

`current` is a `Ref{Int}` for the same reason [`AlgebraicWorkspace`](@ref)
carries one.
"""
@concrete struct FacetItemWorkspace <: AbstractWorkspace
    Ke
    slot_buffers   # NamedTuple of element-local state buffers keyed by slot name
    re
    cell
    ivh
    element        # the surface cache
    sensitivity    # SensitivityBuffers, or `nothing`
    dofs           # augmented dof vector, or `nothing`
    items          # the resolved FacetItems; shared, read-only during a sweep
    current        # Ref{Int}: the index of the item being processed
end

function Ferrite.reinit!(ws::FacetItemWorkspace, index::Int)
    ws.current[] = index
    reinit!(ws.cell, ws.items[index].cellid)
    _refresh_dof_head!(ws.dofs, ws.cell)
    return ws
end

"The local facet indices of the [`FacetItem`](@ref) `ws` is positioned on."
@inline current_facets(ws::FacetItemWorkspace) = ws.items[ws.current[]].local_facets

function duplicate_for_device(device::AbstractCPUDevice, ws::FacetItemWorkspace)
    return create_facet_item_workspace(
        duplicate_for_device(device, ws.element),
        ws.items,                                    # shared: read-only during a sweep
        ws.cell.dh,
        duplicate_for_device(device, ws.ivh),
        keys(ws.slot_buffers);
        needs_sensitivity = ws.sensitivity !== nothing,
        global_dofs = _declared_global_dofs(ws.dofs, ws.cell.dh),
    )
end

"""
    create_facet_item_workspace(cache, items, sdh, ivh, slots; needs_sensitivity = true, global_dofs = ())

Create a single [`FacetItemWorkspace`](@ref). Buffer sizing is
[`create_assembly_workspace`](@ref)'s: the same `allocate_element_*` hooks on
the surface cache, padded by the [`facet_item_global_dofs`](@ref) declaration.
"""
function create_facet_item_workspace(cache, items, sdh, ivh, slots::NTuple{N, Symbol} = (:u,);
        needs_sensitivity::Bool = true, global_dofs = ()) where {N}
    n = length(global_dofs)
    slot_buffers = NamedTuple{slots}(ntuple(_ -> pad_element_vector(allocate_element_unknown_vector(cache, sdh), n), N))
    return FacetItemWorkspace(
        pad_element_matrix(allocate_element_matrix(cache, sdh), n),
        slot_buffers,
        pad_element_vector(allocate_element_residual_vector(cache, sdh), n),
        CellCache(sdh),
        ivh,
        cache,
        needs_sensitivity ? create_sensitivity_buffers(cache, sdh, n) : nothing,
        _augmented_dof_vector(sdh, global_dofs),
        items,
        Ref(0),
    )
end

####################################
## Drivers
####################################

execute_single_task!(task::AssemblyTask, ws::FacetItemWorkspace) = execute_kind!(task.kind, task, ws)

"""
    facet_item_walk!(req, task, ws, statesₑ)

The facet walk of one item: `req` over the item's declared facets, in
declaration order, with the facet parameters queried SEPARATELY per facet. The
DECLARED set is the membership statement — a cache states its facets by
declaring them, never by re-deriving them per kernel call.

No [`reinit_values!`](@ref) call: a facet kernel reinitializes its own
`FacetValues` for the local facet index it was handed.
"""
function facet_item_walk!(req, task, ws, statesₑ)
    for lfi in current_facets(ws)
        pᵦ = query_facet_parameters(ws.element, ws.cell, lfi, task.p)
        assemble_facet!(req, ws.element, _facet_args(ws, statesₑ, pᵦ, task.ctx), lfi)
    end
    return nothing
end

execute_kind!(kind::PrimalKind, task, ws::FacetItemWorkspace) = primal_facet_item_sweep!(kind, task, ws)

"""
    primal_facet_item_sweep!(kind, task, ws)

The primal driver body of the facet item family: [`primal_cell_sweep!`](@ref)
with the volumetric kernel taken out and the facet walk restricted to the
item's own facets. Scatters ONCE per item through [`scatter_local!`](@ref) —
one scatter for all of a cell's declared facets, which is what the item shape
buys.

[`assert_facet_item_route`](@ref) runs first, so a kind this cache cannot serve
on this route fails naming the kernel to write instead of as a `MethodError`
inside the walk.
"""
function primal_facet_item_sweep!(kind, task, ws)
    assert_facet_item_route(kind, ws.element)
    assembles_matrix(kind) && fill!(ws.Ke, zero(eltype(ws.Ke)))
    assembles_vector(kind) && fill!(ws.re, zero(eltype(ws.re)))
    if depends_on_unknowns(kind)
        statesₑ = load_slots!(ws, task.states)
        @timeit_debug "assemble facet item" facet_item_walk!(materialize_request(kind, ws), task, ws, statesₑ)
    else
        @timeit_debug "assemble facet item" facet_item_walk!(materialize_request(kind, ws), task, ws, (;))
    end
    scatter_local!(kind, task.inner_assembler, ws)
end

execute_kind!(kind::SensitivityKind, task, ws::FacetItemWorkspace) = sensitivity_facet_item_sweep!(kind, task, ws)

"""
    sensitivity_facet_item_sweep!(kind, task, ws)

The [`sensitivity_cell_sweep!`](@ref) counterpart of the facet item family: a
facet-item term DOES enter the sensitivity sweeps. The request is bound over
[`SensitivityBuffers`](@ref) and scattered through [`scatter_request!`](@ref);
nothing is written back into `u`.

Analytic only — facet kernels have no automatic-differentiation fallback in any
sweep. Declaring the kind makes [`validate_facet_item_cache`](@ref) demand the
kernel at setup instead of letting the sweep reach a missing method.
"""
function sensitivity_facet_item_sweep!(kind, task, ws)
    statesₑ = load_slots!(ws, task.states)
    req = materialize_request(kind, ws, task)
    @timeit_debug "assemble facet item sensitivity" facet_item_walk!(req, task, ws, statesₑ)
    scatter_request!(req, task.inner_assembler, scatter_address(ws))
    return nothing
end

_item_family(ws::FacetItemWorkspace) = :facets
_family_reduction_sweep(kind, task, ws::FacetItemWorkspace) = functional_facet_item_sweep(kind, task, ws)

"""
    functional_facet_item_sweep(kind, task, ws) -> value

The [`functional_cell_sweep`](@ref) counterpart of the facet item family:
gathers the state slots without writing anything back and folds what
[`evaluate_facet_functional`](@ref) returns over the item's declared facets, in
declaration order. The item contributes `nothing` when every one of its facets
does, which is what keeps the untyped fold's type scan going.

The per-facet parameter query and the absent [`reinit_values!`](@ref) call are
[`facet_item_walk!`](@ref)'s; only the destination differs — a returned value
instead of the request buffers.
"""
function functional_facet_item_sweep(kind, task, ws)
    statesₑ = load_slots!(ws, task.states)
    total = initial_partial(kind)
    for lfi in current_facets(ws)
        pᵦ = query_facet_parameters(ws.element, ws.cell, lfi, task.p)
        val = evaluate_facet_functional(kind, ws.element, _facet_args(ws, statesₑ, pᵦ, task.ctx), lfi)
        val === nothing || (total = _reduce_partials(total, val))
    end
    return total
end

# Condensed internal state on facet items is not supported: `q` is per owning
# CELL, and a facet item shares its cell with the cell family's own item for
# that cell, so both condensing would write the same range twice. The kind's
# `reduction_families` declaration says the same thing structurally; the decline
# is spelled here too because its own cell driver is more specific than the
# derived route.
execute_kind!(::CondensationKind, task, ws::FacetItemWorkspace) = nothing

# ∂F/∂q is the block over the CONDENSED internal state, which no facet item
# owns — the family has no internal dof block at all.
execute_kind!(::JacobianKind{:q}, task, ws::FacetItemWorkspace) = nothing

# Quadrature evaluation writes the per-quadrature-point data of a cell, which
# the cell family's own sweep over that same cell already does.
execute_kind!(::QuadratureEvaluationKind, task, ws::FacetItemWorkspace) = nothing

####################################
## Setup
####################################

"""
    resolve_facet_items(index, sdh, declared) -> Vector{FacetItem}

The [`facet_items`](@ref) declaration of subdomain `index` grouped into work
items: the declared facets sorted, then collected per owning cell. Sorting is
what makes the item order — and with it the coloring, the scatter order and
the floating-point summation — reproducible from a `Set{FacetIndex}`, whose
own iteration order is not.

Throws unless every facet's cell lies in `sdh.cellset` (a facet whose cell this
subdomain does not own has no local system here), its local index exists on
that cell, and no facet is declared twice.
"""
function resolve_facet_items(index::Int, sdh::SubDofHandler, declared)
    grid = get_grid(sdh.dh)
    items = FacetItem[]
    # Sorted, so a cell's facets arrive consecutively and ascending: the run of
    # equal cell ids IS the item, and a duplicate is the previous entry.
    for (cellid, lfi) in sort!([(f[1], f[2]) for f in declared])
        cellid in sdh.cellset || throw(ArgumentError(
            "Subdomain $index declares the facet item `FacetIndex($cellid, $lfi)`, whose cell " *
            "is not in the subdomain's cellset. A facet item's local system is its owning " *
            "cell's, so the cell must belong to the `SubDofHandler` that declares it."))
        nf = Ferrite.nfacets(getcells(grid, cellid))
        1 <= lfi <= nf || throw(ArgumentError(
            "Subdomain $index declares the facet item `FacetIndex($cellid, $lfi)`, but cell " *
            "$cellid has $nf facets."))
        if !isempty(items) && last(items).cellid == cellid
            lfi == last(last(items).local_facets) && throw(ArgumentError(
                "Subdomain $index declares the facet `FacetIndex($cellid, $lfi)` twice. An item " *
                "assembles each of its facets once, so a repeated facet would contribute twice."))
            push!(last(items).local_facets, lfi)
        else
            push!(items, FacetItem(cellid, [lfi]))
        end
    end
    return items
end

"""
    validate_facet_item_cache(cache, declared_requests = ())

Setup-time consistency check for facet item caches, the
[`validate_element_cache`](@ref) counterpart. The mandatory
[`ResidualRequest`](@ref) facet kernel must exist, and every kind
[`provides_analytic`](@ref) claims must have a matching
[`assemble_facet!`](@ref) method.

A DECLARED sensitivity kind is checked harder — the kernel must exist, not
merely be trait-consistent — because facet contributions have no
automatic-differentiation fallback, unlike a volumetric cache whose
[`ADElementCache`](@ref) decoration would serve it from the residual. Every
declared kind additionally passes [`assert_facet_item_route`](@ref), the
route-level check the sweep repeats per item.

The probes run against [`unwrap`](@ref), the type an author would have written
the kernel on: a decorator's forwarding methods answer `hasmethod` for everyone
and would make every probe pass vacuously. A composite cache recurses into its
inners from there, for the same reason: its blanket fan-out method answers
every probe.
"""
function validate_facet_item_cache(cache, declared_requests::Tuple = ())
    _validate_facet_item_kernels(cache, declared_requests)
    return nothing
end

function _validate_facet_item_kernels(cache, declared_requests::Tuple)
    T = typeof(cache)
    D = unwrap(T)
    hasmethod(assemble_facet!, Tuple{ResidualRequest, D, FacetArgs, Int}) || throw(ArgumentError(
        "$(T) implements no `assemble_facet!(::ResidualRequest, ::$(nameof(D)), ::FacetArgs, " *
        "::Int)` method. The residual kernel is mandatory on the facet-item route, as it is " *
        "for every other item family."))
    for kind in _primal_validatable_kinds()
        _assert_trait_backed(T, kind, assemble_facet!, FacetArgs, (Int,))
    end
    for K in declared_requests
        has_cell_request(K) || continue
        kind = validation_instance(K)
        _assert_trait_backed(T, kind, assemble_facet!, FacetArgs, (Int,))
        assert_facet_item_route(kind, cache)
        K <: SensitivityKind && _assert_facet_analytic(D, kind)
    end
    return nothing
end

"""
    assert_facet_item_route(kind, cache)

Assert that `cache` can serve `kind` on the facet-item route. The generic
method passes: every other kind reaches its own [`assemble_facet!`](@ref)
method, or a `MethodError` naming it.

[`WeightedJacobianKind`](@ref) is the one kind that could take more than one,
and takes exactly one — the FUSED `assemble_facet!(::WeightedJacobianRequest,
…)` kernel. Nothing composes per-slot facet Jacobians behind the kernel's back,
so a cache without the fused kernel is rejected naming that kernel.

Checked at setup for every DECLARED kind ([`validate_facet_item_cache`](@ref))
and once per item in [`primal_facet_item_sweep!`](@ref) for the kind actually
swept, which is the declare-to-check model the kernel probes follow.
"""
assert_facet_item_route(kind, cache) = nothing

function assert_facet_item_route(kind::WeightedJacobianKind, cache)
    provides_analytic(typeof(cache), kind) || _throw_no_weighted_facet_item_route(typeof(cache), kind)
    return nothing
end

@noinline _throw_no_weighted_facet_item_route(T::Type, ::WeightedJacobianKind{slots}) where {slots} =
    throw(ArgumentError(
        "$(T) declares no fused weighted facet kernel, which is the only route a weighted " *
        "Jacobian sweep takes on the facet-item route. Implement " *
        "`assemble_facet!(::WeightedJacobianRequest, ::$(nameof(T)), ::FacetArgs, ::Int)` and " *
        "declare it through `provides_analytic(::Type{<:$(nameof(T))}, ::WeightedJacobianKind)`; " *
        "the kernel reads the weights of $(slots) from `req.weights` and forms the combination " *
        "itself. Facet kernels have no automatic-differentiation fallback, and this route " *
        "composes no per-slot `JacobianRequest{slot}` kernels behind the kernel's back."))

function _assert_facet_analytic(D::Type, kind)
    hasmethod(assemble_facet!, Tuple{request_type(kind), D, FacetArgs, Int}) && return nothing
    throw(ArgumentError(
        "$(nameof(D)) implements no `assemble_facet!(::$(request_type(kind)), ::$(nameof(D)), " *
        "::FacetArgs, ::Int)` method, and facet kernels have no automatic-differentiation " *
        "fallback in any sweep — a surface cache serves a request analytically or not at all. " *
        "A facet-item term DOES enter the sensitivity sweeps, so implement the kernel for " *
        "$(typeof(kind)) (declared via `provides_analytic`) or drop the kind from the " *
        "operator's declared requests."))
end

"""
    setup_facet_item_caches(strategy, integrator, dh, protocol, ivh; slots, needs_sensitivity, facet_item_global_dof_sets)

The `SubdomainCache`s of the facet item family, one per subdomain that
declares [`facet_items`](@ref): resolve and validate the declaration, build the
cache [`setup_facet_item_cache`](@ref) names, derive the partition, and build
the shared [`FacetItemWorkspace`](@ref).

Unlike the algebraic family, nothing has to be resolved before the
[`InternalVariableHandler`](@ref): a facet item owns no condensed internal state
(its owning cell's `q` belongs to the cell family's item for that cell), so the
declaration cannot change the `[ū | q_cells | q_items]` layout.
"""
function setup_facet_item_caches(strategy, integrator, dh, protocol, ivh;
        slots::NTuple{<:Any, Symbol}, needs_sensitivity::Bool, facet_item_global_dof_sets)
    device = strategy.device
    caches = SubdomainCache[]
    for (index, (sdh, gdofs)) in enumerate(zip(dh.subdofhandlers, facet_item_global_dof_sets))
        declared = facet_items(integrator, sdh)
        isempty(declared) && continue
        items = resolve_facet_items(index, sdh, declared)
        cache = setup_facet_item_cache(integrator, sdh)
        validate_facet_item_cache(cache, get_declared_kinds(protocol))
        partition = compute_partition(strategy, FacetItems(sdh, items))
        ws = create_facet_item_workspace(cache, items, sdh, ivh, slots; needs_sensitivity, global_dofs = gdofs)
        dc = setup_device_instances(device, ws, n_workers(device, partition))
        push!(caches, SubdomainCache(FacetItemDomain(sdh, cache, items), dc, partition))
    end
    return caches
end
