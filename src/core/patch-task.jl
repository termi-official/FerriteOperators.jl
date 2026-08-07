####################################
## Patch items (experimental)
####################################
#
# Work items that are SETS of cells with a patch-local dof numbering — the
# local-BVP layer. Cell kernels are reused unchanged; the scatter target is
# patch-local. Solving the delivered local system is deliberately NOT part of
# this contract: the caller owns the solve, the sinks and the item-lifetime
# state slots below carry its inputs and results.
#
# Experimental: this surface may still change.

####################################
## Term restrictions
####################################

"""
    WholePatch()

Term restriction: integrate over every cell of the patch.
"""
struct WholePatch end

"""
    CellGroup(id)

Term restriction: integrate only over the patch cells carrying the group tag
`id` (see the `groups` argument of [`PatchItems`](@ref)). Group membership is
tested inside the single pass over the patch's cells, so a restricted term
accumulates in the same ascending cell order as an unrestricted one.
"""
struct CellGroup
    id::Int
end

@inline patch_term_active(::WholePatch, group::Int) = true
@inline patch_term_active(r::CellGroup, group::Int) = r.id == group

"""
    PatchTerm(restriction, data = nothing)

One additive contribution of a patch request. `restriction` is
[`WholePatch`](@ref) or [`CellGroup`](@ref); `data` is the element-specific
term payload handed to [`assemble_patch_cell!`](@ref) (`nothing` selects the
element's ordinary cell kernel).

A request carries a *tuple* of terms. All terms of one request are evaluated
in a single pass over the patch's cells and accumulate into the same element
buffer in tuple order — the accumulation order is part of the contract.
Terms that must fuse at quadrature-point level (rather than at cell level)
belong in ONE term whose `data` carries both sources.
"""
struct PatchTerm{R, D}
    restriction::R
    data::D
end
PatchTerm(restriction) = PatchTerm(restriction, nothing)

# The implicit request of the plain "assemble every cell with the element's own
# kernel" entry points.
const WHOLE_PATCH_TERMS = (PatchTerm(WholePatch(), nothing),)

# Tuple recursions; the compiler unrolls them (terms are short, concrete tuples)
# so each term is dispatched monomorphically and no term value ever enters a
# kernel as a `Union`.
@inline any_patch_term_active(::Tuple{}, group::Int) = false
@inline any_patch_term_active(terms::Tuple, group::Int) =
    patch_term_active(first(terms).restriction, group) || any_patch_term_active(Base.tail(terms), group)

"""
    whole_patch_terms(terms::Tuple)

The [`WholePatch`](@ref)-restricted subtuple of `terms`, order preserved.
Restrictions are tuple *type* information, so this folds at compile time —
elements that fuse several terms per quadrature point use it to select a
monomorphic loop for non-group cells instead of threading `Union`-typed
values through the quadrature loop.
"""
@inline whole_patch_terms(::Tuple{}) = ()
@inline function whole_patch_terms(terms::Tuple)
    rest = whole_patch_terms(Base.tail(terms))
    return first(terms).restriction isa WholePatch ? (first(terms), rest...) : rest
end

####################################
## The item provider
####################################

"""
    PatchDofPartition(prescribed, free)

The view-derived dof partition of one patch, in PATCH-LOCAL numbering:
`prescribed` are the dofs the caller's local BVP constrains (homogeneously —
this layer carries no inhomogeneities), `free` is their sorted complement.
"""
struct PatchDofPartition
    prescribed::Vector{Int}
    free::Vector{Int}
end

"""
    PatchItems(sdh, cellsets; groups, prescribed_facets, field)

Work-item provider over patches: item `i` is the cell set `cellsets[i]` (cells
of `sdh`) with its own contiguous patch-local dof numbering. The injection map
back to global dofs is [`patch_dofs`](@ref).

Keyword arguments:

- `groups`: per patch, a group tag per cell (aligned with the sorted cell list
  of that patch), the index a [`CellGroup`](@ref) restriction selects on.
  Defaults to the cell ids themselves, i.e. `CellGroup(cellid)` restricts to a
  single cell. The natural tagging for a multiscale patch is the parent coarse
  cell.
- `prescribed_facets`: per patch, an iterable of `FacetIndex` on the patch's
  cells that the local BVP prescribes. FO does NOT classify the boundary —
  which facets are cut boundary is caller geometry — it only resolves the
  given classification into the dof partition ([`patch_free_dofs`](@ref) /
  [`patch_prescribed_dofs`](@ref)). Defaults to no prescribed dofs.
- `field`: the field the partition and [`patch_vertex_dofs`](@ref) resolve
  against. Defaults to the `sdh`'s only field.

Experimental: the patch item surface may change with the local-BVP work.
"""
struct PatchItems{SDH <: SubDofHandler}
    sdh::SDH
    patches::Vector{Vector{Int}}
    groups::Vector{Vector{Int}}
    dofs::Vector{Vector{Int}}        # per patch: local → global dof (the injection)
    dofmaps::Vector{Dict{Int, Int}}  # per patch: global → local dof
    partitions::Vector{PatchDofPartition}
    field::Symbol
end

function PatchItems(
        sdh::SubDofHandler, cellsets;
        groups = nothing, prescribed_facets = nothing, field::Symbol = _only_field(sdh)
    )
    dh = sdh.dh
    patches = [sort!(collect(Int, cs)) for cs in cellsets]
    for cells in patches
        issubset(cells, sdh.cellset) || throw(ArgumentError(
                "patch cells $(setdiff(cells, sdh.cellset)) are not part of the SubDofHandler's cellset"))
    end
    dofs = map(patches) do cells
        gd = Int[]
        for c in cells
            append!(gd, celldofs(dh, c))
        end
        sort!(unique!(gd))
    end
    dofmaps = [Dict{Int, Int}(g => l for (l, g) in pairs(d)) for d in dofs]
    tags = groups === nothing ? [copy(cells) for cells in patches] :
        [collect(Int, g) for g in groups]
    length(tags) == length(patches) || throw(DimensionMismatch(
            "expected $(length(patches)) group vectors, got $(length(tags))"))
    for (i, g) in pairs(tags)
        length(g) == length(patches[i]) || throw(DimensionMismatch(
                "patch $i has $(length(patches[i])) cells but $(length(g)) group tags"))
    end
    provider = PatchItems(sdh, patches, tags, dofs, dofmaps, PatchDofPartition[], field)
    facets = prescribed_facets === nothing ? Iterators.repeated(FacetIndex[]) : prescribed_facets
    for (i, fs) in zip(1:length(patches), facets)
        push!(provider.partitions, _partition_from_facets(provider, i, fs))
    end
    return provider
end

function _only_field(sdh::SubDofHandler)
    names = Ferrite.getfieldnames(sdh)
    length(names) == 1 || throw(ArgumentError(
            "the SubDofHandler carries fields $(names); pass `field = :name` explicitly"))
    return first(names)
end

# Local dofs of a field on one local entity, expanded over the field's
# components exactly as Ferrite's Dirichlet machinery does.
function _entity_local_dofs(sdh::SubDofHandler, field::Symbol, entitydofs, entity::Int)
    idx = Ferrite.find_field(sdh, field)
    ip = Ferrite.getfieldinterpolation(sdh, idx)
    ncomp = Ferrite.n_dbc_components(ip)
    ip isa Ferrite.VectorizedInterpolation && (ip = ip.ip)
    offset = Ferrite.field_offset(sdh, idx)
    return [(d - 1) * ncomp + c + offset for d in entitydofs(ip)[entity] for c in 1:ncomp]
end

function _partition_from_facets(provider::PatchItems, i::Int, facets)
    dh = provider.sdh.dh
    dofmap = provider.dofmaps[i]
    prescribed = Int[]
    for fi in facets
        cellid, lfi = fi[1], fi[2]
        insorted(cellid, provider.patches[i]) || throw(ArgumentError(
                "prescribed facet $fi is on cell $cellid, which is not part of patch $i"))
        gdofs = celldofs(dh, cellid)
        for ld in _entity_local_dofs(provider.sdh, provider.field, Ferrite.dirichlet_facetdof_indices, lfi)
            push!(prescribed, dofmap[gdofs[ld]])
        end
    end
    return _partition(prescribed, length(provider.dofs[i]))
end

function _partition(prescribed::Vector{Int}, ndofs::Int)
    sort!(unique!(prescribed))
    free = setdiff(1:ndofs, prescribed)
    return PatchDofPartition(prescribed, free)
end

npatches(provider::PatchItems) = length(provider.patches)

"Global dofs of patch `i` in patch-local order (the local→global injection)."
patch_dofs(provider::PatchItems, i::Int) = provider.dofs[i]

"Number of dofs of patch `i`."
patch_ndofs(provider::PatchItems, i::Int) = length(provider.dofs[i])

"Cells of patch `i`, ascending — the order every patch sweep visits them in."
patch_cells(provider::PatchItems, i::Int) = provider.patches[i]

"Group tag of each cell of patch `i`, aligned with [`patch_cells`](@ref)."
patch_cell_groups(provider::PatchItems, i::Int) = provider.groups[i]

"Patch-local dofs of patch `i` the local BVP prescribes (sorted)."
patch_prescribed_dofs(provider::PatchItems, i::Int) = provider.partitions[i].prescribed

"Patch-local dofs of patch `i` the local BVP solves for (sorted)."
patch_free_dofs(provider::PatchItems, i::Int) = provider.partitions[i].free

"""
    augment_prescribed_dofs!(provider, i, localdofs)

Add `localdofs` (patch-local) to patch `i`'s prescribed set and recompute the
free set. The facet classification is not a closed rule: callers pin extra
dofs the local BVP needs to be well posed (e.g. floating components of a
perforated patch that no prescribed facet touches). Setup-time mutation —
call before any sweep reads the partition.
"""
function augment_prescribed_dofs!(provider::PatchItems, i::Int, localdofs)
    prescribed = append!(copy(provider.partitions[i].prescribed), localdofs)
    all(d -> 1 <= d <= patch_ndofs(provider, i), prescribed) || throw(ArgumentError(
            "prescribed dofs must be patch-local indices in 1:$(patch_ndofs(provider, i))"))
    provider.partitions[i] = _partition(prescribed, patch_ndofs(provider, i))
    return provider
end

"""
    patch_vertices(provider, i) -> Vector{Int}

Grid vertex (node) ids of patch `i`, ascending. With
[`patch_vertex_dofs`](@ref) this is the correspondence a caller needs to
restrict its own vertex-indexed data (e.g. a fine→coarse vertex map) to the
patch.
"""
function patch_vertices(provider::PatchItems, i::Int)
    grid = Ferrite.get_grid(provider.sdh.dh)
    nodes = Int[]
    for c in provider.patches[i]
        append!(nodes, Ferrite.vertices(getcells(grid, c)))
    end
    return sort!(unique!(nodes))
end

"""
    patch_vertex_dofs(provider, i) -> Dict{Int, Vector{Int}}

Grid vertex (node) id → the patch-local dofs of the provider's field at that
vertex, for patch `i`. Computed per call (no provider state), so it is safe to
call concurrently.
"""
function patch_vertex_dofs(provider::PatchItems, i::Int)
    dh = provider.sdh.dh
    grid = Ferrite.get_grid(dh)
    dofmap = provider.dofmaps[i]
    out = Dict{Int, Vector{Int}}()
    for c in provider.patches[i]
        gdofs = celldofs(dh, c)
        for (lvi, node) in pairs(Ferrite.vertices(getcells(grid, c)))
            haskey(out, node) && continue
            lds = _entity_local_dofs(provider.sdh, provider.field, Ferrite.vertexdof_indices, lvi)
            out[node] = [dofmap[gdofs[ld]] for ld in lds]
        end
    end
    return out
end

compute_partition(::SequentialScheduling, provider::PatchItems) = (collect(1:npatches(provider)),)

####################################
## Item-lifetime state
####################################

"""
    PatchItemStates{S}(nitems)

Per-item state slots of element type `S`, indexed by item. Two uses, one
mechanism — they differ only in the caller's invalidation policy:

- *item-lifetime state* that must survive across sweeps (a retained
  factorization, a reduction snapshot),
- the *solve→scatter payload channel* within one sweep (what a local solve
  hands to the phase that emits its result).

Freshness contract: FO never writes and never invalidates these slots. The
caller stores with [`set_item_state!`](@ref), tests with
[`has_item_state`](@ref), and drops stale content with
[`invalidate_item_state!`](@ref) when whatever the slot was derived from
changes. Slots are indexed by ITEM, so items processed by different workers
touch disjoint slots — item lifetime is not worker lifetime, and a slot must
never be handed to a worker-lifetime cache.
"""
struct PatchItemStates{S}
    slots::Vector{S}
    valid::Vector{Bool}
end
PatchItemStates{S}(nitems::Int) where {S} = PatchItemStates(Vector{S}(undef, nitems), fill(false, nitems))

Base.length(st::PatchItemStates) = length(st.valid)

"Is item `i`'s state slot filled and not invalidated?"
has_item_state(st::PatchItemStates, i::Int) = st.valid[i]

"""
    item_state(st, i)

Item `i`'s state. Throws when the slot is empty or was invalidated — guard
with [`has_item_state`](@ref).
"""
function item_state(st::PatchItemStates, i::Int)
    st.valid[i] || throw(ArgumentError("item $i has no valid state; check `has_item_state` first"))
    return st.slots[i]
end

"Store `s` as item `i`'s state and mark it valid."
set_item_state!(st::PatchItemStates, i::Int, s) = (st.slots[i] = s; st.valid[i] = true; st)

"Drop item `i`'s state (the caller's invalidation trigger fired)."
invalidate_item_state!(st::PatchItemStates, i::Int) = (st.valid[i] = false; st)

"Drop every item's state."
invalidate_item_states!(st::PatchItemStates) = (fill!(st.valid, false); st)

####################################
## Sinks
####################################

"""
Supertype of the patch assembly sinks: where a patch's assembled matrix or
vector goes. Matrix-vs-vector is the request dimension
([`PatchMatrixKind`](@ref) / [`PatchVectorKind`](@ref)); the sink is the
scatter mode within it.
"""
abstract type AbstractPatchSink end

"""
    PatchLocalSink(dest)

Patch-local sink: patch `i` is assembled into `dest[i]`, indexed by
patch-local dofs and zeroed by the driver. `dest[i]` is a matrix for a
[`PatchMatrixKind`](@ref) and a vector for a [`PatchVectorKind`](@ref);
scatter is by scalar indexing, so a sparse `dest[i]` GROWS its pattern for
entries it does not yet hold.
"""
struct PatchLocalSink{D} <: AbstractPatchSink
    dest::D
end

"""
    PatchAssemblerSink(dest)

Matrix sink through a Ferrite assembler per item (a cached column walk instead
of a binary search per entry). `dest[i]` must be a sparse matrix over patch
`i`'s dofs whose pattern already holds every element entry; use
[`PatchLocalSink`](@ref) when the pattern must grow.
"""
struct PatchAssemblerSink{D} <: AbstractPatchSink
    dest::D
end

"""
    PatchGlobalVectorSink(dest)

Additive global-vector sink: patch `i`'s assembled vector is accumulated into
`dest` through the injection, `dest[patch_dofs(provider, i)[l]] += v[l]`.
"""
struct PatchGlobalVectorSink{V <: AbstractVector} <: AbstractPatchSink
    dest::V
end

"""
    PatchTripletSink{Tv}(columns)
    PatchTripletSink(columns)

Ordered COO sink for vector-valued patch results: patch `i` emits the triplets
`(patch_dofs(provider, i)[l], columns[i], v[l])` in ascending `l`. Duplicate
`(row, col)` entries are SUMMED by `sparse(sink, m, n)` — patch-additive
quantities (a corrector basis assembled from overlapping patches) rely on that.

The emission order is the item order, so a sequential sweep is reproducible
bit-for-bit. Callers running their own parallel chunks collect into one sink
per chunk and `append!` them back in chunk order to keep that property.
[`emit_patch_column!`](@ref) is public: a local SOLVE's result is emitted the
same way an assembled vector is.
"""
struct PatchTripletSink{Tv} <: AbstractPatchSink
    columns::Vector{Int}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Tv}
end
PatchTripletSink{Tv}(columns) where {Tv} = PatchTripletSink(collect(Int, columns), Int[], Int[], Tv[])
PatchTripletSink(columns) = PatchTripletSink{Float64}(columns)

"""
    emit_patch_column!(sink::PatchTripletSink, rows, col, values)

Append `(rows[l], col, values[l])` for every `l`, in order.
"""
function emit_patch_column!(sink::PatchTripletSink, rows, col::Int, values)
    length(rows) == length(values) || throw(DimensionMismatch(
            "$(length(rows)) rows but $(length(values)) values"))
    for (r, v) in zip(rows, values)
        push!(sink.I, r)
        push!(sink.J, col)
        push!(sink.V, v)
    end
    return sink
end

Base.empty!(s::PatchTripletSink) = (empty!(s.I); empty!(s.J); empty!(s.V); s)
Base.append!(a::PatchTripletSink, b::PatchTripletSink) =
    (append!(a.I, b.I); append!(a.J, b.J); append!(a.V, b.V); a)
SparseArrays.sparse(s::PatchTripletSink, m::Int, n::Int) = sparse(s.I, s.J, s.V, m, n)

####################################
## Requests
####################################

"""
    PatchMatrixKind(terms, sink)

Assemble a patch-local MATRIX from a tuple of [`PatchTerm`](@ref)s. Terms
reach the element as `JacobianRequest{:u}` over the element buffer; a term
with `nothing` data runs the element's ordinary Jacobian kernel (including the
AD fallback), a term with data goes to [`assemble_patch_cell!`](@ref).
"""
struct PatchMatrixKind{T <: Tuple, S <: AbstractPatchSink}
    terms::T
    sink::S
end

"""
    PatchVectorKind(terms, sink)

Assemble a patch-local VECTOR from a tuple of [`PatchTerm`](@ref)s — the
right-hand side of a local BVP. Terms reach the element as `ResidualRequest`
over the element buffer; a term with `nothing` data runs the element's
ordinary residual kernel, a term with data goes to
[`assemble_patch_cell!`](@ref).
"""
struct PatchVectorKind{T <: Tuple, S <: AbstractPatchSink}
    terms::T
    sink::S
end

const PatchAssemblyKind = Union{PatchMatrixKind, PatchVectorKind}

patch_element_kind(::PatchMatrixKind) = JacobianKind()
patch_element_kind(::PatchVectorKind) = ResidualKind()

"""
    assemble_patch_cell!(req, cache, args::KernelArgs, data)

Element kernel for one [`PatchTerm`](@ref) on one cell: accumulate the term's
contribution into `req`'s buffer, exactly as `assemble_cell!` does. `data` is
the term payload from the request; the `nothing` payload dispatches to the
element's ordinary cell kernel, so unrestricted patch assembly reuses existing
elements unchanged.

Term kernels are analytic — there is no AD fallback for a custom payload.
"""
function assemble_patch_cell! end

####################################
## Execution
####################################

# Wraps a cell workspace; the item is a patch index resolved through the
# provider.
@concrete struct PatchAssemblyWorkspace <: AbstractWorkspace
    provider
    current   # Ref{Int}: the patch index of the item being processed
    inner     # the per-worker cell AssemblyWorkspace
    buffer    # patch-sized scratch for sinks that do not own patch-local storage
    ldofs     # the current cell's dofs in patch-local numbering
end

Ferrite.reinit!(ws::PatchAssemblyWorkspace, patchid::Int) = (ws.current[] = patchid; nothing)

execute_single_task!(task::AssemblyTask, ws::PatchAssemblyWorkspace) = execute_kind!(task.kind, task, ws)

# --- sink protocol ---------------------------------------------------------
# `patch_target` yields the zeroed patch-local accumulator, `patch_scatter`
# the object cell contributions are scattered through, `patch_emit!` publishes
# the finished patch quantity.

function patch_target(sink::PatchLocalSink, ws, pid)
    dest = sink.dest[pid]
    fill!(dest, zero(eltype(dest)))
    return dest
end
function patch_target(sink::PatchAssemblerSink, ws, pid)
    dest = sink.dest[pid]
    fill!(nonzeros(dest), zero(eltype(dest)))
    return dest
end
patch_target(::Union{PatchGlobalVectorSink, PatchTripletSink}, ws, pid) =
    (v = _patch_buffer(ws, pid); fill!(v, zero(eltype(v))); v)

function _patch_buffer(ws::PatchAssemblyWorkspace, pid)
    n = patch_ndofs(ws.provider, pid)
    length(ws.buffer) < n && resize!(ws.buffer, n)
    return view(ws.buffer, 1:n)
end

patch_scatter(::AbstractPatchSink, target) = target
patch_scatter(::PatchAssemblerSink, target) = start_assemble(target; fillzero = false)

@inline function scatter_patch_cell!(target::AbstractMatrix, ldofs, Ke)
    for (j, lj) in pairs(ldofs), (i, li) in pairs(ldofs)
        target[li, lj] += Ke[i, j]
    end
    return nothing
end
@inline scatter_patch_cell!(assembler::Ferrite.AbstractAssembler, ldofs, Ke) =
    (assemble!(assembler, ldofs, Ke); nothing)
@inline function scatter_patch_cell!(target::AbstractVector, ldofs, re)
    for (i, li) in pairs(ldofs)
        target[li] += re[i]
    end
    return nothing
end

patch_emit!(::Union{PatchLocalSink, PatchAssemblerSink}, provider, pid, target) = nothing
function patch_emit!(sink::PatchGlobalVectorSink, provider, pid, target)
    for (l, g) in pairs(patch_dofs(provider, pid))
        sink.dest[g] += target[l]
    end
    return nothing
end
patch_emit!(sink::PatchTripletSink, provider, pid, target) =
    (emit_patch_column!(sink, patch_dofs(provider, pid), sink.columns[pid], target); nothing)

# Matrix-vs-vector is the request dimension; not every scatter mode exists in
# both. Fail at the sweep, not with an unrelated method error per cell.
_check_sink(::PatchMatrixKind, ::Union{PatchLocalSink, PatchAssemblerSink}) = nothing
_check_sink(::PatchVectorKind, ::Union{PatchLocalSink, PatchGlobalVectorSink, PatchTripletSink}) = nothing
_check_sink(kind, sink) = throw(ArgumentError("$(typeof(sink).name.name) is not a valid sink for $(typeof(kind).name.name)"))

# --- the driver ------------------------------------------------------------

# One pass over the patch's cells; per cell the active terms accumulate into the
# element buffer in tuple order before a single scatter into the patch target.
function execute_kind!(kind::PatchAssemblyKind, task, ws::PatchAssemblyWorkspace)
    pid = ws.current[]
    provider = ws.provider
    dofmap = provider.dofmaps[pid]
    iws = ws.inner
    ldofs = ws.ldofs
    target = patch_target(kind.sink, ws, pid)
    scatter = patch_scatter(kind.sink, target)
    for (k, cellid) in pairs(provider.patches[pid])
        group = provider.groups[pid][k]
        any_patch_term_active(kind.terms, group) || continue
        reinit!(iws.cell, cellid)
        _zero_element_buffer!(kind, iws)
        reinit_values!(iws.element, iws.cell, patch_element_kind(kind))
        pₑ = query_cell_parameters(iws.element, iws.cell, task.p)
        statesₑ = load_slots!(iws, task.states)
        run_patch_terms!(kind, kind.terms, group, iws, statesₑ, pₑ, task.ctx)
        cdofs = celldofs(iws.cell)
        length(ldofs) == length(cdofs) || resize!(ldofs, length(cdofs))
        for (i, g) in pairs(cdofs)
            ldofs[i] = dofmap[g]
        end
        scatter_patch_cell!(scatter, ldofs, _element_buffer(kind, iws))
    end
    patch_emit!(kind.sink, provider, pid, target)
    return nothing
end

_element_buffer(::PatchMatrixKind, iws) = iws.Ke
_element_buffer(::PatchVectorKind, iws) = iws.re
_zero_element_buffer!(kind, iws) = fill!(_element_buffer(kind, iws), 0.0)

@inline run_patch_terms!(kind, ::Tuple{}, group::Int, iws, statesₑ, pₑ, ctx) = nothing
@inline function run_patch_terms!(kind, terms::Tuple, group::Int, iws, statesₑ, pₑ, ctx)
    term = first(terms)
    patch_term_active(term.restriction, group) &&
        run_patch_term!(kind, term.data, iws, statesₑ, pₑ, ctx)
    return run_patch_terms!(kind, Base.tail(terms), group, iws, statesₑ, pₑ, ctx)
end

# The `nothing` payload is the element's ordinary cell kernel (AD fallback
# included); a real payload goes to the analytic term kernel.
@inline run_patch_term!(kind::PatchMatrixKind, ::Nothing, iws, statesₑ, pₑ, ctx) =
    v2_cell_kernel!(JacobianKind(), iws.element, iws, statesₑ, pₑ, ctx)
@inline run_patch_term!(kind::PatchVectorKind, ::Nothing, iws, statesₑ, pₑ, ctx) =
    v2_cell_kernel!(ResidualKind(), iws.element, iws, statesₑ, pₑ, ctx)
@inline run_patch_term!(kind::PatchMatrixKind, data, iws, statesₑ, pₑ, ctx) =
    assemble_patch_cell!(JacobianRequest{:u}(iws.Ke), iws.element, _v2_args(iws, statesₑ, pₑ, ctx), data)
@inline run_patch_term!(kind::PatchVectorKind, data, iws, statesₑ, pₑ, ctx) =
    assemble_patch_cell!(ResidualRequest(iws.re), iws.element, _v2_args(iws, statesₑ, pₑ, ctx), data)

"""
    assemble_patches!(kind, op, provider::PatchItems, states, p, ctx = nothing)

Run one patch sweep of `kind` ([`PatchMatrixKind`](@ref) or
[`PatchVectorKind`](@ref)) over every item of `provider`, reusing the
operator's element caches. Each patch is visited once, its cells in ascending
order; the kind's sink receives the patch-local result. Sequential CPU only.

Patch sweeps are pure evaluation: condensed element unknowns are gathered but
never written back, unlike the global sweeps.

Experimental: part of the patch item family; the local BVP itself (partition,
solve, item state) is the caller's — see [`patch_free_dofs`](@ref) and
[`PatchItemStates`](@ref).
"""
function assemble_patches!(kind::PatchAssemblyKind, op, provider::PatchItems, states::NamedTuple, p, ctx = nothing)
    _check_declared_slots(op.engine, states)
    _check_sink(kind, kind.sink)
    op.engine.strategy.device isa SequentialCPUDevice || throw(
        ArgumentError(
            "patch assembly currently supports SequentialCPUDevice only (got $(typeof(op.engine.strategy.device)))"
        )
    )
    sc = findfirst(sc -> sc.domain.sdh === provider.sdh, op.engine.subdomain_caches)
    sc === nothing && throw(ArgumentError("the provider's SubDofHandler is not part of the operator"))
    ws = PatchAssemblyWorkspace(provider, Ref(0), first(op.engine.subdomain_caches[sc].device_cache), Float64[], Int[])
    task = AssemblyTask(kind, nothing, states, p, ctx)
    execute_on_device!(task, op.engine.strategy.device, (ws,), compute_partition(SequentialScheduling(), provider))
    return kind.sink
end

"""
    assemble_patch_matrices!(dest, op, provider::PatchItems, states, p, ctx = nothing)

Assemble the patch-local Jacobian of every patch into `dest[i]` (a matrix of
size `patch_ndofs(provider, i)` square, zeroed here), reusing the operator's
cell kernels; rows/columns follow [`patch_dofs`](@ref). Shorthand for
`assemble_patches!` with one whole-patch term and a [`PatchLocalSink`](@ref).

Experimental: part of the patch item family.
"""
function assemble_patch_matrices!(dest::AbstractVector, op, provider::PatchItems, states::NamedTuple, p, ctx = nothing)
    length(dest) == npatches(provider) || throw(
        DimensionMismatch(
            "expected $(npatches(provider)) patch targets, got $(length(dest))"
        )
    )
    assemble_patches!(PatchMatrixKind(WHOLE_PATCH_TERMS, PatchLocalSink(dest)), op, provider, states, p, ctx)
    return dest
end

assemble_patch_matrices!(dest::AbstractVector, op, provider::PatchItems, u::AbstractVector, p) =
    assemble_patch_matrices!(dest, op, provider, (u = u,), p, nothing)
