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

"Number of patches (items) of the provider."
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

# Coloring is race freedom by construction over the ITEM adjacency graph, which
# a patch provider does not carry (whether two patches share dofs is caller
# geometry). Say so instead of failing with a missing method.
compute_partition(::ColoredScheduling, provider::PatchItems) = throw(ArgumentError(
        "colored scheduling needs the adjacency of the items, which `PatchItems` does not carry. " *
        "Use `SequentialScheduling()` and chunk the patches yourself (`patch_chunks`)."))

"""
    patch_chunks(provider::PatchItems, nchunks::Int) -> Vector{UnitRange{Int}}

Split the items of `provider` into at most `nchunks` CONTIGUOUS ascending
ranges of nearly equal length. Contiguity and order are the contract: a caller
that processes the chunks in parallel and merges its per-chunk collectors in
chunk order reproduces the item order of a sequential sweep exactly.

Empty chunks are dropped, so fewer patches than chunks yields fewer ranges.
"""
function patch_chunks(provider::PatchItems, nchunks::Int)
    nchunks >= 1 || throw(ArgumentError("need at least one chunk, got $nchunks"))
    n = npatches(provider)
    chunks = UnitRange{Int}[]
    lo = 1
    for c in 1:nchunks
        len = fld(n - lo + 1, nchunks - c + 1)
        len == 0 && continue
        push!(chunks, lo:(lo + len - 1))
        lo += len
    end
    return chunks
end

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

Not thread safe: overlapping patches hit the same entries of `dest` with a
non-atomic read-modify-write. Parallel callers collect into a
[`PatchTripletSink`](@ref) per chunk instead and reduce afterwards.
"""
struct PatchGlobalVectorSink{V <: AbstractVector} <: AbstractPatchSink
    dest::V
end

"""
    PatchTripletSink{Tv}(columns)
    PatchTripletSink(columns)
    PatchTripletSink{Tv}()

Ordered COO sink for vector-valued patch results: patch `i` emits the triplets
`(patch_dofs(provider, i)[l], columns[i], v[l])` in ascending `l`. Duplicate
`(row, col)` entries are SUMMED by `sparse(sink, m, n)` — patch-additive
quantities (a corrector basis assembled from overlapping patches) rely on that.

`columns` is the item → column map the sweep tail of [`assemble_patches!`](@ref)
needs, one entry per patch. The column-less form is for callers emitting through
[`emit_patch_column!`](@ref) themselves, which names the column per call and is
therefore what a many-columns-per-patch result uses.

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
PatchTripletSink{Tv}() where {Tv} = PatchTripletSink{Tv}(Int[])
PatchTripletSink() = PatchTripletSink{Float64}()

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

"""
    PatchCallbackKind(f)

Request kind of the per-patch callback route: `f(ws, patchid)` runs once per
patch with the patch workspace `ws` positioned on it. Everything a local BVP
does per patch — assembling targets with [`assemble_patch_target!`](@ref),
factorizing, solving, emitting — happens inside `f`; FO only drives the sweep.

[`foreach_patch`](@ref) is the entry point.
"""
@concrete struct PatchCallbackKind
    f
end

# The element request dimension: matrix-valued requests reach the element as
# `JacobianRequest{:u}`, vector-valued ones as `ResidualRequest`. A patch
# request states its dimension by type; a bare patch target states it by being
# a matrix (or an assembler over one) rather than a vector.
patch_element_kind(::PatchMatrixKind) = JacobianKind()
patch_element_kind(::PatchVectorKind) = ResidualKind()
patch_element_kind(::AbstractMatrix) = JacobianKind()
patch_element_kind(::Ferrite.AbstractAssembler) = JacobianKind()
patch_element_kind(::AbstractVector) = ResidualKind()
patch_element_kind(target) = throw(ArgumentError(
        "a patch target must be a patch-local matrix, a patch-local vector, or a Ferrite " *
        "assembler over a patch-local matrix; got $(typeof(target))"))

"""
    assemble_patch_cell!(req, cache, args, data)

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

"""
    PatchAssemblyWorkspace

Per-worker workspace of a patch sweep: it wraps a cell workspace and resolves
the item — a patch index — through the provider. `Ferrite.reinit!(ws, patchid)`
positions it on a patch, [`current_patch`](@ref) reports where it stands, and
[`assemble_patch_target!`](@ref) runs the element kernels on it.

Build one per worker with [`patch_workspace`](@ref), or copy an existing one
with `duplicate_for_device`. Independent workspaces share only the provider,
which patch sweeps read and never write.

`current` is a `Ref{Int}` because a patch item is resolved through the
provider rather than carried into the kernel, so the driver has to park the
index somewhere between `reinit!` and the kernels. That makes it a CPU-scoped
positioning mechanism: patch sweeps are sequential CPU only
([`assemble_patches!`](@ref), [`foreach_patch`](@ref)), and a device without
host-side references needs a different way to say which patch a worker is on.
"""
@concrete struct PatchAssemblyWorkspace <: AbstractWorkspace
    provider
    current   # Ref{Int}: the patch index of the item being processed
    inner     # the per-worker cell AssemblyWorkspace
    buffer    # patch-sized scratch for sinks that do not own patch-local storage
    ldofs     # the current cell's dofs in patch-local numbering
end

Ferrite.reinit!(ws::PatchAssemblyWorkspace, patchid::Int) = (ws.current[] = patchid; nothing)

"The patch index `ws` is positioned on, set by `Ferrite.reinit!(ws, patchid)`."
current_patch(ws::PatchAssemblyWorkspace) = ws.current[]

"The [`PatchItems`](@ref) provider `ws` resolves its patch index against."
patch_provider(ws::PatchAssemblyWorkspace) = ws.provider

function duplicate_for_device(device::AbstractCPUDevice, ws::PatchAssemblyWorkspace)
    return PatchAssemblyWorkspace(
        ws.provider,                                    # shared: read-only during a sweep
        Ref(ws.current[]),
        duplicate_for_device(device, ws.inner),
        copy(ws.buffer),
        copy(ws.ldofs),
    )
end

"""
    patch_workspace(op, provider::PatchItems)

An INDEPENDENT [`PatchAssemblyWorkspace`](@ref) over `provider`, built from the
operator's element caches for the provider's `SubDofHandler`. Every call
duplicates those caches, so `n` calls give one workspace per worker; the only
shared object is the provider.

The patch entry points build their own workspace — this is the constructor a
caller scheduling patches itself needs.
"""
function patch_workspace(op, provider::PatchItems)
    sc = op.engine.subdomain_caches[_patch_subdomain(op, provider)]
    inner = duplicate_for_device(op.engine.strategy.device, first(sc.device_cache))
    return _patch_workspace(provider, inner)
end

_patch_workspace(provider, inner) = PatchAssemblyWorkspace(provider, Ref(0), inner, Float64[], Int[])

function _patch_subdomain(op, provider::PatchItems)
    i = findfirst(sc -> sc.domain.sdh === provider.sdh, op.engine.subdomain_caches)
    i === nothing && throw(ArgumentError("the provider's SubDofHandler is not part of the operator"))
    return i
end

execute_single_task!(task::AssemblyTask, ws::PatchAssemblyWorkspace) = execute_kind!(task.kind, task, ws)

# --- sink protocol ---------------------------------------------------------

"""
    patch_target(sink, ws, pid) -> target

The ZEROED patch-local accumulator patch `pid` is assembled into: the sink's
own per-patch storage, or the patch-sized scratch of `ws` for sinks that hold
none. First of the three methods a patch sink implements, with
[`patch_scatter`](@ref) and [`patch_emit!`](@ref).
"""
function patch_target end

"""
    patch_scatter(sink, target) -> scatter

The object cell contributions are scattered through — `target` itself, unless
the sink assembles through a Ferrite assembler over it. Second of the three
methods a patch sink implements.
"""
function patch_scatter end

"""
    patch_emit!(sink, provider, pid, target)

Publish patch `pid`'s finished quantity: a no-op for sinks whose `target`
already is the destination, an injection into a global vector or a column of
triplets otherwise. Third of the three methods a patch sink implements.
"""
function patch_emit! end

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

# The request dimension also has to reach the sink's per-item dest: a
# `PatchMatrixKind` over a vector-valued dest would scatter element matrices
# through the vector scatter and produce garbage instead of failing. One check
# per sweep, on the first dest.
_check_sink_dimension(kind, sink) = nothing
function _check_sink_dimension(kind::PatchAssemblyKind, sink::Union{PatchLocalSink, PatchAssemblerSink})
    isempty(sink.dest) && return nothing
    wants_matrix = kind isa PatchMatrixKind
    (first(sink.dest) isa AbstractMatrix) == wants_matrix || throw(ArgumentError(
            "$(typeof(kind).name.name) assembles a patch-local $(wants_matrix ? "matrix" : "vector"), " *
            "but the sink's dest holds $(typeof(first(sink.dest))). Match the dest to the kind's dimension."))
    return nothing
end

# A sink indexed by item must span the provider. Fail at the sweep, not with a
# BoundsError on whichever patch first runs past the end.
_check_sink_extent(::AbstractPatchSink, provider) = nothing
function _check_sink_extent(sink::PatchTripletSink, provider)
    length(sink.columns) == npatches(provider) || throw(ArgumentError(
            "the sink carries $(length(sink.columns)) column indices but the provider has " *
            "$(npatches(provider)) patches; `PatchTripletSink` maps item → column."))
    return nothing
end

# --- the shared per-patch assembly body ------------------------------------

"""
    assemble_patch_target!(target, terms, ws::PatchAssemblyWorkspace, states, p, ctx = nothing)

ACCUMULATE the tuple of [`PatchTerm`](@ref)s `terms` over the patch `ws` is
positioned on into `target`, which is a patch-local matrix (the element
Jacobian kernels run), a patch-local vector (the residual kernels run), or a
Ferrite assembler over a patch-local matrix. `target` is NOT zeroed here — the
caller owns its initial state, and repeated calls accumulate.

One pass over the patch's cells in ascending order; per cell the active terms
accumulate into the element buffer in tuple order before a single scatter. This
is the body the kind × sink pipelines run per patch, callable directly: a local
BVP assembles its matrix with one call and its `N` right-hand sides with `N`
calls carrying per-column term data, all on the same workspace.

Position `ws` with `Ferrite.reinit!(ws, patchid)` first —
[`foreach_patch`](@ref) has already done that for the workspace it hands out.
Slot names are not validated here; the entry points do that once per sweep.
"""
assemble_patch_target!(target, terms, ws::PatchAssemblyWorkspace, states::NamedTuple, p, ctx = nothing) =
    assemble_patch_target!(patch_element_kind(target), target, terms, ws, states, p, ctx)

# The pipeline route is the second caller: it takes the element kind from the
# request rather than from the target, so the request's dimension stays
# authoritative there.
function assemble_patch_target!(ekind, target, terms::Tuple, ws::PatchAssemblyWorkspace, states::NamedTuple, p, ctx)
    pid = ws.current[]
    provider = ws.provider
    dofmap = provider.dofmaps[pid]
    iws = ws.inner
    ldofs = ws.ldofs
    for (k, cellid) in pairs(provider.patches[pid])
        group = provider.groups[pid][k]
        any_patch_term_active(terms, group) || continue
        reinit!(iws.cell, cellid)
        _zero_element_buffer!(ekind, iws)
        reinit_values!(iws.element, iws.cell, ekind)
        pₑ = query_cell_parameters(iws.element, iws.cell, p)
        statesₑ = load_slots!(iws, states)
        run_patch_terms!(ekind, terms, group, iws, statesₑ, pₑ, ctx)
        cdofs = celldofs(iws.cell)
        length(ldofs) == length(cdofs) || resize!(ldofs, length(cdofs))
        for (i, g) in pairs(cdofs)
            ldofs[i] = dofmap[g]
        end
        scatter_patch_cell!(target, ldofs, _element_buffer(ekind, iws))
    end
    return target
end

_element_buffer(::JacobianKind, iws) = iws.Ke
_element_buffer(::ResidualKind, iws) = iws.re
_zero_element_buffer!(ekind, iws) = fill!(_element_buffer(ekind, iws), 0.0)

@inline run_patch_terms!(ekind, ::Tuple{}, group::Int, iws, statesₑ, pₑ, ctx) = nothing
@inline function run_patch_terms!(ekind, terms::Tuple, group::Int, iws, statesₑ, pₑ, ctx)
    term = first(terms)
    patch_term_active(term.restriction, group) &&
        run_patch_term!(ekind, term.data, iws, statesₑ, pₑ, ctx)
    return run_patch_terms!(ekind, Base.tail(terms), group, iws, statesₑ, pₑ, ctx)
end

# The `nothing` payload is the element's ordinary cell kernel (AD fallback
# included); a real payload goes to the analytic term kernel. Both reach the
# element through the same kind → request association as the cell driver.
@inline run_patch_term!(ekind, ::Nothing, iws, statesₑ, pₑ, ctx) =
    v2_cell_kernel!(ekind, iws.element, iws, statesₑ, pₑ, ctx)
@inline run_patch_term!(ekind, data, iws, statesₑ, pₑ, ctx) =
    assemble_patch_cell!(materialize_request(ekind, iws), iws.element,
                         _v2_args(iws, statesₑ, pₑ, ctx), data)

# --- the drivers -----------------------------------------------------------

function execute_kind!(kind::PatchAssemblyKind, task, ws::PatchAssemblyWorkspace)
    pid = ws.current[]
    target = patch_target(kind.sink, ws, pid)
    assemble_patch_target!(patch_element_kind(kind), patch_scatter(kind.sink, target),
                           kind.terms, ws, task.states, task.p, task.ctx)
    patch_emit!(kind.sink, ws.provider, pid, target)
    return nothing
end

execute_kind!(kind::PatchCallbackKind, task, ws::PatchAssemblyWorkspace) =
    (kind.f(ws, ws.current[]); nothing)

"""
    assemble_patches!(kind, op, provider::PatchItems, states, p, ctx = nothing)

Run one patch sweep of `kind` ([`PatchMatrixKind`](@ref) or
[`PatchVectorKind`](@ref)) over every item of `provider`, reusing the
operator's element caches. Each patch is visited once, its cells in ascending
order; the kind's sink receives the patch-local result.

Sequential CPU only, and structurally so: the sink rides inside the shared
`kind`, so every worker of a parallel device would scatter through the same
sink object. [`foreach_patch`](@ref) is the route whose collectors the caller
owns and can therefore schedule — see the parallel section of the patch
documentation.

Patch sweeps are pure evaluation: condensed element unknowns are gathered but
never written back, unlike the global sweeps.

Experimental: part of the patch item family; the local BVP itself (partition,
solve, item state) is the caller's — see [`patch_free_dofs`](@ref) and
[`PatchItemStates`](@ref).
"""
function assemble_patches!(kind::PatchAssemblyKind, op, provider::PatchItems, states::NamedTuple, p, ctx = nothing)
    _check_declared_slots(op.engine, states)
    _check_sink(kind, kind.sink)
    _check_sink_dimension(kind, kind.sink)
    _check_sink_extent(kind.sink, provider)
    op.engine.strategy.device isa SequentialCPUDevice || throw(
        ArgumentError(
            "`assemble_patches!` supports SequentialCPUDevice only (got $(typeof(op.engine.strategy.device))): " *
            "the sink rides inside the shared kind. Drive the patches through `foreach_patch` and own the collectors."
        )
    )
    ws = _patch_workspace(provider, first(op.engine.subdomain_caches[_patch_subdomain(op, provider)].device_cache))
    task = AssemblyTask(kind, nothing, states, p, ctx)
    execute_on_device!(task, op.engine.strategy.device, (ws,), compute_partition(op.engine.strategy, provider))
    return kind.sink
end

"""
    foreach_patch(f, op, provider::PatchItems, states, p, ctx = nothing)

Call `f(ws, patchid)` once per patch of `provider`, in item order, with `ws` a
[`PatchAssemblyWorkspace`](@ref) already positioned on `patchid`. Inside `f` the
caller owns the patch: assemble any number of targets with
[`assemble_patch_target!`](@ref) (passing `states`, `p` and `ctx` on),
factorize, solve, retain the factorization in a [`PatchItemStates`](@ref) slot,
and emit results through a sink — [`emit_patch_column!`](@ref) writes one column
per call, so an `N`-column local basis is `N` emissions.

`states` is validated against the operator's declared slots once per sweep
rather than per patch.

Sequential: FO cannot duplicate what `f` writes into, because the collectors
and the retained state are the caller's. Scheduling patches is therefore the
caller's too, and [`patch_workspace`](@ref), [`patch_chunks`](@ref) and
`duplicate_for_device` are the seams for it — see the parallel section of the
patch documentation.

Experimental: part of the patch item family.
"""
function foreach_patch(f, op, provider::PatchItems, states::NamedTuple, p, ctx = nothing)
    _check_declared_slots(op.engine, states)
    op.engine.strategy.device isa SequentialCPUDevice || throw(
        ArgumentError(
            "`foreach_patch` supports SequentialCPUDevice only (got $(typeof(op.engine.strategy.device))): " *
            "the callback's collectors are the caller's, so FO cannot duplicate them per worker. " *
            "Schedule the patches yourself with `patch_chunks` and `patch_workspace`."
        )
    )
    ws = _patch_workspace(provider, first(op.engine.subdomain_caches[_patch_subdomain(op, provider)].device_cache))
    task = AssemblyTask(PatchCallbackKind(f), nothing, states, p, ctx)
    execute_on_device!(task, op.engine.strategy.device, (ws,), compute_partition(op.engine.strategy, provider))
    return nothing
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
