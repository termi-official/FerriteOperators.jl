####################################
## Patch items (experimental)
####################################
#
# Work items that are SETS of cells with a patch-local dof numbering — the
# local-BVP layer. Cell kernels are reused unchanged; the scatter target is
# patch-local. FO does not solve the local system: the caller owns the solve,
# the sinks and `ItemStates` slots carry its inputs and results.
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
`id` (see the `groups` argument of [`PatchItems`](@ref)). Tested inside the
single pass over the patch's cells, so a restricted term accumulates in the
same ascending cell order as an unrestricted one.
"""
struct CellGroup
    id::Int
end

"""
    patch_term_active(restriction, group::Int) -> Bool

Whether a term carrying `restriction` contributes on a patch cell tagged
`group`: always for [`WholePatch`](@ref), only on a matching tag for
[`CellGroup`](@ref). A downstream restriction type is a method here.
"""
@inline patch_term_active(::WholePatch, group::Int) = true
@inline patch_term_active(r::CellGroup, group::Int) = r.id == group

"""
    PatchTerm(restriction, data = nothing)

One additive contribution of a patch request. `restriction` is
[`WholePatch`](@ref) or [`CellGroup`](@ref); `data` is the element-specific
term payload handed to [`assemble_patch_cell!`](@ref) (`nothing` selects the
element's ordinary cell kernel).

A request carries a *tuple* of terms, all evaluated in one pass over the
patch's cells and accumulated into the same element buffer in tuple order —
that order is contract. Terms that must fuse per quadrature point rather than
per cell belong in ONE term whose `data` carries both sources.
"""
struct PatchTerm{R, D}
    restriction::R
    data::D
end
PatchTerm(restriction) = PatchTerm(restriction, nothing)

# Tuple recursions, unrolled by the compiler (terms are short, concrete
# tuples): each term dispatches monomorphically, no term value reaches a kernel
# as a `Union`.
"""
    any_patch_term_active(terms::Tuple, group::Int) -> Bool

Whether any term of a patch request contributes on a cell tagged `group` — the
gate that lets a patch sweep skip a cell no term touches.
"""
@inline any_patch_term_active(::Tuple{}, group::Int) = false
@inline any_patch_term_active(terms::Tuple, group::Int) =
    patch_term_active(first(terms).restriction, group) || any_patch_term_active(Base.tail(terms), group)

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
  of that patch), what a [`CellGroup`](@ref) restriction selects on. Defaults
  to the cell ids, so `CellGroup(cellid)` restricts to one cell; a multiscale
  patch typically tags by parent coarse cell.
- `prescribed_facets`: per patch, an iterable of `FacetIndex` on the patch's
  cells that the local BVP prescribes. FO does NOT classify the boundary —
  which facets are cut boundary is caller geometry — it only resolves the
  given classification into the dof partition ([`patch_free_dofs`](@ref) /
  [`patch_prescribed_dofs`](@ref)). Defaults to no prescribed dofs.
- `field`: the field the partition and [`patch_vertex_dofs`](@ref) resolve
  against. Defaults to the `sdh`'s only field.

Experimental: the patch item surface may change in a minor release.
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
free set — for dofs the local BVP needs pinned that no prescribed facet
touches (e.g. floating components of a perforated patch). Setup-time mutation:
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
[`patch_vertex_dofs`](@ref), what a caller needs to restrict its own
vertex-indexed data (e.g. a fine→coarse vertex map) to the patch.
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
vertex, for patch `i`. Computed per call, so concurrent calls are safe.
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

# Coloring needs the ITEM adjacency graph, which a patch provider does not carry
# (whether two patches share dofs is caller geometry). Say so instead of failing
# with a missing method.
compute_partition(::ColoredScheduling, provider::PatchItems) = throw(ArgumentError(
        "colored scheduling needs the adjacency of the items, which `PatchItems` does not carry. " *
        "Drive the patches yourself: `foreach_patch(...; items)` selects the subset, " *
        "`patch_chunks` splits it for workers."))

# The item list of a sweep, ascending and duplicate free. `items` is whatever
# the caller named; the ascending order is what makes a chunked sweep's merge
# reproduce the sequential one.
_patch_items(provider::PatchItems, items::AbstractUnitRange{<:Integer}) =
    (_check_patch_items(provider, items); items)
_patch_items(provider::PatchItems, items) =
    (its = sort!(collect(Int, items)); _check_patch_items(provider, its); its)

function _check_patch_items(provider::PatchItems, its)
    isempty(its) && return nothing
    n = npatches(provider)
    (first(its) >= 1 && last(its) <= n) || throw(ArgumentError(
            "`items` must be patch indices in 1:$n, got $(first(its)) … $(last(its))"))
    allunique(its) || throw(ArgumentError("`items` must not name a patch twice"))
    return nothing
end

"""
    patch_chunks(provider::PatchItems, nchunks::Int; items = 1:npatches(provider))

Split `items` (default: every patch) into at most `nchunks` CONTIGUOUS
ascending pieces of nearly equal length. Contiguity and order are contract:
merging per-chunk collectors in chunk order reproduces the item order of the
sequential sweep over the same `items` exactly. Empty chunks are dropped, so
fewer items than chunks yields fewer pieces.

A chunk is a slice of the item list, so the default yields `UnitRange{Int}`s of
patch indices and an explicit subset yields the ascending item ids themselves.
"""
function patch_chunks(provider::PatchItems, nchunks::Int; items = 1:npatches(provider))
    nchunks >= 1 || throw(ArgumentError("need at least one chunk, got $nchunks"))
    its = _patch_items(provider, items)
    n = length(its)
    chunks = typeof(its[1:0])[]
    lo = 1
    for c in 1:nchunks
        len = fld(n - lo + 1, nchunks - c + 1)
        len == 0 && continue
        push!(chunks, its[lo:(lo + len - 1)])
        lo += len
    end
    return chunks
end

####################################
## Sinks
####################################

"""
Supertype of the patch assembly sinks: where a patch's finished quantity goes
once the caller has assembled (and solved) it — a whole patch-local vector
through [`patch_emit!`](@ref), or one named column at a time through
[`emit_patch_column!`](@ref). A downstream scatter mode is a new sink type, not
a fork of the per-patch body.
"""
abstract type AbstractPatchSink end

"""
    PatchGlobalVectorSink(dest)

Additive global-vector sink: patch `i`'s patch-local vector is accumulated into
`dest` through the injection, `dest[patch_dofs(provider, i)[l]] += v[l]`.

Not thread safe: overlapping patches read-modify-write the same entries of
`dest` non-atomically. Parallel callers collect into one
[`PatchTripletSink`](@ref) per chunk instead and reduce afterwards.
"""
struct PatchGlobalVectorSink{V <: AbstractVector} <: AbstractPatchSink
    dest::V
end

"""
    PatchTripletSink{Tv}()
    PatchTripletSink()

Ordered COO sink for vector-valued patch results: every
[`emit_patch_column!`](@ref) appends `(rows[l], col, values[l])` in order, and
`sparse(sink, m, n)` SUMS duplicate `(row, col)` entries — what patch-additive
quantities (a corrector basis over overlapping patches) rely on. The column is
named per emission, so one patch may carry as many columns as its local
problem has right-hand sides.

Emission follows the sweep's item order, so a sequential sweep is reproducible
bit-for-bit; parallel callers collect one sink per chunk and `append!` them
back in chunk order to keep that.
"""
struct PatchTripletSink{Tv} <: AbstractPatchSink
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Tv}
end
PatchTripletSink{Tv}() where {Tv} = PatchTripletSink(Int[], Int[], Tv[])
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
    PatchCallbackKind(f)

Request kind of the per-patch callback route: `f(ws, patchid)` runs once per
patch with the workspace positioned on it, and everything a local BVP does per
patch happens inside `f` — FO only drives the sweep.
[`foreach_patch`](@ref) is the entry point.
"""
@concrete struct PatchCallbackKind
    f
end

# The element request dimension follows the target: a matrix (or an assembler
# over one) gets the Jacobian kernels, a vector the residual kernels.
patch_element_kind(::AbstractMatrix) = JacobianKind()
patch_element_kind(::Ferrite.AbstractAssembler) = JacobianKind()
patch_element_kind(::AbstractVector) = ResidualKind()
patch_element_kind(target) = throw(ArgumentError(
        "a patch target must be a patch-local matrix, a patch-local vector, or a Ferrite " *
        "assembler over a patch-local matrix; got $(typeof(target))"))

"""
    PatchArgs(states, cell, p, ctx, provider, patch, group, ldofs)

The argument bundle a patch term kernel's third parameter receives — the four
fields of [`CellArgs`](@ref) plus where in the patch the cell sits:

| field | meaning |
|---|---|
| `provider` | the [`PatchItems`](@ref) the sweep runs over |
| `patch` | the item index of the patch being assembled |
| `group` | the current cell's group tag, what a [`CellGroup`](@ref) restriction selects on |
| `ldofs` | the current cell's dofs in PATCH-LOCAL numbering, aligned with `celldofs(args.cell)` |

`ldofs` is the driver's scratch, refilled per cell and scattered through after
the kernels run: read it, never retain it past the call — the same lifetime
`args.cell` has. It is the window a term payload indexes patch-local data
through, so a kernel needs no per-patch handle of its own.

Closed, like `CellArgs` and `FacetArgs`: the patch item family's one args
record, not an extension channel.
"""
struct PatchArgs{States <: NamedTuple, Cell, P, Ctx, Prov}
    states::States
    cell::Cell
    p::P
    ctx::Ctx
    provider::Prov
    patch::Int
    group::Int
    ldofs::Vector{Int}
end

"""
    assemble_patch_cell!(req, cache, args::PatchArgs, data)

Element kernel for one [`PatchTerm`](@ref) on one cell: accumulate the term's
contribution into `req`'s buffer, exactly as `assemble_cell!` does. `data` is
the term payload; a term whose data is `nothing` bypasses this entry point and
runs the element's ordinary cell kernel over [`CellArgs`](@ref) instead, so
unrestricted patch assembly reuses existing elements unchanged.

`args` is a [`PatchArgs`](@ref) — `CellArgs`'s four fields plus the patch, the
cell's group tag and its patch-local dof window.

Term kernels are analytic — no AD fallback for a custom payload.
"""
function assemble_patch_cell! end

####################################
## Execution
####################################

"""
    PatchAssemblyWorkspace

Per-worker workspace of a patch sweep: a cell workspace plus the provider the
item — a patch index — is resolved through. `Ferrite.reinit!(ws, patchid)`
positions it on a patch, [`assemble_patch_target!`](@ref) runs the element
kernels on it.

Build one per worker with [`patch_workspace`](@ref) or copy one with
`duplicate_for_device`; independent workspaces share only the provider, which
patch sweeps read and never write.

`current` is a `Ref{Int}` because the item is resolved through the provider
between `reinit!` and the kernels rather than handed to them. That makes it a
CPU-scoped positioning mechanism — one reason [`foreach_patch`](@ref) is
sequential CPU only.
"""
@concrete struct PatchAssemblyWorkspace <: AbstractWorkspace
    provider
    current   # Ref{Int}: the patch index of the item being processed
    inner     # the per-worker cell AssemblyWorkspace
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
        copy(ws.ldofs),
    )
end

"""
    patch_workspace(op, provider::PatchItems)

An INDEPENDENT [`PatchAssemblyWorkspace`](@ref) over `provider`, built from the
operator's element caches for the provider's `SubDofHandler`. Every call
duplicates those caches, so `n` calls give `n` workers sharing only the
provider — the constructor a caller scheduling patches itself needs.
"""
function patch_workspace(op, provider::PatchItems)
    sc = op.engine.subdomain_caches[_patch_subdomain(op, provider)]
    inner = duplicate_for_device(op.engine.strategy.device, first(sc.device_cache))
    return _patch_workspace(provider, inner)
end

_patch_workspace(provider, inner) = PatchAssemblyWorkspace(provider, Ref(0), inner, Int[])

function _patch_subdomain(op, provider::PatchItems)
    i = findfirst(sc -> sc.domain isa AssemblyDomain && sc.domain.sdh === provider.sdh,
                  op.engine.subdomain_caches)
    i === nothing && throw(ArgumentError("the provider's SubDofHandler is not part of the operator"))
    # Rejected where a patch sweep binds to the subdomain rather than in the
    # scatter, where the drop would leave no trace.
    first(op.engine.subdomain_caches[i].device_cache).dofs === nothing || throw(ArgumentError(
        "Patch assembly does not support elements declaring `global_dofs`: a patch's dof map is " *
        "built from `celldofs`, so the declared tail of each element's local system has no " *
        "patch-local number and would be silently dropped. Build the patch operator on an " *
        "integrator without a `global_dofs` declaration."))
    return i
end

execute_single_task!(task::AssemblyTask, ws::PatchAssemblyWorkspace) = execute_kind!(task.kind, task, ws)

# --- sink protocol ---------------------------------------------------------

"""
    patch_emit!(sink, provider, pid, v)

Publish patch `pid`'s finished patch-local vector `v` into `sink`, mapped
through the injection. [`emit_patch_column!`](@ref) is the column-naming form a
many-columns-per-patch result emits through instead.
"""
function patch_emit! end

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

function patch_emit!(sink::PatchGlobalVectorSink, provider, pid, v)
    for (l, g) in pairs(patch_dofs(provider, pid))
        sink.dest[g] += v[l]
    end
    return nothing
end

# --- the shared per-patch assembly body ------------------------------------

"""
    assemble_patch_target!(target, terms, ws::PatchAssemblyWorkspace, states, p, ctx = nothing)

ACCUMULATE the tuple of [`PatchTerm`](@ref)s `terms` over the patch `ws` is
positioned on into `target` — a patch-local matrix (the Jacobian kernels run),
a patch-local vector (the residual kernels run), or a Ferrite assembler over a
patch-local matrix. `target` is NOT zeroed here, so repeated calls accumulate.

One pass over the patch's cells in ascending order, the active terms of each
cell accumulating into the element buffer before a single scatter. Callable
directly: a local BVP assembles its matrix with one call and its `N` right-hand
sides with `N` calls carrying per-column term data, on one workspace.

Term payloads reach [`assemble_patch_cell!`](@ref) over a [`PatchArgs`](@ref)
carrying the cell's group tag and patch-local dof window.

Position `ws` with `Ferrite.reinit!(ws, patchid)` first;
[`foreach_patch`](@ref) has already done that. Slot names are validated by the
entry points once per sweep, not here.
"""
function assemble_patch_target!(target, terms::Tuple, ws::PatchAssemblyWorkspace, states::NamedTuple, p, ctx = nothing)
    ekind = patch_element_kind(target)
    pid = ws.current[]
    provider = ws.provider
    dofmap = provider.dofmaps[pid]
    iws = ws.inner
    ldofs = ws.ldofs
    for (k, cellid) in pairs(provider.patches[pid])
        group = provider.groups[pid][k]
        any_patch_term_active(terms, group) || continue
        reinit!(iws, cellid)
        cdofs = celldofs(iws.cell)
        length(ldofs) == length(cdofs) || resize!(ldofs, length(cdofs))
        for (i, g) in pairs(cdofs)
            ldofs[i] = dofmap[g]
        end
        _zero_element_buffer!(ekind, iws)
        reinit_values!(iws.element, iws.cell, ekind)
        pₑ = query_cell_parameters(iws.element, iws.cell, p)
        statesₑ = load_slots!(iws, states)
        args = PatchArgs(statesₑ, iws.cell, pₑ, ctx, provider, pid, group, ldofs)
        run_patch_terms!(ekind, terms, iws, args)
        scatter_patch_cell!(target, ldofs, _element_buffer(ekind, iws))
    end
    return target
end

_element_buffer(::JacobianKind, iws) = iws.Ke
_element_buffer(::ResidualKind, iws) = iws.re
_zero_element_buffer!(ekind, iws) = fill!(_element_buffer(ekind, iws), 0.0)

@inline run_patch_terms!(ekind, ::Tuple{}, iws, args::PatchArgs) = nothing
@inline function run_patch_terms!(ekind, terms::Tuple, iws, args::PatchArgs)
    term = first(terms)
    patch_term_active(term.restriction, args.group) && run_patch_term!(ekind, term.data, iws, args)
    return run_patch_terms!(ekind, Base.tail(terms), iws, args)
end

# Both payload routes reach the element through the same kind → request
# association as the cell driver. A `nothing` payload is the element's ordinary
# cell kernel, so it gets the ordinary `CellArgs`.
@inline run_patch_term!(ekind, ::Nothing, iws, args::PatchArgs) =
    cell_kernel!(ekind, iws.element, iws, args.states, args.p, args.ctx)
@inline run_patch_term!(ekind, data, iws, args::PatchArgs) =
    assemble_patch_cell!(materialize_request(ekind, iws), iws.element, args, data)

# --- the driver ------------------------------------------------------------

execute_kind!(kind::PatchCallbackKind, task, ws::PatchAssemblyWorkspace) =
    (kind.f(ws, ws.current[]); nothing)

"""
    foreach_patch(f, op, provider::PatchItems, states, p, ctx = nothing; items = 1:npatches(provider))

Call `f(ws, patchid)` once per patch named by `items` (default: every patch of
`provider`), in ascending item order, with `ws` a
[`PatchAssemblyWorkspace`](@ref) already positioned on `patchid`. Inside `f` the
caller owns the patch: assemble targets with [`assemble_patch_target!`](@ref)
(passing `states`, `p` and `ctx` on), factorize, solve, retain the
factorization in an [`ItemStates`](@ref) slot, and emit through a sink —
[`emit_patch_column!`](@ref) writes one column per call, so an `N`-column local
basis is `N` emissions.

`items` is what a partial re-solve names: only those patches are visited, and
what the untouched ones contributed to a previous sweep is untouched too, since
FO writes nothing outside `f`.

Sequential: FO cannot duplicate what `f` writes into, since the collectors and
the retained state are the caller's — scheduling is the caller's too, through
[`patch_workspace`](@ref), [`patch_chunks`](@ref) and `duplicate_for_device`
(see the parallel section of the patch documentation).

Experimental: part of the patch item family.
"""
function foreach_patch(f, op, provider::PatchItems, states::NamedTuple, p, ctx = nothing;
                       items = 1:npatches(provider))
    _check_declared_slots(op.engine, states)
    op.engine.strategy.device isa SequentialCPUDevice || throw(
        ArgumentError(
            "`foreach_patch` supports SequentialCPUDevice only (got $(typeof(op.engine.strategy.device))): " *
            "the callback's collectors are the caller's, so FO cannot duplicate them per worker. " *
            "Schedule the patches yourself with `patch_chunks` and `patch_workspace`."
        )
    )
    its = _patch_items(provider, items)
    ws = patch_workspace(op, provider)
    task = AssemblyTask(PatchCallbackKind(f), nothing, states, p, ctx)
    execute_on_device!(task, op.engine.strategy.device, (ws,), (its,))
    return nothing
end
