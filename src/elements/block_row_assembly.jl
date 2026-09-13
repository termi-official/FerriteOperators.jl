####################################
## The BLOCK-ROW assembly level
####################################

####################################
## The cell-with-neighbours item
####################################

"""
    CellNeighbourWindow

One item's dof window as a VIEW into the `(slot, k)` window table
[`CellNeighbourCursor`](@ref) carries — no staging and no per-item copy. `n` is
the prefix read: the whole `(1 + Nf)·Nb` gather window, or the leading `Nb` of
it that the item SCATTERS.

!!! warning "Experimental surface"
    Internal to the block-row action; it may change in a minor release.
"""
struct CellNeighbourWindow{I <: Integer, M <: AbstractMatrix{I}} <: AbstractVector{I}
    windows::M
    slot::Int
    n::Int
end
Base.size(w::CellNeighbourWindow) = (w.n,)
Base.IndexStyle(::Type{<:CellNeighbourWindow}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(w::CellNeighbourWindow, i::Int) = w.windows[w.slot, i]

"""
    CellNeighbourCursor

What a [`BlockRowAssembly`](@ref) action positions on: ONE cell of the
subdomain, gathering its own and its `Nf` facet neighbours' dofs and scattering
its own rows alone.

It wraps whatever [`default_assembly_iterator`](@ref) builds for the kind and
handler — a host `Ferrite.CellCache` or the device cell cursor — so ONE type
serves the host, `KernelAbstractions.CPU()` and CUDA, and the sequential CPU
device (which positions in place and discards the returned iterator) is served
by the inner cursor's own positioning.

`windows` is the `(slot, k)` window table, `slot` STRIDE-1 so a warp of
consecutive slots reads adjacent dofs, and `slots[cellid]` is the row of it this
item reads. A BOUNDARY facet's `Nb` window entries repeat the own cell's dofs:
the gather is fixed-width and every entry must address live memory, and the
block those entries pair with is zero and skipped.

[`iterator_scatter_address`](@ref) answers the own-cell PREFIX of that same row
— the two-window shape [`element_scatter_length`](@ref) names.

!!! warning "Experimental surface"
    Internal to the block-row action; it may change in a minor release.
"""
struct CellNeighbourCursor{IT, W <: AbstractMatrix, S <: AbstractVector}
    inner::IT
    windows::W
    slots::S
    nwindow::Int
    nrows::Int
end

Ferrite.cellid(c::CellNeighbourCursor) = Ferrite.cellid(c.inner)
iterator_handler(c::CellNeighbourCursor) = iterator_handler(c.inner)

@inline block_row_slot(c::CellNeighbourCursor) = Int(@inbounds c.slots[Ferrite.cellid(c)])
@inline iterator_dofs(c::CellNeighbourCursor) =
    CellNeighbourWindow(c.windows, block_row_slot(c), c.nwindow)
@inline iterator_scatter_address(c::CellNeighbourCursor) =
    CellNeighbourWindow(c.windows, block_row_slot(c), c.nrows)

@inline position_iterator(c::CellNeighbourCursor, item, flags::Ferrite.UpdateFlags) =
    CellNeighbourCursor(position_iterator(c.inner, item, flags), c.windows, c.slots, c.nwindow, c.nrows)

# `windows`/`slots` are read-only between workers; only the inner cursor stages.
duplicate_for_device(device::AbstractCPUDevice, c::CellNeighbourCursor) =
    CellNeighbourCursor(duplicate_for_device(device, c.inner), c.windows, c.slots, c.nwindow, c.nrows)
setup_device_instances(device::AbstractGPUDevice, c::CellNeighbourCursor, n_instances::Int) =
    CellNeighbourCursor(setup_device_instances(device, c.inner, n_instances),
                        adapt_shared(device, c.windows), adapt_shared(device, c.slots),
                        c.nwindow, c.nrows)
device_worker_view(c::CellNeighbourCursor, worker) =
    CellNeighbourCursor(device_worker_view(c.inner, worker), c.windows, c.slots, c.nwindow, c.nrows)
decorate_device_iterator(c::CellNeighbourCursor, sdh::SubDofHandler) =
    CellNeighbourCursor(decorate_device_iterator(c.inner, sdh), c.windows, c.slots, c.nwindow, c.nrows)

"""
    CellNeighbourItems(sdh)

The work-item provider of the [`BlockRowAssembly`](@ref) action: the cells of
one `SubDofHandler`, each carrying its facet neighbours. Items are CELL IDS, as
[`CellItems`](@ref)'s are — the neighbour table is the cache's
([`BlockRowAssemblyCache`](@ref)), addressed by the item's own slot.

**One colour, and why it is valid.** Every item scatters its OWN cell's `Nb`
rows and nothing else ([`iterator_scatter_address`](@ref) on
[`CellNeighbourCursor`](@ref)), so the item → owned-rows map is injective and
the whole item set satisfies [`compute_partition`](@ref)'s promise: no two items
of one chunk share a SCATTER dof. Three things that promise does NOT say:

1. It is about the scatter address, not the gather window. Items of the one
   chunk DO share gather dofs — that is what a neighbour window is — and the
   gather is read-only.
2. It therefore rests on [`element_scatter_length`](@ref). Without that
   declaration the matrix-free action addresses the scatter through the GATHER
   window, and this colouring would be invalid.
3. It does not depend on how the mesh is ordered or partitioned, and no
   colouring algorithm runs.

Elect [`ColoredScheduling`](@ref): the scatter is then a plain `+=` and the
action is bitwise repeatable. [`SequentialScheduling`](@ref) is also correct
(the atomic scatter resolves collisions that cannot occur) and buys nothing.

!!! warning "Experimental surface"
    Internal to the block-row action; it may change in a minor release.
"""
struct CellNeighbourItems{SDH <: SubDofHandler}
    sdh::SDH
end

compute_partition(::SequentialScheduling, provider::CellNeighbourItems) = (_cell_chunk(provider.sdh.cellset),)
compute_partition(::ColoredScheduling, provider::CellNeighbourItems) = (_cell_chunk(provider.sdh.cellset),)

####################################
## The cache
####################################

"""
    BlockRowAssemblyCache <: AbstractElementCacheDecorator

The cache a `storage = `[`BlockRowAssembly`](@ref)`()` operator runs: the
wrapped bilinear cache and the subdomain's condensed matrix ROWS, in blocks.

`K` is `(slot, 1 + Nf, Nb, Nb)` with `slot` LEADING and stride-1 — consecutive
workers are consecutive element slots at the same lane, the thread ordering the
`(slot, i, j)` layout of [`ElementAssemblyCache`](@ref) was measured for. Facet
`0` (index `1`) is the cell's DIAGONAL block; facet `f` is the block coupling
this cell's rows to the neighbour across its local facet `f`.

`neighbours[slot, f]` is that neighbour's cell id, `0` where the facet is a
boundary of the SUBDOMAIN — which is the whole of the bookkeeping the action
needs, the facet being the column index itself. `slots[cellid]` is the cell's
row, `0` outside this subdomain.

Two windows, not one ([`element_scatter_length`](@ref)): the action GATHERS
`(1 + Nf)·Nb` dofs (`local_size`) and SCATTERS `Nb` (`row_size`), which is what
makes [`CellNeighbourItems`](@ref)'s one-colour partition valid.

The FILL is two-sided and runs on the HOST — see [`BlockRowAssembly`](@ref).

!!! warning "Experimental surface"
    This decorator and the elections that build it may change in a minor
    release.
"""
struct BlockRowAssemblyCache{Inner, KT, NT, ST, WT, R, SDH, MI, ND, NB, NF} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
    K::KT
    neighbours::NT
    slots::ST
    windows::WT
    route::R
    sdh::SDH          # the HOST subdomain the fill walks; `nothing` on a device instance
    mass::MI          # the fused mass integrator, or `nothing`
    local_size::Val{ND}
    row_size::Val{NB}
    nfacets::Val{NF}
end

element_local_length(c::BlockRowAssemblyCache) = c.local_size
element_scatter_length(c::BlockRowAssemblyCache) = c.row_size

# The stored blocks ARE the element matrix; the workspace's two vectors carry the
# cache's two windows, so the fixed-width gather and the row scatter each hit a
# buffer of their own extent.
allocate_element_matrix(c::BlockRowAssemblyCache, sdh) = zeros(element_value_type(c), 0, 0)
allocate_element_unknown_vector(c::BlockRowAssemblyCache, sdh) = zeros(element_value_type(c), _extent(c.local_size))
allocate_element_residual_vector(c::BlockRowAssemblyCache, sdh) = zeros(element_value_type(c), _extent(c.row_size))

@inline _extent(::Val{N}) where {N} = N

# The wrapped cache's values objects are consumed by the FILL, which positions
# the element's OWN two-sided iterator; the action visits no quadrature point.
reinit_values!(::BlockRowAssemblyCache, cell, ::MatrixFreeActionKind) = nothing
item_update_flags(::MatrixFreeActionKind, ::BlockRowAssemblyCache) =
    Ferrite.UpdateFlags(nodes = false, coords = false, dofs = true)

assembly_iterator(kind, c::BlockRowAssemblyCache, sdh) =
    CellNeighbourCursor(default_assembly_iterator(kind, sdh), c.windows, c.slots,
                        _extent(c.local_size), _extent(c.row_size))
# The generic default body, stated here because the decorator's own forward would
# otherwise hand back the WRAPPED element's device iterator — its two-sided one,
# which is the FILL's and not the action's.
device_assembly_iterator(kind, c::BlockRowAssemblyCache, sdh, device_sdh) =
    decorate_device_iterator(assembly_iterator(kind, c, device_sdh), sdh)
item_provider(kind, ::BlockRowAssemblyCache, sdh) = CellNeighbourItems(sdh)

duplicate_for_device(device, c::BlockRowAssemblyCache) =
    BlockRowAssemblyCache(duplicate_for_device(device, c.inner), c.K, c.neighbours, c.slots, c.windows,
                          c.route, c.sdh, c.mass, c.local_size, c.row_size, c.nfacets)
# The fill runs on the HOST mirror and its result is copied here, so the device
# instance drops the two members only a fill reads — neither is `isbits`.
setup_device_instances(device::AbstractGPUDevice, c::BlockRowAssemblyCache, n) =
    BlockRowAssemblyCache(setup_device_instances(device, c.inner, n),
                          adapt_shared(device, c.K), adapt_shared(device, c.neighbours),
                          adapt_shared(device, c.slots), adapt_shared(device, c.windows),
                          c.route, nothing, nothing, c.local_size, c.row_size, c.nfacets)
device_worker_view(c::BlockRowAssemblyCache, worker) =
    BlockRowAssemblyCache(device_worker_view(c.inner, worker), c.K, c.neighbours, c.slots, c.windows,
                          c.route, c.sdh, c.mass, c.local_size, c.row_size, c.nfacets)

####################################
## Setup
####################################

function with_action_storage(cache, storage::BlockRowAssembly, sdh::SubDofHandler)
    nb = ndofs_per_cell(sdh)
    nl = length(allocate_element_residual_vector(cache, sdh))
    nl == 2nb || throw(ArgumentError(
        "$(typeof(cache)) declares an element-local system of $(nl) rows over a subdomain with " *
        "$(nb) dofs per cell, and `BlockRowAssembly()` condenses a TWO-SIDED local system: it " *
        "reads the $(2nb) × $(2nb) matrix of an item spanning two cells and writes its four " *
        "blocks into the two cells' rows. Elect `storage = ElementAssembly()` for a cell-square " *
        "element, or declare `allocate_element_residual_vector` as `2 * ndofs_per_cell`."))
    return BlockRowAssemblyCache(cache, sdh, element_matrix_fill_route(typeof(cache)),
                                 storage.premultiply_inverse_mass)
end

function BlockRowAssemblyCache(cache, sdh::SubDofHandler, route, mass)
    T    = element_value_type(cache)
    grid = get_grid(sdh.dh)
    nb   = ndofs_per_cell(sdh)
    nf   = Ferrite.nfacets(getcells(grid, first(sdh.cellset)))
    slots = zeros(Int32, getncells(grid))
    for (slot, cellid) in enumerate(sdh.cellset)
        slots[cellid] = slot
    end
    neighbours, windows = _cell_neighbour_tables(sdh, grid, slots, nb, nf)
    K = zeros(T, length(sdh.cellset), 1 + nf, nb, nb)
    return BlockRowAssemblyCache(cache, K, neighbours, slots, windows, route, sdh, mass,
                                 Val((1 + nf) * nb), Val(nb), Val(nf))
end

# The topology-derived neighbour table and the gather window it lays out, built
# once on the host. A facet whose neighbour is outside this subdomain is a
# boundary HERE: the store has no row for that cell, so the coupling is not this
# subdomain's to hold.
function _cell_neighbour_tables(sdh::SubDofHandler, grid, slots, nb::Int, nf::Int)
    top        = Ferrite.ExclusiveTopology(grid)
    nslots     = length(sdh.cellset)
    neighbours = zeros(Int32, nslots, nf)
    windows    = zeros(Int, nslots, (1 + nf) * nb)
    own        = Vector{Int}(undef, nb)
    other      = Vector{Int}(undef, nb)
    for (slot, cellid) in enumerate(sdh.cellset)
        celldofs!(own, sdh.dh, cellid)
        windows[slot, 1:nb] .= own
        for f in 1:nf
            found = Ferrite.getneighborhood(top, grid, FacetIndex(cellid, f))
            length(found) <= 1 || throw(ArgumentError(
                "Facet $f of cell $cellid has $(length(found)) neighbours. `BlockRowAssembly()` " *
                "keeps ONE block per facet, so it covers conforming meshes only."))
            c = (isempty(found) || slots[first(found)[1]] == 0) ? 0 : first(found)[1]
            # A boundary facet's window repeats the own cell's dofs: the gather is
            # fixed-width and reads it, the action skips the zero block it pairs with.
            celldofs!(other, sdh.dh, c == 0 ? cellid : c)
            neighbours[slot, f] = c
            windows[slot, (f * nb + 1):((f + 1) * nb)] .= other
        end
    end
    return neighbours, windows
end

####################################
## The action
####################################

"""
    apply_element_action!(yₑ, cache::BlockRowAssemblyCache, uₑ, args)

The ELEMENT level's action over ONE matrix row block: `yₑ += Σ_f K[slot, 1+f]·uₑ[window(f)]`,
`f = 0` the diagonal block and `f > 0` the neighbour across local facet `f`,
skipped where that facet has no neighbour in the subdomain. `yₑ` is the cell's
own `Nb` rows; `uₑ` is the whole `(1 + Nf)·Nb` gather window.
"""
@inline apply_element_action!(yₑ, cache::BlockRowAssemblyCache, uₑ, args::CellArgs) =
    _block_row_action!(yₑ, cache.K, cache.neighbours, block_row_slot(args.cell), uₑ,
                       cache.row_size, cache.nfacets)

@inline function _block_row_action!(yₑ, K, neighbours, slot, uₑ, ::Val{NB}, ::Val{NF}) where {NB, NF}
    for i in 1:NB
        @inbounds yₑ[i] += _block_row_dot(K, neighbours, slot, uₑ, i, Val(NB), Val(NF))
    end
    return nothing
end

"""
    element_action_row(cache::BlockRowAssemblyCache, uₑ, args, i)

The ELEMENT level's row: row `i` of this cell's block row against `uₑ`,
accumulated in ONE register and returned by value — what a
[`LanesPerElement`](@ref) lane owns. `i` runs over the SCATTER extent
([`element_scatter_length`](@ref)), not the gather window.
"""
@inline element_action_row(cache::BlockRowAssemblyCache, uₑ, args::CellArgs, i::Int) =
    _block_row_dot(cache.K, cache.neighbours, block_row_slot(args.cell), uₑ, i,
                   cache.row_size, cache.nfacets)

@inline function _block_row_dot(K, neighbours, slot, uₑ, i::Int, ::Val{NB}, ::Val{NF}) where {NB, NF}
    acc = zero(eltype(K))
    for j in 1:NB
        @inbounds acc += K[slot, 1, i, j] * uₑ[j]
    end
    for f in 1:NF
        (@inbounds neighbours[slot, f]) == 0 && continue
        offset = f * NB
        for j in 1:NB
            @inbounds acc += K[slot, 1 + f, i, j] * uₑ[offset + j]
        end
    end
    return acc
end

####################################
## The fill
####################################

"""
    fill_block_rows!(cache::BlockRowAssemblyCache, p, ctx)

Refill the subdomain's block rows from the wrapped element's own TWO-SIDED
traversal — [`assembly_iterator`](@ref)/[`item_provider`](@ref) as the element
declares them — condensing every item's `2Nb × 2Nb` system into the two cells it
spans: `LL → K[slotL, 1]`, `LR → K[slotL, 1+fL]`, `RL → K[slotR, 1+fR]`,
`RR → K[slotR, 1]`. Blocks ACCUMULATE, so a cell's diagonal block collects the
contribution of every item it takes part in.

It runs on the HOST and outside the engine's sweep: the action's items are cells
and the fill's are two-sided, and one operator resolves one traversal per sweep
([`BlockRowAssembly`](@ref) states the consequence).
"""
function fill_block_rows!(cache::BlockRowAssemblyCache, p, ctx)
    sdh   = cache.sdh
    inner = cache.inner
    T     = eltype(cache.K)
    nb    = _extent(cache.row_size)
    kind  = QuadratureDataKind()
    fill!(cache.K, zero(T))
    Kₑ    = zeros(T, 2nb, 2nb)
    dofs  = Vector{Int}(undef, nb)
    it    = assembly_iterator(kind, inner, sdh)
    flags = item_update_flags(kind, inner)
    for chunk in compute_partition(SequentialScheduling(), item_provider(kind, inner, sdh))
        for item in chunk
            cell = position_iterator(it, item, flags)
            reinit_values!(inner, cell, kind)
            args = CellArgs((;), cell, query_cell_parameters(inner, cell, p), ctx)
            fill_quadrature_data!(inner, args)
            fill!(Kₑ, zero(T))
            _fill_pair_matrix!(cache.route, Kₑ, inner, args)
            _condense_pair!(cache, Kₑ, cell, dofs)
        end
    end
    return nothing
end

_fill_pair_matrix!(::MatrixKernelFill, Kₑ, inner, args::CellArgs) =
    (assemble_cell!(JacobianRequest{:u}(Kₑ), inner, args); nothing)

function _fill_pair_matrix!(::ActionKernelFill, Kₑ, inner, args::CellArgs)
    T, n = eltype(Kₑ), size(Kₑ, 1)
    for j in 1:n
        apply_element_action!((@view Kₑ[:, j]), inner, UnitElementVector{T}(n, j), args)
    end
    return nothing
end

# Which two cells the item spans and how they see each other. `cellid` names the
# LEFT one by contract; the RIGHT one is identified by the cell whose dofs the
# window's second half starts with, searched over the left cell's own neighbours.
# Over a discontinuous space a dof belongs to exactly one cell, so this is an
# identification and not a guess — and `iterator_dofs` demands that space anyway.
function _condense_pair!(cache::BlockRowAssemblyCache, Kₑ, cell, dofs)
    nb = _extent(cache.row_size)
    nf = _extent(cache.nfacets)
    window = iterator_dofs(cell)
    length(window) == 2nb || throw(ArgumentError(
        "A `BlockRowAssembly()` fill item spans a dof window of $(length(window)) entries over a " *
        "subdomain with $(nb) dofs per cell; the condensation reads `[celldofs(L); celldofs(R)]`."))
    left = Ferrite.cellid(cell)
    sl = Int(@inbounds cache.slots[left])
    sl == 0 && throw(ArgumentError(
        "A `BlockRowAssembly()` fill item names cell $(left), which is outside the subdomain the " *
        "store was built for."))
    celldofs!(dofs, cache.sdh.dh, left)
    dofs[1] == window[1] || throw(ArgumentError(
        "A `BlockRowAssembly()` fill item's dof window does not start with `celldofs($(left))`, " *
        "the cell its iterator names through `Ferrite.cellid`. The condensation reads the window " *
        "as `[celldofs(L); celldofs(R)]`."))
    fl, right = 0, 0
    for f in 1:nf
        c = Int(@inbounds cache.neighbours[sl, f])
        c == 0 && continue
        celldofs!(dofs, cache.sdh.dh, c)
        if dofs[1] == window[nb + 1]
            fl, right = f, c
            break
        end
    end
    fl == 0 && throw(ArgumentError(
        "A `BlockRowAssembly()` fill item couples cell $(left) to a cell that is not one of its " *
        "facet neighbours inside this subdomain. The store keeps one block per facet, so a " *
        "coupling it has no column for cannot be condensed."))
    sr = Int(@inbounds cache.slots[right])
    fr = findfirst(f -> cache.neighbours[sr, f] == left, 1:nf)
    @views begin
        cache.K[sl, 1, :, :]      .+= Kₑ[1:nb, 1:nb]
        cache.K[sl, 1 + fl, :, :] .+= Kₑ[1:nb, (nb + 1):(2nb)]
        cache.K[sr, 1 + fr, :, :] .+= Kₑ[(nb + 1):(2nb), 1:nb]
        cache.K[sr, 1, :, :]      .+= Kₑ[(nb + 1):(2nb), (nb + 1):(2nb)]
    end
    return nothing
end

####################################
## The post-fill transform
####################################

"""
    finalize_action_storage!(cache, device)

Transform what a matrix-free `storage` election keeps, once the fill sweep that
wrote it has finished. A no-op for every shipped cache but
[`BlockRowAssemblyCache`](@ref), whose `premultiply_inverse_mass` election is
applied here.

It is a POST-SWEEP hook and not a per-item one (the `_pack_symmetric!` route):
the transform is per matrix ROW, and a row is complete only after every item
that contributes to it has run.

!!! warning "Experimental surface"
    Internal to the matrix-free action; it may change in a minor release.
"""
finalize_action_storage!(cache, device) = nothing

finalize_action_storage!(cache::BlockRowAssemblyCache, device) =
    _premultiply_inverse_mass!(cache, cache.mass)

_premultiply_inverse_mass!(::BlockRowAssemblyCache, ::Nothing) = nothing

# `M` is block diagonal by cell over a discontinuous space, so `M⁻¹K` has the
# SAME block-row sparsity: one cell's inverse mass block left-scales that cell's
# whole row. The inverse is formed once per cell, here, and never at action time.
function _premultiply_inverse_mass!(cache::BlockRowAssemblyCache, integrator)
    sdh  = cache.sdh
    nb   = _extent(cache.row_size)
    nf   = _extent(cache.nfacets)
    T    = eltype(cache.K)
    mass  = setup_element_cache(integrator, sdh)
    it    = assembly_iterator(nothing, mass, sdh)
    flags = item_update_flags(nothing, mass)
    Mₑ    = zeros(T, nb, nb)
    B     = zeros(T, nb, nb)
    C     = zeros(T, nb, nb)
    for (slot, cellid) in enumerate(sdh.cellset)
        cell = position_iterator(it, cellid, flags)
        reinit_values!(mass, cell)
        fill!(Mₑ, zero(T))
        assemble_cell!(JacobianRequest{:u}(Mₑ), mass,
                       CellArgs((;), cell, query_cell_parameters(mass, cell, nothing), nothing))
        Minv = inv(Mₑ)
        for f in 0:nf
            (f == 0 || cache.neighbours[slot, f] != 0) || continue
            # Through contiguous scratch: `K`'s block view is strided, not
            # column-contiguous, so neither operand of the product may be it.
            B .= @view cache.K[slot, 1 + f, :, :]
            mul!(C, Minv, B)
            (@view cache.K[slot, 1 + f, :, :]) .= C
        end
    end
    return nothing
end

####################################
## The host mirror
####################################

# A CPU device's workers SHARE the store the fill wrote; a GPU device's is the
# copy `adapt_shared` made at setup, and the refill is a host round trip.
_upload_action_storage!(::AbstractCPUDevice, host, device_cache) = nothing
_upload_action_storage!(::AbstractGPUDevice, host::BlockRowAssemblyCache, device_cache) =
    (copyto!(_block_row_cache(device_cache.element).K, host.K); nothing)

_block_row_cache(c::BlockRowAssemblyCache) = c
_block_row_cache(d::AbstractElementCacheDecorator) = _block_row_cache(d.inner)
