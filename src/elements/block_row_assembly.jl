####################################
## The BLOCK-ROW assembly level
####################################

####################################
## The cell-with-neighbours item
####################################

"""
    CellNeighbourWindow

One item's dof window as a VIEW into the materialized `(slot, k)` window table a
[`CellNeighbourCursor`](@ref) carries where the subdomain's dofs are NOT
cell-contiguous — no staging and no per-item copy. `n` is the prefix read: the
whole `(1 + Nf)·Nb` gather window, or the leading `Nb` of it that the item
SCATTERS.

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
    ContiguousCellDofs(neighbours, Val(Nb))

The window source a [`CellNeighbourCursor`](@ref) carries where the subdomain's
dofs are CELL-CONTIGUOUS with uniform stride `Nb` — `celldofs(c) == (c-1)Nb+1 : c·Nb`
for every cell of the subdomain, which `DiscontinuousLagrange` over a single
`SubDofHandler` satisfies. Window entry `(f, j)` is then `(cell(f) - 1)·Nb + j`:
ARITHMETIC on the per-slot neighbour list, and no `(slot, k)` table is built,
uploaded or read at all.

`neighbours` is [`BlockRowAssemblyCache`](@ref)'s own table — `Int32` CELL ids,
bounded at setup — and `stride` carries `Nb` as a compile-time constant, so the
`(f, j)` recovery is a `divrem` by a literal. The derived dof index is `Int`, the
cell id widening before it multiplies: unlike the table's entries, it is computed
and never stored, so its width is register traffic and not DRAM traffic.
(Measured: narrowing this arithmetic to `Int32` moves the RTX 2080 DG benchmark
by less than 1% at either order.)

The election is `_dof_window_source`'s, made ONCE against the HOST
`SubDofHandler`. It is strictly stronger than the affine-offset test the device
cell cursor's dof stride rests on: a renumbered handler keeps affine
`cell_dofs_offset` while its `cell_dofs` are no longer the offsets themselves.

!!! warning "Experimental surface"
    Internal to the block-row action; it may change in a minor release.
"""
struct ContiguousCellDofs{NT <: AbstractMatrix, NB}
    neighbours::NT
    stride::Val{NB}
end

"""
    ContiguousDofWindow

One item's dof window DERIVED rather than read: entry `(f, j)` of the window is
`(cell(f) - 1)·Nb + j`, with `cell(0)` the item's own cell and `cell(f)` the
neighbour across its local facet `f`. The counterpart of
[`CellNeighbourWindow`](@ref) for a [`ContiguousCellDofs`](@ref) source.

A BOUNDARY facet (`neighbours[slot, f] == 0`) derives the own cell's dofs: the
gather is fixed-width and every entry must address live memory, and the block
those entries pair with is zero and skipped.

!!! warning "Experimental surface"
    Internal to the block-row action; it may change in a minor release.
"""
struct ContiguousDofWindow{NT <: AbstractMatrix, NB} <: AbstractVector{Int}
    neighbours::NT
    slot::Int
    cellid::Int
    n::Int
    stride::Val{NB}
end
Base.size(w::ContiguousDofWindow) = (w.n,)
Base.IndexStyle(::Type{<:ContiguousDofWindow}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(w::ContiguousDofWindow{<:Any, NB}, k::Int) where {NB}
    f, j = divrem(k - 1, NB)
    c = f == 0 ? w.cellid : Int(w.neighbours[w.slot, f])
    return (ifelse(c == 0, w.cellid, c) - 1) * NB + j + 1
end

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

`windows` is the item's dof-window SOURCE, elected once at setup: a
[`ContiguousCellDofs`](@ref) descriptor deriving every entry arithmetically, or
the materialized `(slot, k)` table with `slot` STRIDE-1 so a warp of consecutive
slots reads adjacent dofs. `slots[cellid]` is the row either one is addressed by.

[`iterator_scatter_address`](@ref) answers the own-cell PREFIX of that same row
— the two-window shape [`element_scatter_length`](@ref) names.

!!! warning "Experimental surface"
    Internal to the block-row action; it may change in a minor release.
"""
struct CellNeighbourCursor{IT, W, S <: AbstractVector}
    inner::IT
    windows::W
    slots::S
    nwindow::Int
    nrows::Int
end

Ferrite.cellid(c::CellNeighbourCursor) = Ferrite.cellid(c.inner)
iterator_handler(c::CellNeighbourCursor) = iterator_handler(c.inner)

@inline block_row_slot(c::CellNeighbourCursor) = Int(@inbounds c.slots[Ferrite.cellid(c)])
@inline iterator_dofs(c::CellNeighbourCursor) = _dof_window(c.windows, c, c.nwindow)
@inline iterator_scatter_address(c::CellNeighbourCursor) = _dof_window(c.windows, c, c.nrows)

# One method each, so neither window source pays for the other's branch.
@inline _dof_window(windows::AbstractMatrix, c::CellNeighbourCursor, n::Int) =
    CellNeighbourWindow(windows, block_row_slot(c), n)
@inline _dof_window(windows::ContiguousCellDofs, c::CellNeighbourCursor, n::Int) =
    ContiguousDofWindow(windows.neighbours, block_row_slot(c), Ferrite.cellid(c), n, windows.stride)

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
row, `0` outside this subdomain. Both are `Int32`, the cell count bounded at
setup.

`windows` is the item's dof-window source, elected against this subdomain's dof
layout by `_dof_window_source`: a [`ContiguousCellDofs`](@ref) descriptor where
the arithmetic derivation applies, the materialized table otherwise.

Two windows, not one ([`element_scatter_length`](@ref)): the action GATHERS
`(1 + Nf)·Nb` dofs (`local_size`) and SCATTERS `Nb` (`row_size`), which is what
makes [`CellNeighbourItems`](@ref)'s one-colour partition valid.

The FILL is two-sided and runs on the HOST — see [`BlockRowAssembly`](@ref).

!!! warning "Experimental surface"
    This decorator and the elections that build it may change in a minor
    release.
"""
struct BlockRowAssemblyCache{Inner, KT, NT, ST, WT, R, SDH, MI, SCT, DT, ND, NB, NF} <:
        AbstractElementCacheDecorator{Inner}
    inner::Inner
    K::KT
    neighbours::NT
    slots::ST
    windows::WT
    route::R
    sdh::SDH          # the HOST subdomain the fill walks; `nothing` on a device instance
    mass::MI          # `RateFormIntegrator`'s mass term, fused at finalize, or `nothing`
    scratch::SCT      # per-worker `2Nb × 2Nb` fill scratch (the engine-driven fill)
    fill_dofs::DT     # per-worker `Nb` fill scratch, `_condense_pair!`'s temp dof buffer
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
# `dofs = false`: `CellNeighbourCursor`'s `iterator_dofs` derives its window
# from `c.windows`/`c.slots` (the block-row tables), never from the inner
# cursor's own staged `celldofs` buffer — asking for it bought a `celldofs!`
# per item per `mul!` and discarded the result.
item_update_flags(::MatrixFreeActionKind, ::BlockRowAssemblyCache) =
    Ferrite.UpdateFlags(nodes = false, coords = false, dofs = false)

# Parameter queries are NOT routed through the block-row action: the decorator's
# blanket forward would otherwise hand `cell` — a `CellNeighbourCursor`, the
# action's own item shape — to the WRAPPED element's `query_cell_parameters`,
# which expects the shape ITS traversal positions. Passthrough, always; the
# fill queries the wrapped element directly, on its own two-sided item
# (`fill_quadrature_data!`, `fill_block_rows!`).
query_cell_parameters(::BlockRowAssemblyCache, cell, p) = p

assembly_iterator(::MatrixFreeActionKind, c::BlockRowAssemblyCache, sdh) =
    CellNeighbourCursor(default_assembly_iterator(MatrixFreeActionKind(), sdh), c.windows, c.slots,
                        _extent(c.local_size), _extent(c.row_size))
# The generic default body, stated here because the decorator's own forward would
# otherwise hand back the WRAPPED element's device iterator — its two-sided one,
# which is the FILL's and not the action's.
device_assembly_iterator(::MatrixFreeActionKind, c::BlockRowAssemblyCache, sdh, device_sdh) =
    decorate_device_iterator(assembly_iterator(MatrixFreeActionKind(), c, device_sdh), sdh)
item_provider(::MatrixFreeActionKind, ::BlockRowAssemblyCache, sdh) = CellNeighbourItems(sdh)

# The declared fill kind: the WRAPPED element's own two-sided traversal, not
# the action's cell-with-neighbours one — this is what makes `QuadratureDataKind`
# resolve its OWN pair here instead of narrowing only the cache (the spelling
# rule `assembly_iterator`'s docstring states) and inheriting the action's.
additional_iteration_kinds(::MatrixFreeAction, ::BlockRowAssemblyCache) = (QuadratureDataKind(),)

assembly_iterator(::QuadratureDataKind, c::BlockRowAssemblyCache, sdh) =
    assembly_iterator(QuadratureDataKind(), c.inner, sdh)
device_assembly_iterator(::QuadratureDataKind, c::BlockRowAssemblyCache, sdh, device_sdh) =
    device_assembly_iterator(QuadratureDataKind(), c.inner, sdh, device_sdh)

"""
    BlockRowFillItems(inner)

The FILL's item provider: `inner`, the wrapped element's own two-sided
provider, coloured REGARDLESS of the scheduling policy requested.

The condensation accumulates into `K` across items that share a cell and runs
with no assembler ([`QuadratureDataKind`](@ref) scatters nothing), so it has no
atomic fallback the way a `SequentialScheduling` sweep's DOF scatter does — a
multi-worker device would race on a shared `SequentialScheduling` chunk. The
colouring `inner` provides for [`ColoredScheduling`](@ref) already argues the
promise this needs (no two items of one chunk share a cell), so both scheduling
policies resolve to it here.

!!! warning "Experimental surface"
    Internal to the block-row fill; it may change in a minor release.
"""
struct BlockRowFillItems{P}
    inner::P
end
compute_partition(::SequentialScheduling, p::BlockRowFillItems) = compute_partition(ColoredScheduling(), p.inner)
compute_partition(::ColoredScheduling, p::BlockRowFillItems) = compute_partition(ColoredScheduling(), p.inner)

item_provider(::QuadratureDataKind, c::BlockRowAssemblyCache, sdh) =
    BlockRowFillItems(item_provider(QuadratureDataKind(), c.inner, sdh))

duplicate_for_device(device, c::BlockRowAssemblyCache) =
    BlockRowAssemblyCache(duplicate_for_device(device, c.inner), c.K, c.neighbours, c.slots, c.windows,
                          c.route, c.sdh, c.mass, similar(c.scratch), similar(c.fill_dofs),
                          c.local_size, c.row_size, c.nfacets)
# `mass` is read only by the HOST-side `finalize_action_storage!`, so it drops
# to `nothing` on every device instance. `sdh` is different: a HOST-resident
# `KernelAbstractionsDevice(KA.CPU())` runs the fill THROUGH this very instance
# (`_engine_driven_fill`), and `fill_quadrature_data!`/`_condense_pair!` read
# `cache.sdh.dh` to identify the pair's two cells — dropping it there is a
# `FieldError`, not a savings, since a real GPU never reaches it anyway. A real
# GPU device's `sdh` is `nothing`: it has no `InterfaceValues` to run the fill
# with, so it never reaches this cache through the fill at all — the store
# arrives already filled, from the host mirror's `copyto!`.
# `scratch`/`fill_dofs` stay batched either way, exactly as
# `ElementAssemblyCache.scratch` does.
function setup_device_instances(device::AbstractGPUDevice, c::BlockRowAssemblyCache, n)
    neighbours = adapt_shared(device, c.neighbours)
    return BlockRowAssemblyCache(setup_device_instances(device, c.inner, n),
                                 adapt_shared(device, c.K), neighbours,
                                 adapt_shared(device, c.slots),
                                 _device_window_source(device, c.windows, neighbours),
                                 c.route, (_engine_driven_fill(device) ? c.sdh : nothing), nothing,
                                 setup_device_instances(device, c.scratch, n),
                                 setup_device_instances(device, c.fill_dofs, n),
                                 c.local_size, c.row_size, c.nfacets)
end

# The descriptor is rebuilt around the neighbour table this very cache already
# moved, so the cache holds ONE copy of it rather than two.
_device_window_source(device, windows, neighbours) = adapt_shared(device, windows)
_device_window_source(device, windows::ContiguousCellDofs, neighbours) =
    ContiguousCellDofs(neighbours, windows.stride)
device_worker_view(c::BlockRowAssemblyCache, worker) =
    BlockRowAssemblyCache(device_worker_view(c.inner, worker), c.K, c.neighbours, c.slots, c.windows,
                          c.route, c.sdh, c.mass,
                          device_worker_view(c.scratch, worker), device_worker_view(c.fill_dofs, worker),
                          c.local_size, c.row_size, c.nfacets)

####################################
## Setup
####################################

# Cross-subdomain coupling is NOT supported. `_cell_neighbour_table` treats a
# neighbour OUTSIDE this `SubDofHandler`'s cellset the same as a mesh boundary —
# slot `0`, no block kept for it — so the store's SHAPE drops such a coupling
# structurally, silently. An element whose fill still names that neighbour (a
# pair item spanning the subdomain seam) has no slot to condense into and dies
# at FILL time instead (`_condense_pair!`'s "not one of its facet neighbours
# inside this subdomain"): this election only sees the subdomain's shape, not
# the fill's item set, so it cannot refuse the coupling any earlier than that.
with_action_storage(cache, storage::BlockRowAssembly, sdh::SubDofHandler) =
    _block_row_storage(cache, sdh, nothing)

# `mass` is [`RateFormIntegrator`](@ref)'s, handed over by the rate form's own
# element cache; `nothing` for the bare store.
function _block_row_storage(cache, sdh::SubDofHandler, mass)
    nb = ndofs_per_cell(sdh)
    _assert_dense_element_matrix(cache, "BlockRowAssembly")
    nl = length(allocate_element_residual_vector(cache, sdh))
    nl == 2nb || throw(ArgumentError(
        "$(typeof(cache)) declares an element-local system of $(nl) rows over a subdomain with " *
        "$(nb) dofs per cell, and `BlockRowAssembly()` condenses a TWO-SIDED local system: it " *
        "reads the $(2nb) × $(2nb) matrix of an item spanning two cells and writes its four " *
        "blocks into the two cells' rows. Elect `storage = ElementAssembly()` for a cell-square " *
        "element, or declare `allocate_element_residual_vector` as `2 * ndofs_per_cell`."))
    _assert_cell_disjoint_dofs(sdh)
    return BlockRowAssemblyCache(cache, sdh, element_matrix_fill_route(typeof(cache), "BlockRowAssembly"),
                                 mass)
end

# The first dof two of `sdh`'s cells share, or `0` where none do — the promise a
# DISCONTINUOUS space keeps by construction and a continuous one does not. One
# walk over the subdomain's cells, cheap next to the topology walk
# `BlockRowAssemblyCache` already does. `seen` is an ARGUMENT so that a caller
# spanning several subdomains (a rate form's cell-block mass) can carry one
# bitvector across them and catch a dof shared over a subdomain SEAM, which no
# per-subdomain walk can see.
function _shared_cell_dof(sdh::SubDofHandler, seen = falses(ndofs(sdh.dh)))
    dofs = Vector{Int}(undef, ndofs_per_cell(sdh))
    for cellid in sdh.cellset
        celldofs!(dofs, sdh.dh, cellid)
        for d in dofs
            seen[d] && return d
            seen[d] = true
        end
    end
    return 0
end

# `CellNeighbourItems`'s one-colour partition is valid only because every item
# scatters its OWN cell's dofs and no two cells of the subdomain share a dof
# (its docstring's injectivity claim). Over a CONTINUOUS space two
# face-neighbours share the dofs on their common facet, so two different items
# would scatter into the same row through a plain, non-atomic `+=` — a silent
# lost-update race under `PolyesterDevice`/`ColoredScheduling` (P1-1 of the
# do/gpu-dg adversarial review). Checked once, at setup.
function _assert_cell_disjoint_dofs(sdh::SubDofHandler)
    d = _shared_cell_dof(sdh)
    d == 0 && return nothing
    throw(ArgumentError(
        "`BlockRowAssembly()` requires a DISCONTINUOUS space: dof $(d) is shared by two " *
        "cells of this subdomain. Its one-colour partition scatters every item's OWN cell " *
        "rows through a plain, non-atomic `+=`, which is injective only when every dof " *
        "belongs to exactly one cell — the promise a discontinuous space keeps and a " *
        "continuous one does not. Elect `storage = ElementAssembly()` for a continuous " *
        "space."))
end

function BlockRowAssemblyCache(cache, sdh::SubDofHandler, route, mass)
    T    = element_value_type(cache)
    grid = get_grid(sdh.dh)
    nb   = ndofs_per_cell(sdh)
    nf   = Ferrite.nfacets(getcells(grid, first(sdh.cellset)))
    # `neighbours`/`slots` store CELL ids narrowed to `Int32`; refuse the mesh
    # that would not fit rather than truncating it silently.
    getncells(grid) <= typemax(Int32) || throw(ArgumentError(
        "`BlockRowAssembly()` keeps its neighbour and slot maps as `Int32` cell ids, and this " *
        "grid has $(getncells(grid)) cells."))
    slots = zeros(Int32, getncells(grid))
    for (slot, cellid) in enumerate(sdh.cellset)
        slots[cellid] = slot
    end
    neighbours = _cell_neighbour_table(sdh, grid, slots, nf)
    windows = _dof_window_source(sdh, neighbours, nb, nf)
    K = zeros(T, length(sdh.cellset), 1 + nf, nb, nb)
    scratch = zeros(T, 2nb, 2nb)
    fill_dofs = Vector{Int}(undef, nb)
    return BlockRowAssemblyCache(cache, K, neighbours, slots, windows, route, sdh, mass, scratch, fill_dofs,
                                 Val((1 + nf) * nb), Val(nb), Val(nf))
end

# The topology-derived neighbour table, built once on the host. A facet whose
# neighbour is outside this subdomain is a boundary HERE: the store has no row
# for that cell, so the coupling is not this subdomain's to hold.
function _cell_neighbour_table(sdh::SubDofHandler, grid, slots, nf::Int)
    top        = Ferrite.ExclusiveTopology(grid)
    neighbours = zeros(Int32, length(sdh.cellset), nf)
    for (slot, cellid) in enumerate(sdh.cellset)
        for f in 1:nf
            found = Ferrite.getneighborhood(top, grid, FacetIndex(cellid, f))
            length(found) <= 1 || throw(ArgumentError(
                "Facet $f of cell $cellid has $(length(found)) neighbours. `BlockRowAssembly()` " *
                "keeps ONE block per facet, so it covers conforming meshes only."))
            neighbours[slot, f] = (isempty(found) || slots[first(found)[1]] == 0) ? 0 : first(found)[1]
        end
    end
    return neighbours
end

"""
    _dof_window_source(sdh, neighbours, nb, nf)

Which of the two dof-window sources this subdomain's action reads, decided ONCE
against the HOST `SubDofHandler`: a [`ContiguousCellDofs`](@ref) descriptor
where every cell's dofs are `(c-1)Nb+1 : c·Nb`, the materialized `(slot, k)`
table otherwise.

The table is the fallback and not the default because it is the action's largest
index stream — `(1 + Nf)·Nb` entries per cell, fetched and then chased into `u`
— where the derivation reads only the neighbour cell id the action already
loads to skip its boundary blocks.

!!! warning "Experimental surface"
    Internal to the block-row action; it may change in a minor release.
"""
function _dof_window_source(sdh::SubDofHandler, neighbours, nb::Int, nf::Int)
    _cell_contiguous_dofs(sdh, nb) || return _dof_window_table(sdh, neighbours, nb, nf)
    return ContiguousCellDofs(neighbours, Val(nb))
end

# The whole of the arithmetic derivation's premise, checked exhaustively over
# the cells whose dofs the windows address — every cell of the subdomain, a
# boundary facet repeating its own cell and every neighbour having a slot. It is
# NOT the affine-offset test the device cell cursor's dof stride rests on: that
# one reads `cell_dofs_offset` alone and a renumbered handler passes it.
function _cell_contiguous_dofs(sdh::SubDofHandler, nb::Int)
    dofs = Vector{Int}(undef, nb)
    for cellid in sdh.cellset
        celldofs!(dofs, sdh.dh, cellid)
        base = (cellid - 1) * nb
        for j in 1:nb
            (@inbounds dofs[j]) == base + j || return false
        end
    end
    return true
end

# The `(slot, k)` table the non-contiguous fallback reads, `slot` STRIDE-1 so a
# warp of consecutive slots reads adjacent dofs. `Int32` wherever the dof count
# fits it — the entries are the action's per-entry index stream, so their width
# is DRAM traffic.
function _dof_window_table(sdh::SubDofHandler, neighbours, nb::Int, nf::Int)
    IT      = ndofs(sdh.dh) <= typemax(Int32) ? Int32 : Int
    windows = zeros(IT, length(sdh.cellset), (1 + nf) * nb)
    dofs    = Vector{Int}(undef, nb)
    for (slot, cellid) in enumerate(sdh.cellset)
        celldofs!(dofs, sdh.dh, cellid)
        windows[slot, 1:nb] .= dofs
        for f in 1:nf
            c = Int(neighbours[slot, f])
            # A boundary facet's window repeats the own cell's dofs: the gather is
            # fixed-width and reads it, the action skips the zero block it pairs with.
            celldofs!(dofs, sdh.dh, c == 0 ? cellid : c)
            windows[slot, (f * nb + 1):((f + 1) * nb)] .= dofs
        end
    end
    return windows
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
    fill_quadrature_data!(cache::BlockRowAssemblyCache, args)

The FILL half of the `QuadratureDataKind` sweep [`additional_iteration_kinds`](@ref)
declaration resolves onto the wrapped element's own pair items: condense
`args.cell`'s `2Nb × 2Nb` system into the two cells it spans, exactly
[`fill_block_rows!`](@ref)'s per-item body — `cache.scratch`/`cache.fill_dofs`
are this method's per-worker counterparts of that function's local buffers, so
neither allocates per item.

Blocks ACCUMULATE (a cell's diagonal block collects every item it takes part
in), so the CALLER zeroes `cache.K` once before the sweep; this method does not,
being called once per item.
"""
function fill_quadrature_data!(cache::BlockRowAssemblyCache, args::CellArgs)
    fill_quadrature_data!(cache.inner, args)
    fill!(cache.scratch, zero(eltype(cache.scratch)))
    _fill_pair_matrix!(cache.route, cache.scratch, cache.inner, args)
    _condense_pair!(cache, cache.scratch, args.cell, cache.fill_dofs)
    return nothing
end

"""
    fill_block_rows!(cache::BlockRowAssemblyCache, p, ctx)

Refill the subdomain's block rows from the wrapped element's own TWO-SIDED
traversal — [`assembly_iterator`](@ref)/[`item_provider`](@ref) as the element
declares them — condensing every item's `2Nb × 2Nb` system into the two cells it
spans: `LL → K[slotL, 1]`, `LR → K[slotL, 1+fL]`, `RL → K[slotR, 1+fR]`,
`RR → K[slotR, 1]`. Blocks ACCUMULATE, so a cell's diagonal block collects the
contribution of every item it takes part in.

It runs on the HOST, outside the engine's sweep, and is the GPU/host-mirror
route [`update_operator!`](@ref) keeps for a device with no `InterfaceValues`
of its own. A CPU-resident device instead runs [`fill_quadrature_data!`](@ref)
through the ordinary [`QuadratureDataKind`](@ref) engine sweep, the two
routes doing the same per-item work.
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
    fr === nothing && throw(ArgumentError(
        "A `BlockRowAssembly()` fill item couples cell $(left) to cell $(right), which does not " *
        "list $(left) among ITS OWN facet neighbours inside this subdomain — the neighbour table " *
        "is not symmetric for this pair, so the coupling's second block has no column to condense " *
        "into."))
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
    finalize_action_storage!(cache, device, p, ctx)

Transform what a matrix-free `storage` election keeps, once the fill sweep that
wrote it has finished. `p` and `ctx` are the refill's own — the pair
[`update_operator!`](@ref) was called with, and the initial pair
[`setup_operator`](@ref) fills with — so a transform reading a second term's
kernels evaluates them at the same point the fill did.

A no-op for every shipped cache but [`BlockRowAssemblyCache`](@ref), where it is
where [`RateFormIntegrator`](@ref) fuses `M⁻¹` into the store.

It is a POST-SWEEP hook and not a per-item one (the `_pack_symmetric!` route):
the transform is per matrix ROW, and a row is complete only after every item
that contributes to it has run.

!!! warning "Experimental surface"
    Internal to the matrix-free action; it may change in a minor release.
"""
finalize_action_storage!(cache, device, p, ctx) = nothing

finalize_action_storage!(cache::BlockRowAssemblyCache, device, p, ctx) =
    _premultiply_inverse_mass!(cache, cache.mass, p, ctx)

_premultiply_inverse_mass!(::BlockRowAssemblyCache, ::Nothing, p, ctx) = nothing

# `M` is block diagonal by cell over a discontinuous space — the space this
# store already demands (`_assert_cell_disjoint_dofs`) — so `M⁻¹K` has the SAME
# block-row sparsity: one cell's inverse mass left-scales that cell's whole row.
# `ElementInverse` hands the inverse per cell, in the shape the mass's
# `element_matrix_structure` declares; nothing is inverted at action time.
function _premultiply_inverse_mass!(cache::BlockRowAssemblyCache, integrator, p, ctx)
    nb = _extent(cache.row_size)
    T  = eltype(cache.K)
    B  = zeros(T, nb, nb)
    C  = zeros(T, nb, nb)
    foreach_element_mass(ElementInverse(integrator), cache.sdh, p, ctx) do slot, cellid, cell, Minv
        _scale_block_row!(cache, slot, Minv, B, C)
    end
    return nothing
end

function _scale_block_row!(cache::BlockRowAssemblyCache, slot::Int, Minv::AbstractMatrix, B, C)
    for f in _stored_facets(cache, slot)
        # Through contiguous scratch: `K`'s block view is strided, not
        # column-contiguous, so neither operand of the product may be it.
        B .= @view cache.K[slot, 1 + f, :, :]
        mul!(C, Minv, B)
        (@view cache.K[slot, 1 + f, :, :]) .= C
    end
    return nothing
end

function _scale_block_row!(cache::BlockRowAssemblyCache, slot::Int, minv::AbstractVector, B, C)
    for f in _stored_facets(cache, slot)
        for j in axes(cache.K, 4), i in eachindex(minv)
            @inbounds cache.K[slot, 1 + f, i, j] *= minv[i]
        end
    end
    return nothing
end

# The facet columns this slot keeps a block for: its own diagonal block and
# every facet with a neighbour inside the subdomain.
_stored_facets(cache::BlockRowAssemblyCache, slot::Int) =
    Iterators.filter(f -> f == 0 || cache.neighbours[slot, f] != 0, 0:_extent(cache.nfacets))

####################################
## The host mirror
####################################

# Whether a refill can run the ENGINE-driven pair sweep instead of the
# HOST-mirror walk below: a device that is genuinely HOST-resident, since the
# fill reads host `InterfaceValues` — `KernelAbstractionsDevice(KA.CPU())`
# included, sharing `AbstractGPUDevice`'s launch machinery but running on the
# host; a real GPU backend is not.
_engine_driven_fill(::AbstractCPUDevice) = true
_engine_driven_fill(::AbstractGPUDevice) = false
_engine_driven_fill(device::KernelAbstractionsDevice) = nameof(typeof(device.backend)) === :CPU

# The fill's own pair-item traversal is read from the device only where the
# fill itself runs through the engine; on any other device it fills through the
# host mirror instead (`fill_block_rows!`) and never touches this alternate's
# device cache at all (P1-3 — see `device_needs_alternate_cache`'s docstring).
device_needs_alternate_cache(device, ::QuadratureDataKind, ::BlockRowAssemblyCache) =
    _engine_driven_fill(device)

# The element cache an ALTERNATE kind's resolved `device_cache` carries, its
# shape depending on the device: a `Tuple`/`Vector` of per-worker duplicates
# (`AbstractCPUDevice`, `setup_device_instances`) or one batched workspace
# (`KernelAbstractionsDevice`) — both read-only here, so any one worker's answers.
_alternate_fill_element(dc) = dc.element
_alternate_fill_element(dc::Union{Tuple, AbstractVector}) = first(dc).element

# A CPU device's workers SHARE the store the fill wrote; a GPU device's is the
# copy `adapt_shared` made at setup, and the refill is a host round trip.
_upload_action_storage!(::AbstractCPUDevice, host, device_cache) = nothing
_upload_action_storage!(::AbstractGPUDevice, host::BlockRowAssemblyCache, device_cache) =
    (copyto!(_block_row_cache(device_cache.element).K, host.K); nothing)

_block_row_cache(c::BlockRowAssemblyCache) = c
_block_row_cache(d::AbstractElementCacheDecorator) = _block_row_cache(d.inner)
