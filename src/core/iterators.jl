## Cell iterators for assembling rectangular (transfer/prolongation) operators:
## SameGridCellIterator for two DofHandlers on the *same* grid (p-multigrid),
## NestedGridCellIterator for a fine grid nested inside a coarse one (geometric multigrid).

####################################
## SameGridCellCache      ##
####################################

"""
    SameGridCellCache

Cache for iterating the cells of a shared grid, e.g. to assemble a transfer
(prolongation / restriction) operator between two DofHandlers on the **same**
grid and the **same** set of cells (polynomial p-multigrid). The geometry is
shared; each DofHandler provides its own dof ids.

Accessors: `cellid`, `getnodes`, `getcoordinates`, plus `getrowdofs` (from
`dh_row`, typically the fine/test space) and `getcolumndofs` (from `dh_col`,
the coarse/trial space).
"""
mutable struct SameGridCellCache{X, G <: AbstractGrid,
                                          DH_row <: AbstractDofHandler,
                                          DH_col <: AbstractDofHandler}
    const grid::G
    const dh_row::DH_row
    const dh_col::DH_col
    cellid::Int
    const nodes::Vector{Int}
    const coords::Vector{X}
    const rdofs::Vector{Int}
    const cdofs::Vector{Int}
end

function SameGridCellCache(dh_row::DofHandler, dh_col::DofHandler)
    @assert length(dh_row.subdofhandlers) == length(dh_col.subdofhandlers) == 1 "Only a single subdofhandler is allowed for iterations."
    return SameGridCellCache(
        dh_row.subdofhandlers[1],
        dh_col.subdofhandlers[1],
    )
end

function SameGridCellCache(sdh_row::SubDofHandler, sdh_col::SubDofHandler)
    @assert get_grid(sdh_row.dh) === get_grid(sdh_col.dh) "Both SubDofHandlers must share the same grid"
    @assert sdh_row.cellset == sdh_col.cellset "Both SubDofHandlers must have the same cellset"
    grid = get_grid(sdh_row.dh)
    sdim = getspatialdim(grid)
    T    = get_coordinate_eltype(grid)
    X    = Vec{sdim, T}
    nN   = Ferrite.nnodes_per_cell(grid, first(sdh_row.cellset))
    return SameGridCellCache(
        grid, sdh_row, sdh_col, -1,
        zeros(Int, nN), zeros(X, nN),
        zeros(Int, ndofs_per_cell(sdh_row)),
        zeros(Int, ndofs_per_cell(sdh_col)),
    )
end

function Ferrite.reinit!(tc::SameGridCellCache, i::Int)
    tc.cellid = i
    cellnodes!(tc.nodes, tc.grid, i)
    getcoordinates!(tc.coords, tc.grid, i)
    celldofs!(tc.rdofs, tc.dh_row, i)
    celldofs!(tc.cdofs, tc.dh_col, i)
    return tc
end

Ferrite.cellid(tc::SameGridCellCache)      = tc.cellid
Ferrite.getnodes(tc::SameGridCellCache)    = tc.nodes
Ferrite.getcoordinates(tc::SameGridCellCache) = tc.coords
"""
    getrowdofs(tc) -> Vector{Int}
    getcolumndofs(tc) -> Vector{Int}

The dof vectors indexing the rectangular element matrix: rows from the
test-space DofHandler, columns from the trial-space one. For a nested cache
that is the fine cell's dofs and its parent coarse cell's.
"""
getrowdofs(tc::SameGridCellCache)  = tc.rdofs

@doc (@doc getrowdofs)
getcolumndofs(tc::SameGridCellCache) = tc.cdofs

duplicate_for_device(device::AbstractCPUDevice, tc::SameGridCellCache) = SameGridCellCache(tc.dh_row, tc.dh_col)

# So that the usual `reinit!(cv, tc)` loop pattern works with this cache.
function Ferrite.reinit!(cv::Ferrite.AbstractCellValues, tc::SameGridCellCache)
    cell = Ferrite.reinit_needs_cell(cv) ? getcells(tc.grid, tc.cellid) : nothing
    return Ferrite.reinit!(cv, cell, tc.coords)
end


####################################
## SameGridCellIterator   ##
####################################

"""
    SameGridCellIterator(dh_row, dh_col [, cellset])
    SameGridCellIterator(sdh_row::SubDofHandler, sdh_col::SubDofHandler)

Iterates the cells of a shared grid, reinitialising and returning a
[`SameGridCellCache`](@ref) per step. `dh_row` (row / test / fine space) and
`dh_col` (column / trial / coarse space) must live on the **same** grid object;
from a pair of `SubDofHandler`s the iteration is restricted to their common
`cellset`.

!!! warning
    Stateful – do not collect or broadcast over this iterator.
"""
struct SameGridCellIterator{CC <: SameGridCellCache, IC}
    cc::CC
    set::IC
end

function SameGridCellIterator(
        dh_row::DofHandler, dh_col::DofHandler,
        set::Union{IntegerCollection, Nothing} = nothing,
    )
    if set === nothing
        set = 1:getncells(get_grid(dh_row))
    end
    return SameGridCellIterator(SameGridCellCache(dh_row, dh_col), set)
end

function SameGridCellIterator(sdh_row::SubDofHandler, sdh_col::SubDofHandler)
    @assert sdh_row.cellset == sdh_col.cellset "SubDofHandlers must share the same cellset"
    return SameGridCellIterator(
        SameGridCellCache(sdh_row, sdh_col), sdh_row.cellset,
    )
end

@inline _getset(it::SameGridCellIterator)   = it.set
@inline _getcache(it::SameGridCellIterator) = it.cc

function Base.iterate(it::SameGridCellIterator, state...)
    res = iterate(_getset(it), state...)
    res === nothing && return nothing
    item, next_state = res
    reinit!(_getcache(it), item)
    return (_getcache(it), next_state)
end

Base.IteratorSize(::Type{<:SameGridCellIterator{CC, IC}}) where {CC, IC} =
    Base.IteratorSize(IC)
Base.IteratorEltype(::Type{<:SameGridCellIterator}) = Base.HasEltype()
Base.eltype(::Type{<:SameGridCellIterator{CC}}) where {CC} = CC
Base.length(it::SameGridCellIterator) = length(_getset(it))


########################################
## NestedGridCellCache        ##
########################################

"""
    NestedGridCellCache

Cache for iterating **fine** cells, e.g. to assemble a transfer operator
between a fine and a coarse grid where every fine cell is a child of exactly
one coarse cell. The caller supplies the mesh-hierarchy data:
- `fine2coarse :: Vector{Int}` – maps `fine_cell_id → coarse_cell_id`.
- `child_ref_coords :: Vector{Vector{Vec{dim,T}}}` – for each fine cell the reference
  coordinates of that cell's nodes *inside the parent (coarse) reference element*.

Accessors: `cellid`/`coarse_cellid`, `get_fine_nodes`/`get_coarse_nodes`,
`get_fine_coordinates`/`get_coarse_coordinates`, `getrowdofs` (fine dofs, the
rows of the resulting matrix), `getcolumndofs` (coarse dofs, the columns) and
`get_child_ref_coords`.
"""
mutable struct NestedGridCellCache{
        X_f, X_c,
        G_f <: AbstractGrid, G_c <: AbstractGrid,
        DH_f <: AbstractDofHandler, DH_c <: AbstractDofHandler,
    }
    # Fine side
    const fine_grid::G_f
    const dh_fine::DH_f
    fine_cellid::Int
    const fine_nodes::Vector{Int}
    const fine_coords::Vector{X_f}
    const fine_dofs::Vector{Int}
    # Coarse side
    const coarse_grid::G_c
    const dh_coarse::DH_c
    coarse_cellid::Int
    const coarse_nodes::Vector{Int}
    const coarse_coords::Vector{X_c}
    const coarse_dofs::Vector{Int}
    # Mapping data
    const fine2coarse::Vector{Int}
    # Indexed as child_ref_coords[fine_cell_id], each entry a Vector{Vec{dim,T}}.
    const child_ref_coords::Vector{Vector{X_c}}
end

function NestedGridCellCache(
        dh_fine::DofHandler, dh_coarse::DofHandler,
        fine2coarse::Vector{Int},
        child_ref_coords::Vector{<:AbstractVector},
    )
    @assert length(dh_fine.subdofhandlers) == length(dh_coarse.subdofhandlers) == 1 "Only a single subdofhandler is allowed for iterations."
    return NestedGridCellCache(
        dh_fine.subdofhandlers[1],
        dh_coarse.subdofhandlers[1],
        fine2coarse,
        child_ref_coords,
    )
end

function NestedGridCellCache(
        sdh_fine::SubDofHandler, sdh_coarse::SubDofHandler,
        fine2coarse::Vector{Int},
        child_ref_coords::Vector{<:AbstractVector},
    )
    fine_grid   = get_grid(sdh_fine.dh)
    coarse_grid = get_grid(sdh_coarse.dh)
    sdim_f = getspatialdim(fine_grid)
    sdim_c = getspatialdim(coarse_grid)
    @assert sdim_f == sdim_c "Fine and coarse grids must have the same spatial dimension"
    T_f = get_coordinate_eltype(fine_grid)
    T_c = get_coordinate_eltype(coarse_grid)
    X_f = Vec{sdim_f, T_f}
    X_c = Vec{sdim_c, T_c}
    nN   = Ferrite.nnodes_per_cell(fine_grid, first(sdh_fine.cellset))
    return NestedGridCellCache(
        fine_grid,   sdh_fine,   -1, zeros(Int, nN), zeros(X_f, nN), zeros(Int, ndofs_per_cell(sdh_fine)),
        coarse_grid, sdh_coarse, -1, zeros(Int, nN), zeros(X_c, nN), zeros(Int, ndofs_per_cell(sdh_coarse)),
        fine2coarse,
        [convert(Vector{X_c}, v) for v in child_ref_coords],
    )
end

function Ferrite.reinit!(tc::NestedGridCellCache, fine_id::Int)
    tc.fine_cellid   = fine_id
    tc.coarse_cellid = tc.fine2coarse[fine_id]

    # Fine geometry
    cellnodes!(tc.fine_nodes,   tc.fine_grid,   fine_id)
    getcoordinates!(tc.fine_coords, tc.fine_grid, fine_id)

    # Coarse geometry
    coarse_id = tc.coarse_cellid
    cellnodes!(tc.coarse_nodes,   tc.coarse_grid,   coarse_id)
    getcoordinates!(tc.coarse_coords, tc.coarse_grid, coarse_id)

    # Dofs
    celldofs!(tc.fine_dofs,   tc.dh_fine,   fine_id)
    celldofs!(tc.coarse_dofs, tc.dh_coarse, coarse_id)
    return tc
end

Ferrite.cellid(tc::NestedGridCellCache)              = tc.fine_cellid
coarse_cellid(tc::NestedGridCellCache)       = tc.coarse_cellid
get_fine_nodes(tc::NestedGridCellCache)      = tc.fine_nodes
get_coarse_nodes(tc::NestedGridCellCache)    = tc.coarse_nodes
"""
    get_fine_coordinates(tc) -> Vector{<:Vec}
    get_coarse_coordinates(tc) -> Vector{<:Vec}

The node coordinates of the current fine cell and of its parent coarse cell —
the two geometries a nested transfer kernel maps between — both refreshed by
`Ferrite.reinit!` on the [`NestedGridCellCache`](@ref).
"""
get_fine_coordinates(tc::NestedGridCellCache)   = tc.fine_coords

@doc (@doc get_fine_coordinates)
get_coarse_coordinates(tc::NestedGridCellCache) = tc.coarse_coords
getrowdofs(tc::NestedGridCellCache)          = tc.fine_dofs
getcolumndofs(tc::NestedGridCellCache)       = tc.coarse_dofs

duplicate_for_device(device::AbstractCPUDevice, tc::NestedGridCellCache) = NestedGridCellCache(tc.dh_fine, tc.dh_coarse, tc.fine2coarse, tc.child_ref_coords)

"""
    get_child_ref_coords(tc::NestedGridCellCache)

The current fine cell's nodes in the reference frame of its parent coarse
cell, where coarse-grid shape functions are evaluated at fine-grid quadrature
points.
"""
get_child_ref_coords(tc::NestedGridCellCache) = tc.child_ref_coords[tc.fine_cellid]

# So that the usual `reinit!(cv, tc)` loop pattern works with nested grids.
function Ferrite.reinit!(cv::Ferrite.AbstractCellValues, tc::NestedGridCellCache)
    cell = Ferrite.reinit_needs_cell(cv) ? getcells(tc.fine_grid, tc.fine_cellid) : nothing
    return Ferrite.reinit!(cv, cell, tc.fine_coords)
end


##########################################
## NestedGridCellIterator               ##
##########################################

"""
    NestedGridCellIterator(dh_fine, dh_coarse, fine2coarse, child_ref_coords [, cellset])

Iterator over the **fine** cells of two hierarchically nested grids, e.g. for
assembling a transfer operator between them. `dh_fine`/`dh_coarse` are the
DofHandlers of the two grids, `fine2coarse` and `child_ref_coords` are as in
[`NestedGridCellCache`](@ref), and `cellset` restricts the iteration (default:
all fine cells).

!!! warning
    Stateful – do not collect or broadcast over this iterator.
"""
struct NestedGridCellIterator{CC <: NestedGridCellCache, IC}
    cc::CC
    set::IC
end

function NestedGridCellIterator(
        dh_fine::DofHandler, dh_coarse::DofHandler,
        fine2coarse::Vector{Int},
        child_ref_coords::Vector{<:AbstractVector},
        set::Union{IntegerCollection, Nothing} = nothing,
    )
    if set === nothing
        set = 1:getncells(get_grid(dh_fine))
    end
    cache = NestedGridCellCache(dh_fine, dh_coarse, fine2coarse, child_ref_coords)
    return NestedGridCellIterator(cache, set)
end

function NestedGridCellIterator(
        sdh_fine::SubDofHandler, sdh_coarse::SubDofHandler,
        fine2coarse::Vector{Int},
        child_ref_coords::Vector{<:AbstractVector},
    )
    @assert length(sdh_fine.cellset) >= length(sdh_coarse.cellset) "The fine cellset must have more cells than the coarse cellset!"
    cache = NestedGridCellCache(sdh_fine, sdh_coarse, fine2coarse, child_ref_coords)
    return NestedGridCellIterator(cache, sdh_fine.cellset)
end

@inline _getset(it::NestedGridCellIterator)   = it.set
@inline _getcache(it::NestedGridCellIterator) = it.cc

function Base.iterate(it::NestedGridCellIterator, state...)
    res = iterate(_getset(it), state...)
    res === nothing && return nothing
    item, next_state = res
    reinit!(_getcache(it), item)
    return (_getcache(it), next_state)
end

Base.IteratorSize(::Type{<:NestedGridCellIterator{CC, IC}}) where {CC, IC} =
    Base.IteratorSize(IC)
Base.IteratorEltype(::Type{<:NestedGridCellIterator}) = Base.HasEltype()
Base.eltype(::Type{<:NestedGridCellIterator{CC}}) where {CC} = CC
Base.length(it::NestedGridCellIterator) = length(_getset(it))
