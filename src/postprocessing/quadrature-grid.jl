"""
    VTKQuadratureGrid{T, sdim}

A VTK-compatible "grid" where every node is a quadrature point and every
cell is a single-node `VTK_VERTEX`. Used as the mesh backing a
[`VTKQuadratureFile`](@ref).

Build with [`VTKQuadratureGrid(dh, qrc)`](@ref).
"""
struct VTKQuadratureGrid{T, sdim}
    cells::Vector{VTKBase.MeshCell{VTKBase.VTKCellType, Tuple{Int}}}
    nodes::Vector{Vec{sdim, T}}
end

Ferrite.getnnodes(qgrid::VTKQuadratureGrid)  = length(qgrid.nodes)
Ferrite.getnodes(qgrid::VTKQuadratureGrid)   = qgrid.nodes

function Ferrite.create_vtk_griddata(qgrid::VTKQuadratureGrid{T, sdim}) where {T, sdim}
    coords = reshape(reinterpret(T, qgrid.nodes), (sdim, Ferrite.getnnodes(qgrid)))
    return coords, qgrid.cells
end

"""
    VTKQuadratureGrid(dh::AbstractDofHandler, qrc) -> VTKQuadratureGrid

Build a quadrature-point "grid" from `dh` and `qrc`.  Nodes are emitted in
ascending cell-ID order (matching the layout of a [`QVector`](@ref) built
with the same `dh`/`qrc`), so the flat `QVector` data maps directly to VTK
point data without reordering.
"""
function VTKQuadratureGrid(dh::AbstractDofHandler, qrc)
    grid   = get_grid(dh)
    T      = get_coordinate_eltype(grid)
    sdim   = getspatialdim(grid)
    ncells = getncells(grid)

    # Map cellid → SubDofHandler (nothing for cells not in any subdomain)
    sdh_for_cell = Vector{Any}(nothing, ncells)
    for sdh in dh.subdofhandlers
        for cellid in sdh.cellset
            sdh_for_cell[cellid] = sdh
        end
    end

    # Cache one geometry-only CellValues per SubDofHandler
    cv_cache = Dict{Any, CellValues}()
    for sdh in dh.subdofhandlers
        ip_geo        = Ferrite.geometric_interpolation(typeof(get_first_cell(sdh)))
        qr            = getquadraturerule(qrc, sdh)
        cv_cache[sdh] = CellValues(qr, ip_geo)
    end

    nodes = Vec{sdim, T}[]
    cells = VTKBase.MeshCell{VTKBase.VTKCellType, Tuple{Int}}[]
    pidx  = 1
    for cellid in 1:ncells
        sdh = sdh_for_cell[cellid]
        sdh === nothing && continue
        cv     = cv_cache[sdh]
        coords = getcoordinates(grid, cellid)
        reinit!(cv, coords)
        for qp in 1:getnquadpoints(cv)
            push!(nodes, spatial_coordinate(cv, qp, coords))
            push!(cells, VTKBase.MeshCell(VTKBase.VTKCellTypes.VTK_VERTEX, (pidx,)))
            pidx += 1
        end
    end
    return VTKQuadratureGrid(cells, nodes)
end
