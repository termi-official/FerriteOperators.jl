
# Utility which holds partial grid information for device assembly.
# Follows the pattern from Thunderbolt.jl.
struct DeviceGrid{
    sdim,
    C <: Ferrite.AbstractCell,
    T <: Real,
    CellDataType <: AbstractVector{C},
    NodeDataType <: AbstractVector,
} <: Ferrite.AbstractGrid{sdim}
    cells::CellDataType
    nodes::NodeDataType
end

function DeviceGrid(
    cells::CellDataType,
    nodes::NodeDataType,
) where {
    C <: Ferrite.AbstractCell,
    CellDataType <: AbstractArray{C, 1},
    NodeDataType <: AbstractArray{<:Node{dim, T}},
} where {dim, T}
    return DeviceGrid{dim, C, T, CellDataType, NodeDataType}(cells, nodes)
end

Ferrite.get_coordinate_type(::DeviceGrid{sdim, <:Any, T}) where {sdim, T} = Vec{sdim, T}

@inline Ferrite.getcells(grid::DeviceGrid, v::Ti) where {Ti <: Integer} = grid.cells[v]
@inline Ferrite.getcells(grid::DeviceGrid, v::Int) = grid.cells[v]
@inline Ferrite.getnodes(grid::DeviceGrid, v::Integer) = grid.nodes[v]
@inline Ferrite.getnodes(grid::DeviceGrid, v::Int) = grid.nodes[v]

function _getcoordinates(grid::DeviceGrid, e::Ti) where {Ti <: Integer}
    CT = Ferrite.get_coordinate_type(grid)
    cell = getcells(grid, e)
    N = length(Ferrite.get_node_ids(cell))
    x = MVector{N, CT}(undef)
    node_ids = Ferrite.get_node_ids(cell)
    for i in 1:length(x)
        x[i] = getnodes(grid, node_ids[i]).x
    end
    return SVector(x...)
end

Ferrite.getcoordinates(grid::DeviceGrid, e::Integer) = _getcoordinates(grid, e)
Ferrite.getcoordinates(grid::DeviceGrid, e::Int) = _getcoordinates(grid, e)

# Build DeviceGrid from a Ferrite grid with CPU arrays
function build_device_grid(grid::Ferrite.AbstractGrid)
    cells_vec = [getcells(grid, i) for i in 1:getncells(grid)]
    nodes_vec = [getnodes(grid, i) for i in 1:getnnodes(grid)]
    return DeviceGrid(cells_vec, nodes_vec)
end
