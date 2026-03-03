## GPU-compatible grid ##
# Mirrors Ferrite's Grid but stores cells and nodes as GPU arrays.
# Keeps cell structs (AbstractCell) on GPU so that get_node_ids, nnodes etc. work.

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

@inline Ferrite.getcells(grid::DeviceGrid, v::Integer) = grid.cells[v]
@inline Ferrite.getnodes(grid::DeviceGrid, v::Integer) = grid.nodes[v]

function _getcoordinates(grid::DeviceGrid, e::Ti) where {Ti <: Integer}
    CT = Ferrite.get_coordinate_type(grid)
    cell = getcells(grid, e)
    N = nnodes(cell)
    x = MVector{N, CT}(undef)
    node_ids = get_node_ids(cell)
    for i in 1:length(x)
        x[i] = get_node_coordinate(grid, node_ids[i])
    end
    return SVector(x...)
end

Ferrite.getcoordinates(grid::DeviceGrid, e::Integer) = _getcoordinates(grid, e)
