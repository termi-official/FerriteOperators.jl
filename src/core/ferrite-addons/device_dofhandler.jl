## GPU-compatible DofHandler data ##
# Mirrors Ferrite's DofHandler internals on GPU: flat cell_dofs + offsets.

struct DeviceDofHandlerData{
    sdim,
    GridType <: Ferrite.AbstractGrid{sdim},
    IndexType,
    IndexVectorType <: AbstractVector{IndexType},
    Ti <: Integer,
} <: Ferrite.AbstractDofHandler
    grid::GridType
    cell_dofs::IndexVectorType
    cell_dofs_offset::IndexVectorType
    ndofs::Ti
end

Ferrite.get_grid(dh::DeviceDofHandlerData) = dh.grid
cell_dof_offset(dh::DeviceDofHandlerData, i::Integer) = dh.cell_dofs_offset[i]

## GPU-compatible SubDofHandler ##

struct DeviceSubDofHandler{
    Ti <: Integer,
    IndexType,
    IndexVectorType <: AbstractVector{IndexType},
    DHDataType <: DeviceDofHandlerData,
} <: Ferrite.AbstractDofHandler
    cellset::IndexVectorType
    ndofs_per_cell::Ti
    dh_data::DHDataType
end

Ferrite.ndofs_per_cell(sdh::DeviceSubDofHandler) = sdh.ndofs_per_cell
Ferrite.get_grid(sdh::DeviceSubDofHandler) = sdh.dh_data.grid

function celldofsview(sdh::DeviceSubDofHandler, i::Ti) where {Ti <: Integer}
    offset = cell_dof_offset(sdh.dh_data, i)
    ndofs = ndofs_per_cell(sdh)
    return @view sdh.dh_data.cell_dofs[offset:(offset + ndofs - one(Ti))]
end

"""
    DeviceSubDofHandler(sdh::SubDofHandler, device::AbstractGPUDevice)

Build GPU-compatible `DeviceSubDofHandler` from a CPU `SubDofHandler`.
Adapts the grid (cells + nodes), flat DOF arrays, and cellset to GPU.
"""
function DeviceSubDofHandler(sdh::SubDofHandler, device::AbstractGPUDevice)
    Ti = index_type(device)
    (; dh) = sdh
    grid = get_grid(dh)
    backend = default_backend(device)

    # Build DeviceGrid from CPU Grid
    device_grid = DeviceGrid(
        Adapt.adapt(backend, grid.cells),
        Adapt.adapt(backend, grid.nodes),
    )

    # Build DeviceDofHandlerData from CPU DofHandler
    cell_dofs = Adapt.adapt(backend, Ti.(dh.cell_dofs))
    cell_dofs_offset = Adapt.adapt(backend, Ti.(dh.cell_dofs_offset))
    dh_data = DeviceDofHandlerData(device_grid, cell_dofs, cell_dofs_offset, Ti(dh.ndofs))

    # Build DeviceSubDofHandler
    cellset_gpu = Adapt.adapt(backend, Ti.(collect(sdh.cellset)))
    return DeviceSubDofHandler(cellset_gpu, Ti(sdh.ndofs_per_cell), dh_data)
end
