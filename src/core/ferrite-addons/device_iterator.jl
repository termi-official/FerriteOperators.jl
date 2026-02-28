## DeviceCellCacheFactory ##
# Pre-allocates pools for coords, dofs, cellid on GPU.
# Same Factory → materialize pattern as local caches (Ke/ue/re).

struct DeviceCellCacheFactory{CP, DP, CIP, SDH}
    coords_pool::CP      # GPU array (nnodes_per_cell, total_nthreads)
    dofs_pool::DP         # GPU array (ndofs_per_cell, total_nthreads)
    cellid_pool::CIP      # GPU array (total_nthreads,)
    device_sdh::SDH       # DeviceSubDofHandler (for grid/dof lookups in reinit!)
end

"""
    DeviceCellCacheFactory(sdh::SubDofHandler, device::AbstractGPUDevice; total_nthreads)

Build a `DeviceCellCacheFactory` from a CPU `SubDofHandler`.
Allocates per-thread pools on GPU and adapts the grid/dofhandler.
"""
function DeviceCellCacheFactory(sdh::SubDofHandler, device::AbstractGPUDevice; total_nthreads)
    Ti = index_type(device)
    grid = get_grid(sdh.dh)

    # Build DeviceSubDofHandler (adapts grid + dof arrays to GPU)
    device_sdh = DeviceSubDofHandler(sdh, device)

    # Pool dimensions
    first_cell = getcells(grid, first(sdh.cellset))
    nnpc = length(Ferrite.get_node_ids(first_cell))
    N = Ferrite.ndofs_per_cell(sdh)
    CT = get_coordinate_type(grid)

    # Allocate GPU pools
    coords_pool = _gpu_zeros(device, CT, nnpc, total_nthreads)
    dofs_pool   = _gpu_zeros(device, Ti, N, total_nthreads)
    cellid_pool = _gpu_zeros(device, Ti, total_nthreads)

    return DeviceCellCacheFactory(coords_pool, dofs_pool, cellid_pool, device_sdh)
end

## DeviceCellCache ##
# Immutable struct — fields are views into pools (contents mutable).
# Created once per thread via materialize(), updated per cell via reinit!().

struct DeviceCellCache{CoordsView, DofsView, CellidView, SDHType}
    coords::CoordsView      # view into coords_pool[:, tid]
    dofs::DofsView           # view into dofs_pool[:, tid]
    cellid_ref::CellidView   # view into cellid_pool[tid:tid] — 1-element, mutable scalar
    sdh::SDHType             # DeviceSubDofHandler (for grid/dof lookups)
end

@inline function materialize(factory::DeviceCellCacheFactory, tid)
    DeviceCellCache(
        view(factory.coords_pool, :, tid),
        view(factory.dofs_pool, :, tid),
        view(factory.cellid_pool, tid:tid),
        factory.device_sdh,
    )
end

## reinit! — same logic as CPU CellCache, mutates view contents ##
@inline function Ferrite.reinit!(cc::DeviceCellCache, i::Integer)
    cc.cellid_ref[1] = i
    grid = cc.sdh.dh_data.grid
    # Load coordinates from grid nodes
    # Access cell.nodes tuple directly (avoids dynamic dispatch through get_node_ids)
    cell = grid.cells[i]
    node_ids = cell.nodes
    _fill_coords!(cc.coords, grid.nodes, node_ids)
    # Load DOFs from flat cell_dofs
    offset = cc.sdh.dh_data.cell_dofs_offset[i]
    ndofs = cc.sdh.ndofs_per_cell
    for k in 1:ndofs
        cc.dofs[k] = cc.sdh.dh_data.cell_dofs[offset + k - 1]
    end
    return cc
end

# Unroll coord copy using NTuple type info (compile-time length)
@generated function _fill_coords!(coords, nodes, node_ids::NTuple{N, T}) where {N, T}
    exprs = [:(coords[$k] = nodes[node_ids[$k]].x) for k in 1:N]
    return quote
        $(exprs...)
        nothing
    end
end

## Ferrite-compatible accessors ##
@inline Ferrite.getcoordinates(cc::DeviceCellCache) = cc.coords
@inline Ferrite.celldofs(cc::DeviceCellCache) = cc.dofs
@inline Ferrite.cellid(cc::DeviceCellCache) = cc.cellid_ref[1]

# CellValues reinit! dispatch: extract coordinates and forward
@inline Ferrite.reinit!(cv::CellValues, cc::DeviceCellCache) = reinit!(cv, cc.coords)
