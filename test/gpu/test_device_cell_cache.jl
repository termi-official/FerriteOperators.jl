using Test
using StaticArrays
using Adapt
import KernelAbstractions as KA
using FerriteOperators
import FerriteOperators:
    DeviceCellCacheFactory, DeviceCellCache,
    DeviceGrid, DeviceDofHandlerData, DeviceSubDofHandler,
    materialize, default_backend, _gpu_zeros

## GPU kernel: materialize cell cache, reinit for each cell, write coords/dofs/cellid ##
KA.@kernel function _test_cell_cache_kernel!(
    out_coords, out_dofs, out_cellids,
    factory, @Const(cellset), ncells
)
    tid = KA.@index(Global, Linear)

    if tid <= ncells
        cc = materialize(factory, tid)
        cellid = cellset[tid]
        reinit!(cc, cellid)

        # Write coordinates to output
        for k in 1:size(out_coords, 1)
            out_coords[k, tid] = getcoordinates(cc)[k]
        end
        # Write DOFs to output
        for k in 1:size(out_dofs, 1)
            out_dofs[k, tid] = celldofs(cc)[k]
        end
        # Write cellid to output
        out_cellids[tid] = Ferrite.cellid(cc)
    end
end

## CPU reference: use Ferrite's CellCache to get ground truth ##
function compute_cpu_reference(sdh)
    grid = Ferrite.get_grid(sdh.dh)
    cells_vec = collect(sdh.cellset)
    ncells = length(cells_vec)
    first_cell = getcells(grid, first(cells_vec))
    nnpc = length(Ferrite.get_node_ids(first_cell))
    N = ndofs_per_cell(sdh)
    CT = Ferrite.get_coordinate_type(grid)

    ref_coords = Matrix{CT}(undef, nnpc, ncells)
    ref_dofs = Matrix{Int}(undef, N, ncells)
    ref_cellids = Vector{Int}(undef, ncells)

    cc = CellCache(sdh)
    for (ci, cellid) in enumerate(cells_vec)
        reinit!(cc, cellid)
        coords = getcoordinates(cc)
        dofs = celldofs(cc)
        for k in 1:nnpc
            ref_coords[k, ci] = coords[k]
        end
        for k in 1:N
            ref_dofs[k, ci] = dofs[k]
        end
        ref_cellids[ci] = Ferrite.cellid(cc)
    end

    return ref_coords, ref_dofs, ref_cellids
end

## Test function ##
function test_device_cell_cache(backend, device)
    Ti = FerriteOperators.index_type(device)

    # Create a 4×4 quad mesh (16 cells, 25 nodes)
    grid = generate_grid(Quadrilateral, (4, 4))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    sdh = dh.subdofhandlers[1]

    ncells = getncells(grid)
    total_nthreads = ncells

    # Build factory
    factory = DeviceCellCacheFactory(sdh, device; total_nthreads)

    @testset "Factory construction" begin
        @test factory.device_sdh isa DeviceSubDofHandler
        @test size(factory.coords_pool) == (4, ncells)   # 4 nodes per quad
        @test size(factory.dofs_pool) == (4, ncells)      # 4 dofs per cell (scalar Lagrange-1)
        @test length(factory.cellid_pool) == ncells
    end

    # CPU reference
    ref_coords, ref_dofs, ref_cellids = compute_cpu_reference(sdh)

    # GPU outputs
    CT = Ferrite.get_coordinate_type(grid)
    out_coords_gpu  = _gpu_zeros(device, CT, 4, ncells)
    out_dofs_gpu    = _gpu_zeros(device, Ti, 4, ncells)
    out_cellids_gpu = _gpu_zeros(device, Ti, ncells)

    # Cellset on GPU
    cellset_gpu = Adapt.adapt(backend, Ti.(collect(sdh.cellset)))

    # Launch kernel (one thread per cell)
    kernel = _test_cell_cache_kernel!(backend, ncells)
    kernel(out_coords_gpu, out_dofs_gpu, out_cellids_gpu,
           factory, cellset_gpu, Ti(ncells);
           ndrange=ncells)
    KA.synchronize(backend)

    # Compare against CPU reference
    out_coords  = Array(out_coords_gpu)
    out_dofs    = Array(out_dofs_gpu)
    out_cellids = Array(out_cellids_gpu)

    @testset "Coordinates" begin
        for ci in 1:ncells, k in 1:4
            @test out_coords[k, ci] ≈ ref_coords[k, ci]
        end
    end

    @testset "DOFs" begin
        for ci in 1:ncells, k in 1:4
            @test out_dofs[k, ci] == Ti(ref_dofs[k, ci])
        end
    end

    @testset "Cell IDs" begin
        for ci in 1:ncells
            @test out_cellids[ci] == Ti(ref_cellids[ci])
        end
    end
end

## Detect backend and run ##
roc_available = try
    using AMDGPU
    include(joinpath(@__DIR__, "..", "..", "ext", "FerriteOperatorsAMDGPUExt.jl"))
    AMDGPU.functional()
catch; false end

cuda_available = try
    using CUDA
    include(joinpath(@__DIR__, "..", "..", "ext", "FerriteOperatorsCUDAExt.jl"))
    CUDA.functional()
catch; false end

@testset "Device Cell Cache" begin
    if roc_available
        @testset "ROC" begin
            test_device_cell_cache(AMDGPU.ROCBackend(), RocDevice())
        end
    else
        @info "AMDGPU not available, skipping ROC"
    end

    if cuda_available
        @testset "CUDA" begin
            test_device_cell_cache(CUDA.CUDABackend(), CudaDevice())
        end
    else
        @info "CUDA not available, skipping CUDA"
    end
end
