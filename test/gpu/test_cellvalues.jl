using Test
using StaticArrays
using Adapt
import KernelAbstractions as KA
using FerriteOperators
import FerriteOperators:
    DeviceCellCacheFactory, DeviceCellCache,
    DeviceCellValuesFactory, DeviceCellValues,
    DeviceGrid, DeviceDofHandlerData, DeviceSubDofHandler,
    materialize, default_backend, _gpu_zeros,
    assemble_element!, duplicate_for_device, resolve_launch_config

## GPU kernel: materialize element cache, reinit cell + cellvalues, assemble Ke ##
KA.@kernel function _test_assembly_kernel!(
    out_Ke, out_detJdV,
    cell_factory, element_factory,
    @Const(cellset), ncells, nbf, nqp
)
    tid = KA.@index(Global, Linear)

    if tid <= ncells
        # Materialize per-thread caches
        cc = materialize(cell_factory, tid)
        element_cache = materialize(element_factory, tid)

        cellid = cellset[tid]
        reinit!(cc, cellid)

        # Zero out local Ke
        for j in 1:nbf, i in 1:nbf
            out_Ke[i, j, tid] = zero(eltype(out_Ke))
        end

        # Create a view into out_Ke[:, :, tid] for assemble_element!
        Ke_view = @view out_Ke[:, :, tid]
        assemble_element!(Ke_view, cc, element_cache, zero(eltype(out_Ke)))

        # Also write detJdV for verification
        cv = element_cache.cellvalues
        for q in 1:nqp
            out_detJdV[q, tid] = getdetJdV(cv, q)
        end
    end
end

## CPU reference: compute Ke and detJdV for each cell ##
function compute_cpu_assembly_reference(sdh, D)
    grid = Ferrite.get_grid(sdh.dh)
    cells_vec = collect(sdh.cellset)
    ncells = length(cells_vec)

    ip     = Ferrite.getfieldinterpolation(sdh, :u)
    ip_geo = Ferrite.geometric_interpolation(typeof(getcells(grid, first(cells_vec))))
    qr     = QuadratureRule{RefQuadrilateral}(2)
    cv     = CellValues(qr, ip, ip_geo)
    nbf    = getnbasefunctions(cv)
    nqp    = getnquadpoints(cv)

    ref_Ke     = zeros(nbf, nbf, ncells)
    ref_detJdV = zeros(nqp, ncells)

    cc = CellCache(sdh)
    element_cache = FerriteOperators.SimpleBilinearDiffusionElementCache(D, cv)

    for (ci, cellid) in enumerate(cells_vec)
        reinit!(cc, cellid)
        Ke = zeros(nbf, nbf)
        assemble_element!(Ke, cc, element_cache, 0.0)
        ref_Ke[:, :, ci] .= Ke
        for q in 1:nqp
            ref_detJdV[q, ci] = getdetJdV(cv, q)
        end
    end

    return ref_Ke, ref_detJdV, nbf, nqp
end

## Test function ##
function test_device_cellvalues(backend, device)
    Ti = FerriteOperators.index_type(device)

    # Create a 4x4 quad mesh (16 cells, 25 nodes)
    grid = generate_grid(Quadrilateral, (4, 4))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    sdh = dh.subdofhandlers[1]

    D = 2.5  # diffusion coefficient
    ncells = getncells(grid)

    # Resolve device launch config — gives concrete threads/blocks
    resolved = resolve_launch_config(device, ncells)
    nt = FerriteOperators.total_nthreads(resolved)

    # Build CPU element cache first (to extract CellValues)
    ip     = Ferrite.getfieldinterpolation(sdh, :u)
    ip_geo = Ferrite.geometric_interpolation(Quadrilateral)
    qr     = QuadratureRule{RefQuadrilateral}(2)
    cv     = CellValues(qr, ip, ip_geo)
    cpu_element_cache = FerriteOperators.SimpleBilinearDiffusionElementCache(D, cv)

    nbf = getnbasefunctions(cv)
    nqp = getnquadpoints(cv)

    # Build factories — duplicate_for_device dispatches: GPU → DeviceCellValuesFactory
    cell_factory    = DeviceCellCacheFactory(sdh, resolved; total_nthreads=nt)
    element_factory = duplicate_for_device(resolved, cpu_element_cache)

    @testset "Factory construction" begin
        @test element_factory isa FerriteOperators.SimpleBilinearDiffusionElementCache{<:DeviceCellValuesFactory}
        @test element_factory.D == D
        @test size(element_factory.cellvalues.detJdV_pool) == (nqp, nt)
        @test size(element_factory.cellvalues.dNdx_pool) == (nbf, nqp, nt)
    end

    # CPU reference
    ref_Ke, ref_detJdV, _, _ = compute_cpu_assembly_reference(sdh, D)

    # GPU outputs
    out_Ke_gpu      = _gpu_zeros(device, Float64, nbf, nbf, ncells)
    out_detJdV_gpu  = _gpu_zeros(device, Float64, nqp, ncells)

    # Cellset on GPU
    cellset_gpu = Adapt.adapt(backend, Ti.(collect(sdh.cellset)))

    # Launch kernel (one thread per cell)
    kernel = _test_assembly_kernel!(backend, ncells)
    kernel(out_Ke_gpu, out_detJdV_gpu,
           cell_factory, element_factory,
           cellset_gpu, Ti(ncells), Ti(nbf), Ti(nqp);
           ndrange=ncells)
    KA.synchronize(backend)

    # Compare against CPU reference
    out_Ke     = Array(out_Ke_gpu)
    out_detJdV = Array(out_detJdV_gpu)

    @testset "detJdV" begin
        for ci in 1:ncells, q in 1:nqp
            @test out_detJdV[q, ci] ≈ ref_detJdV[q, ci]
        end
    end

    @testset "Element matrices (Ke)" begin
        for ci in 1:ncells, j in 1:nbf, i in 1:nbf
            @test out_Ke[i, j, ci] ≈ ref_Ke[i, j, ci] atol=1e-12
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

@testset "Device CellValues + Assembly" begin
    if roc_available
        @testset "ROC" begin
            test_device_cellvalues(AMDGPU.ROCBackend(), RocDevice())
        end
    else
        @info "AMDGPU not available, skipping ROC"
    end

    if cuda_available
        @testset "CUDA" begin
            test_device_cellvalues(CUDA.CUDABackend(), CudaDevice())
        end
    else
        @info "CUDA not available, skipping CUDA"
    end
end
