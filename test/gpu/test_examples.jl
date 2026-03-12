using Test
using Adapt
import KernelAbstractions as KA
using FerriteOperators
import LinearAlgebra: mul!, norm

## End-to-end test: setup_operator → update_operator! → verify assembly ##
function test_bilinear_diffusion_assembly(device)
    # Setup: 4x4 quad mesh, scalar Lagrange, diffusion D=2.5
    grid = generate_grid(Quadrilateral, (4, 4))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    D = 2.5
    integrator = FerriteOperators.SimpleBilinearDiffusionIntegrator(
        D,
        QuadratureRuleCollection(2),
        :u,
    )

    # CPU reference: use ElementAssemblyStrategy on SequentialCPUDevice
    cpu_strategy = ElementAssemblyStrategy(SequentialCPUDevice())
    cpu_op = setup_operator(cpu_strategy, integrator, dh)
    update_operator!(cpu_op, 0.0)

    # GPU: use ElementAssemblyStrategy on GPU device
    gpu_strategy = ElementAssemblyStrategy(device)
    gpu_op = setup_operator(gpu_strategy, integrator, dh)
    update_operator!(gpu_op, 0.0)

    # Compare element matrices: copy GPU data back to CPU
    cpu_Ke_data = cpu_op.A.element_matrices.data
    gpu_Ke_data = Array(gpu_op.A.element_matrices.data)

    @testset "Element matrices match CPU" begin
        @test length(cpu_Ke_data) == length(gpu_Ke_data)
        @test cpu_Ke_data ≈ gpu_Ke_data atol=1e-6
    end

    # Idempotency: re-assemble should give same result
    update_operator!(gpu_op, 0.0)
    gpu_Ke_data2 = Array(gpu_op.A.element_matrices.data)

    @testset "Idempotency" begin
        @test gpu_Ke_data ≈ gpu_Ke_data2 atol=1e-14
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

@testset "Bilinear Diffusion Assembly (GPU)" begin
    if roc_available
        @testset "ROC" begin
            test_bilinear_diffusion_assembly(RocDevice())
        end
    else
        @info "AMDGPU not available, skipping ROC"
    end

    if cuda_available
        @testset "CUDA" begin
            test_bilinear_diffusion_assembly(CudaDevice())
        end
    else
        @info "CUDA not available, skipping CUDA"
    end
end
