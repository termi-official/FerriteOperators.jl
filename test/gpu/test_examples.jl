using Adapt
import KernelAbstractions as KA
import LinearAlgebra: mul!, norm

## End-to-end test: setup_operator → update_operator! → verify assembly ##
function test_bilinear_diffusion_assembly(device; atol=nothing, rtol=nothing)
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

    @testset "Element matrices" begin
        @test length(cpu_Ke_data) == length(gpu_Ke_data)
        @test isapprox(cpu_Ke_data, gpu_Ke_data; atol=something(atol, 0), rtol=something(rtol, 0))
    end

    # Idempotency: re-assemble should give same result
    update_operator!(gpu_op, 0.0)
    gpu_Ke_data2 = Array(gpu_op.A.element_matrices.data)

    @testset "Idempotency" begin
        @test gpu_Ke_data ≈ gpu_Ke_data2 atol=1e-14
    end

    # Matrix-free product: mul!(y, op, x) must match CPU
    @testset "Matrix-free product" begin
        ndofs = Ferrite.ndofs(dh)
        x = KA.zeros(KA.backend(device), Float64, ndofs)
        copyto!(x, collect(1.0:ndofs).^2)
        y_gpu = KA.zeros(KA.backend(device), Float64, ndofs)
        mul!(y_gpu, gpu_op.A, x)

        x_cpu = collect(1.0:ndofs).^2
        y_cpu = zeros(ndofs)
        mul!(y_cpu, cpu_op.A, x_cpu)

        @test isapprox(Array(y_gpu), y_cpu; atol=something(atol, 0), rtol=something(rtol, 0))
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

## Test ea_collapse! on GPU ##
function test_ea_collapse(device; rtol=1e-10)
    grid = generate_grid(Quadrilateral, (4, 4))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    # Create EAVector and fill with known per-element data
    eavec = FerriteOperators.EAVector(dh)
    fill!(eavec, 0.0)
    # Write 1.0 into every per-element slot
    eavec.data.data .= 1.0

    # CPU reference collapse
    b_cpu = zeros(Ferrite.ndofs(dh))
    FerriteOperators.ea_collapse!(b_cpu, eavec, SequentialCPUDevice())

    # GPU collapse
    backend = KA.backend(device)
    gpu_eavec = Adapt.adapt(backend, eavec)
    b_gpu = KA.zeros(backend, Float64, Ferrite.ndofs(dh))
    FerriteOperators.ea_collapse!(b_gpu, gpu_eavec, device)

    @test isapprox(Array(b_gpu), b_cpu; rtol=rtol)
end

@testset "Bilinear Diffusion Assembly (GPU)" begin
    if roc_available
        @testset "ROC" begin
            test_bilinear_diffusion_assembly(RocDevice())
        end
    else
        @info "AMDGPU not available, skipping ROC"
    end

    if cuda_available
        @testset "CUDA (Float32)" begin
            test_bilinear_diffusion_assembly(CudaDevice(), rtol=1e-5)
        end
        @testset "CUDA (Float64)" begin
            test_bilinear_diffusion_assembly(CudaDevice{Float64, Int32}(Int32(0), Int32(0)), atol=1e-10, rtol=0.0)
        end
    else
        @info "CUDA not available, skipping CUDA"
    end
end

@testset "EA Collapse (GPU)" begin
    if cuda_available
        @testset "CUDA" begin
            test_ea_collapse(CudaDevice{Float64, Int32}(Int32(64), Int32(4)))
        end
    else
        @info "CUDA not available, skipping"
    end
end
