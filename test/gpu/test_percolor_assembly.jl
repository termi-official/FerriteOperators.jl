## PerColorAssemblyStrategy GPU tests ##

function test_bilinear_diffusion_percolor(device; atol=nothing, rtol=nothing)
    dh, integrator = setup_diffusion_problem()

    # CPU reference: PerColor on SequentialCPUDevice
    cpu_strategy = PerColorAssemblyStrategy(SequentialCPUDevice())
    cpu_op = setup_operator(cpu_strategy, integrator, dh)
    update_operator!(cpu_op, 0.0)

    # GPU: PerColor on GPU device
    gpu_strategy = PerColorAssemblyStrategy(device)
    gpu_op = setup_operator(gpu_strategy, integrator, dh)
    update_operator!(gpu_op, 0.0)

    # Compare assembled sparse matrices: copy GPU sparse → CPU sparse
    cpu_K = cpu_op.A
    gpu_K = SparseMatrixCSC(gpu_op.A)

    @testset "Sparse matrix assembly" begin
        @test isapprox(cpu_K, gpu_K; atol=something(atol, 0), rtol=something(rtol, 0))
    end

    @testset "Idempotency" begin
        update_operator!(gpu_op, 0.0)
        gpu_K2 = SparseMatrixCSC(gpu_op.A)
        @test isapprox(gpu_K, gpu_K2; atol=1e-14)
    end
end

@testset "PerColorAssemblyStrategy (GPU)" begin
    if cuda_available
        @testset "CUDA (Float32)" begin
            test_bilinear_diffusion_percolor(CudaDevice(), rtol=1e-5)
        end
        @testset "CUDA (Float64)" begin
            test_bilinear_diffusion_percolor(CudaDevice{Float64, Int32}(Int32(0), Int32(0)), atol=1e-10, rtol=0.0)
        end
    else
        @info "CUDA not available, skipping"
    end

    if roc_available
        @testset "ROC" begin
            test_bilinear_diffusion_percolor(RocDevice())
        end
    else
        @info "AMDGPU not available, skipping ROC"
    end
end
