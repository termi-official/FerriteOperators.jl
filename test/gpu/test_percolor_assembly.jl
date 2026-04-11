## PerColorAssemblyStrategy GPU tests ##

# --- Bilinear: sparse matrix assembly ---
function test_bilinear_percolor(device, dh, integrator; atol=nothing, rtol=nothing)
    # CPU reference
    cpu_strategy = PerColorAssemblyStrategy(SequentialCPUDevice())
    cpu_op = setup_operator(cpu_strategy, integrator, dh)
    update_operator!(cpu_op, 0.0)

    # GPU
    gpu_strategy = PerColorAssemblyStrategy(device)
    gpu_op = setup_operator(gpu_strategy, integrator, dh)
    update_operator!(gpu_op, 0.0)

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

# --- Nonlinear: Full Assembly (J, J+R, R) ---
function test_nonlinear_percolor(device, dh, integrator, u_gpu; atol=nothing, rtol=nothing)
    backend = KA.backend(device)
    n = ndofs(dh)
    u_cpu = Array(u_gpu)

    # CPU reference
    cpu_strategy = PerColorAssemblyStrategy(SequentialCPUDevice())
    cpu_op = setup_operator(cpu_strategy, integrator, dh)

    cpu_op.J .= NaN
    update_linearization!(cpu_op, u_cpu, 0.0)
    J_baseline = copy(cpu_op.J)

    residual_cpu = zeros(n)
    cpu_op.J .= NaN
    update_linearization!(cpu_op, residual_cpu, u_cpu, 0.0)
    residual_baseline = copy(residual_cpu)

    # GPU — u, residual are all GPU
    gpu_strategy = PerColorAssemblyStrategy(device)
    gpu_op = setup_operator(gpu_strategy, integrator, dh)

    @testset "J only" begin
        update_linearization!(gpu_op, u_gpu, 0.0)
        gpu_J = SparseMatrixCSC(gpu_op.J)
        @test isapprox(J_baseline, gpu_J; atol=something(atol, 0), rtol=something(rtol, 0))
    end

    @testset "J+R" begin
        residual_gpu = KA.zeros(backend, Float64, n)
        update_linearization!(gpu_op, residual_gpu, u_gpu, 0.0)
        gpu_J = SparseMatrixCSC(gpu_op.J)
        @test isapprox(J_baseline, gpu_J; atol=something(atol, 0), rtol=something(rtol, 0))
        @test isapprox(residual_baseline, Array(residual_gpu); atol=something(atol, 0), rtol=something(rtol, 0))
    end

    @testset "R only" begin
        residual_gpu = KA.zeros(backend, Float64, n)
        gpu_op(residual_gpu, u_gpu, 0.0)
        @test isapprox(residual_baseline, Array(residual_gpu); atol=something(atol, 0), rtol=something(rtol, 0))
    end

    @testset "Idempotency" begin
        update_linearization!(gpu_op, u_gpu, 0.0)
        gpu_J = SparseMatrixCSC(gpu_op.J)
        @test isapprox(J_baseline, gpu_J; atol=something(atol, 0), rtol=something(rtol, 0))
    end
end

@testset "PerColorAssemblyStrategy (GPU)" begin
    run_on_backends(cuda_f32=false) do device
        @testset "Bilinear Diffusion" begin
            dh, integrator = setup_diffusion_problem()
            test_bilinear_percolor(device, dh, integrator, atol=1e-10)
        end

        @testset "Bilinear Mass" begin
            dh, integrator = setup_mass_problem()
            test_bilinear_percolor(device, dh, integrator, atol=1e-10)
        end

        @testset "Nonlinear Hyperelasticity" begin
            dh, integrator, u = setup_hyperelasticity_problem(device)
            test_nonlinear_percolor(device, dh, integrator, u, atol=1e-10)
        end
    end
end
