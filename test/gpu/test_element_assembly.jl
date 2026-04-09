## ElementAssemblyStrategy GPU tests ##

# --- Bilinear: test element matrices + matrix-free product ---
function test_bilinear_ea(device, dh, integrator; atol=nothing, rtol=nothing)
    # CPU reference
    cpu_strategy = ElementAssemblyStrategy(SequentialCPUDevice())
    cpu_op = setup_operator(cpu_strategy, integrator, dh)
    update_operator!(cpu_op, 0.0)

    # GPU
    gpu_strategy = ElementAssemblyStrategy(device)
    gpu_op = setup_operator(gpu_strategy, integrator, dh)
    update_operator!(gpu_op, 0.0)

    cpu_Ke_data = cpu_op.A.element_matrices.data
    gpu_Ke_data = Array(gpu_op.A.element_matrices.data)

    @testset "Element matrices" begin
        @test length(cpu_Ke_data) == length(gpu_Ke_data)
        @test isapprox(cpu_Ke_data, gpu_Ke_data; atol=something(atol, 0), rtol=something(rtol, 0))
    end

    @testset "Idempotency" begin
        update_operator!(gpu_op, 0.0)
        gpu_Ke_data2 = Array(gpu_op.A.element_matrices.data)
        @test gpu_Ke_data ≈ gpu_Ke_data2 atol=1e-14
    end

    @testset "Matrix-free product" begin
        n = Ferrite.ndofs(dh)
        x = KA.zeros(KA.backend(device), Float64, n)
        copyto!(x, collect(1.0:n) .^ 2)
        y_gpu = KA.zeros(KA.backend(device), Float64, n)
        mul!(y_gpu, gpu_op.A, x)

        x_cpu = collect(1.0:n) .^ 2
        y_cpu = zeros(n)
        mul!(y_cpu, cpu_op.A, x_cpu)

        @test isapprox(Array(y_gpu), y_cpu; atol=something(atol, 0), rtol=something(rtol, 0))
    end
end

# --- Nonlinear: test J, J+R, R via ElementAssemblyStrategy ---
function test_nonlinear_ea(device, dh, integrator, u_gpu; atol=nothing, rtol=nothing)
    backend = KA.backend(device)
    n = ndofs(dh)
    u_cpu = Array(u_gpu)

    # CPU reference
    cpu_strategy = ElementAssemblyStrategy(SequentialCPUDevice())
    cpu_op = setup_operator(cpu_strategy, integrator, dh)

    update_linearization!(cpu_op, u_cpu, 0.0)
    yref = zero(u_cpu)
    mul!(yref, cpu_op.J, u_cpu)

    residual_cpu = zeros(n)
    update_linearization!(cpu_op, residual_cpu, u_cpu, 0.0)
    residual_baseline = copy(residual_cpu)

    # GPU — u, residual, mul! vectors are all GPU
    gpu_strategy = ElementAssemblyStrategy(device)
    gpu_op = setup_operator(gpu_strategy, integrator, dh)

    @testset "J + mul!" begin
        update_linearization!(gpu_op, u_gpu, 0.0)
        y_gpu = KA.zeros(backend, Float64, n)
        mul!(y_gpu, gpu_op.J, u_gpu)
        @test isapprox(yref, Array(y_gpu); atol=something(atol, 0), rtol=something(rtol, 0))
    end

    @testset "J+R" begin
        residual_gpu = KA.zeros(backend, Float64, n)
        update_linearization!(gpu_op, residual_gpu, u_gpu, 0.0)
        y_gpu = KA.zeros(backend, Float64, n)
        mul!(y_gpu, gpu_op.J, u_gpu)
        @test isapprox(yref, Array(y_gpu); atol=something(atol, 0), rtol=something(rtol, 0))
        @test isapprox(residual_baseline, Array(residual_gpu); atol=something(atol, 0), rtol=something(rtol, 0))
    end

    @testset "R only" begin
        residual_gpu = KA.zeros(backend, Float64, n)
        gpu_op(residual_gpu, u_gpu, 0.0)
        @test isapprox(residual_baseline, Array(residual_gpu); atol=something(atol, 0), rtol=something(rtol, 0))
    end

    @testset "Idempotency" begin
        update_linearization!(gpu_op, u_gpu, 0.0)
        y_gpu = KA.zeros(backend, Float64, n)
        mul!(y_gpu, gpu_op.J, u_gpu)
        @test isapprox(yref, Array(y_gpu); atol=something(atol, 0), rtol=something(rtol, 0))
    end
end

# --- EA Collapse ---
function test_ea_collapse(device; rtol=1e-10)
    dh, _ = setup_diffusion_problem()

    eavec = FerriteOperators.EAVector(dh)
    fill!(eavec, 0.0)
    eavec.data.data .= 1.0

    b_cpu = zeros(Ferrite.ndofs(dh))
    FerriteOperators.ea_collapse!(b_cpu, eavec, SequentialCPUDevice())

    backend = KA.backend(device)
    gpu_eavec = Adapt.adapt(backend, eavec)
    b_gpu = KA.zeros(backend, Float64, Ferrite.ndofs(dh))
    FerriteOperators.ea_collapse!(b_gpu, gpu_eavec, device)

    @test isapprox(Array(b_gpu), b_cpu; rtol=rtol)
end

@testset "ElementAssemblyStrategy (GPU)" begin
    run_on_backends(cuda_f32=false) do device
        @testset "Bilinear Diffusion" begin
            dh, integrator = setup_diffusion_problem()
            test_bilinear_ea(device, dh, integrator, atol=1e-10)
        end

        @testset "Bilinear Mass" begin
            dh, integrator = setup_mass_problem()
            test_bilinear_ea(device, dh, integrator, atol=1e-10)
        end

        @testset "Nonlinear Hyperelasticity" begin
            dh, integrator, u = setup_hyperelasticity_problem(device)
            test_nonlinear_ea(device, dh, integrator, u, atol=1e-10)
        end

        @testset "EA Collapse" begin
            test_ea_collapse(device)
        end
    end
end
