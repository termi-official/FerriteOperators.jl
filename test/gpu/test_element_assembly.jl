## ElementAssemblyStrategy GPU tests ##

function test_bilinear_diffusion_ea(device; atol=nothing, rtol=nothing)
    dh, integrator = setup_diffusion_problem()

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
    if cuda_available
        @testset "CUDA (Float32)" begin
            test_bilinear_diffusion_ea(CudaDevice(), rtol=1e-5)
        end
        @testset "CUDA (Float64)" begin
            test_bilinear_diffusion_ea(CudaDevice{Float64, Int32}(Int32(0), Int32(0)), atol=1e-10, rtol=0.0)
        end
        @testset "EA Collapse" begin
            test_ea_collapse(CudaDevice{Float64, Int32}(Int32(64), Int32(4)))
        end
    else
        @info "CUDA not available, skipping"
    end

    if roc_available
        @testset "ROC" begin
            test_bilinear_diffusion_ea(RocDevice())
        end
    else
        @info "AMDGPU not available, skipping ROC"
    end
end
