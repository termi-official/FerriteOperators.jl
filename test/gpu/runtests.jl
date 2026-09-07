# CUDA equivalence tests for `KernelAbstractionsDevice`, in their own
# environment so the main suite — which covers the same device path on the
# `KernelAbstractions.CPU()` backend — carries no GPU dependency and needs no
# GPU on CI. Run with
#
#     julia --project=test/gpu test/gpu/runtests.jl
#
# The [sources] section wires the repo in by path, so this needs Julia ≥ 1.11.
using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using SparseArrays
using LinearAlgebra
using CUDA
import CUDA: CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR
import KernelAbstractions as KA

@test CUDA.functional()

const Tv = Float32
const Ti = Int32

function hex_testbed(dims = (5, 5, 5))
    grid = generate_grid(Hexahedron, dims,
                         Vec{3}((-1.0f0, -1.0f0, -1.0f0)), Vec{3}((1.0f0, 1.0f0, 1.0f0)))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    return dh
end

cuda_device() = KernelAbstractionsDevice(CUDABackend(); value_type = Tv, index_type = Ti,
                                         items_per_worker = 2, max_workgroup_size = 256)

function cuda_strategy(; matrix_type = nothing)
    return AssemblyStrategy(FullAssembly(StandardOperatorSpecification(; matrix_type)),
                            ColoredScheduling(), cuda_device())
end

sequential_strategy() = AssemblyStrategy(SequentialCPUDevice{Tv, Ti}())

@testset "CUDA assembly equivalence" begin
    dh  = hex_testbed()
    qrc = QuadratureRuleCollection(2)

    @testset "bilinear $(nameof(typeof(integrator)))" for integrator in (
            SimpleBilinearDiffusionIntegrator(2.5, qrc, :u),
            SimpleBilinearMassIntegrator(1.7, qrc, :u))

        reference = setup_operator(sequential_strategy(), integrator, dh)
        update_operator!(reference, nothing)

        strategy = cuda_strategy(; matrix_type = CuSparseMatrixCSC{Tv, Ti})
        device   = setup_operator(strategy, integrator, dh)
        @test device.A isa CuSparseMatrixCSC
        update_operator!(device, nothing)

        @test SparseMatrixCSC(device.A) ≈ reference.A rtol = 1.0f-4

        # Coloring fixes the accumulation order per entry, so a repeated sweep
        # reproduces the previous one exactly.
        first_run = Array(nonzeros(device.A))
        update_operator!(device, nothing)
        @test first_run == Array(nonzeros(device.A))
    end

    @testset "linear form" begin
        integrator = SimpleLinearIntegrator(3.1, qrc, :u)

        reference = setup_operator(sequential_strategy(), integrator, dh)
        update_operator!(reference, nothing)

        device = setup_operator(cuda_strategy(), integrator, dh)
        @test device.b isa CuVector{Tv}
        update_operator!(device, nothing)

        @test Array(device.b) ≈ reference.b rtol = 1.0f-4

        first_run = Array(device.b)
        update_operator!(device, nothing)
        @test first_run == Array(device.b)
    end

    @testset "per-sweep host allocations stay O(1)" begin
        op = setup_operator(cuda_strategy(; matrix_type = CuSparseMatrixCSC{Tv, Ti}),
                            SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(op, nothing)
        update_operator!(op, nothing)
        @test (@allocated update_operator!(op, nothing)) < 200_000
    end

    @testset "rejects a CSR device matrix" begin
        strategy = cuda_strategy(; matrix_type = CuSparseMatrixCSR{Tv, Ti})
        err = @test_throws ArgumentError setup_operator(
            strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        @test occursin("start_assemble", err.value.msg)
    end
end
