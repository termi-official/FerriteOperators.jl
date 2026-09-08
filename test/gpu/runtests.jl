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

# Deterministic, dependency-free probes, matching `test/test_matrix_free.jl`.
probe(n, k) = Tv[sin(Tv(0.7) * k * i + Tv(0.3) * k) for i in 1:n]
wobble(node, d) = Tv(sin(2.7 * node + 1.3 * d))

# Perturbed interior nodes: the matrix-free element evaluates the Jacobian per
# quadrature point and must not be validated on an affine mesh.
function distorted_hex_testbed(order, dims = (4, 4, 4); distortion = 0.15f0)
    grid = generate_grid(Hexahedron, dims,
                         Vec{3}((-1.0f0, -1.0f0, -1.0f0)), Vec{3}((1.0f0, 1.0f0, 1.0f0)))
    h = 2.0f0 / maximum(dims)
    nodes = [Ferrite.Node(Vec{3, Tv}(ntuple(d -> node.x[d] +
                (all(abs.(node.x) .< 1 - 1.0f-4) ? distortion * h * wobble(i, d) : 0.0f0), 3)))
             for (i, node) in enumerate(Ferrite.getnodes(grid))]
    dh = DofHandler(Grid(Ferrite.getcells(grid), nodes))
    add!(dh, :u, Lagrange{RefHexahedron, order}())
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
    # The element precision is the integrator's election; the device's
    # `value_type` above is the global system's. Here they agree.
    qrc = QuadratureRuleCollection(Tv, 2)

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

@testset "CUDA matrix-free action" begin
    # ONE element definition, two execution mappings and two quadrature-data
    # elections, all selected on the strategy side. The scatter is atomic, so
    # the device result is compared with a tolerance rather than bitwise.
    @testset "p = $p, $(nameof(typeof(mapping))), $(nameof(typeof(storage)))" for p in 1:3,
            mapping in (WorkerPerElement(), CooperativeElement()),
            storage in (Stored(), Recompute())

        dh  = distorted_hex_testbed(p)
        qrc = QuadratureRuleCollection(Tv, p + 1)

        assembled = setup_operator(sequential_strategy(),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(ndofs(dh), 7)
        reference = assembled.A * u

        strategy = AssemblyStrategy(MatrixFreeAction(; element_mapping = mapping, storage),
                                    SequentialScheduling(), cuda_device())
        op = setup_operator(strategy, SumFactorizedDiffusionIntegrator(Tv(2.5), qrc, :u), dh)
        @test size(op) == (ndofs(dh), ndofs(dh))
        @test eltype(op) === Tv

        ud = CuVector(u)
        yd = CUDA.zeros(Tv, ndofs(dh))
        mul!(yd, op, ud)
        @test Array(yd) ≈ reference rtol = 1.0f-3

        # The action is linear, so the five-argument form is the same sweep
        # with the accumulator scaled.
        base = CuVector(probe(ndofs(dh), 9))
        y2 = copy(base)
        mul!(y2, op, ud, -1.0f0, 1.0f0)
        @test Array(y2) ≈ Array(base) .- reference rtol = 1.0f-3
    end

    # The second consumer of the sum-factorization core on the device: a scalar
    # pointwise map on the interpolated value, over the same contractions.
    @testset "mass action, p = $p, $(nameof(typeof(mapping))), $(nameof(typeof(storage)))" for p in 1:3,
            mapping in (WorkerPerElement(), CooperativeElement()),
            storage in (Stored(), Recompute())

        dh  = distorted_hex_testbed(p)
        qrc = QuadratureRuleCollection(Tv, p + 1)

        assembled = setup_operator(sequential_strategy(),
                                   SimpleBilinearMassIntegrator(1.7, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(ndofs(dh), 7)

        strategy = AssemblyStrategy(MatrixFreeAction(; element_mapping = mapping, storage),
                                    SequentialScheduling(), cuda_device())
        op = setup_operator(strategy, SumFactorizedMassIntegrator(1.7, qrc, :u), dh)
        yd = CUDA.zeros(Tv, ndofs(dh))
        mul!(yd, op, CuVector(u))
        @test Array(yd) ≈ assembled.A * u rtol = 1.0f-3
    end

    # The ELEMENT level on the device: dense per-cell matrices in a
    # (cell, i, j) store, gathered, multiplied and scattered atomically.
    @testset "ELEMENT level, p = $p, $(nameof(typeof(integrator)))" for p in 1:3,
            integrator in (:sumfact, :assembled)

        dh  = distorted_hex_testbed(p)
        qrc = QuadratureRuleCollection(Tv, p + 1)
        assembled = setup_operator(sequential_strategy(),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(ndofs(dh), 7)

        # Both fill routes: the sum-factorized cache builds its matrices from the
        # action, the standard cache from its own element-matrix kernel.
        term = integrator === :sumfact ? SumFactorizedDiffusionIntegrator(Tv(2.5), qrc, :u) :
                                         SimpleBilinearDiffusionIntegrator(2.5, qrc, :u)
        strategy = AssemblyStrategy(MatrixFreeAction(; storage = ElementAssembly()),
                                    SequentialScheduling(), cuda_device())
        op = setup_operator(strategy, term, dh)
        yd = CUDA.zeros(Tv, ndofs(dh))
        mul!(yd, op, CuVector(u))
        @test Array(yd) ≈ assembled.A * u rtol = 1.0f-3
    end

    @testset "per-mul! host allocations stay O(1)" begin
        dh  = distorted_hex_testbed(2, (6, 6, 6))
        op  = setup_operator(AssemblyStrategy(MatrixFreeAction(), SequentialScheduling(), cuda_device()),
                             SumFactorizedDiffusionIntegrator(Tv(2.5), QuadratureRuleCollection(Tv, 3), :u), dh)
        u = CUDA.rand(Tv, ndofs(dh))
        y = CUDA.zeros(Tv, ndofs(dh))
        mul!(y, op, u)
        mul!(y, op, u)
        @test (@allocated mul!(y, op, u)) < 200_000
    end
end
