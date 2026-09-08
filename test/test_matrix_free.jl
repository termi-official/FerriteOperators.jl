using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using LinearAlgebra
using Polyester
# FerriteKAExt — which supplies the device handler and every Adapt rule the
# device kernels build on — is triggered by these four together.
import Adapt, GPUArrays, GPUArraysCore
import KernelAbstractions as KA

# Deterministic, dependency-free and version-stable probes, in the style of
# `check_derivatives`' own.
probe(::Type{T}, n, k) where {T} = T[sin(T(0.7) * k * i + T(0.3) * k) for i in 1:n]
wobble(::Type{T}, node, d) where {T} = T(sin(2.7 * node + 1.3 * d))

# Perturbed interior nodes: the element must not assume an affine map, so every
# testbed here is a genuinely distorted mesh.
function distorted_testbed(cellT, interpolation, ::Type{T}, dims, p; distortion = 0.15) where {T}
    dim = length(dims)
    grid = generate_grid(cellT, dims, Vec{dim}(ntuple(_ -> -one(T), dim)),
                         Vec{dim}(ntuple(_ -> one(T), dim)))
    h = T(2) / maximum(dims)
    nodes = [Ferrite.Node(Vec{dim, T}(ntuple(d -> node.x[d] +
                (all(abs.(node.x) .< 1 - 1.0f-4) ? T(distortion) * h * wobble(T, i, d) : zero(T)), dim)))
             for (i, node) in enumerate(Ferrite.getnodes(grid))]
    dh = DofHandler(Grid(Ferrite.getcells(grid), nodes))
    add!(dh, :u, interpolation(p))
    close!(dh)
    return dh
end

matrix_free_ka(::Type{T}, mapping; scheduling = SequentialScheduling()) where {T} = AssemblyStrategy(
    MatrixFreeAction(; element_mapping = mapping), scheduling,
    KernelAbstractionsDevice(KA.CPU(); value_type = T, index_type = Int,
                             items_per_worker = 2, max_workgroup_size = 8))

@testset "MatrixFreeAction" begin
    # ONE element definition under every execution mapping the strategy axis
    # offers, against the assembled matrix of the same bilinear form.
    @testset "action matches the assembled operator ($label, $T, p = $p)" for
            (label, cellT, interpolation, dims) in (
                ("quad", Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), (4, 3)),
                ("hex",  Hexahedron,    o -> Lagrange{RefHexahedron, o}(),    (3, 2, 2))),
            T in (Float64, Float32), p in 1:3

        dh   = distorted_testbed(cellT, interpolation, T, dims, p)
        qrc  = QuadratureRuleCollection(T, p + 1)
        rtol = T === Float32 ? 1.0f-3 : 1.0e-11

        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice{T, Int}()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(T, ndofs(dh), 7)
        reference = assembled.A * u

        integrator = SumFactorizedDiffusionIntegrator(T(2.5), qrc, :u)
        @testset "$arm" for (arm, strategy) in (
                ("sequential", AssemblyStrategy(SequentialCPUDevice{T, Int}(); form = MatrixFreeAction())),
                ("polyester",  AssemblyStrategy(PolyesterDevice{T, Int}(4); form = MatrixFreeAction(),
                                                scheduling = ColoredScheduling())),
                ("KA worker-per-element", matrix_free_ka(T, WorkerPerElement())),
                ("KA cooperative",        matrix_free_ka(T, CooperativeElement())))
            op = setup_operator(strategy, integrator, dh)
            @test size(op) == (ndofs(dh), ndofs(dh))
            @test size(op, 1) == ndofs(dh)
            @test eltype(op) === T
            y = zeros(T, ndofs(dh))
            mul!(y, op, u)
            @test y ≈ reference rtol = rtol
        end
    end

    @testset "colored scatter on the device backend" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 11)
        integrator = SumFactorizedDiffusionIntegrator(2.5, qrc, :u)
        # The vector scatter is atomic under `SequentialScheduling` and plain
        # under `ColoredScheduling`; both are race-free and must agree.
        for mapping in (WorkerPerElement(), CooperativeElement())
            op = setup_operator(matrix_free_ka(Float64, mapping; scheduling = ColoredScheduling()),
                                integrator, dh)
            y = zeros(ndofs(dh))
            mul!(y, op, u)
            @test y ≈ assembled.A * u rtol = 1.0e-11
        end
    end

    @testset "the sequential arm is deterministic and allocation-free" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (4, 4, 4), 2)
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u), dh)
        u = probe(Float64, ndofs(dh), 13)
        y, z = zeros(ndofs(dh)), zeros(ndofs(dh))
        mul!(y, op, u)
        mul!(z, op, u)
        # No atomics on this arm, and the item order is fixed, so the two sweeps
        # agree bit for bit.
        @test y == z
        @test (@allocated mul!(y, op, u)) == 0
    end

    @testset "an anisotropic tensor gives a symmetric operator" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        D = SymmetricTensor{2, 3}((2.0, 0.3, -0.2, 1.4, 0.1, 3.1))
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(D, QuadratureRuleCollection(3), :u), dh)
        u = probe(Float64, ndofs(dh), 3)
        v = probe(Float64, ndofs(dh), 4)
        Au, Av = zeros(ndofs(dh)), zeros(ndofs(dh))
        mul!(Au, op, u)
        mul!(Av, op, v)
        @test dot(v, Au) ≈ dot(u, Av) rtol = 1.0e-12
        # An isotropic tensor is the scalar case, which the assembled reference covers.
        isotropic = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                                   SumFactorizedDiffusionIntegrator(2.5 * one(SymmetricTensor{2, 3}),
                                                                    QuadratureRuleCollection(3), :u), dh)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u), dh)
        update_operator!(assembled, nothing)
        y = zeros(ndofs(dh))
        mul!(y, isotropic, u)
        @test y ≈ assembled.A * u rtol = 1.0e-11
    end

    @testset "the five-argument mul! scales as LinearAlgebra promises" begin
        dh = distorted_testbed(Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), Float64, (4, 4), 2)
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u), dh)
        u = probe(Float64, ndofs(dh), 17)
        base = probe(Float64, ndofs(dh), 19)
        action = zeros(ndofs(dh))
        mul!(action, op, u)
        for (α, β) in ((1.0, 0.0), (-1.0, 1.0), (2.0, 0.5), (0.0, 3.0))
            y = copy(base)
            mul!(y, op, u, α, β)
            @test y ≈ α .* action .+ β .* base rtol = 1.0e-10
        end
        @test op * u ≈ action rtol = 1.0e-12
    end

    @testset "evaluate! is the action with parameters and a context" begin
        dh = distorted_testbed(Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), Float64, (3, 3), 1)
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(2), :u), dh)
        u = probe(Float64, ndofs(dh), 23)
        y, z = zeros(ndofs(dh)), zeros(ndofs(dh))
        mul!(y, op, u)
        evaluate!(op, z, u, nothing)
        @test y == z
        residual = zeros(ndofs(dh))
        update_linearization!(op, residual, u, nothing)
        @test residual == y
        # Nothing is stored, so there is nothing to update.
        @test update_operator!(op, nothing) === nothing
    end
end

####################################
## Capability walls
####################################

@testset "MatrixFreeAction capability walls" begin
    dh = distorted_testbed(Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), Float64, (3, 3), 1)
    qrc = QuadratureRuleCollection(2)
    sum_factorized = SumFactorizedDiffusionIntegrator(2.5, qrc, :u)
    assembled_form = SimpleBilinearDiffusionIntegrator(2.5, qrc, :u)

    @testset "the cooperative mapping needs a KernelAbstractions device" begin
        for device in (SequentialCPUDevice(), PolyesterDevice())
            strategy = AssemblyStrategy(device;
                form = MatrixFreeAction(; element_mapping = CooperativeElement()),
                scheduling = ColoredScheduling())
            err = @test_throws ArgumentError setup_operator(strategy, sum_factorized, dh)
            @test occursin("CooperativeElement", err.value.msg)
            @test occursin("KernelAbstractionsDevice", err.value.msg)
        end
    end

    @testset "a cache without the action entry is refused" begin
        strategy = AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction())
        err = @test_throws ArgumentError setup_operator(strategy, assembled_form, dh)
        @test occursin("apply_element_action!", err.value.msg)
    end

    @testset "a cache without the cooperative entries is refused" begin
        err = @test_throws ArgumentError setup_operator(
            matrix_free_ka(Float64, CooperativeElement()), assembled_form, dh)
        # `setup_operator` reports the action entry first; the cooperative half
        # of the check is what the direct call below exercises.
        @test occursin("apply_element_action!", err.value.msg)
        cache_type = typeof(setup_element_cache(assembled_form, dh.subdofhandlers[1]))
        err = @test_throws ArgumentError FerriteOperators._assert_cooperative_element(
            CooperativeElement(), cache_type)
        @test occursin("cooperative_lattice_dim", err.value.msg)
        @test occursin("WorkerPerElement", err.value.msg)
    end

    @testset "only the bilinear family takes the matrix-free form" begin
        strategy = AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction())
        err = @test_throws ArgumentError setup_operator(
            strategy, SimpleLinearIntegrator(3.1, qrc, :u), dh)
        @test occursin("no argument to act on", err.value.msg)
    end

    @testset "the action needs the :u slot" begin
        strategy = AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction())
        err = @test_throws ArgumentError setup_operator(strategy, sum_factorized, dh; slots = (:v,))
        @test occursin(":u", err.value.msg)
    end

    @testset "the matrix-free element has no element matrix" begin
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice()), sum_factorized, dh)
        err = @test_throws ArgumentError update_operator!(op, nothing)
        @test occursin("forms no element matrix", err.value.msg)
        @test occursin("MatrixFreeAction", err.value.msg)
    end

    @testset "the cooperative kernel serves the action kind only" begin
        device = FerriteOperators.with_element_mapping(
            KernelAbstractionsDevice(KA.CPU()), CooperativeElement())
        task = FerriteOperators.AssemblyTask(FerriteOperators.BilinearKind(), nothing, (;), nothing, nothing)
        err = @test_throws ArgumentError FerriteOperators.execute_on_device!(
            task, device, nothing, ())
        @test occursin("BilinearKind", err.value.msg)
    end
end
