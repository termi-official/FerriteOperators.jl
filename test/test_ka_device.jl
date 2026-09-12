using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using SparseArrays
using LinearAlgebra
# FerriteKAExt — `distribute_to_workers`, the device handler and every Adapt
# rule the device kernel builds on — is triggered by these four together, not by
# KernelAbstractions alone.
import Adapt, GPUArrays, GPUArraysCore
import KernelAbstractions as KA

# The CPU backend runs the same kernels as a GPU backend, so it covers the
# device path where there is no GPU. `test/gpu/` runs these on `CUDABackend()`.
ka_device(::Type{T} = Float64, ::Type{I} = Int) where {T, I} =
    KernelAbstractionsDevice(KA.CPU(); value_type = T, index_type = I,
                             items_per_worker = 2, max_workgroup_size = 8)
ka_strategy(args...) = AssemblyStrategy(ka_device(args...); scheduling = ColoredScheduling())

function quad_testbed(::Type{T} = Float64; dims = (6, 5)) where {T}
    grid = generate_grid(Quadrilateral, dims,
                         Vec{2}((-one(T), -one(T))), Vec{2}((one(T), one(T))))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    return dh
end

function hex_testbed(::Type{T} = Float64; dims = (3, 3, 3)) where {T}
    grid = generate_grid(Hexahedron, dims,
                         Vec{3}((-one(T), -one(T), -one(T))), Vec{3}((one(T), one(T), one(T))))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    return dh
end

@testset "KernelAbstractionsDevice" begin
    # Both elections are `T` here.
    @testset "matches the sequential result ($name, $T, $label)" for
            (label, testbed) in (("quad", quad_testbed), ("hex", hex_testbed)),
            T in (Float64, Float32),
            (name, build) in (("diffusion", q -> SimpleBilinearDiffusionIntegrator(2.5, q, :u)),
                              ("mass",      q -> SimpleBilinearMassIntegrator(1.7, q, :u)),
                              ("linear",    q -> SimpleLinearIntegrator(3.1, q, :u)))

        integrator = build(QuadratureRuleCollection(T, 2))
        dh  = testbed(T)
        I   = T === Float32 ? Int32 : Int
        rtol = T === Float32 ? 1.0f-5 : 1.0e-12

        # Without more than one colour the synchronization is never exercised.
        @test length(Ferrite.create_coloring(Ferrite.get_grid(dh))) > 1

        reference = setup_operator(AssemblyStrategy(SequentialCPUDevice{T, I}()), integrator, dh)
        update_operator!(reference, nothing)
        device = setup_operator(ka_strategy(T, I), integrator, dh)
        update_operator!(device, nothing)

        target = FerriteOperators.operator_payload(device)
        @test eltype(target) === T
        @test target ≈ FerriteOperators.operator_payload(reference) rtol = rtol

        # Coloring fixes the accumulation order, so a repeat is exact.
        first_run = copy(target)
        update_operator!(device, nothing)
        @test first_run == FerriteOperators.operator_payload(device)
    end

    @testset "mixed local and global precision" begin
        # The element evaluates in its collection's scalar, the device
        # accumulates in its `value_type`, and the scatter converts entry-wise.
        dh = quad_testbed(Float64)
        for (Te, Tg, Ig) in ((Float64, Float32, Int32), (Float32, Float64, Int))
            integrator = SimpleBilinearDiffusionIntegrator(2.5, QuadratureRuleCollection(Te, 2), :u)
            reference  = setup_operator(AssemblyStrategy(SequentialCPUDevice{Tg, Ig}()), integrator, dh)
            update_operator!(reference, nothing)
            op = setup_operator(ka_strategy(Tg, Ig), integrator, dh)
            update_operator!(op, nothing)

            payload = FerriteOperators.operator_payload(op)
            @test eltype(payload) === Tg
            @test payload ≈ FerriteOperators.operator_payload(reference) rtol = 1.0f-5
        end
    end

    @testset "per-sweep host allocations stay O(1)" begin
        dh = hex_testbed(Float32; dims = (6, 6, 6))
        op = setup_operator(ka_strategy(Float32, Int32),
                            SimpleBilinearDiffusionIntegrator(2.5, QuadratureRuleCollection(Float32, 2), :u), dh)
        update_operator!(op, nothing)
        update_operator!(op, nothing)
        # Nothing is transferred or rebuilt per sweep — the workspaces, the
        # coloring and the device handler were all built at setup — so what is
        # left is the per-color kernel object and the distributed assembler, and
        # the count must not scale with the 216 cells.
        allocations = @allocated update_operator!(op, nothing)
        @test allocations < 200_000
    end

    @testset "launch geometry" begin
        device = ka_device()
        # Monotone: no barrier launches more workers than `n_workers` sized for.
        @test prod(FerriteOperators.launch_geometry(device, 4)) == 2
        @test prod(FerriteOperators.launch_geometry(device, 100)) ≥ prod(FerriteOperators.launch_geometry(device, 40))
        @test FerriteOperators.launch_geometry(device, 0) == (1, 0)
        counts = [prod(FerriteOperators.launch_geometry(device, n)) for n in 1:400]
        @test issorted(counts)
        @test FerriteOperators.n_workers(device, [collect(1:40), collect(1:7)]) == prod(FerriteOperators.launch_geometry(device, 40))
    end

    @testset "lane launch geometry" begin
        device = ka_device()   # items_per_worker = 2, max_workgroup_size = 8
        # 40 items -> 24 slots; a block of 8 lanes fills a group by itself.
        @test FerriteOperators.lane_launch_geometry(device, 8, 40) == (8, 24, 24)
        # A block of 3 leaves room for a second in the same 8-wide group.
        @test FerriteOperators.lane_launch_geometry(device, 3, 40) == (6, 12, 24)
        # The slot count is the grid-stride mapping's worker count, unchanged.
        for n in (1, 7, 40, 400)
            for nlanes in (1, 3, 8)
                workgroup, blocks, n_slots = FerriteOperators.lane_launch_geometry(device, nlanes, n)
                @test n_slots == prod(FerriteOperators.launch_geometry(device, n))
                @test workgroup % nlanes == 0
                @test workgroup ≤ max(device.max_workgroup_size, nlanes)
                @test workgroup * blocks ≥ nlanes * n_slots
            end
        end
        @test FerriteOperators.lane_launch_geometry(device, 8, 0) == (8, 0, 0)
    end
end

####################################
## Scope walls
####################################

# One offending declaration each, over the plain diffusion cache.
struct FacetWallIntegrator <: AbstractBilinearIntegrator
    qrc::QuadratureRuleCollection
    facetset::Any
end
FerriteOperators.setup_element_cache(m::FacetWallIntegrator, sdh::SubDofHandler) =
    FerriteOperators.setup_element_cache(SimpleBilinearDiffusionIntegrator(1.0, m.qrc, :u), sdh)
FerriteOperators.facet_items(m::FacetWallIntegrator, ::SubDofHandler) = m.facetset

struct AlgebraicWallIntegrator <: AbstractBilinearIntegrator
    qrc::QuadratureRuleCollection
end
FerriteOperators.setup_element_cache(m::AlgebraicWallIntegrator, sdh::SubDofHandler) =
    FerriteOperators.setup_element_cache(SimpleBilinearDiffusionIntegrator(1.0, m.qrc, :u), sdh)
FerriteOperators.algebraic_items(::AlgebraicWallIntegrator, dh) = ([1],)

struct GlobalDofWallIntegrator <: AbstractBilinearIntegrator
    qrc::QuadratureRuleCollection
end
FerriteOperators.setup_element_cache(m::GlobalDofWallIntegrator, sdh::SubDofHandler) =
    FerriteOperators.setup_element_cache(SimpleBilinearDiffusionIntegrator(1.0, m.qrc, :u), sdh)
FerriteOperators.global_dofs(::GlobalDofWallIntegrator, ::SubDofHandler) = (1,)

struct NonlinearWallIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
end
FerriteOperators.setup_element_cache(m::NonlinearWallIntegrator, sdh::SubDofHandler) =
    FerriteOperators.setup_element_cache(SimpleBilinearDiffusionIntegrator(1.0, m.qrc, :u), sdh)

# Not `KernelAbstractions.CPU`, so the host-resident exemption does not apply.
struct FakeGPUBackend end

@testset "KernelAbstractionsDevice scope walls" begin
    qrc = QuadratureRuleCollection(2)
    dh  = quad_testbed()
    bilinear = SimpleBilinearDiffusionIntegrator(1.0, qrc, :u)

    @testset "requires ColoredScheduling" begin
        strategy = AssemblyStrategy(ka_device())   # SequentialScheduling by default
        err = @test_throws ArgumentError setup_operator(strategy, bilinear, dh)
        @test occursin("ColoredScheduling", err.value.msg)
        @test occursin("race", err.value.msg)
    end

    @testset "rejects a blocked specification" begin
        strategy = AssemblyStrategy(
            FullAssembly(BlockedOperatorSpecification([ndofs(dh)], SparseMatrixCSC{Float64, Int})),
            ColoredScheduling(), ka_device())
        err = @test_throws ArgumentError setup_operator(strategy, bilinear, dh)
        @test occursin("BlockedOperatorSpecification", err.value.msg)
    end

    @testset "rejects constraints on the specification" begin
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(Ferrite.get_grid(dh), "left"), (x, t) -> 0.0))
        close!(ch)
        strategy = AssemblyStrategy(
            FullAssembly(StandardOperatorSpecification(; constraint_handler = ch)),
            ColoredScheduling(), ka_device())
        err = @test_throws ArgumentError setup_operator(strategy, bilinear, dh)
        @test occursin("constraint_handler", err.value.msg)
    end

    @testset "rejects a matrix type it cannot assemble into" begin
        strategy = AssemblyStrategy(
            FullAssembly(StandardOperatorSpecification(; matrix_type = Matrix{Float64})),
            ColoredScheduling(), ka_device())
        err = @test_throws ArgumentError setup_operator(strategy, bilinear, dh)
        @test occursin("start_assemble", err.value.msg)

        mismatched = AssemblyStrategy(
            FullAssembly(StandardOperatorSpecification(; matrix_type = SparseMatrixCSC{Float32, Int32})),
            ColoredScheduling(), ka_device())
        err = @test_throws ArgumentError setup_operator(mismatched, bilinear, dh)
        @test occursin("element type", err.value.msg)
    end

    @testset "rejects a silent host matrix on a non-CPU GPU-class device" begin
        # No `matrix_type` named resolves to the host `SparseMatrixCSC`, which a
        # real accelerator device must not get silently. Caught through the full
        # `setup_operator` route, `assert_device_supported` running before any
        # device cache is built and so needing nothing from `FakeGPUBackend`
        # beyond its type name.
        fake_device = KernelAbstractionsDevice(FakeGPUBackend(); value_type = Float64, index_type = Int)
        strategy = AssemblyStrategy(FullAssembly(), ColoredScheduling(), fake_device)
        err = @test_throws ArgumentError setup_operator(strategy, bilinear, dh)
        @test occursin("matrix_type", err.value.msg)
        @test occursin("host", err.value.msg)

        # A linear integrator allocates a VECTOR, never this matrix type, so the
        # same device and the same (absent) `matrix_type` is not an error.
        # Checked directly: a full `setup_operator` round trip on
        # `FakeGPUBackend` would need real backend support past this point.
        spec = StandardOperatorSpecification()
        linear = SimpleLinearIntegrator(3.1, qrc, :u)
        @test FerriteOperators._assert_no_silent_host_matrix(fake_device, spec, linear) === nothing

        # The `KernelAbstractions.CPU()` debug backend is genuinely host-resident.
        @test FerriteOperators._assert_no_silent_host_matrix(ka_device(), spec, bilinear) === nothing
    end

    @testset "rejects nonlinear integrators" begin
        err = @test_throws ArgumentError setup_operator(ka_strategy(), NonlinearWallIntegrator(qrc), dh)
        @test occursin("bilinear and linear forms only", err.value.msg)
    end

    @testset "rejects facet items" begin
        integrator = FacetWallIntegrator(qrc, getfacetset(Ferrite.get_grid(dh), "left"))
        err = @test_throws ArgumentError setup_operator(ka_strategy(), integrator, dh)
        @test occursin("facet item family", err.value.msg)
    end

    @testset "rejects algebraic items" begin
        err = @test_throws ArgumentError setup_operator(ka_strategy(), AlgebraicWallIntegrator(qrc), dh)
        @test occursin("algebraic item family", err.value.msg)
    end

    @testset "rejects global_dofs declarations" begin
        err = @test_throws ArgumentError setup_operator(ka_strategy(), GlobalDofWallIntegrator(qrc), dh)
        @test occursin("global_dofs", err.value.msg)
    end

    @testset "rejects condensed internal state" begin
        ivh = InternalVariableHandler(cumsum(zeros(Int, getncells(Ferrite.get_grid(dh)) + 1)), nothing, ndofs(dh), 0)
        err = @test_throws ArgumentError FerriteOperators.assert_device_internal_state_supported(ka_device(), ivh)
        @test occursin("condensed internal state", err.value.msg)
    end

    @testset "rejects transfer and patch operators" begin
        err = @test_throws ArgumentError setup_transfer_operator(
            ka_strategy(), MassProlongatorIntegrator(qrc, :u), dh, dh)
        @test occursin("sequential", lowercase(err.value.msg))
    end

    @testset "rejects state-dependent sweeps" begin
        op = setup_operator(ka_strategy(), bilinear, dh)
        err = @test_throws ArgumentError evaluate!(op, zeros(ndofs(dh)), zeros(ndofs(dh)), nothing)
        @test occursin("state slots", err.value.msg)
    end

    @testset "rejects value-returning sweeps" begin
        op = setup_evaluation_operator(ka_strategy(), bilinear, dh)
        err = @test_throws ArgumentError evaluate_functional(
            op, FunctionalKind{:anything}(), (u = zeros(ndofs(dh)),), nothing)
        @test occursin("value-returning", err.value.msg)
    end
end
