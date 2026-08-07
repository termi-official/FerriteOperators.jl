using FerriteOperators
using Test
using LinearAlgebra
using SparseArrays
using SparseArrays: getcolptr

# Transient diffusion, r(u, u̇) = ∫ (u̇ v + ∇u⋅∇v) dΩ — the ∂F/∂du block is the
# mass matrix and the ∂F/∂u block the stiffness matrix, both known in closed
# form from the bundled bilinear integrators.
struct StageDiffusionIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct StageDiffusionCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::StageDiffusionIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return StageDiffusionCache(CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::StageDiffusionCache) =
    StageDiffusionCache(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.reinit_values!(c::StageDiffusionCache, cell) = reinit!(c.cv, cell)

function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::StageDiffusionCache, args::KernelArgs)
    (; cv) = cache
    uₑ, duₑ = args.states.u, args.states.du
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        u̇  = function_value(cv, qp, duₑ)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (u̇ * shape_value(cv, qp, i) + ∇u ⋅ shape_gradient(cv, qp, i)) * dΩ
        end
    end
end

# Same physics, but the analytic-Jacobian declaration is written against the
# `JacobianKind` UnionAll while only the `:u` kernel exists — the claim a slot
# sweep must not take at face value.
struct BlanketClaimIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct BlanketClaimCache{C <: StageDiffusionCache} <: AbstractVolumetricElementCache
    inner::C
end
FerriteOperators.setup_element_cache(m::BlanketClaimIntegrator, sdh::SubDofHandler) =
    BlanketClaimCache(FerriteOperators.setup_element_cache(StageDiffusionIntegrator(m.qrc, m.field_name), sdh))
FerriteOperators.duplicate_for_device(device, c::BlanketClaimCache) =
    BlanketClaimCache(FerriteOperators.duplicate_for_device(device, c.inner))
FerriteOperators.reinit_values!(c::BlanketClaimCache, cell) = reinit_values!(c.inner, cell)
FerriteOperators.assemble_cell!(req::ResidualRequest, c::BlanketClaimCache, args::KernelArgs) =
    assemble_cell!(req, c.inner, args)
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, c::BlanketClaimCache, args::KernelArgs)
    (; cv) = c.inner
    for qp in 1:getnquadpoints(cv), i in 1:getnbasefunctions(cv), j in 1:getnbasefunctions(cv)
        req.K[i, j] += (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) * getdetJdV(cv, qp)
    end
end
FerriteOperators.provides_analytic(::Type{<:BlanketClaimCache}, ::JacobianKind) = true

@testset "Components and stage blocks" begin
    grid = generate_grid(Quadrilateral, (3, 2))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(2)
    n   = ndofs(dh)

    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op  = setup_operator(strategy, StageDiffusionIntegrator(qrc, :u), dh; slots = (:u, :du))
    Mop = setup_operator(strategy, FerriteOperators.SimpleBilinearMassIntegrator(1.0, qrc, :u), dh)
    Kop = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
    update_operator!(Mop, nothing)
    update_operator!(Kop, nothing)

    u   = sin.(0.3 .* (1:n))
    du  = cos.(0.2 .* (1:n))
    Δt  = 0.25
    ctx = TimeIntegrationContext(1.0, Δt, Δt)

    @testset "component bag shares one pattern" begin
        comps = allocate_components(op, (:M, :K))
        @test keys(comps) == (:M, :K)
        @test getcolptr(comps.M) === getcolptr(comps.K)
        @test rowvals(comps.M) === rowvals(comps.K)
        @test comps.M.nzval !== comps.K.nzval
        @test all(iszero, comps.K.nzval)
        @test_throws ArgumentError allocate_components(op, (:M, :M))
        @test_throws ArgumentError allocate_components(op, ())

        # The components are plain system matrices: the existing assembly
        # entry points fill them without knowing about the bag.
        FerriteOperators.assemble_into!(FerriteOperators.BilinearKind(), (comps.M,), Mop, (;), nothing, nothing)
        FerriteOperators.assemble_into!(FerriteOperators.BilinearKind(), (comps.K,), Kop, (;), nothing, nothing)
        @test comps.M ≈ Mop.A
        @test comps.K ≈ Kop.A

        # Backward Euler Newton matrix, monolithically known.
        W = share_pattern(comps.M)
        combine!(W, comps, (M = 1 / Δt, K = 1.0))
        @test W ≈ Mop.A ./ Δt .+ Kop.A rtol = 1e-12

        # A weight subset combines only what it names.
        combine!(W, comps, (K = 2.0,))
        @test W ≈ 2.0 .* Kop.A rtol = 1e-12

        @test_throws ArgumentError combine!(W, comps, (;))
        @test_throws ArgumentError combine!(W, comps, (nope = 1.0,))

        # A foreign pattern is rejected instead of silently combined.
        foreign = spdiagm(0 => ones(n))
        @test_throws ArgumentError combine!(W, (M = foreign,), (M = 1.0,))
        @test_throws DimensionMismatch combine!(W, (M = spdiagm(0 => ones(n + 1)),), (M = 1.0,))

        # Complex target, real components, complex weights.
        λ  = 0.7 + 1.3im
        Wc = share_pattern(comps.M, ComplexF64)
        combine!(Wc, comps, (M = 1.0 + 0.0im, K = Δt * λ))
        @test eltype(Wc) == ComplexF64
        @test Matrix(Wc) ≈ Matrix(Mop.A) .+ (Δt * λ) .* Matrix(Kop.A) rtol = 1e-12
    end

    @testset "Jacobian w.r.t. a named slot" begin
        states = (u = u, du = du)
        Ju  = FerriteOperators.create_system_matrix(op.engine.strategy, dh)
        Jdu = share_pattern(Ju)
        assemble_slot_jacobian!(Ju,  op, JacobianKind{:u}(),  states, nothing, ctx)
        assemble_slot_jacobian!(Jdu, op, JacobianKind{:du}(), states, nothing, ctx)
        @test Ju  ≈ Kop.A rtol = 1e-12
        @test Jdu ≈ Mop.A rtol = 1e-12
        # The default kind stays the `:u` Newton path.
        @test JacobianKind() === JacobianKind{:u}()

        # A reconstructed slot is frozen by contract — its Jacobian is the
        # solver's chain rule, not an assemblable quantity.
        @test_throws ArgumentError assemble_slot_jacobian!(
            Jdu, op, JacobianKind{:du}(), (u = u, du = AffineRate(1 / Δt, u)), nothing, ctx)
        # A slot the sweep does not carry cannot be differentiated either.
        @test_throws ArgumentError assemble_slot_jacobian!(
            Jdu, op, JacobianKind{:v}(), (u = u, du = du), nothing, ctx)

        # A blanket `provides_analytic` claim without the slot's kernel fails
        # at the entry point, not as a per-cell MethodError.
        bop = setup_operator(strategy, BlanketClaimIntegrator(qrc, :u), dh; slots = (:u, :du))
        assemble_slot_jacobian!(Ju, bop, JacobianKind{:u}(), states, nothing, ctx)
        @test Ju ≈ Kop.A rtol = 1e-12
        @test_throws ArgumentError assemble_slot_jacobian!(
            Jdu, bop, JacobianKind{:du}(), states, nothing, ctx)
    end

    @testset "stage-block action" begin
        s3 = sqrt(3) / 6
        A  = [0.25 0.25-s3; 0.25+s3 0.25]
        c  = [0.5 - s3, 0.5 + s3]
        sbop = StageBlockOperator(op, A, c, Δt)
        @test size(sbop) == (2n, 2n)

        z    = [sin.(0.11 .* (1:n)), cos.(0.07 .* (1:n))]
        k    = [cos.(0.05 .* (1:n)), sin.(0.09 .* (1:n))]
        sts  = [(u = z[i], du = k[i]) for i in 1:2]
        ctxs = [TimeIntegrationContext(1.0 + c[i] * Δt, Δt, A[i, i] * Δt) for i in 1:2]
        assemble_stages!(sbop, op, sts, nothing, ctxs)
        # Stage Jacobians of this element are state-independent.
        @test all(i -> sbop.Ju[i] ≈ Kop.A, 1:2)
        @test all(i -> sbop.Jdu[i] ≈ Mop.A, 1:2)

        block = zeros(2n, 2n)
        for i in 1:2, j in 1:2
            rows = ((i - 1) * n + 1):(i * n)
            cols = ((j - 1) * n + 1):(j * n)
            block[rows, cols] = Δt * A[i, j] .* Matrix(sbop.Ju[i])
            i == j && (block[rows, cols] .+= Matrix(sbop.Jdu[i]))
        end

        x = sin.(0.13 .* (1:(2n)))
        y = zeros(2n)
        mul!(y, sbop, x)
        @test y ≈ block * x rtol = 1e-12

        y0 = copy(y)
        mul!(y, sbop, x, 2.0, 3.0)
        @test y ≈ 2.0 .* (block * x) .+ 3.0 .* y0 rtol = 1e-12

        @test_throws DimensionMismatch mul!(zeros(n), sbop, x)
        @test_throws DimensionMismatch assemble_stages!(sbop, op, sts[1:1], nothing, ctxs)
        @test_throws DimensionMismatch StageBlockOperator(op, A, [0.5], Δt)
    end

    @testset "transformed Radau stage matrix" begin
        # The diagonalized scheme needs ONE stage-independent component pair
        # and a complex combine! per eigenvalue — no stage-block machinery.
        comps = allocate_components(op, (:Ju, :Jdu))
        states = (u = u, du = du)
        assemble_slot_jacobian!(comps.Ju,  op, JacobianKind{:u}(),  states, nothing, ctx)
        assemble_slot_jacobian!(comps.Jdu, op, JacobianKind{:du}(), states, nothing, ctx)

        λ = 3.6378342527444957 + 3.0805293910256707im   # a Radau IIA(3) eigenvalue of A⁻¹
        W = share_pattern(comps.Ju, ComplexF64)
        combine!(W, comps, (Jdu = 1.0 + 0.0im, Ju = Δt * λ))
        @test Matrix(W) ≈ Matrix(Mop.A) .+ (Δt * λ) .* Matrix(Kop.A) rtol = 1e-12

        # And it solves: the complex stage system is a normal sparse solve.
        b = ComplexF64.(sin.(0.17 .* (1:n)), cos.(0.19 .* (1:n)))
        @test W * (W \ b) ≈ b rtol = 1e-10
    end
end
