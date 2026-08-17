using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using SparseArrays

include(joinpath(@__DIR__, "fixture_elements.jl"))

# Transient diffusion element: r(u, u̇) = ∫ (u̇ v + ∇u⋅∇v) dΩ. It reads the
# rate through a slot and never encodes a time-integration scheme.
const TransientDiffusionCache = CVCache{:affine_rate}
TransientDiffusionIntegrator(qrc, field_name) = CVIntegrator{:affine_rate}(qrc, field_name)

function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::TransientDiffusionCache, args)
    transient_diffusion_residual!(req.r, cache, args)
end

@testset "AffineRate slots" begin
    grid = generate_grid(Quadrilateral, (4, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(2)
    n   = ndofs(dh)

    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op = setup_operator(strategy, TransientDiffusionIntegrator(qrc, :u), dh; slots = (:u, :du))

    # Reference mass and stiffness from the bundled bilinear integrators.
    Mop = setup_operator(strategy, SimpleBilinearMassIntegrator(1.0, qrc, :u), dh)
    Kop = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
    update_operator!(Mop, nothing)
    update_operator!(Kop, nothing)

    u     = sin.(0.3 .* (1:n))
    uprev = cos.(0.2 .* (1:n))
    Δt    = 0.25
    ctx   = TimeIntegrationContext(1.0, Δt, Δt)

    @testset "Backward Euler equivalence" begin
        r = zeros(n)
        evaluate!(op, r, (u = u, du = AffineRate(1 / Δt, uprev)), nothing, ctx)
        @test r ≈ Mop.A * (u .- uprev) ./ Δt .+ Kop.A * u rtol = 1e-12
    end

    @testset "plain vector source in the same slot" begin
        # The kernel sees slot values only, so a materialized rate vector and
        # its reconstruction are interchangeable at the element interface.
        r = zeros(n)
        evaluate!(op, r, (u = u, du = (u .- uprev) ./ Δt), nothing, ctx)
        @test r ≈ Mop.A * (u .- uprev) ./ Δt .+ Kop.A * u rtol = 1e-12
    end

    @testset "AffineRate without a preceding :u slot" begin
        @test_throws ArgumentError evaluate!(
            op, zeros(n), (du = AffineRate(1 / Δt, uprev), u = u), nothing, ctx)
        opdu = setup_operator(strategy, TransientDiffusionIntegrator(qrc, :u), dh; slots = (:du,))
        @test_throws ArgumentError evaluate!(
            opdu, zeros(n), (du = AffineRate(1 / Δt, uprev),), nothing, ctx)
    end

    @testset "Jacobian is ∂F/∂u at frozen slot values" begin
        # The AD sweep seeds only the `:u` buffer, so the reconstructed rate
        # slot stays at its primal value and J is the stiffness alone. The
        # solver adds the `slope*M` term through its per-slot weights — that
        # chain rule is not the framework's job.
        s = 1 / Δt
        update_linearization!(op, (u = u, du = AffineRate(s, uprev)), nothing, ctx)
        @test op.J ≈ Kop.A rtol = 1e-12
        @test !(op.J ≈ Kop.A + s .* Mop.A)
    end
end
