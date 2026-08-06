using FerriteOperators
using Test
import LinearAlgebra: mul!, dot, norm
using SparseArrays
using Polyester

# A v2-native nonlinear element: r(u, p) = ∫ ∇v⋅∇u dΩ − ∫ p v dΩ.
# The scalar source p is the differentiable parameter (and doubles as the
# bare time in the ∂F/∂t test). Only the residual kernel exists — every
# derivative is exercised through the AD fallback.
struct SourceDiffusionIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct SourceDiffusionCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::SourceDiffusionIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return SourceDiffusionCache(CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::SourceDiffusionCache) =
    SourceDiffusionCache(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.implements_v2_kernels(::Type{<:SourceDiffusionCache}) = true
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::SourceDiffusionCache, args::KernelArgs)
    (; cv) = cache
    uₑ = args.states.u
    q  = args.p
    reinit!(cv, args.cell)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u - q * shape_value(cv, qp, i)) * dΩ
        end
    end
end

@testset "Sensitivities" begin
    grid = generate_grid(Quadrilateral, (4, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc  = QuadratureRuleCollection(2)
    n    = ndofs(dh)

    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op = setup_operator(strategy, SourceDiffusionIntegrator(qrc, :u), dh)

    u = sin.(0.3 .* (1:n))
    p = 1.7

    @testset "AD Jacobian equals assembled stiffness" begin
        update_linearization!(op, u, p)
        Kop = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        update_operator!(Kop, nothing)
        @test op.J ≈ Kop.A rtol = 1e-13
    end

    # r(u, p) = K u − p b  ⇒  b = r(u, 0) − r(u, 1)
    r0 = zeros(n); residual!(op, r0, u, 0.0)
    r1 = zeros(n); residual!(op, r1, u, 1.0)
    b = r0 .- r1

    @testset "residual structure" begin
        rp = zeros(n); residual!(op, rp, u, p)
        @test rp ≈ r0 .- p .* b rtol = 1e-13
    end

    @testset "parameter jacobian" begin
        B = zeros(n, 1)
        update_parameter_jacobian!(B, op, u, p)
        @test vec(B) ≈ -b rtol = 1e-12

        # finite-difference cross-check
        h = 1e-6
        rplus = zeros(n); residual!(op, rplus, u, p + h)
        rminus = zeros(n); residual!(op, rminus, u, p - h)
        @test vec(B) ≈ (rplus .- rminus) ./ 2h rtol = 1e-6

        @test_throws DimensionMismatch update_parameter_jacobian!(zeros(n, 2), op, u, p)
    end

    @testset "parameter VJP" begin
        B = zeros(n, 1)
        update_parameter_jacobian!(B, op, u, p)
        λ = cos.(0.7 .* (1:n))
        g = zeros(1)
        parameter_vjp!(g, op, λ, u, p)
        @test g ≈ B' * λ rtol = 1e-12
    end

    @testset "time sensitivity (bare-time parameter)" begin
        t = 0.9
        g = zeros(n)
        time_sensitivity!(g, op, u, t)
        @test g ≈ -b rtol = 1e-12
    end

    @testset "fused J+r via AD fallback" begin
        # SourceDiffusionCache declares no analytic Jacobian, so this exercises
        # the AD-fused JacobianResidualKind branch (J and primal r in one sweep).
        r_fused = zeros(n)
        update_linearization!(op, r_fused, u, p)
        r_ref = zeros(n); residual!(op, r_ref, u, p)
        @test r_fused ≈ r_ref rtol = 1e-13

        v = cos.(0.11 .* (1:n))
        h = 1e-6
        rplus = zeros(n); residual!(op, rplus, u .+ h .* v, p)
        rminus = zeros(n); residual!(op, rminus, u .- h .* v, p)
        @test op.J * v ≈ (rplus .- rminus) ./ 2h rtol = 1e-6
    end

    @testset "parallel strategy consistency" begin
        pstrategy = PerColorAssemblyStrategy(PolyesterDevice(2))
        pop = setup_operator(pstrategy, SourceDiffusionIntegrator(qrc, :u), dh)
        Bs = zeros(n, 1); update_parameter_jacobian!(Bs, op, u, p)
        Bp = zeros(n, 1); update_parameter_jacobian!(Bp, pop, u, p)
        @test Bs ≈ Bp rtol = 1e-13
        gs = zeros(n); time_sensitivity!(gs, op, u, 0.9)
        gp = zeros(n); time_sensitivity!(gp, pop, u, 0.9)
        @test gs ≈ gp rtol = 1e-13
        # VJP accumulates in parameter space: coloring gives no isolation, so
        # the scatter must go atomic (and warn once) — results must still match.
        λ = cos.(0.7 .* (1:n))
        vs = zeros(1); parameter_vjp!(vs, op, λ, u, p)
        vp = zeros(1)
        @test_logs (:warn, r"no isolation for parameter-space") match_mode = :any parameter_vjp!(vp, pop, λ, u, p)
        @test vs ≈ vp rtol = 1e-12
    end

    @testset "bilinear element inside a nonlinear operator" begin
        # Regression for the trait-consistency finding: a v2 bilinear element
        # must carry a residual kernel so it composes into nonlinear operators.
        ndi = NonlinearMultiDomainIntegrator(Dict(
            dh.subdofhandlers[1] => FerriteOperators.SimpleBilinearDiffusionIntegrator(1.3, qrc, :u),
        ))
        dop = setup_operator(strategy, ndi, dh)
        r = zeros(n); residual!(dop, r, u, nothing)
        update_linearization!(dop, u, nothing)
        @test r ≈ dop.J * u rtol = 1e-12
    end

    @testset "sensitivity sweeps never mutate u" begin
        ucopy = copy(u)
        update_parameter_jacobian!(zeros(n, 1), op, u, p)
        parameter_vjp!(zeros(1), op, ones(n), u, p)
        time_sensitivity!(zeros(n), op, u, 0.9)
        @test u == ucopy
    end
end

@testset "Analytic vs AD cross-checks (hyperelasticity)" begin
    struct NeoHookeanSens
        E::Float64
        ν::Float64
    end
    function (mat::NeoHookeanSens)(F)
        (; E, ν) = mat
        μ = E / (2(1 + ν))
        λ = (E * ν) / ((1 + ν) * (1 - 2ν))
        C = tdot(F)
        Ic = tr(C)
        J = sqrt(det(C))
        return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
    end

    grid = generate_grid(Hexahedron, (2, 2, 2))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(dh)
    qrc  = QuadratureRuleCollection(2)
    n    = ndofs(dh)

    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    integrator = FerriteOperators.SimpleHyperelasticityIntegrator(NeoHookeanSens(10.0, 0.3), qrc, :u)
    op = setup_operator(strategy, integrator, dh)

    u = 0.05 .* sin.(0.3 .* (1:n))

    # analytic v2 Jacobian against a central finite difference of the residual
    update_linearization!(op, u, 0.0)
    v = cos.(0.11 .* (1:n))
    h = 1e-6
    rplus = zeros(n); residual!(op, rplus, u .+ h .* v, 0.0)
    rminus = zeros(n); residual!(op, rminus, u .- h .* v, 0.0)
    @test op.J * v ≈ (rplus .- rminus) ./ 2h rtol = 1e-5

    # fused J+r path must agree with the split calls
    r_fused = zeros(n)
    update_linearization!(op, r_fused, u, 0.0)
    r_split = zeros(n); residual!(op, r_split, u, 0.0)
    @test r_fused ≈ r_split rtol = 1e-14
end
