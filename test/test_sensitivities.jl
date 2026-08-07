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

    @testset "condensed operators are rejected loudly" begin
        vgrid = generate_grid(Hexahedron, (1, 1, 1))
        vdh = DofHandler(vgrid)
        add!(vdh, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(vdh)
        vint = FerriteOperators.SimpleCondensedLinearViscoelasticity(
            FerriteOperators.MaxwellParameters(), qrc, :u, :εᵛ)
        vop = setup_operator(strategy, vint, vdh; slots = (:u, :uprev))
        vu = zeros(unknown_size(vop))
        @test_throws ArgumentError update_parameter_jacobian!(zeros(residual_size(vop), 1), vop, vu, 1.0)
        @test_throws ArgumentError parameter_vjp!(zeros(1), vop, zeros(residual_size(vop)), vu, 1.0)
        @test_throws ArgumentError time_sensitivity!(zeros(residual_size(vop)), vop, vu, 0.5)
    end

    @testset "undeclared slots error loudly" begin
        @test_throws ArgumentError update_linearization!(op, zeros(n), (u = u, uprev = copy(u)), p, nothing)
    end

    @testset "sensitivity sweeps never mutate u" begin
        ucopy = copy(u)
        update_parameter_jacobian!(zeros(n, 1), op, u, p)
        parameter_vjp!(zeros(1), op, ones(n), u, p)
        time_sensitivity!(zeros(n), op, u, 0.9)
        @test u == ucopy
    end
end

# --- Sensitivity admissibility refinements and method selection ---

# Analytic sensitivity kernels win per cache: prove selection via a counter.
const ANALYTIC_TS_CALLS = Ref(0)
FerriteOperators.provides_analytic(::Type{<:SourceDiffusionCache}, ::FerriteOperators.TimeSensitivityKind) = true
function FerriteOperators.assemble_cell!(req::TimeSensitivityRequest, cache::SourceDiffusionCache, args::KernelArgs)
    ANALYTIC_TS_CALLS[] += 1
    # ∂F/∂t with p ≡ t (bare-time): ∂/∂t of −t·∫v = −∫v dΩ
    (; cv) = cache
    reinit!(cv, args.cell)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        for i in 1:getnbasefunctions(cv)
            req.g[i] -= shape_value(cv, qp, i) * dΩ
        end
    end
end

# Condensed admissibility: analytic parameter kernel on the viscoelastic cache
# (trivially zero — the material is parameter-independent), and an
# insensitivity declaration for the VJP kind.
FerriteOperators.provides_analytic(::Type{<:FerriteOperators.SimpleCondensedLinearViscoelasticityCache}, ::FerriteOperators.ParameterJacobianKind) = true
FerriteOperators.assemble_cell!(req::ParameterJacobianRequest, ::FerriteOperators.SimpleCondensedLinearViscoelasticityCache, args::KernelArgs) = nothing
FerriteOperators.internal_state_insensitive(::Type{<:FerriteOperators.SimpleCondensedLinearViscoelasticityCache}, ::FerriteOperators.ParameterVJPKind) = true

@testset "Sensitivity admissibility and methods" begin
    grid = generate_grid(Quadrilateral, (4, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(2)
    n   = ndofs(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op = setup_operator(strategy, SourceDiffusionIntegrator(qrc, :u), dh)
    u = sin.(0.3 .* (1:n))

    @testset "analytic time kernel is selected and agrees with AD-by-construction" begin
        ANALYTIC_TS_CALLS[] = 0
        g_analytic = zeros(n)
        time_sensitivity!(g_analytic, op, u, 0.9)
        @test ANALYTIC_TS_CALLS[] == getncells(grid)
        # −b, independently derived from residual structure
        r0 = zeros(n); residual!(op, r0, u, 0.0)
        r1 = zeros(n); residual!(op, r1, u, 1.0)
        @test g_analytic ≈ -(r0 .- r1) rtol = 1e-12
    end

    @testset "FD method agrees with the derivative (and bypasses kernels)" begin
        ANALYTIC_TS_CALLS[] = 0
        g_fd = zeros(n)
        time_sensitivity!(g_fd, op, u, 0.9; method = FiniteDifferenceSensitivity())
        @test ANALYTIC_TS_CALLS[] == 0     # operator-level differencing, kernels untouched
        g_ref = zeros(n)
        time_sensitivity!(g_ref, op, u, 0.9)
        @test g_fd ≈ g_ref rtol = 1e-6
    end

    @testset "condensed: analytic parameter kernel is admissible" begin
        vgrid = generate_grid(Hexahedron, (1, 1, 1))
        vdh = DofHandler(vgrid)
        add!(vdh, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(vdh)
        vint = FerriteOperators.SimpleCondensedLinearViscoelasticity(
            FerriteOperators.MaxwellParameters(), qrc, :u, :εᵛ)
        vop = setup_operator(strategy, vint, vdh; slots = (:u, :uprev))
        vu = 1e-4 .* sin.(0.2 .* (1:unknown_size(vop)))
        vuprev = zeros(unknown_size(vop))
        vctx = TimeIntegrationContext(0.0, 0.1, 0.1)

        # analytic (trivially zero) parameter kernel: accepted, runs, B stays 0
        B = zeros(residual_size(vop), 1)
        update_parameter_jacobian!(B, vop, (u = vu, uprev = vuprev), 1.0, vctx)
        @test iszero(B)

        # insensitivity declaration: VJP runs through AD with zero result
        # (the material has no parameter dependence)
        gv = zeros(1)
        parameter_vjp!(gv, vop, ones(residual_size(vop)), (u = vu, uprev = vuprev), 1.0, vctx)
        @test abs(gv[1]) < 1e-12

        # time sensitivity via AD is still rejected (no analytic kernel, no
        # declaration for that kind) …
        @test_throws ArgumentError time_sensitivity!(zeros(residual_size(vop)), vop, (u = vu, uprev = vuprev), 0.5, vctx)
        # … but FD is admissible: primal evaluations on a protected copy.
        vu_before = copy(vu)
        g = zeros(residual_size(vop))
        time_sensitivity!(g, vop, (u = vu, uprev = vuprev), 0.5, vctx; method = FiniteDifferenceSensitivity())
        @test vu == vu_before                       # trial write-back never leaked
        @test norm(g) < 1e-8                        # no explicit t-dependence via p
    end
end

struct NeoHookeanState
    E::Float64
    ν::Float64
end
function (mat::NeoHookeanState)(F)
    (; E, ν) = mat
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    C = tdot(F)
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

@testset "State JVP/VJP actions" begin
    grid = generate_grid(Quadrilateral, (4, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(2)
    n   = ndofs(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op = setup_operator(strategy, SourceDiffusionIntegrator(qrc, :u), dh)
    u = sin.(0.3 .* (1:n))
    p = 1.7

    update_linearization!(op, u, p)   # materialize J for reference

    @testset "JVP matches materialized J·v" begin
        v  = cos.(0.11 .* (1:n))
        Jv = zeros(n)
        state_jvp!(Jv, op, v, u, p)
        @test Jv ≈ op.J * v rtol = 1e-12
    end

    @testset "VJP matches materialized Jᵀλ" begin
        λ = cos.(0.7 .* (1:n))
        g = zeros(n)
        state_vjp!(g, op, λ, u, p)
        @test g ≈ op.J' * λ rtol = 1e-12
    end

    @testset "nonlinear element, parallel consistency, no mutation" begin
        hgrid = generate_grid(Hexahedron, (2, 2, 2))
        hdh = DofHandler(hgrid)
        add!(hdh, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(hdh)
        hint = FerriteOperators.SimpleHyperelasticityIntegrator(NeoHookeanState(10.0, 0.3), qrc, :u)
        hop = setup_operator(strategy, hint, hdh)
        hn = ndofs(hdh)
        hu = 0.05 .* sin.(0.3 .* (1:hn))
        hv = cos.(0.17 .* (1:hn))
        update_linearization!(hop, hu, 0.0)   # analytic J
        Jv = zeros(hn)
        hu_before = copy(hu)
        state_jvp!(Jv, hop, hv, hu, 0.0)      # AD directional sweep
        @test Jv ≈ hop.J * hv rtol = 1e-10
        @test hu == hu_before

        pop = setup_operator(PerColorAssemblyStrategy(PolyesterDevice(2)), hint, hdh)
        Jvp = zeros(hn)
        state_jvp!(Jvp, pop, hv, hu, 0.0)
        @test Jvp ≈ Jv rtol = 1e-13
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

# --- Declared request kinds (setup-scoped validation) and preallocated sweeps ---

# A cache whose trait CLAIMS an analytic parameter Jacobian without providing
# the kernel: legal while the kind is undeclared (checked lazily at the entry
# points), loud once declared at setup.
struct BogusClaimCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.assemble_cell!(::ResidualRequest, ::BogusClaimCache, ::KernelArgs) = nothing
FerriteOperators.provides_analytic(::Type{BogusClaimCache}, ::ParameterJacobianKind) = true

@testset "Declared request kinds" begin
    grid = generate_grid(Quadrilateral, (4, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(2)
    n   = ndofs(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

    @testset "trait check is scoped to declared kinds" begin
        @test isnothing(FerriteOperators.validate_element_cache(BogusClaimCache()))
        @test_throws ArgumentError FerriteOperators.validate_element_cache(
            BogusClaimCache(), (ParameterJacobianKind,))
    end

    @testset "declaration is normalized, stored, and does not change results" begin
        # instances normalize to their UnionAll kind type
        op = setup_operator(strategy, SourceDiffusionIntegrator(qrc, :u), dh;
                            requests = (ParameterVJPKind(zeros(n)), TimeSensitivityKind))
        @test op.engine.requests == (ParameterVJPKind, TimeSensitivityKind)
        u = sin.(0.3 .* (1:n))
        λ = ones(n)
        g = zeros(1); parameter_vjp!(g, op, λ, u, 1.7)
        B = zeros(n, 1); update_parameter_jacobian!(B, op, u, 1.7)
        @test g ≈ B' * λ rtol = 1e-12
    end

    @testset "declared inadmissible kinds fail at setup, not first use" begin
        vgrid = generate_grid(Hexahedron, (1, 1, 1))
        vdh = DofHandler(vgrid)
        add!(vdh, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(vdh)
        vint = FerriteOperators.SimpleCondensedLinearViscoelasticity(
            FerriteOperators.MaxwellParameters(), qrc, :u, :εᵛ)
        # condensed state, no analytic StateVJP kernel, no insensitivity declaration
        @test_throws ArgumentError setup_operator(strategy, vint, vdh;
            slots = (:u, :uprev), requests = (StateVJPKind,))
        # time sensitivities stay declarable: the FD escape is a call-time choice
        vop = setup_operator(strategy, vint, vdh;
            slots = (:u, :uprev), requests = (TimeSensitivityKind,))
        @test vop.engine.requests == (TimeSensitivityKind,)
        # kinds made admissible above (analytic kernel / insensitivity) pass setup
        vop2 = setup_operator(strategy, vint, vdh;
            slots = (:u, :uprev), requests = (ParameterJacobianKind, ParameterVJPKind))
        @test vop2.engine.requests == (ParameterJacobianKind, ParameterVJPKind)
    end
end

@testset "Preallocated AD sweeps" begin
    qrc = QuadratureRuleCollection(2)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

    @testset "chunked AD Jacobian (local size > chunk) equals assembled stiffness" begin
        # 27 dofs per cell forces ForwardDiff's chunk mode through the
        # per-worker jacobian config.
        grid = generate_grid(Hexahedron, (1, 1, 1))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron, 2}())
        close!(dh)
        op = setup_operator(strategy, SourceDiffusionIntegrator(qrc, :u), dh)
        u = sin.(0.3 .* (1:ndofs(dh)))
        update_linearization!(op, u, 1.7)
        Kop = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        update_operator!(Kop, nothing)
        @test op.J ≈ Kop.A rtol = 1e-13
    end

    @testset "chunked state VJP matches materialized Jᵀλ" begin
        # 24 dofs per cell: chunk-mode gradient over the preallocated Dual
        # residual buffer.
        grid = generate_grid(Hexahedron, (2, 2, 2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(dh)
        hint = FerriteOperators.SimpleHyperelasticityIntegrator(NeoHookeanState(10.0, 0.3), qrc, :u)
        hop = setup_operator(strategy, hint, dh)
        hn = ndofs(dh)
        hu = 0.05 .* sin.(0.3 .* (1:hn))
        λ = cos.(0.7 .* (1:hn))
        update_linearization!(hop, hu, 0.0)
        g = zeros(hn)
        state_vjp!(g, hop, λ, hu, 0.0)
        @test g ≈ hop.J' * λ rtol = 1e-10
    end

    @testset "state and time sweeps do not allocate per cell" begin
        grid = generate_grid(Quadrilateral, (4, 3))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        n = ndofs(dh)
        op = setup_operator(strategy, SourceDiffusionIntegrator(qrc, :u), dh)
        u = sin.(0.3 .* (1:n)); p = 1.7
        v = cos.(0.11 .* (1:n)); λ = cos.(0.7 .* (1:n))
        Jv = zeros(n); g = zeros(n); r = zeros(n)
        for _ in 1:2   # warmup: compilation + lazy parameter-buffer sizing
            state_jvp!(Jv, op, v, u, p)
            state_vjp!(g, op, λ, u, p)
            time_sensitivity!(g, op, u, 0.9)
            update_linearization!(op, r, u, p)
        end
        # A per-cell allocation regression shows up as ≳10 KiB on 12 cells
        # (measured: 0 B for the sweeps, ~400 B assembler setup for fused J+r).
        @test @allocated(state_jvp!(Jv, op, v, u, p)) == 0
        @test @allocated(state_vjp!(g, op, λ, u, p)) == 0
        @test @allocated(time_sensitivity!(g, op, u, 0.9)) == 0
        @test @allocated(update_linearization!(op, r, u, p)) < 1024
    end
end
