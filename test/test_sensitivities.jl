using FerriteOperators
using Test
import LinearAlgebra: mul!, dot, norm
using SparseArrays
using Polyester

# A v2-native nonlinear element: r(u, p) = ∫ ∇v⋅∇u dΩ − ∫ p v dΩ.
# The scalar source p is the differentiable parameter. Only the residual
# kernel exists — every derivative is exercised through the AD fallback.
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
FerriteOperators.reinit_values!(c::SourceDiffusionCache, cell) = reinit!(c.cv, cell)
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::SourceDiffusionCache, args)
    (; cv) = cache
    uₑ = args.states.u
    q  = args.p
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u - q * shape_value(cv, qp, i)) * dΩ
        end
    end
end

# A v2-native element with EXPLICIT time dependence, read where the contract
# puts time: r(u, ctx) = ∫ ∇v⋅∇u dΩ − t ∫ v dΩ with t = evaluation_time(ctx).
# Hence ∂F/∂t = −∫ v dΩ, the same vector the parameter Jacobian of
# `SourceDiffusionCache` produces. The `analytic` type parameter selects
# between the AD fallback and an analytic ∂F/∂t kernel.
struct TimeSourceDiffusionIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
    analytic::Bool
end
TimeSourceDiffusionIntegrator(qrc, field_name) = TimeSourceDiffusionIntegrator(qrc, field_name, false)
struct TimeSourceDiffusionCache{analytic, CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::TimeSourceDiffusionIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    cv     = CellValues(qr, ip, ip_geo)
    return TimeSourceDiffusionCache{m.analytic, typeof(cv)}(cv)
end
FerriteOperators.duplicate_for_device(device, c::TimeSourceDiffusionCache{a}) where {a} =
    TimeSourceDiffusionCache{a, typeof(c.cv)}(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.reinit_values!(c::TimeSourceDiffusionCache, cell) = reinit!(c.cv, cell)
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::TimeSourceDiffusionCache, args)
    (; cv) = cache
    uₑ = args.states.u
    t  = evaluation_time(args.ctx)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u - t * shape_value(cv, qp, i)) * dΩ
        end
    end
end

# Analytic ∂F/∂t kernel, selected per cache; the counter proves the selection.
const ANALYTIC_TS_CALLS = Ref(0)
FerriteOperators.provides_analytic(::Type{<:TimeSourceDiffusionCache{true}}, ::FerriteOperators.TimeSensitivityKind) = true
function FerriteOperators.assemble_cell!(req::TimeSensitivityRequest, cache::TimeSourceDiffusionCache{true}, args)
    ANALYTIC_TS_CALLS[] += 1
    (; cv) = cache
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        for i in 1:getnbasefunctions(cv)
            req.g[i] -= shape_value(cv, qp, i) * dΩ
        end
    end
end

# Any context works for a stationary sweep; ∂F/∂t only reads the time.
stationary_ctx(t) = TimeIntegrationContext(t, 1.0, 1.0)

# Shared condensed-viscoelasticity fixture: single hex, vector displacement
# plus hidden per-QP εᵛ, slots (:u, :uprev).
function setup_visco_operator(strategy, qrc; kwargs...)
    vgrid = generate_grid(Hexahedron, (1, 1, 1))
    vdh = DofHandler(vgrid)
    add!(vdh, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(vdh)
    vint = FerriteOperators.SimpleCondensedLinearViscoelasticity(
        FerriteOperators.MaxwellParameters(), qrc, :u, :εᵛ)
    return setup_operator(strategy, vint, vdh; slots = (:u, :uprev), kwargs...)
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
    r0 = zeros(n); evaluate!(op, r0, u, 0.0)
    r1 = zeros(n); evaluate!(op, r1, u, 1.0)
    b = r0 .- r1

    @testset "residual structure" begin
        rp = zeros(n); evaluate!(op, rp, u, p)
        @test rp ≈ r0 .- p .* b rtol = 1e-13
    end

    @testset "parameter jacobian" begin
        B = zeros(n, 1)
        update_parameter_jacobian!(B, op, u, p)
        @test vec(B) ≈ -b rtol = 1e-12

        # finite-difference cross-check
        h = 1e-6
        rplus = zeros(n); evaluate!(op, rplus, u, p + h)
        rminus = zeros(n); evaluate!(op, rminus, u, p - h)
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

    @testset "time sensitivity seeds through the context" begin
        # The element reads `evaluation_time(args.ctx)`, so ∂F/∂t is the
        # source vector −b — and the AD sweep must find it there.
        top = setup_operator(strategy, TimeSourceDiffusionIntegrator(qrc, :u), dh)
        g = zeros(n)
        time_sensitivity!(g, top, (u = u,), nothing, stationary_ctx(0.9))
        @test g ≈ -b rtol = 1e-12

        # the FD method differences primal evaluations at perturbed contexts
        gfd = zeros(n)
        time_sensitivity!(gfd, top, (u = u,), nothing, stationary_ctx(0.9); method = FiniteDifferenceSensitivity())
        @test gfd ≈ g rtol = 1e-6
    end

    @testset "time sensitivity without a context is loud" begin
        top = setup_operator(strategy, TimeSourceDiffusionIntegrator(qrc, :u), dh)
        @test_throws ArgumentError time_sensitivity!(zeros(n), top, (u = u,), nothing, nothing)
        @test_throws ArgumentError time_sensitivity!(zeros(n), top, (u = u,), nothing, nothing;
                                                     method = FiniteDifferenceSensitivity())
    end

    @testset "fused J+r via AD fallback" begin
        # SourceDiffusionCache declares no analytic Jacobian, so this exercises
        # the AD-fused JacobianResidualKind branch (J and primal r in one sweep).
        r_fused = zeros(n)
        update_linearization!(op, r_fused, u, p)
        r_ref = zeros(n); evaluate!(op, r_ref, u, p)
        @test r_fused ≈ r_ref rtol = 1e-13

        v = cos.(0.11 .* (1:n))
        h = 1e-6
        rplus = zeros(n); evaluate!(op, rplus, u .+ h .* v, p)
        rminus = zeros(n); evaluate!(op, rminus, u .- h .* v, p)
        @test op.J * v ≈ (rplus .- rminus) ./ 2h rtol = 1e-6
    end

    @testset "parallel strategy consistency" begin
        pstrategy = PerColorAssemblyStrategy(PolyesterDevice(2))
        pop = setup_operator(pstrategy, SourceDiffusionIntegrator(qrc, :u), dh)
        Bs = zeros(n, 1); update_parameter_jacobian!(Bs, op, u, p)
        Bp = zeros(n, 1); update_parameter_jacobian!(Bp, pop, u, p)
        @test Bs ≈ Bp rtol = 1e-13
        tops = setup_operator(strategy, TimeSourceDiffusionIntegrator(qrc, :u), dh)
        topp = setup_operator(pstrategy, TimeSourceDiffusionIntegrator(qrc, :u), dh)
        gs = zeros(n); time_sensitivity!(gs, tops, (u = u,), nothing, stationary_ctx(0.9))
        gp = zeros(n); time_sensitivity!(gp, topp, (u = u,), nothing, stationary_ctx(0.9))
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
        r = zeros(n); evaluate!(dop, r, u, nothing)
        update_linearization!(dop, u, nothing)
        @test r ≈ dop.J * u rtol = 1e-12
    end

    @testset "condensed operators are rejected loudly" begin
        vop = setup_visco_operator(strategy, qrc)
        vu = zeros(unknown_size(vop))
        @test_throws ArgumentError update_parameter_jacobian!(zeros(residual_size(vop), 1), vop, vu, 1.0)
        @test_throws ArgumentError parameter_vjp!(zeros(1), vop, zeros(residual_size(vop)), vu, 1.0)
        @test_throws ArgumentError time_sensitivity!(zeros(residual_size(vop)), vop, (u = vu,), 1.0, stationary_ctx(0.5))
    end

    @testset "undeclared slots error loudly" begin
        @test_throws ArgumentError update_linearization!(op, zeros(n), (u = u, uprev = copy(u)), p, nothing)
    end

    @testset "sensitivity sweeps never mutate u" begin
        ucopy = copy(u)
        update_parameter_jacobian!(zeros(n, 1), op, u, p)
        parameter_vjp!(zeros(1), op, ones(n), u, p)
        top = setup_operator(strategy, TimeSourceDiffusionIntegrator(qrc, :u), dh)
        time_sensitivity!(zeros(n), top, (u = u,), nothing, stationary_ctx(0.9))
        @test u == ucopy
    end
end

# --- Sensitivity admissibility refinements and method selection ---

# Condensed admissibility: analytic parameter kernel on the viscoelastic cache
# (trivially zero — the material is parameter-independent), and an
# insensitivity declaration for the VJP kind.
FerriteOperators.provides_analytic(::Type{<:FerriteOperators.SimpleCondensedLinearViscoelasticityCache}, ::FerriteOperators.ParameterJacobianKind) = true
FerriteOperators.assemble_cell!(req::ParameterJacobianRequest, ::FerriteOperators.SimpleCondensedLinearViscoelasticityCache, args) = nothing
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
    aop = setup_operator(strategy, TimeSourceDiffusionIntegrator(qrc, :u, true), dh)
    dop = setup_operator(strategy, TimeSourceDiffusionIntegrator(qrc, :u), dh)
    u = sin.(0.3 .* (1:n))
    ctx = stationary_ctx(0.9)

    @testset "analytic time kernel is selected and agrees with the AD fallback" begin
        ANALYTIC_TS_CALLS[] = 0
        g_analytic = zeros(n)
        time_sensitivity!(g_analytic, aop, (u = u,), nothing, ctx)
        @test ANALYTIC_TS_CALLS[] == getncells(grid)
        # the same element without the analytic kernel, differentiated
        g_ad = zeros(n)
        time_sensitivity!(g_ad, dop, (u = u,), nothing, ctx)
        @test g_analytic ≈ g_ad rtol = 1e-12
    end

    @testset "FD method agrees with the derivative (and bypasses kernels)" begin
        ANALYTIC_TS_CALLS[] = 0
        g_fd = zeros(n)
        time_sensitivity!(g_fd, aop, (u = u,), nothing, ctx; method = FiniteDifferenceSensitivity())
        @test ANALYTIC_TS_CALLS[] == 0     # operator-level differencing, kernels untouched
        g_ref = zeros(n)
        time_sensitivity!(g_ref, aop, (u = u,), nothing, ctx)
        @test g_fd ≈ g_ref rtol = 1e-6
    end

    @testset "condensed: analytic parameter kernel is admissible" begin
        vop = setup_visco_operator(strategy, qrc)
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
        @test_throws ArgumentError time_sensitivity!(zeros(residual_size(vop)), vop, (u = vu, uprev = vuprev), 1.0, vctx)
        # … but FD is admissible: primal evaluations on a protected copy.
        vu_before = copy(vu)
        g = zeros(residual_size(vop))
        time_sensitivity!(g, vop, (u = vu, uprev = vuprev), 1.0, vctx; method = FiniteDifferenceSensitivity())
        @test vu == vu_before                       # trial write-back never leaked
        @test norm(g) < 1e-8                        # the residual reads γ̃, not the evaluation time
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
    grid = generate_grid(Hexahedron, (2, 2, 2))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(dh)
    qrc  = QuadratureRuleCollection(2)
    n    = ndofs(dh)

    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    integrator = FerriteOperators.SimpleHyperelasticityIntegrator(NeoHookeanState(10.0, 0.3), qrc, :u)
    op = setup_operator(strategy, integrator, dh)

    u = 0.05 .* sin.(0.3 .* (1:n))

    # analytic v2 Jacobian against a central finite difference of the residual
    update_linearization!(op, u, 0.0)
    v = cos.(0.11 .* (1:n))
    h = 1e-6
    rplus = zeros(n); evaluate!(op, rplus, u .+ h .* v, 0.0)
    rminus = zeros(n); evaluate!(op, rminus, u .- h .* v, 0.0)
    @test op.J * v ≈ (rplus .- rminus) ./ 2h rtol = 1e-5

    # fused J+r path must agree with the split calls
    r_fused = zeros(n)
    update_linearization!(op, r_fused, u, 0.0)
    r_split = zeros(n); evaluate!(op, r_split, u, 0.0)
    @test r_fused ≈ r_split rtol = 1e-14
end

# --- Declared request kinds (setup-scoped validation) and preallocated sweeps ---

# A cache whose trait CLAIMS an analytic parameter Jacobian without providing
# the kernel: legal while the kind is undeclared (checked lazily at the entry
# points), loud once declared at setup.
struct BogusClaimCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.assemble_cell!(::ResidualRequest, ::BogusClaimCache, args) = nothing
FerriteOperators.reinit_values!(::BogusClaimCache, cell) = nothing
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
        # condensed state, no analytic StateVJP kernel, no insensitivity declaration
        @test_throws ArgumentError setup_visco_operator(strategy, qrc; requests = (StateVJPKind,))
        # time sensitivities stay declarable: the FD escape is a call-time choice
        vop = setup_visco_operator(strategy, qrc; requests = (TimeSensitivityKind,))
        @test vop.engine.requests == (TimeSensitivityKind,)
        # kinds made admissible above (analytic kernel / insensitivity) pass setup
        vop2 = setup_visco_operator(strategy, qrc; requests = (ParameterJacobianKind, ParameterVJPKind))
        @test vop2.engine.requests == (ParameterJacobianKind, ParameterVJPKind)
    end
end

# --- The kernel-args channel protocol at setup ---

# Two caches differing only in how open their residual kernel's args parameter
# is, plus an args family unrelated to `KernelArgs` that carries the same
# channels.
struct LooseArgsCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.assemble_cell!(::ResidualRequest, ::LooseArgsCache, args) = nothing
FerriteOperators.reinit_values!(::LooseArgsCache, cell) = nothing

struct PinnedArgsCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.assemble_cell!(::ResidualRequest, ::PinnedArgsCache, args::KernelArgs) = nothing
FerriteOperators.reinit_values!(::PinnedArgsCache, cell) = nothing

struct ForeignArgs{S, C, P, Sc, Cx}
    states::S
    cell::C
    p::P
    scratch::Sc
    ctx::Cx
end

@testset "Kernel-args channel protocol" begin
    @testset "validation queries the operator family's own args type" begin
        @test isnothing(FerriteOperators.validate_element_cache(LooseArgsCache()))
        @test isnothing(FerriteOperators.validate_element_cache(LooseArgsCache(), (), ForeignArgs))
        # the pinned kernel does not exist as far as another args family is concerned
        @test_throws ArgumentError FerriteOperators.validate_element_cache(
            PinnedArgsCache(), (), ForeignArgs)
    end

    @testset "pinned kernels earn an advisory warning, loose ones do not" begin
        @test_logs FerriteOperators.validate_element_cache(LooseArgsCache())
        @test_logs (:warn, r"pins its residual kernel") FerriteOperators.validate_element_cache(PinnedArgsCache())
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
        # ctx-seeded ∂F/∂t: the context rebuild per cell must stay on the stack
        top = setup_operator(strategy, TimeSourceDiffusionIntegrator(qrc, :u), dh)
        u = sin.(0.3 .* (1:n)); p = 1.7
        ctx = stationary_ctx(0.9)
        v = cos.(0.11 .* (1:n)); λ = cos.(0.7 .* (1:n))
        Jv = zeros(n); g = zeros(n); r = zeros(n)
        states = (u = u,)
        for _ in 1:2   # warmup: compilation + lazy parameter-buffer sizing
            state_jvp!(Jv, op, v, u, p)
            state_vjp!(g, op, λ, u, p)
            time_sensitivity!(g, top, states, nothing, ctx)
            update_linearization!(op, r, u, p)
        end
        # A per-cell allocation regression shows up as ≳10 KiB on 12 cells
        # (measured: 0 B for the sweeps, ~400 B assembler setup for fused J+r).
        @test @allocated(state_jvp!(Jv, op, v, u, p)) == 0
        @test @allocated(state_vjp!(g, op, λ, u, p)) == 0
        @test @allocated(time_sensitivity!(g, top, states, nothing, ctx)) == 0
        @test @allocated(update_linearization!(op, r, u, p)) < 1024
    end
end
