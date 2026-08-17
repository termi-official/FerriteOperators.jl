using FerriteOperators
using FerriteOperatorsExampleElements
using Test

include(joinpath(@__DIR__, "fixture_elements.jl"))

# Diffusion with a scalar source; the analytic Jacobian kernel can be scaled
# to emulate a WRONG analytic implementation (exact for scale = 1).
struct CheckerDiffusionIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
    jac_scale::Float64
end
struct CheckerDiffusionCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
    jac_scale::Float64
end
function FerriteOperators.setup_element_cache(m::CheckerDiffusionIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return CheckerDiffusionCache(CellValues(qr, ip, ip_geo), m.jac_scale)
end
FerriteOperators.duplicate_for_device(device, c::CheckerDiffusionCache) =
    CheckerDiffusionCache(FerriteOperators.duplicate_for_device(device, c.cv), c.jac_scale)
FerriteOperators.reinit_values!(c::CheckerDiffusionCache, cell) = reinit!(c.cv, cell)
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::CheckerDiffusionCache, args)
    (; cv) = cache
    uₑ = args.states.u
    q  = _source(args.p, cellid(args.cell))
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u - q * shape_value(cv, qp, i)) * dΩ
        end
    end
end
FerriteOperators.provides_analytic(::Type{<:CheckerDiffusionCache}, ::JacobianKind) = true
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, cache::CheckerDiffusionCache, args)
    (; cv, jac_scale) = cache
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        for i in 1:getnbasefunctions(cv)
            ∇Nᵢ = shape_gradient(cv, qp, i)
            for j in 1:getnbasefunctions(cv)
                req.K[i, j] += jac_scale * (∇Nᵢ ⋅ shape_gradient(cv, qp, j)) * dΩ
            end
        end
    end
end

# Parameter bag with a static per-cell field and ONE differentiable scalar:
# θ = (E,), the field never enters any parameter Jacobian.
struct SplitParams{V <: AbstractVector, T}
    field::V
    E::T
end
_source(p::Real, cellid) = p
_source(p::SplitParams, cellid) = p.E * p.field[cellid]
FerriteOperators.parameter_vector(p::SplitParams) = [p.E]
FerriteOperators.rebuild_parameters(p::SplitParams, θ) = SplitParams(p.field, θ[1])

function setup_checker_operator(jac_scale)
    grid = generate_grid(Quadrilateral, (4, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op = setup_operator(strategy, CheckerDiffusionIntegrator(QuadratureRuleCollection(2), :u, jac_scale), dh)
    return op, dh
end

# Same diffusion element, but with the source scaled by the evaluation time —
# a residual with genuine ∂F/∂t through the context channel.
struct TimedCheckerIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct TimedCheckerCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::TimedCheckerIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return TimedCheckerCache(CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::TimedCheckerCache) =
    TimedCheckerCache(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.reinit_values!(c::TimedCheckerCache, cell) = reinit!(c.cv, cell)
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::TimedCheckerCache, args)
    (; cv) = cache
    uₑ = args.states.u
    q  = args.p * evaluation_time(args.ctx)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u - q * shape_value(cv, qp, i)) * dΩ
        end
    end
end

# Transient diffusion, r(u, u̇) = ∫ (u̇ v + ∇u⋅∇v) dΩ, with a hand-fused scheme
# matrix as its analytic weighted kernel. The cache's `params` is a `w_scale`
# that detunes that kernel exactly as `jac_scale` detunes the Jacobian above
# (exact for scale = 1).
const WeightedCheckerCache = CVCache{:weighted_checker}
WeightedCheckerIntegrator(qrc, field_name, w_scale) =
    CVIntegrator{:weighted_checker}(qrc, field_name, w_scale)

function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::WeightedCheckerCache, args)
    transient_diffusion_residual!(req.r, cache, args)
end
FerriteOperators.provides_analytic(::Type{<:WeightedCheckerCache}, ::WeightedJacobianKind) = true
function FerriteOperators.assemble_cell!(req::WeightedJacobianRequest, cache::WeightedCheckerCache, args)
    analytic_weighted_jacobian!(req.K, cache.cv, req.weights, cache.params)
end

function setup_weighted_checker_operator(w_scale)
    grid = generate_grid(Quadrilateral, (4, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op = setup_operator(strategy, WeightedCheckerIntegrator(QuadratureRuleCollection(2), :u, w_scale), dh;
                        slots = (:u, :du))
    return op, dh
end

@testset "Derivative checker" begin
    @testset "correct analytic Jacobian and AD sensitivities pass" begin
        op, dh = setup_checker_operator(1.0)
        u = sin.(0.3 .* (1:ndofs(dh)))
        res = check_derivatives(op, u, 1.7)
        @test res.passed
        # every check but the ones needing a context resp. weights
        @test res.checks.time_sensitivity.skipped !== nothing
        @test occursin("seed through ctx", res.checks.time_sensitivity.skipped)
        @test occursin("no `weights` given", res.checks.weighted_jacobian.skipped)
        @test all(c.skipped === nothing for (name, c) in pairs(res.checks)
                  if name ∉ (:time_sensitivity, :weighted_jacobian, :weighted_jacobian_routes))
        @test res.checks.jacobian.err < 1e-7
        @test res.checks.parameter_jacobian.err < 1e-7
    end

    @testset "the time check runs with a context" begin
        grid = generate_grid(Quadrilateral, (4, 3))
        dh   = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
        op = setup_operator(strategy, TimedCheckerIntegrator(QuadratureRuleCollection(2), :u), dh)
        u = sin.(0.3 .* (1:ndofs(dh)))
        res = check_derivatives(op, (u = u,), 1.7, TimeIntegrationContext(0.9, 0.1, 0.1))
        @test res.passed
        @test res.checks.time_sensitivity.skipped === nothing
        @test res.checks.time_sensitivity.err < 1e-6
    end

    @testset "wrong analytic Jacobian is detected" begin
        op, dh = setup_checker_operator(1.01)
        u = sin.(0.3 .* (1:ndofs(dh)))
        res = check_derivatives(op, u, 1.7)
        @test !res.passed
        @test !res.checks.jacobian.passed
        # the residual-derived paths are untouched by the broken Jacobian kernel
        @test res.checks.parameter_jacobian.passed
        @test res.checks.state_jvp.passed
    end

    @testset "weighted Jacobian: fused vs composed vs FD" begin
        Δt = 0.25
        ctx = TimeIntegrationContext(1.0, Δt, Δt)
        weights = (u = 1.0, du = 1 / (0.5 * Δt))

        op, dh = setup_weighted_checker_operator(1.0)
        n = ndofs(dh)
        states = (u = sin.(0.3 .* (1:n)), du = cos.(0.2 .* (1:n)))

        res = check_derivatives(op, states, nothing, ctx; weights)
        @test res.passed
        @test res.checks.weighted_jacobian.skipped === nothing
        @test res.checks.weighted_jacobian.err < 1e-6
        @test res.checks.weighted_jacobian_routes.skipped === nothing
        @test res.checks.weighted_jacobian_routes.err < 1e-10

        # without weights there is nothing to check
        skipped = check_derivatives(op, states, nothing, ctx)
        @test occursin("no `weights` given", skipped.checks.weighted_jacobian.skipped)
        @test skipped.checks.weighted_jacobian_routes.skipped !== nothing

        # complex weights never reach the real finite-difference referee
        cplx = check_derivatives(op, states, nothing, ctx; weights = (u = 1.0 + 0im, du = 2.0 + 0im))
        @test occursin("complex weights", cplx.checks.weighted_jacobian.skipped)

        # a reconstructed slot cannot be perturbed independently
        rate = check_derivatives(op, (u = states.u, du = AffineRate(1 / Δt, states.u)), nothing, ctx; weights)
        @test occursin("reconstructed source", rate.checks.weighted_jacobian.skipped)
    end

    @testset "wrong analytic weighted kernel is detected" begin
        Δt = 0.25
        ctx = TimeIntegrationContext(1.0, Δt, Δt)
        weights = (u = 1.0, du = 1 / (0.5 * Δt))
        op, dh = setup_weighted_checker_operator(1.01)
        n = ndofs(dh)
        states = (u = sin.(0.3 .* (1:n)), du = cos.(0.2 .* (1:n)))

        res = check_derivatives(op, states, nothing, ctx; weights)
        @test !res.passed
        @test !res.checks.weighted_jacobian.passed
        # the composed route derives from the residual, so the routes disagree too
        @test !res.checks.weighted_jacobian_routes.passed
        # the residual-derived paths are untouched by the broken weighted kernel
        @test res.checks.jacobian.passed
        @test res.checks.state_jvp.passed
    end

    @testset "differentiable/static parameter split" begin
        op, dh = setup_checker_operator(1.0)
        n = ndofs(dh)
        u = sin.(0.3 .* (1:n))
        p = SplitParams(collect(range(0.5, 2.0; length = getncells(Ferrite.get_grid(dh)))), 1.3)

        # θ covers only E: the parameter Jacobian has ONE column.
        @test length(parameter_vector(p)) == 1
        B = zeros(n, 1)
        update_parameter_jacobian!(B, op, u, p)
        @test !iszero(B)
        @test_throws DimensionMismatch update_parameter_jacobian!(zeros(n, 2), op, u, p)

        res = check_derivatives(op, (u = u,), p)
        @test res.checks.parameter_jacobian.passed
        @test res.checks.parameter_vjp.passed
        @test res.checks.jacobian.passed
    end

    @testset "condensed operator: consistent tangent vs FD, inadmissible kinds skipped" begin
        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
        vop = visco_testbed(strategy, QuadratureRuleCollection(2)).op
        vu = 1e-3 .* sin.(0.2 .* (1:unknown_size(vop)))
        vstates = (u = vu, uprev = zeros(unknown_size(vop)))
        vctx = TimeIntegrationContext(0.0, 0.1, 0.1)

        vu_before = copy(vu)
        res = check_derivatives(vop, vstates, MaxwellParameters(), vctx)
        @test res.passed
        @test res.checks.jacobian.passed              # condensed tangent vs FD through local solves
        @test res.checks.jacobian.skipped === nothing
        @test res.checks.state_jvp.skipped !== nothing    # condensed: state actions unsupported
        @test res.checks.parameter_jacobian.skipped !== nothing  # no parameter_vector for MaxwellParameters
        @test res.checks.time_sensitivity.skipped !== nothing     # condensed: AD-from-residual inadmissible
        @test vu == vu_before                         # caller state protected
    end
end
