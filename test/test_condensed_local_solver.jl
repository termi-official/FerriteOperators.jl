using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using Polyester
using LinearAlgebra: norm
using SparseArrays: nnz

include(joinpath(@__DIR__, "fixture_elements.jl"))

# Reference root of the element-local stage problem
#     q = q₀ + γ̃ σ|σ|ⁿ⁻¹/η,  σ = α(u − q)
# by bisection, independent of the element's Newton iteration.
function reference_internal_state(mat, u, q₀, γ̃)
    (; α, η, n) = mat
    R(q) = q - q₀ - γ̃ * (α * (u - q)) * abs(α * (u - q))^(n - 1) / η
    lo, hi = min(q₀, u), max(q₀, u)
    for _ in 1:200
        mid = (lo + hi) / 2
        R(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

####################################
## Doubles for the local-conditions seam
####################################

# A condensed element whose LOCAL model depends explicitly on the evaluation
# time — the case that separates the frozen-q ∂F/∂t partial from the total:
#
#   r(u, q) = ∫ (∇u ⋅ ∇δu + α (u − q) δu) dΩ
#   L(q)    = q − qprev − (γ̃/τ) (a(t) u − q),        a(t) = 1 + sin(t)
#
# linear in q, so the local solve is closed form. θ = (α, τ): ∂F/∂τ|_q and
# ∂F/∂t|_q are both ZERO, so whatever those sensitivities report is the
# ∂F/∂q·dq/d· correction alone. It declares the mandatory residual and its
# analytic `Consistent` tangent, and NO analytic parameter or time kernel —
# those are exactly what `local_conditions!` has to serve.
struct TimedRelaxationParameters{T}
    α::T
    τ::T
end
TimedRelaxationParameters(; α = 0.8, τ = 1.3) = TimedRelaxationParameters(promote(α, τ)...)
FerriteOperators.parameter_vector(p::TimedRelaxationParameters) = [p.α, p.τ]
FerriteOperators.rebuild_parameters(::TimedRelaxationParameters, θ) = TimedRelaxationParameters(θ[1], θ[2])

_tr_amplitude(t) = 1 + sin(t)

struct TimedRelaxationIntegrator <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    params::TimedRelaxationParameters{Float64}
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct TimedRelaxationCache{CV <: CellValues} <: FerriteOperators.AbstractVolumetricElementCache
    params::TimedRelaxationParameters{Float64}
    cv::CV
    correctors::ItemStates{Vector{Float64}}
    ncalls::Base.RefValue{Int}      # `local_conditions!` calls, so a test can see the route run
end
FerriteOperators.setup_element_cache(m::TimedRelaxationIntegrator, sdh::SubDofHandler) =
    TimedRelaxationCache(m.params,
        CellValues(getquadraturerule(m.qrc, sdh), Ferrite.getfieldinterpolation(sdh, m.field_name),
                   FerriteOperators.geometric_subdomain_interpolation(sdh)),
        ItemStates{Vector{Float64}}(getncells(Ferrite.get_grid(sdh.dh))), Ref(0))
# The stores and the call counter are shared, `ItemStates`' own rule.
FerriteOperators.duplicate_for_device(device, c::TimedRelaxationCache) =
    TimedRelaxationCache(c.params, FerriteOperators.duplicate_for_device(device, c.cv), c.correctors, c.ncalls)
FerriteOperators.reinit_values!(c::TimedRelaxationCache, cell) = reinit!(c.cv, cell)
Ferrite.getnquadpoints(c::TimedRelaxationCache) = getnquadpoints(c.cv)
FerriteOperators.has_internal_state(::Type{<:TimedRelaxationCache}) = true
FerriteOperators.get_number_of_internal_dofs_per_element(m, c::TimedRelaxationCache, sdh) =
    [getnquadpoints(c.cv) for _ in sdh.cellset]
FerriteOperators.provides_analytic(::Type{<:TimedRelaxationCache}, ::JacobianKind{:u}) = true
FerriteOperators.provides_analytic(::Type{<:TimedRelaxationCache}, ::JacobianResidualKind) = true
FerriteOperators.invalidate_correctors!(c::TimedRelaxationCache) =
    (FerriteOperators.invalidate_item_states!(c.correctors); nothing)

_tr_params(c::TimedRelaxationCache, ::Nothing) = c.params
_tr_params(::TimedRelaxationCache, p) = p

function FerriteOperators.condense_cell!(c::TimedRelaxationCache, args::CellArgs, weights::NamedTuple)
    (; τ) = _tr_params(c, args.p)
    cv, γ̃, a = c.cv, stage_scaling(args.ctx), _tr_amplitude(evaluation_time(args.ctx))
    k = γ̃ / τ
    w = get(weights, :u, 1.0)
    dqdu = zeros(getnquadpoints(cv))
    for qp in 1:getnquadpoints(cv)
        u = function_value(cv, qp, args.states.u)
        args.states.q[qp] = (args.states.qprev[qp] + k * a * u) / (1 + k)
        dqdu[qp] = w * k * a / (1 + k)
    end
    FerriteOperators.set_item_state!(c.correctors, cellid(args.cell), dqdu)
    return CondensationReport(true, getnquadpoints(cv), 0, 0, cellid(args.cell), 0, 0.0, 1.0)
end

function FerriteOperators.local_conditions!(L, c::TimedRelaxationCache, args::CellArgs)
    (; τ) = _tr_params(c, args.p)
    cv, γ̃, a = c.cv, stage_scaling(args.ctx), _tr_amplitude(evaluation_time(args.ctx))
    c.ncalls[] += 1
    for qp in 1:getnquadpoints(cv)
        u = function_value(cv, qp, args.states.u)
        L[qp] = args.states.q[qp] - args.states.qprev[qp] - (γ̃ / τ) * (a * u - args.states.q[qp])
    end
    return L
end

const _TRJacobianLike = Union{JacobianRequest{:u, Consistent}, JacobianResidualRequest{Consistent}}
for R in (:ResidualRequest, :(JacobianRequest{:u, Consistent}), :(JacobianResidualRequest{Consistent}))
    @eval FerriteOperators.assemble_cell!(req::$R, c::TimedRelaxationCache, args::CellArgs) = _tr_assemble!(req, c, args)
end
function _tr_assemble!(req, c::TimedRelaxationCache, args::CellArgs)
    (; α) = _tr_params(c, args.p)
    cv = c.cv
    dqdu = req isa _TRJacobianLike ? FerriteOperators.item_state(c.correctors, cellid(args.cell)) : nothing
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        u  = function_value(cv, qp, args.states.u)
        ∇u = function_gradient(cv, qp, args.states.u)
        q  = args.states.q[qp]
        if req isa Union{ResidualRequest, JacobianResidualRequest}
            for i in 1:getnbasefunctions(cv)
                req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u + α * (u - q) * shape_value(cv, qp, i)) * dΩ
            end
        end
        if req isa _TRJacobianLike
            ∂σ∂u = α * (1 - dqdu[qp])
            for i in 1:getnbasefunctions(cv), j in 1:getnbasefunctions(cv)
                req.K[i, j] += (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j) +
                                ∂σ∂u * shape_value(cv, qp, i) * shape_value(cv, qp, j)) * dΩ
            end
        end
    end
end

function timed_relaxation_testbed(strategy, qrc, dims = (2, 2); params = TimedRelaxationParameters())
    grid = generate_grid(Quadrilateral, dims)
    dh = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
    op = setup_operator(strategy, TimedRelaxationIntegrator(params, qrc, :u), dh; slots = (:u, :q, :qprev))
    return (; op, dh, grid)
end

# The test-only wrapper pattern: everything of the power-law cache forwarded
# EXCEPT its analytic `ParameterJacobianKind` claim, plus the local conditions
# it never needed. The wrapper's ∂F/∂θ therefore has to come out of the
# generic route, and the element it wraps is the reference for what that route
# must produce.
struct HiddenAnalyticCache{C} <: FerriteOperators.AbstractVolumetricElementCache
    inner::C
    ncalls::Base.RefValue{Int}
end
FerriteOperators.duplicate_for_device(device, c::HiddenAnalyticCache) =
    HiddenAnalyticCache(FerriteOperators.duplicate_for_device(device, c.inner), c.ncalls)
FerriteOperators.query_cell_parameters(c::HiddenAnalyticCache, cell, p) =
    FerriteOperators.query_cell_parameters(c.inner, cell, p)
FerriteOperators.reinit_values!(c::HiddenAnalyticCache, cell) = FerriteOperators.reinit_values!(c.inner, cell)
FerriteOperators.reinit_values!(c::HiddenAnalyticCache, cell, kind) = FerriteOperators.reinit_values!(c.inner, cell, kind)
Ferrite.getnquadpoints(c::HiddenAnalyticCache) = getnquadpoints(c.inner)
FerriteOperators.has_internal_state(::Type{<:HiddenAnalyticCache}) = true
FerriteOperators.get_number_of_internal_dofs_per_element(m, c::HiddenAnalyticCache, sdh) =
    FerriteOperators.get_number_of_internal_dofs_per_element(m, c.inner, sdh)
FerriteOperators.condense_cell!(c::HiddenAnalyticCache, args, weights) =
    FerriteOperators.condense_cell!(c.inner, args, weights)
FerriteOperators.invalidate_correctors!(c::HiddenAnalyticCache) = FerriteOperators.invalidate_correctors!(c.inner)
FerriteOperators.provides_analytic(::Type{<:HiddenAnalyticCache}, ::JacobianKind{:u}) = true
FerriteOperators.provides_analytic(::Type{<:HiddenAnalyticCache}, ::JacobianResidualKind) = true
for R in (:ResidualRequest, :(JacobianRequest{:u, Consistent}), :(JacobianResidualRequest{Consistent}))
    @eval FerriteOperators.assemble_cell!(req::$R, c::HiddenAnalyticCache, args::CellArgs) =
        FerriteOperators.assemble_cell!(req, c.inner, args)
end
function FerriteOperators.local_conditions!(L, c::HiddenAnalyticCache, args::CellArgs)
    mat = args.p === nothing ? c.inner.material_parameters : args.p
    cv, γ̃ = c.inner.cv, stage_scaling(args.ctx)
    c.ncalls[] += 1
    for qp in 1:getnquadpoints(cv)
        u = function_value(cv, qp, args.states.u)
        σ = mat.α * (u - args.states.q[qp])
        L[qp] = args.states.q[qp] - args.states.qprev[qp] - γ̃ * σ * abs(σ)^(mat.n - 1) / mat.η
    end
    return L
end

struct HiddenAnalyticIntegrator{I} <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    inner::I
    ncalls::Base.RefValue{Int}
end
FerriteOperators.setup_element_cache(m::HiddenAnalyticIntegrator, sdh::SubDofHandler) =
    HiddenAnalyticCache(FerriteOperators.setup_element_cache(m.inner, sdh), m.ncalls)

# The same wrapper WITHOUT the hook — the case the admissibility rule still
# has to refuse, since neither of the other two branches applies either.
struct NoHookCache{C} <: FerriteOperators.AbstractVolumetricElementCache
    inner::C
end
FerriteOperators.duplicate_for_device(device, c::NoHookCache) =
    NoHookCache(FerriteOperators.duplicate_for_device(device, c.inner))
FerriteOperators.query_cell_parameters(c::NoHookCache, cell, p) =
    FerriteOperators.query_cell_parameters(c.inner, cell, p)
FerriteOperators.reinit_values!(c::NoHookCache, cell) = FerriteOperators.reinit_values!(c.inner, cell)
FerriteOperators.reinit_values!(c::NoHookCache, cell, kind) = FerriteOperators.reinit_values!(c.inner, cell, kind)
Ferrite.getnquadpoints(c::NoHookCache) = getnquadpoints(c.inner)
FerriteOperators.has_internal_state(::Type{<:NoHookCache}) = true
FerriteOperators.get_number_of_internal_dofs_per_element(m, c::NoHookCache, sdh) =
    FerriteOperators.get_number_of_internal_dofs_per_element(m, c.inner, sdh)
FerriteOperators.condense_cell!(c::NoHookCache, args, weights) =
    FerriteOperators.condense_cell!(c.inner, args, weights)
FerriteOperators.invalidate_correctors!(c::NoHookCache) = FerriteOperators.invalidate_correctors!(c.inner)
FerriteOperators.provides_analytic(::Type{<:NoHookCache}, ::JacobianKind{:u}) = true
FerriteOperators.provides_analytic(::Type{<:NoHookCache}, ::JacobianResidualKind) = true
for R in (:ResidualRequest, :(JacobianRequest{:u, Consistent}), :(JacobianResidualRequest{Consistent}))
    @eval FerriteOperators.assemble_cell!(req::$R, c::NoHookCache, args::CellArgs) =
        FerriteOperators.assemble_cell!(req, c.inner, args)
end

struct NoHookIntegrator{I} <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    inner::I
end
FerriteOperators.setup_element_cache(m::NoHookIntegrator, sdh::SubDofHandler) =
    NoHookCache(FerriteOperators.setup_element_cache(m.inner, sdh))

@testset "Condensed element with a nonlinear local solve" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters(κ = 1.0, α = 1.0, η = 1.0, n = 3.0)

    @testset "condense_internal! hits the root and the pure assembly is exact" begin
        tb  = relaxation_testbed(strategy, qrc; material = mat)
        op, dh, grid = tb.op, tb.dh, tb.grid
        γ̃   = 0.5
        ctx = TimeIntegrationContext(0.0, γ̃, γ̃)

        # A constant field has the same value at every quadrature point, so
        # every local problem has the same known root.
        u = zeros(unknown_size(op)); view(u, 1:ndofs(dh)) .= 1.0
        uprev = zeros(unknown_size(op))
        states = condensed_states(u, uprev)
        r = zeros(residual_size(op))

        report = condense_internal!(op, states, nothing, ctx)
        @test report.converged
        evaluate!(op, r, states, nothing, ctx)   # pure: reads the q condense_internal! wrote

        qref = reference_internal_state(mat, 1.0, 0.0, γ̃)
        q    = view(u, (ndofs(dh)+1):unknown_size(op))   # the [ū; q] tail write-back
        @test all(qi -> isapprox(qi, qref; atol = 1e-11), q)
        # ∇u ≡ 0, so the residual is the exchange term only and sums to α(u−q)|Ω|.
        @test sum(r) ≈ mat.α * (1.0 - qref) * 4.0 rtol = 1e-12
    end

    @testset "consistent tangent through the local solve" begin
        tb = relaxation_testbed(strategy, qrc; material = mat)
        op = tb.op
        n  = unknown_size(tb.op)
        u  = 0.3 .* sin.(0.7 .* (1:n))
        uprev = zeros(n)
        states = condensed_states(u, uprev)
        ctx = TimeIntegrationContext(0.0, 0.4, 0.4)

        # check_derivatives condenses internally at every trial point it
        # probes, so the FD referee is a total — exactly what the analytic
        # `Consistent` kernel (reading the stored `dq/du`) computes.
        res = check_derivatives(op, states, nothing, ctx)
        @test res.passed
        @test res.checks.jacobian.passed
        @test res.checks.jacobian.skipped === nothing
    end

    @testset "outer → inner: the requested tolerance changes local iteration counts" begin
        # A stiff local problem, so the iteration count is tolerance-sensitive.
        stiff = NortonRelaxationParameters(κ = 1.0, α = 1.0, η = 1.0, n = 3.0)
        tb    = relaxation_testbed(strategy, qrc; material = stiff,
                                   local_solver = LocalNewtonSettings(max_iterations = 40, tolerance = 1e-12))
        op    = tb.op
        n     = unknown_size(op)
        uprev = zeros(n)
        base  = TimeIntegrationContext(0.0, 1.0e3, 1.0e3)

        u = zeros(n); view(u, 1:ndofs(tb.dh)) .= 2.0
        tight = condense_internal!(op, condensed_states(u, uprev), nothing, base)

        u2 = zeros(n); view(u2, 1:ndofs(tb.dh)) .= 2.0
        loose = condense_internal!(op, condensed_states(u2, uprev), nothing, InexactLocalSolveContext(base, 1.0e-2))

        @test tight.solves == loose.solves
        @test loose.iterations < tight.iterations
        @test loose.worst_iterations < tight.worst_iterations

        # A requested tolerance tighter than the element's own floor is ignored.
        u3 = zeros(n); view(u3, 1:ndofs(tb.dh)) .= 2.0
        floored = condense_internal!(op, condensed_states(u3, uprev), nothing, InexactLocalSolveContext(base, 1.0e-30))
        @test floored.iterations == tight.iterations

        # The decoration survives the framework's context handling.
        @test evaluation_time(InexactLocalSolveContext(base, 1.0e-2)) == 0.0
        @test local_solve_tolerance(FerriteOperators.with_time(InexactLocalSolveContext(base, 1.0e-2), 3.0)) == 1.0e-2
        @test local_solve_tolerance(base) === nothing
        # ∂F/∂t of a time-independent element, through a custom context type —
        # the FD method condenses internally, so it needs no prior condensation.
        g = zeros(residual_size(op))
        time_sensitivity!(g, op, condensed_states(u, uprev), nothing, InexactLocalSolveContext(base, 1.0e-2);
                          method = FiniteDifferenceSensitivity())
        @test norm(g) < 1e-8
    end

    @testset "inner → outer: the report aggregates per condensation and merges over workers" begin
        tb  = relaxation_testbed(strategy, qrc; material = mat)
        op  = tb.op
        n   = unknown_size(op)
        u   = 0.5 .* sin.(0.9 .* (1:n))
        uprev = zeros(n)
        ctx = TimeIntegrationContext(0.0, 1.0, 1.0)

        nqp    = getnquadpoints(first_element_cache(op))
        ncells = getncells(tb.grid)

        report = condense_internal!(op, condensed_states(u, uprev), nothing, ctx)
        @test report.solves == ncells * nqp
        @test report.iterations ≥ report.solves
        @test report.worst_iterations ≥ 1
        @test report.worst_cell ∈ 1:ncells
        @test report.worst_qp ∈ 1:nqp
        @test report.converged

        # Per-worker partials fold to the same totals under PolyesterDevice(2)
        # — CondensationReport is a monoid (& / + / max-with-argmax / max /
        # min componentwise), so this is a report merge, not a stats reset.
        ptb = relaxation_testbed(PerColorAssemblyStrategy(PolyesterDevice(2)), qrc; material = mat)
        pu  = 0.5 .* sin.(0.9 .* (1:unknown_size(ptb.op)))
        puprev = zeros(unknown_size(ptb.op))
        preport = condense_internal!(ptb.op, condensed_states(pu, puprev), nothing, ctx)
        @test preport.solves == report.solves
        @test preport.iterations == report.iterations
        @test preport.converged == report.converged
    end

    @testset "non-convergence is reported as data, not thrown" begin
        budget = LocalNewtonSettings(max_iterations = 2, tolerance = 1e-12)
        tb  = relaxation_testbed(strategy, qrc; material = mat, local_solver = budget)
        n   = unknown_size(tb.op)
        uprev = zeros(n)
        ctx = TimeIntegrationContext(0.0, 1.0e3, 1.0e3)

        u = zeros(n); view(u, 1:ndofs(tb.dh)) .= 2.0
        report = condense_internal!(tb.op, condensed_states(u, uprev), nothing, ctx)
        @test report.converged == false
        @test report.worst_iterations == 2

        # The same problem within the default budget converges.
        tb2 = relaxation_testbed(strategy, qrc; material = mat)
        u2  = zeros(n); view(u2, 1:ndofs(tb2.dh)) .= 2.0
        report2 = condense_internal!(tb2.op, condensed_states(u2, uprev), nothing, ctx)
        @test report2.converged
        @test report2.worst_iterations > 2
    end
end

@testset "Condensation freshness" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters()

    @testset "uncondensed Consistent Jacobian throws, naming the cell" begin
        tb = relaxation_testbed(strategy, qrc; material = mat)
        op = tb.op
        n  = unknown_size(op)
        u  = 0.3 .* sin.(0.7 .* (1:n))
        uprev = zeros(n)
        states = condensed_states(u, uprev)
        ctx = TimeIntegrationContext(0.0, 0.4, 0.4)

        # The residual reads whatever is in the q slot, so it does not throw —
        # only a `Consistent` Jacobian read of the (unstamped) corrector does.
        r = zeros(residual_size(op))
        evaluate!(op, r, states, nothing, ctx)
        @test_throws ArgumentError update_linearization!(op, r, states, nothing, ctx)

        report = condense_internal!(op, states, nothing, ctx)
        @test report.converged
        update_linearization!(op, r, states, nothing, ctx)   # fine now
    end

    @testset "rollback_state! invalidates; commit_state! does not" begin
        tb = relaxation_testbed(strategy, qrc; material = mat)
        op = tb.op
        n  = unknown_size(op)
        u  = 0.3 .* sin.(0.7 .* (1:n))
        uprev = zeros(n)
        states = condensed_states(u, uprev)
        ctx = TimeIntegrationContext(0.0, 0.4, 0.4)
        r = zeros(residual_size(op))

        condense_internal!(op, states, nothing, ctx)
        update_linearization!(op, r, states, nothing, ctx)   # fine

        committed = zeros(n)
        rollback_state!(op, u, committed)
        @test u == committed
        @test_throws ArgumentError update_linearization!(op, r, states, nothing, ctx)

        condense_internal!(op, states, nothing, ctx)
        commit_state!(op, u, committed)
        update_linearization!(op, r, states, nothing, ctx)   # commit does not invalidate
    end
end

@testset "FrozenQ election" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters()
    tb = relaxation_testbed(strategy, qrc; material = mat)
    op = tb.op
    n  = unknown_size(op)
    u  = 0.3 .* sin.(0.7 .* (1:n))
    uprev = zeros(n)
    states = condensed_states(u, uprev)
    ctx = TimeIntegrationContext(0.0, 0.4, 0.4)
    condense_internal!(op, states, nothing, ctx)

    @testset "refused at construction for the sensitivity kinds" begin
        # ParameterJacobianKind/TimeSensitivityKind carry no type parameter at
        # all — `{FrozenQ}` is an immediate type-application error, so there
        # is no way to even write the election: `CorrectionMode` is a
        # parameter these kinds structurally do not have. (ParameterVJPKind/
        # StateJVPKind/StateVJPKind carry a PAYLOAD parameter instead — λ/v —
        # which is not a CorrectionMode slot either, so the same "no such
        # election" holds; it just isn't a `{FrozenQ}` type-application error
        # for them specifically, since their one parameter is unconstrained.)
        @test_throws Exception FerriteOperators.ParameterJacobianKind{FerriteOperators.FrozenQ}
        @test_throws Exception FerriteOperators.TimeSensitivityKind{FerriteOperators.FrozenQ}
    end

    @testset "accepted for the Newton matrix and produces the partial" begin
        Kc = similar(op.J); Kf = similar(op.J)
        FerriteOperators.assemble_into!(JacobianKind{:u, Consistent}(), (Kc,), op, states, nothing, ctx)
        FerriteOperators.assemble_into!(JacobianKind{:u, FrozenQ}(), (Kf,), op, states, nothing, ctx)
        # The partial drops the ∂F/∂q·dq/du correction, so it disagrees with
        # the total wherever dq/du ≠ 0.
        @test Kc.nzval != Kf.nzval

        # `check_derivatives` treats an elected FrozenQ mismatch as elected,
        # not a failure — it skips the comparison rather than running a
        # doomed one against its (necessarily total) FD referee.
        res = check_derivatives(op, states, nothing, ctx; correction = FrozenQ)
        @test res.checks.jacobian.skipped !== nothing
    end
end

@testset "Corrector election is a construction-time seam" begin
    qrc = QuadratureRuleCollection(2)
    mat = NortonRelaxationParameters()
    vmat = MaxwellParameters()

    @test SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q) isa FerriteOperators.AbstractCondensedNonlinearIntegrator
    @test FerriteOperators.corrector_election(SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q)) isa Stored
    @test FerriteOperators.corrector_election(
        SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q; corrector = Recompute())) isa Recompute
    @test FerriteOperators.corrector_election(
        SimpleCondensedLinearViscoelasticity(vmat, qrc, :u, :εᵛ; corrector = Recompute())) isa Recompute

    # An integrator serving only some elections rejects the rest through the
    # shared, self-naming helper.
    struct UnservedElection <: FerriteOperators.CorrectorElection end
    @test_throws "UnservedElection correctors are not implemented" FerriteOperators.corrector_election_error(UnservedElection())
end

@testset "Recompute() keeps no corrector store and matches Stored()" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters(κ = 1.3, α = 0.8, η = 1.4, n = 2.5)
    ctx      = TimeIntegrationContext(0.0, 0.4, 0.4)

    stored    = relaxation_testbed(strategy, qrc; material = mat)
    recompute = relaxation_testbed(strategy, qrc; material = mat, corrector = Recompute())

    # Structural: the recomputing cache allocates neither corrector store.
    scache = first_element_cache(stored.op).inner
    rcache = first_element_cache(recompute.op).inner
    @test scache.correctors isa FerriteOperators.ItemStates
    @test rcache.correctors === nothing
    @test rcache.param_correctors === nothing

    n = unknown_size(stored.op)
    us = 0.3 .* sin.(0.6 .* (1:n)); ur = copy(us)
    uprev = zeros(n)
    ss = condensed_states(us, uprev); sr = condensed_states(ur, uprev)

    rs = condense_internal!(stored.op, ss, mat, ctx)
    rr = condense_internal!(recompute.op, sr, mat, ctx)
    @test rs.converged && rr.converged
    @test rs.iterations == rr.iterations
    @test us == ur                                   # the same trial q in the tail

    # The recomputed slopes are the SAME arithmetic on the same converged pair
    # — the closed-form `∂R/∂q` at `(u, q)` — so the assembled matrices agree
    # bitwise, not merely to round-off.
    Ks = zeros(residual_size(stored.op)); Kr = similar(Ks)
    update_linearization!(stored.op, Ks, ss, mat, ctx)
    update_linearization!(recompute.op, Kr, sr, mat, ctx)
    @test stored.op.J.nzval == recompute.op.J.nzval
    @test Ks == Kr

    # …and so does the parameter Jacobian, which reads the other corrector.
    nθ = length(parameter_vector(mat))
    Bs = zeros(residual_size(stored.op), nθ); Br = similar(Bs)
    update_parameter_jacobian!(Bs, stored.op, ss, mat, ctx)
    update_parameter_jacobian!(Br, recompute.op, sr, mat, ctx)
    @test Bs == Br

    # The FD referee agrees with the recomputing operator's derivative paths.
    res = check_derivatives(recompute.op, sr, mat, ctx)
    @test res.passed
    @test res.checks.jacobian.passed
    @test res.checks.parameter_jacobian.passed

    # The viscoelastic element makes the same election over a retained Mandel
    # factorization instead of a scalar slope, and it is the same arithmetic
    # there too — `A` is a closed form in the `ℂ` and `γ̃` the kernel already
    # has.
    vstored    = visco_testbed(strategy, qrc)
    vrecompute = visco_testbed(strategy, qrc; corrector = Recompute())
    @test first_element_cache(vstored.op).inner.correctors isa FerriteOperators.ItemStates
    @test first_element_cache(vrecompute.op).inner.correctors === nothing

    m = unknown_size(vstored.op)
    vs = 1.0e-3 .* sin.(0.4 .* (1:m)); vr = copy(vs)
    vprev = zeros(m)
    vctx = TimeIntegrationContext(0.0, 0.5, 0.5)
    condense_internal!(vstored.op, condensed_states(vs, vprev), nothing, vctx)
    condense_internal!(vrecompute.op, condensed_states(vr, vprev), nothing, vctx)
    @test vs == vr
    rvs = zeros(residual_size(vstored.op)); rvr = similar(rvs)
    update_linearization!(vstored.op, rvs, condensed_states(vs, vprev), nothing, vctx)
    update_linearization!(vrecompute.op, rvr, condensed_states(vr, vprev), nothing, vctx)
    @test vstored.op.J.nzval == vrecompute.op.J.nzval
    @test rvs == rvr
end

@testset "Recompute() freshness: the q contract survives, the corrector class does not" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters()
    ctx      = TimeIntegrationContext(0.0, 0.4, 0.4)

    tb = relaxation_testbed(strategy, qrc; material = mat, corrector = Recompute())
    op = tb.op
    n  = unknown_size(op)
    u  = 0.3 .* sin.(0.7 .* (1:n))
    uprev  = zeros(n)
    states = condensed_states(u, uprev)
    r = zeros(residual_size(op))

    # There is no corrector store to be unstamped, so an uncondensed
    # `Consistent` Jacobian does NOT throw the way `Stored()` does — it reads
    # whatever `q` the tail holds. The q contract is the caller's alone here.
    update_linearization!(op, r, states, nothing, ctx)

    # q is still written ONLY by condense_internal!: a sweep at a moved `u`
    # leaves the tail alone, and condensing changes it.
    condense_internal!(op, states, nothing, ctx)
    qafter = copy(view(u, (residual_size(op)+1):n))
    update_linearization!(op, r, states, nothing, ctx)
    @test view(u, (residual_size(op)+1):n) == qafter          # no sweep writes back

    view(u, 1:residual_size(op)) .+= 0.05
    update_linearization!(op, r, states, nothing, ctx)
    @test view(u, (residual_size(op)+1):n) == qafter          # still only condensation writes q
    condense_internal!(op, states, nothing, ctx)
    @test view(u, (residual_size(op)+1):n) != qafter

    # rollback restores `u` and its q tail together, so the corrector a
    # recomputing kernel derives afterwards is the committed point's own.
    condense_internal!(op, states, nothing, ctx)
    update_linearization!(op, r, states, nothing, ctx)
    Jcommitted = copy(op.J.nzval)
    committed  = copy(u)                                    # ū and its q, together
    view(u, 1:residual_size(op)) .+= 0.1
    condense_internal!(op, states, nothing, ctx)
    update_linearization!(op, r, states, nothing, ctx)
    @test op.J.nzval != Jcommitted
    rollback_state!(op, u, committed)
    update_linearization!(op, r, states, nothing, ctx)       # no invalidation needed
    @test op.J.nzval == Jcommitted
end

@testset "JacobianKind{:q} partials at the kernel level" begin
    # ∂F/∂q is a LOCAL (ndofs × nqp) block — no global scatter target exists
    # for it (q's dofs are internal to the cell) — so it is exercised at the
    # kernel level, the supported way to unit-test a kernel without an
    # operator (elements.md, "Unit-testing a kernel").
    qrc  = QuadratureRuleCollection(2)
    integ = SimpleCondensedPowerLawRelaxation(NortonRelaxationParameters(), qrc, :u, :q)
    grid = generate_grid(Quadrilateral, (1, 1))
    dh   = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
    sdh  = dh.subdofhandlers[1]
    cache = FerriteOperators.setup_element_cache(integ, sdh)

    cc = Ferrite.CellCache(dh)
    reinit!(cc, 1)
    FerriteOperators.reinit_values!(cache, cc)

    nqp   = getnquadpoints(cache.cv)
    ndofs = ndofs_per_cell(sdh)
    uₑ = 0.3 .* sin.(1:ndofs)
    qₑ = 0.1 .* cos.(1:nqp)

    K = zeros(ndofs, nqp)
    FerriteOperators.assemble_cell!(JacobianRequest{:q}(K), cache, CellArgs((u = uₑ, q = qₑ), cc, nothing, nothing))

    h = 1e-6
    Kfd = zeros(ndofs, nqp)
    for j in 1:nqp
        qp1 = copy(qₑ); qp1[j] += h
        rp = zeros(ndofs)
        FerriteOperators.assemble_cell!(ResidualRequest(rp), cache, CellArgs((u = uₑ, q = qp1), cc, nothing, nothing))
        qm1 = copy(qₑ); qm1[j] -= h
        rm = zeros(ndofs)
        FerriteOperators.assemble_cell!(ResidualRequest(rm), cache, CellArgs((u = uₑ, q = qm1), cc, nothing, nothing))
        Kfd[:, j] .= (rp .- rm) ./ 2h
    end
    @test K ≈ Kfd rtol = 1e-6

    # Consistent and FrozenQ coincide for ∂F/∂q — q is the seed itself, so
    # there is no dq/dq correction to add or drop.
    Kc = zeros(ndofs, nqp)
    FerriteOperators.assemble_cell!(JacobianRequest{:q, Consistent}(Kc), cache, CellArgs((u = uₑ, q = qₑ), cc, nothing, nothing))
    Kf = zeros(ndofs, nqp)
    FerriteOperators.assemble_cell!(JacobianRequest{:q, FrozenQ}(Kf), cache, CellArgs((u = uₑ, q = qₑ), cc, nothing, nothing))
    @test Kc == Kf == K
end

@testset "JacobianKind{:q} assembled into the rectangular field × internal target" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters(κ = 1.3, α = 0.8, η = 1.4, n = 2.5)
    tb  = relaxation_testbed(strategy, qrc, (2, 2); material = mat)
    op, dh, grid = tb.op, tb.dh, tb.grid
    ivh = op.engine.ivh
    n   = unknown_size(op)
    u   = 0.3 .* sin.(0.6 .* (1:n))
    uprev  = zeros(n)
    states = condensed_states(u, uprev)
    ctx = TimeIntegrationContext(0.0, 0.4, 0.4)
    condense_internal!(op, states, mat, ctx)

    Kq = allocate_internal_jacobian(op)
    @test size(Kq) == (residual_size(op), ndofs(ivh))
    update_internal_jacobian!(Kq, op, states, mat, ctx)

    # Dense hand-rolled reference: the per-cell kernel output scattered by
    # (celldofs, internal range) — the assembly the sweep automates.
    cache = first_element_cache(op).inner
    cc    = Ferrite.CellCache(dh)
    nd    = ndofs_per_cell(dh.subdofhandlers[1])
    ref   = zeros(residual_size(op), ndofs(ivh))
    for cid in 1:getncells(grid)
        reinit!(cc, cid)
        FerriteOperators.reinit_values!(cache, cc)
        range = internal_variable_range(ivh, cid)
        Ke = zeros(nd, length(range))
        FerriteOperators.assemble_cell!(JacobianRequest{:q}(Ke), cache,
            CellArgs((u = u[celldofs(cc)], q = u[range]), cc, mat, ctx))
        ref[celldofs(cc), range .- residual_size(op)] .+= Ke
    end
    @test Matrix(Kq) == ref
    # The pattern carries exactly the cell-local couplings and nothing else.
    @test nnz(Kq) == getncells(grid) * nd * getnquadpoints(cache.cv)

    # The block is the ∂F/∂q of the residual, so it differences the residual
    # w.r.t. the tail — the property a Schur-complement consumer relies on.
    h = 1e-6
    rp = zeros(residual_size(op)); rm = zeros(residual_size(op))
    for j in (1, ndofs(ivh))
        up = copy(u); up[residual_size(op)+j] += h
        evaluate!(op, rp, condensed_states(up, uprev), mat, ctx)
        um = copy(u); um[residual_size(op)+j] -= h
        evaluate!(op, rm, condensed_states(um, uprev), mat, ctx)
        @test Vector(Kq[:, j]) ≈ (rp .- rm) ./ 2h rtol = 1e-6
    end

    # A non-condensed operator has no column space for it.
    vdh = DofHandler(generate_grid(Quadrilateral, (2, 2)))
    add!(vdh, :u, Lagrange{RefQuadrilateral, 1}()^2); close!(vdh)
    plain = setup_operator(strategy, SimpleHyperelasticityIntegrator(NeoHookean(210e3, 0.3), qrc, :u), vdh)
    @test_throws "carries none" allocate_internal_jacobian(plain)
    @test_throws "carries none" update_internal_jacobian!(zeros(ndofs(vdh), 0), plain, (u = zeros(ndofs(vdh)),), nothing, nothing)

    # …and the square slot-Jacobian entry point refuses `:q` outright.
    @test_throws "update_internal_jacobian!" assemble_slot_jacobian!(op.J, op, JacobianKind{:q}(), states, mat, ctx)
end

@testset "q as a slot" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters()
    tb = relaxation_testbed(strategy, qrc; material = mat)
    op, dh = tb.op, tb.dh
    n  = unknown_size(op)
    u  = zeros(n); view(u, 1:ndofs(dh)) .= 1.0
    uprev = zeros(n)
    ctx = TimeIntegrationContext(0.0, 0.5, 0.5)

    condense_internal!(op, condensed_states(u, uprev), nothing, ctx)
    qtail = view(u, (ndofs(dh)+1):n)

    # InternalSource gathers exactly the cell's internal_variable_range.
    ivh = op.engine.ivh
    for cellid in 1:getncells(tb.grid)
        range = internal_variable_range(ivh, cellid)
        gathered = u[range]
        @test gathered == qtail[range .- ndofs(dh)]
    end
end

@testset "local_conditions! admits the θ/t totals on a condensed cache" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)

    @testset "∂F/∂θ and ∂F/∂t against the FD referee, through the hook" begin
        params = TimedRelaxationParameters(α = 0.8, τ = 1.3)
        tb = timed_relaxation_testbed(strategy, qrc; params)
        op = tb.op
        cache = first_element_cache(op).inner
        n = unknown_size(op)
        u = 0.3 .* sin.(0.6 .* (1:n))
        uprev  = zeros(n)
        states = condensed_states(u, uprev)
        ctx = TimeIntegrationContext(0.7, 0.4, 0.4)

        cache.ncalls[] = 0
        res = check_derivatives(op, states, params, ctx)
        @test res.passed
        @test res.checks.parameter_jacobian.passed
        @test res.checks.parameter_jacobian.skipped === nothing
        @test res.checks.parameter_vjp.passed
        @test res.checks.parameter_vjp.skipped === nothing
        @test res.checks.time_sensitivity.passed
        @test res.checks.time_sensitivity.skipped === nothing
        # The generic route RAN — this element has no analytic θ/t kernel, so
        # the totals above could only have come through the hook.
        @test cache.ncalls[] > 0

        # ∂F/∂τ|_q and ∂F/∂t|_q are structurally zero for this element, so both
        # sensitivities are the ∂F/∂q·dq/d· correction alone — nonzero only
        # because the hook supplied it.
        condense_internal!(op, states, params, ctx)
        B = zeros(residual_size(op), 2)
        update_parameter_jacobian!(B, op, states, params, ctx)
        @test norm(B[:, 2]) > 1e-3
        g = zeros(residual_size(op))
        time_sensitivity!(g, op, states, params, ctx)
        @test norm(g) > 1e-3
    end

    @testset "hidden analytic kernels: the hook reproduces what they compute" begin
        mat = NortonRelaxationParameters(κ = 1.3, α = 0.8, η = 1.4, n = 2.5)
        ctx = TimeIntegrationContext(0.0, 0.4, 0.4)
        ref = relaxation_testbed(strategy, qrc; material = mat)

        counter = Ref(0)
        grid = generate_grid(Quadrilateral, (2, 2))
        dh = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
        hidden_op = setup_operator(strategy,
            HiddenAnalyticIntegrator(SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q), counter),
            dh; slots = (:u, :q, :qprev))

        n = unknown_size(ref.op)
        u_ref = 0.3 .* sin.(0.6 .* (1:n)); u_hid = copy(u_ref)
        uprev = zeros(n)
        s_ref = condensed_states(u_ref, uprev); s_hid = condensed_states(u_hid, uprev)
        condense_internal!(ref.op, s_ref, mat, ctx)
        condense_internal!(hidden_op, s_hid, mat, ctx)
        @test u_ref ≈ u_hid

        nθ = length(parameter_vector(mat))
        Bref = zeros(residual_size(ref.op), nθ); Bhid = similar(Bref)
        counter[] = 0
        update_parameter_jacobian!(Bref, ref.op, s_ref, mat, ctx)
        @test counter[] == 0                     # analytic kernel visible: hook untouched
        update_parameter_jacobian!(Bhid, hidden_op, s_hid, mat, ctx)
        @test counter[] > 0                      # analytic kernel hidden: hook ran
        @test Bhid ≈ Bref rtol = 1e-8

        # …and the adjoint pullback contracts the same two factors, against the
        # analytic parameter Jacobian as referee. (The power-law element itself
        # declares no `ParameterVJPKind` kernel, so the hook is what makes the
        # pullback available on the wrapper at all.)
        λ = 0.4 .* cos.(1:residual_size(ref.op))
        ghid = zeros(nθ)
        parameter_vjp!(ghid, hidden_op, λ, s_hid, mat, ctx)
        @test ghid ≈ Bref' * λ rtol = 1e-8
        @test_throws ArgumentError parameter_vjp!(zeros(nθ), ref.op, λ, s_ref, mat, ctx)
    end

    @testset "neither analytic kernels nor the hook: today's rejection, unchanged" begin
        # The power-law cache wrapped WITHOUT `local_conditions!`: the frozen-q
        # AD partial is all the decorator could produce, so the parameter kinds
        # stay refused with the message they have always carried.
        mat = NortonRelaxationParameters()
        grid = generate_grid(Quadrilateral, (2, 2))
        dh = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
        op = setup_operator(strategy,
            NoHookIntegrator(SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q)),
            dh; slots = (:u, :q, :qprev))
        n = unknown_size(op)
        states = condensed_states(0.1 .* sin.(1:n), zeros(n))
        ctx = TimeIntegrationContext(0.0, 0.4, 0.4)
        condense_internal!(op, states, mat, ctx)

        err = @test_throws ArgumentError update_parameter_jacobian!(
            zeros(residual_size(op), 3), op, states, mat, ctx)
        msg = err.value.msg
        @test occursin("NoHookCache carries condensed internal state", msg)
        @test occursin("would compute only the frozen-q partial, missing the ∂F/∂q·dq/d· correction", msg)
        @test occursin("implement the analytic `assemble_cell!` kernel", msg)
        @test occursin("declare `internal_state_insensitive`", msg)
        @test occursin("Implementing `local_conditions!` admits the generic route", msg)

        @test_throws ArgumentError time_sensitivity!(zeros(residual_size(op)), op, states, mat, ctx)
    end
end

@testset "Parameter-sensitivity payoff: exact ∂F/∂θ total on a condensed element" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters(κ = 1.3, α = 0.8, η = 1.4, n = 2.5)
    tb = relaxation_testbed(strategy, qrc; material = mat)
    op = tb.op
    n  = unknown_size(op)
    u  = 0.3 .* sin.(0.6 .* (1:n))
    uprev = zeros(n)
    states = condensed_states(u, uprev)
    ctx = TimeIntegrationContext(0.0, 0.4, 0.4)

    # A condensed cache with a corrector store, migrated to serve
    # ParameterJacobianKind analytically — inadmissible before the phase
    # (AD-from-residual through a local solve), generic-shaped now: the
    # partial (∂F/∂θ|_q, from the pure residual) plus ∂F/∂q·dq/dθ, dq/dθ
    # computed and stored by `condense_cell!` alongside dq/du.
    θ = collect(parameter_vector(mat))
    nθ = length(θ)
    B = zeros(residual_size(op), nθ)
    condense_internal!(op, states, mat, ctx)
    update_parameter_jacobian!(B, op, states, mat, ctx)

    h = 1e-6
    Bfd = zeros(residual_size(op), nθ)
    rp = zeros(residual_size(op)); rm = zeros(residual_size(op))
    for j in 1:nθ
        θp = copy(θ); θp[j] += h
        pp = rebuild_parameters(mat, θp)
        condense_internal!(op, states, pp, ctx)
        evaluate!(op, rp, states, pp, ctx)

        θm = copy(θ); θm[j] -= h
        pm = rebuild_parameters(mat, θm)
        condense_internal!(op, states, pm, ctx)
        evaluate!(op, rm, states, pm, ctx)

        Bfd[:, j] .= (rp .- rm) ./ 2h
    end
    @test B ≈ Bfd rtol = 1e-6

    # The same total, cross-checked through check_derivatives.
    res = check_derivatives(op, states, mat, ctx)
    @test res.checks.parameter_jacobian.passed
    @test res.checks.parameter_jacobian.skipped === nothing
end
