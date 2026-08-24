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

# The element with its FUSED claim withheld: resolves to
# `ADElementCache{FusedFromSplit{…}}`, so the θ/t hook probes must look through
# both wrappers. Hand-written: the split claims, their forwards, duplication.
struct SplitTimedRelaxationCache{C} <: FerriteOperators.AbstractElementCacheDecorator{C}
    inner::C
end
FerriteOperators.duplicate_for_device(device, c::SplitTimedRelaxationCache) =
    SplitTimedRelaxationCache(FerriteOperators.duplicate_for_device(device, c.inner))
FerriteOperators.provides_analytic(::Type{<:SplitTimedRelaxationCache}, ::JacobianKind{:u}) = true
for R in (:ResidualRequest, :(JacobianRequest{:u, Consistent}))
    @eval FerriteOperators.assemble_cell!(req::$R, c::SplitTimedRelaxationCache, args::CellArgs) =
        FerriteOperators.assemble_cell!(req, c.inner, args)
end

struct SplitTimedRelaxationIntegrator <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    inner::TimedRelaxationIntegrator
end
FerriteOperators.setup_element_cache(m::SplitTimedRelaxationIntegrator, sdh::SubDofHandler) =
    SplitTimedRelaxationCache(FerriteOperators.setup_element_cache(m.inner, sdh))

# The test-only wrapper pattern: everything of the power-law cache forwarded
# EXCEPT its analytic `ParameterJacobianKind` claim. `with_hook = true` supplies
# the local conditions the element never needed, so the wrapper's ∂F/∂θ has to
# come out of the generic route and the element it wraps is the reference for
# what that route must produce; `with_hook = false` is the case the
# admissibility rule must refuse, since no other branch applies either.
struct ForwardingCache{with_hook, C} <: FerriteOperators.AbstractVolumetricElementCache
    inner::C
    ncalls::Base.RefValue{Int}
end
ForwardingCache{h}(inner, ncalls) where {h} = ForwardingCache{h, typeof(inner)}(inner, ncalls)
FerriteOperators.duplicate_for_device(device, c::ForwardingCache{h}) where {h} =
    ForwardingCache{h}(FerriteOperators.duplicate_for_device(device, c.inner), c.ncalls)
FerriteOperators.query_cell_parameters(c::ForwardingCache, cell, p) =
    FerriteOperators.query_cell_parameters(c.inner, cell, p)
FerriteOperators.reinit_values!(c::ForwardingCache, cell) = FerriteOperators.reinit_values!(c.inner, cell)
FerriteOperators.reinit_values!(c::ForwardingCache, cell, kind) = FerriteOperators.reinit_values!(c.inner, cell, kind)
Ferrite.getnquadpoints(c::ForwardingCache) = getnquadpoints(c.inner)
FerriteOperators.has_internal_state(::Type{<:ForwardingCache}) = true
FerriteOperators.get_number_of_internal_dofs_per_element(m, c::ForwardingCache, sdh) =
    FerriteOperators.get_number_of_internal_dofs_per_element(m, c.inner, sdh)
FerriteOperators.condense_cell!(c::ForwardingCache, args, weights) =
    FerriteOperators.condense_cell!(c.inner, args, weights)
FerriteOperators.invalidate_correctors!(c::ForwardingCache) = FerriteOperators.invalidate_correctors!(c.inner)
FerriteOperators.provides_analytic(::Type{<:ForwardingCache}, ::JacobianKind{:u}) = true
FerriteOperators.provides_analytic(::Type{<:ForwardingCache}, ::JacobianResidualKind) = true
for R in (:ResidualRequest, :(JacobianRequest{:u, Consistent}), :(JacobianResidualRequest{Consistent}))
    @eval FerriteOperators.assemble_cell!(req::$R, c::ForwardingCache, args::CellArgs) =
        FerriteOperators.assemble_cell!(req, c.inner, args)
end
function FerriteOperators.local_conditions!(L, c::ForwardingCache{true}, args::CellArgs)
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

struct ForwardingIntegrator{with_hook, I} <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    inner::I
    ncalls::Base.RefValue{Int}
end
ForwardingIntegrator{h}(inner, ncalls = Ref(0)) where {h} =
    ForwardingIntegrator{h, typeof(inner)}(inner, ncalls)
FerriteOperators.setup_element_cache(m::ForwardingIntegrator{h}, sdh::SubDofHandler) where {h} =
    ForwardingCache{h}(FerriteOperators.setup_element_cache(m.inner, sdh), m.ncalls)

@testset "Condensed element with a nonlinear local solve" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    mat      = NortonRelaxationParameters(κ = 1.0, α = 1.0, η = 1.0, n = 3.0)

    @testset "condense_internal! hits the root and the pure assembly is exact" begin
        # A constant field has the same value at every quadrature point, so
        # every local problem has the same known root.
        γ̃ = 0.5
        (; op, dh, n, u, states, ctx) = relaxation_case(strategy, qrc; material = mat, field = 1.0, γ̃)
        r = zeros(residual_size(op))

        report = condense_internal!(op, states, nothing, ctx)
        @test report.converged
        evaluate!(op, r, states, nothing, ctx)   # pure: reads the q condense_internal! wrote

        qref = reference_internal_state(mat, 1.0, 0.0, γ̃)
        q    = view(u, (ndofs(dh)+1):n)   # the [ū; q] tail write-back
        @test all(qi -> isapprox(qi, qref; atol = 1e-11), q)
        # ∇u ≡ 0, so the residual is the exchange term only and sums to α(u−q)|Ω|.
        @test sum(r) ≈ mat.α * (1.0 - qref) * 4.0 rtol = 1e-12
    end

    @testset "consistent tangent through the local solve" begin
        (; op, states, ctx) = relaxation_case(strategy, qrc; material = mat)

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
        case  = relaxation_case(strategy, qrc; material = stiff, field = 2.0, γ̃ = 1.0e3,
                                local_solver = LocalNewtonSettings(max_iterations = 40, tolerance = 1e-12))
        (; op, dh, n, uprev) = case
        base = case.ctx
        # Every solve starts from the same trial ū and an uncondensed q tail.
        trial() = (v = zeros(n); view(v, 1:ndofs(dh)) .= 2.0; condensed_states(v, uprev))

        tight = condense_internal!(op, trial(), nothing, base)
        loose = condense_internal!(op, trial(), nothing, InexactLocalSolveContext(base, 1.0e-2))
        @test tight.solves == loose.solves
        @test loose.iterations < tight.iterations
        @test loose.worst_iterations < tight.worst_iterations

        # A requested tolerance tighter than the element's own floor is ignored.
        floored = condense_internal!(op, trial(), nothing, InexactLocalSolveContext(base, 1.0e-30))
        @test floored.iterations == tight.iterations

        # The decoration survives the framework's context handling.
        @test evaluation_time(InexactLocalSolveContext(base, 1.0e-2)) == 0.0
        @test local_solve_tolerance(FerriteOperators.with_time(InexactLocalSolveContext(base, 1.0e-2), 3.0)) == 1.0e-2
        @test local_solve_tolerance(base) === nothing
        # ∂F/∂t of a time-independent element, through a custom context type —
        # the FD method condenses internally, so it needs no prior condensation.
        g = zeros(residual_size(op))
        time_sensitivity!(g, op, case.states, nothing, InexactLocalSolveContext(base, 1.0e-2);
                          method = FiniteDifferenceSensitivity())
        @test norm(g) < 1e-8
    end

    @testset "inner → outer: the report aggregates per condensation and merges over workers" begin
        spread = (material = mat, γ̃ = 1.0, amplitude = 0.5, frequency = 0.9)
        (; op, grid, states, ctx) = relaxation_case(strategy, qrc; spread...)

        nqp    = getnquadpoints(first_element_cache(op))
        ncells = getncells(grid)

        report = condense_internal!(op, states, nothing, ctx)
        @test report.solves == ncells * nqp
        @test report.iterations ≥ report.solves
        @test report.worst_iterations ≥ 1
        @test report.worst_cell ∈ 1:ncells
        @test report.worst_qp ∈ 1:nqp
        @test report.converged

        # Per-worker partials fold to the same totals under PolyesterDevice(2)
        # — CondensationReport is a monoid (& / + / max-with-argmax / max /
        # min componentwise), so this is a report merge, not a stats reset.
        par = relaxation_case(PerColorAssemblyStrategy(PolyesterDevice(2)), qrc; spread...)
        preport = condense_internal!(par.op, par.states, nothing, par.ctx)
        @test preport.solves == report.solves
        @test preport.iterations == report.iterations
        @test preport.converged == report.converged
    end

    @testset "non-convergence is reported as data, not thrown" begin
        stalled = (material = mat, field = 2.0, γ̃ = 1.0e3)
        capped  = relaxation_case(strategy, qrc; stalled...,
                                  local_solver = LocalNewtonSettings(max_iterations = 2, tolerance = 1e-12))
        report = condense_internal!(capped.op, capped.states, nothing, capped.ctx)
        @test report.converged == false
        @test report.worst_iterations == 2

        # The same problem within the default budget converges.
        budgeted = relaxation_case(strategy, qrc; stalled...)
        report2 = condense_internal!(budgeted.op, budgeted.states, nothing, budgeted.ctx)
        @test report2.converged
        @test report2.worst_iterations > 2
    end
end

@testset "Condensation freshness" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    (; op, u, states, ctx) = relaxation_case(strategy, qrc; material = NortonRelaxationParameters())
    check_freshness_contract(op, states, u, ctx)
end

@testset "FrozenQ election" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    (; op, states, ctx) = relaxation_case(strategy, qrc; material = NortonRelaxationParameters())
    condense_internal!(op, states, nothing, ctx)

    @testset "refused at construction for the sensitivity kinds" begin
        # `ParameterJacobianKind` carries no `CorrectionMode` parameter, so the
        # election cannot even be written for it.
        @test_throws Exception FerriteOperators.ParameterJacobianKind{FerriteOperators.FrozenQ}
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

    # Without a request an element elects `Stored()`; what the other election
    # buys is pinned behaviourally below.
    @test FerriteOperators.corrector_election(SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q)) isa Stored

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
    n  = unknown_size(stored.op)
    us = 0.3 .* sin.(0.6 .* (1:n)); ur = copy(us)
    uprev = zeros(n)

    # The viscoelastic element makes the same election over a retained Mandel
    # factorization instead of a scalar slope, and it is the same arithmetic
    # there too — `A` is a closed form in the `ℂ` and `γ̃` the kernel already
    # has.
    vstored    = visco_testbed(strategy, qrc)
    vrecompute = visco_testbed(strategy, qrc; corrector = Recompute())
    m  = unknown_size(vstored.op)
    vs = 1.0e-3 .* sin.(0.4 .* (1:m)); vr = copy(vs)
    vprev = zeros(m)

    @testset "$name" for (name, sop, rop, utrial, urecomputed, uzero, p, c) in (
        ("power-law relaxation", stored.op, recompute.op, us, ur, uprev, mat, ctx),
        ("linear viscoelasticity", vstored.op, vrecompute.op, vs, vr, vprev, nothing,
         TimeIntegrationContext(0.0, 0.5, 0.5)),
    )
        # Structural: the recomputing cache allocates no corrector store.
        @test element_cache_under_decoration(sop).correctors isa FerriteOperators.ItemStates
        @test element_cache_under_decoration(rop).correctors === nothing

        ss = condensed_states(utrial, uzero); sr = condensed_states(urecomputed, uzero)
        rs = condense_internal!(sop, ss, p, c)
        rr = condense_internal!(rop, sr, p, c)
        @test rs.converged && rr.converged
        @test rs.iterations == rr.iterations
        @test utrial == urecomputed                  # the same trial q in the tail

        # The recomputed slopes are the SAME arithmetic on the same converged
        # pair — the closed-form `∂R/∂q` at `(u, q)` — so the assembled
        # matrices agree bitwise, not merely to round-off.
        Ks = zeros(residual_size(sop)); Kr = similar(Ks)
        update_linearization!(sop, Ks, ss, p, c)
        update_linearization!(rop, Kr, sr, p, c)
        @test sop.J.nzval == rop.J.nzval
        @test Ks == Kr
    end

    # Only the power-law element carries a parameter corrector, so the ∂F/∂θ
    # half of the election is its alone.
    @test element_cache_under_decoration(recompute.op).param_correctors === nothing
    ss = condensed_states(us, uprev); sr = condensed_states(ur, uprev)
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
end

@testset "Recompute() freshness: the q contract survives, the corrector class does not" begin
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    (; op, n, u, states, ctx) = relaxation_case(strategy, qrc;
                                                material = NortonRelaxationParameters(), corrector = Recompute())
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
    (; dh, qrc) = scalar_quad_testbed((1, 1))
    integ = SimpleCondensedPowerLawRelaxation(NortonRelaxationParameters(), qrc, :u, :q)
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
    (; op, dh, grid, u, uprev, states, ctx) =
        relaxation_case(strategy, qrc; material = mat, frequency = 0.6)
    ivh = op.engine.ivh
    condense_internal!(op, states, mat, ctx)

    Kq = allocate_internal_jacobian(op)
    @test size(Kq) == (residual_size(op), ndofs(ivh))
    update_internal_jacobian!(Kq, op, states, mat, ctx)

    # Dense hand-rolled reference: the per-cell kernel output scattered by
    # (celldofs, internal range) — the assembly the sweep automates.
    cache = element_cache_under_decoration(op)
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

    check_internal_jacobian_columns(Kq, op, u, uprev, mat, ctx)

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
    (; op, dh, grid, n, u, states, ctx) =
        relaxation_case(strategy, qrc; material = NortonRelaxationParameters(), field = 1.0, γ̃ = 0.5)

    condense_internal!(op, states, nothing, ctx)
    qtail = view(u, (ndofs(dh)+1):n)

    # InternalSource gathers exactly the cell's internal_variable_range.
    ivh = op.engine.ivh
    for cellid in 1:getncells(grid)
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
        cache = element_cache_under_decoration(op)
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

    @testset "a split-analytic condensed element keeps its θ/t route" begin
        params = TimedRelaxationParameters(α = 0.8, τ = 1.3)
        ctx = TimeIntegrationContext(0.7, 0.4, 0.4)
        ref = timed_relaxation_testbed(strategy, qrc; params)

        dh = scalar_quad_testbed((2, 2)).dh
        split_op = setup_operator(strategy,
            SplitTimedRelaxationIntegrator(TimedRelaxationIntegrator(params, qrc, :u)),
            dh; slots = (:u, :q, :qprev))

        # Both wrappers are really there — the θ/t hook sits two levels down.
        outer = first_element_cache(split_op)
        @test outer isa ADElementCache
        @test outer.inner isa FusedFromSplit
        @test FerriteOperators.serves_kind(typeof(outer), ParameterJacobianKind())
        @test FerriteOperators.serves_kind(typeof(outer), TimeSensitivityKind())

        n = unknown_size(ref.op)
        u_ref = 0.3 .* sin.(0.6 .* (1:n)); u_split = copy(u_ref)
        uprev = zeros(n)
        s_ref = condensed_states(u_ref, uprev); s_split = condensed_states(u_split, uprev)
        condense_internal!(ref.op, s_ref, params, ctx)
        condense_internal!(split_op, s_split, params, ctx)
        @test u_ref ≈ u_split

        nθ = length(parameter_vector(params))
        Bref = zeros(residual_size(ref.op), nθ); Bsplit = similar(Bref)
        update_parameter_jacobian!(Bref, ref.op, s_ref, params, ctx)
        update_parameter_jacobian!(Bsplit, split_op, s_split, params, ctx)
        @test Bsplit ≈ Bref rtol = 1e-10
        @test norm(Bsplit[:, 2]) > 1e-3   # the correction the hook supplies

        gref = zeros(residual_size(ref.op)); gsplit = similar(gref)
        time_sensitivity!(gref, ref.op, s_ref, params, ctx)
        time_sensitivity!(gsplit, split_op, s_split, params, ctx)
        @test gsplit ≈ gref rtol = 1e-10
        @test norm(gsplit) > 1e-3

        # The fused Jacobian+residual request runs the two split kernels back
        # to back, so the split operator's linearization matches too.
        rref = zeros(residual_size(ref.op)); rsplit = similar(rref)
        update_linearization!(ref.op, rref, s_ref, params, ctx)
        update_linearization!(split_op, rsplit, s_split, params, ctx)
        @test rsplit ≈ rref rtol = 1e-12
        @test Matrix(split_op.J) ≈ Matrix(ref.op.J) rtol = 1e-12
    end

    @testset "hidden analytic kernels: the hook reproduces what they compute" begin
        mat = NortonRelaxationParameters(κ = 1.3, α = 0.8, η = 1.4, n = 2.5)
        ctx = TimeIntegrationContext(0.0, 0.4, 0.4)
        ref = relaxation_testbed(strategy, qrc; material = mat)

        counter = Ref(0)
        dh = scalar_quad_testbed((2, 2)).dh
        hidden_op = setup_operator(strategy,
            ForwardingIntegrator{true}(SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q), counter),
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

    @testset "neither analytic kernels nor the hook: the parameter kinds stay refused" begin
        # The power-law cache wrapped WITHOUT `local_conditions!`: the frozen-q
        # AD partial is all the decorator can produce, so the parameter kinds
        # are refused with a message naming every route that would admit them.
        mat = NortonRelaxationParameters()
        dh = scalar_quad_testbed((2, 2)).dh
        op = setup_operator(strategy,
            ForwardingIntegrator{false}(SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q)),
            dh; slots = (:u, :q, :qprev))
        n = unknown_size(op)
        states = condensed_states(0.1 .* sin.(1:n), zeros(n))
        ctx = TimeIntegrationContext(0.0, 0.4, 0.4)
        condense_internal!(op, states, mat, ctx)

        err = @test_throws ArgumentError update_parameter_jacobian!(
            zeros(residual_size(op), 3), op, states, mat, ctx)
        msg = err.value.msg
        @test occursin("ForwardingCache carries condensed internal state", msg)
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
    (; op, states, ctx) = relaxation_case(strategy, qrc; material = mat, frequency = 0.6)

    # A condensed cache with a corrector store serving ParameterJacobianKind
    # analytically: the partial (∂F/∂θ|_q, from the pure residual) plus
    # ∂F/∂q·dq/dθ, dq/dθ computed and stored by `condense_cell!` alongside
    # dq/du.
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
