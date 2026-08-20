using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using Polyester
using LinearAlgebra: norm

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

        nqp    = getnquadpoints(first_element_cache(op).cv)
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

@testset "Condensation/corrector elections are construction-time seams" begin
    qrc = QuadratureRuleCollection(2)
    mat = NortonRelaxationParameters()
    vmat = MaxwellParameters()

    @test SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q) isa FerriteOperators.AbstractCondensedNonlinearIntegrator
    @test_throws ArgumentError SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q; condensation = FusedWithResidual())
    @test_throws ArgumentError SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q; corrector = Recompute())
    @test_throws ArgumentError SimpleCondensedLinearViscoelasticity(vmat, qrc, :u, :εᵛ; condensation = FusedWithResidual())
    @test_throws ArgumentError SimpleCondensedLinearViscoelasticity(vmat, qrc, :u, :εᵛ; corrector = Recompute())

    integ = SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q)
    @test FerriteOperators.condensation_election(integ) isa Separate
    @test FerriteOperators.corrector_election(integ) isa Stored
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
