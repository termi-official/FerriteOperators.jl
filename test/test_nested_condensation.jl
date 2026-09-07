using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using Polyester
using LinearAlgebra: norm

include(joinpath(@__DIR__, "fixture_elements.jl"))

####################################
## Nested testbed — a macro bar carrying a micro bar per quadrature point
####################################
# The macroscopic problem is a 1D bar; every macro quadrature point owns a micro
# bar of `micro_elements` [`SimpleRelaxingBar`](@ref) elements with its own
# condensed internal variable per micro quadrature point. `moduli` makes the
# micro bar heterogeneous — an empty declaration is the homogeneous bar the
# single-material law is the reference for. `local_solver` budgets the MICRO
# NEWTON, `micro_solver` the micro material's own local solve one level below.
function nested_testbed(strategy; dims = (3,), micro_elements = 4,
                        material = RelaxingBarParameters(),
                        moduli = [1.0, 2.5, 0.7, 1.8],
                        local_solver = LocalNewtonSettings(),
                        micro_solver = LocalNewtonSettings())
    grid = generate_grid(Line, dims)
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine, 1}())
    close!(dh)
    qrc   = QuadratureRuleCollection(2)
    micro = SimpleRelaxingBar(material, qrc, :u, :q; moduli, local_solver = micro_solver)
    op    = setup_operator(strategy, SimpleNestedHomogenization(micro, micro_elements, qrc, :u, :q; local_solver),
                           dh; slots = (:u, :q, :qprev))
    return (; op, dh, grid, qrc, micro)
end

"""
    nested_case(strategy; kwargs...)

[`nested_testbed`](@ref) plus the trial point the nested tests stand on: a
smooth, non-affine `ū` (so the macroscopic strain differs from quadrature point
to quadrature point), a zero committed state, their states NamedTuple and a
stage context. `kwargs` reach the testbed.
"""
function nested_case(strategy; amplitude = 0.08, frequency = 1.3, kwargs...)
    tb = nested_testbed(strategy; kwargs...)
    n  = unknown_size(tb.op)
    u  = zeros(n)
    view(u, 1:residual_size(tb.op)) .= amplitude .* sin.(frequency .* (1:residual_size(tb.op)))
    uprev = zeros(n)
    return (; tb..., n, u, uprev, states = condensed_states(u, uprev),
              ctx = TimeIntegrationContext(0.0, 0.4, 0.4))
end

# The element cache of one worker, looked through its AD decoration.
worker_caches(op) = [FerriteOperators.unwrap(ws.element)
                     for ws in first(op.engine.subdomain_caches).device_cache]

@testset "Nested condensation: a local problem that is itself a condensed FE problem" begin
    strategy = AssemblyStrategy(SequentialCPUDevice())

    @testset "route A: the whole micro state rides the macro [ū; q] tail" begin
        (; op, dh, grid, u, states, ctx) = nested_case(strategy)
        cache  = element_cache_under_decoration(op)
        nqp    = getnquadpoints(cache)
        nmicro = unknown_size(cache.workspace.op)
        ivh    = op.engine.ivh

        # Per-quadrature-point micro states, not one scalar per quadrature point.
        @test length(internal_variable_range(ivh, 1)) == nqp * nmicro
        @test unknown_size(op) == ndofs(dh) + getncells(grid) * nqp * nmicro

        condense_internal!(op, states, nothing, ctx)

        # The micro state the tail carries for cell 1 is a micro SOLUTION: its
        # driven dofs hold the macroscopic strain of that quadrature point, and
        # its interior is what the micro Newton put there. Built through the
        # documented kernel-testing seam.
        cc = Ferrite.CellCache(dh)
        reinit!(cc, 1)
        FerriteOperators.reinit_values!(cache, cc)
        uₑ    = u[celldofs(cc)]
        slice = reshape(u[internal_variable_range(ivh, 1)], (nmicro, nqp))
        for qp in 1:nqp
            ε̄ = function_gradient(cache.cv, qp, uₑ)[1]
            @test slice[cache.micro.driven, qp] ≈ ε̄ .* cache.micro.driven_coordinates
        end
        @test !all(iszero, slice[cache.micro.free, :])
    end

    @testset "misconfigured micro problems are rejected at setup" begin
        @test_throws "at least two elements" nested_testbed(strategy; micro_elements = 1, moduli = Float64[])
        @test_throws "moduli for a grid of" nested_testbed(strategy; micro_elements = 3)
    end

    @testset "the homogenized tangent against the finite-difference referee" begin
        (; op, states, ctx) = nested_case(strategy)

        # Every finite-difference probe re-runs every micro Newton, so the
        # referee is the TOTAL response — which is what the two composed
        # implicit function theorems (micro q inside the micro tangent, micro ū
        # by its Schur complement) have to reproduce.
        res = check_derivatives(op, states, nothing, ctx)
        @test res.passed
        @test res.checks.jacobian.passed
        @test res.checks.jacobian.skipped === nothing
        @test res.checks.jacobian.err < 1.0e-8
        @test res.checks.fused_residual.passed

        # The micro problem is genuinely solved, not evaluated once: more
        # Newton iterations than quadrature points.
        report = condense_internal!(op, states, nothing, ctx)
        @test report.converged
        @test report.iterations > report.solves
    end

    @testset "the correction mode composes across the two levels" begin
        (; op, n, u, uprev, states, ctx) = nested_case(strategy)
        nres = residual_size(op)
        condense_internal!(op, states, nothing, ctx)

        Kc = similar(op.J); Kf = similar(op.J)
        FerriteOperators.assemble_into!(JacobianKind{:u, Consistent}(), (Kc,), op, states, nothing, ctx)
        FerriteOperators.assemble_into!(JacobianKind{:u, FrozenQ}(), (Kf,), op, states, nothing, ctx)
        @test Kc.nzval != Kf.nzval

        # The elected partial has its own referee: finite differences taken
        # WITHOUT re-condensing hold every micro state fixed, which is what
        # `FrozenQ` names one level up. `check_derivatives` deliberately has no
        # reference for it and says so.
        uw = copy(u)
        statesw = condensed_states(uw, uprev)
        rp = zeros(nres); rm = zeros(nres)
        h = 1.0e-6
        for k in 1:2
            v = [sin(0.7k * i + 0.3k) for i in 1:nres]
            uw .= u; view(uw, 1:nres) .+= h .* v
            evaluate!(op, rp, statesw, nothing, ctx)
            uw .= u; view(uw, 1:nres) .-= h .* v
            evaluate!(op, rm, statesw, nothing, ctx)
            @test Kf * v ≈ (rp .- rm) ./ 2h rtol = 1.0e-6
        end

        res = check_derivatives(op, states, nothing, ctx; correction = FrozenQ)
        @test res.checks.jacobian.skipped !== nothing
    end

    @testset "freshness: condense, roll back, re-condense" begin
        (; op, u, states, ctx) = nested_case(strategy)
        nres = residual_size(op)
        r0 = zeros(nres); rs = zeros(nres); rf = zeros(nres)

        condense_internal!(op, states, nothing, ctx)
        update_linearization!(op, r0, states, nothing, ctx)
        J0        = copy(op.J.nzval)
        committed = copy(u)

        # Condensing again at the same trial point reproduces the same tail bit
        # for bit: the micro Newton starts from the COMMITTED micro state, so
        # phase one is a function of `(ū, qprev)` and not of its own history.
        condense_internal!(op, states, nothing, ctx)
        @test u == committed

        # The documented hazard: moving `ū` in place without re-condensing is
        # silently wrong, not an error — the sweep evaluates the stale micro
        # states at the new macroscopic strain. The move has to change the
        # STRAIN: a rigid translation of the macro field leaves every micro
        # problem where it was.
        view(u, 1:nres) .+= 0.02 .* (1:nres)
        evaluate!(op, rs, states, nothing, ctx)
        condense_internal!(op, states, nothing, ctx)
        evaluate!(op, rf, states, nothing, ctx)
        @test rs != rf
        @test norm(rs - rf) / norm(rf) > 1.0e-3

        # Rolling back restores the trial point AND drops the homogenized
        # tangents, so a `Consistent` sweep refuses until phase one has run.
        rollback_state!(op, u, committed)
        @test_throws ArgumentError update_linearization!(op, rs, states, nothing, ctx)

        r1 = zeros(nres)
        condense_internal!(op, states, nothing, ctx)
        update_linearization!(op, r1, states, nothing, ctx)
        @test u == committed          # the re-condensed micro states, bit for bit
        @test r1 == r0
        @test op.J.nzval == J0
    end

    @testset "the standard freshness contract" begin
        (; op, u, states, ctx) = nested_case(strategy)
        check_freshness_contract(op, states, u, ctx)
    end

    @testset "residual-only condensation solves the same micro problems" begin
        (; op, u, states, ctx) = nested_case(strategy)
        r = zeros(residual_size(op))

        weighted = condense_internal!(op, (u = 1.0,), states, nothing, ctx)
        q_weighted = copy(u)
        residual_only = condense_internal!(op, nothing, states, nothing, ctx)

        # Bitwise: the election governs what is formed AFTER the micro solves.
        @test u == q_weighted
        @test residual_only.iterations == weighted.iterations
        @test residual_only.converged == weighted.converged

        # And what it costs: no homogenized tangent was formed.
        @test_throws ArgumentError update_linearization!(op, r, states, nothing, ctx)
        condense_internal!(op, (u = 1.0,), states, nothing, ctx)
        update_linearization!(op, r, states, nothing, ctx)   # restored
    end

    @testset "the report folds failures of both levels" begin
        # Both budgets at their defaults converge on this problem.
        budgeted = nested_case(strategy)
        ok = condense_internal!(budgeted.op, budgeted.states, nothing, budgeted.ctx)
        @test ok.converged
        @test ok.worst_iterations > 1

        # The MICRO NEWTON capped below what the same trial point needs.
        capped = nested_case(strategy; local_solver = LocalNewtonSettings(max_iterations = 1))
        report = condense_internal!(capped.op, capped.states, nothing, capped.ctx)
        @test report.converged == false
        @test report.worst_iterations == 1
        @test report.worst_cell ∈ 1:getncells(capped.grid)
        @test report.worst_qp ∈ 1:getnquadpoints(element_cache_under_decoration(capped.op))

        # The micro MATERIAL's own local solve capped instead: the level below
        # the micro Newton, folded into the same flag.
        inner = nested_case(strategy; micro_solver = LocalNewtonSettings(max_iterations = 0))
        @test condense_internal!(inner.op, inner.states, nothing, inner.ctx).converged == false
    end

    @testset "a threaded device agrees with the sequential one" begin
        seq = nested_case(strategy)
        par = nested_case(AssemblyStrategy(PolyesterDevice(min_items_per_worker = 1)))

        caches = worker_caches(par.op)
        if Threads.nthreads() > 1
            @test length(caches) > 1
            # Each worker owns its micro operator — matrix, element caches and
            # corrector stores included — while the immutable micro problem and
            # the item-keyed tangent store are shared.
            @test allunique(objectid(c.workspace.op) for c in caches)
            @test allunique(objectid(c.workspace.x) for c in caches)
            @test allunique(objectid(c.workspace.op.J) for c in caches)
            @test all(c -> c.micro === first(caches).micro, caches)
            @test all(c -> c.tangents === first(caches).tangents, caches)
        end

        rseq = condense_internal!(seq.op, seq.states, nothing, seq.ctx)
        rpar = condense_internal!(par.op, par.states, nothing, par.ctx)
        @test rpar.solves == rseq.solves
        @test rpar.iterations == rseq.iterations
        @test rpar.converged == rseq.converged
        @test par.u == seq.u                      # the same micro states in the tail

        r_seq = zeros(residual_size(seq.op)); r_par = zeros(residual_size(par.op))
        update_linearization!(seq.op, r_seq, seq.states, nothing, seq.ctx)
        update_linearization!(par.op, r_par, par.states, nothing, par.ctx)
        @test r_par ≈ r_seq rtol = 1.0e-14
        @test par.op.J.nzval ≈ seq.op.J.nzval rtol = 1.0e-14
    end

    @testset "a homogeneous micro bar is the single-material law" begin
        # With every micro modulus at 1 the micro solution of a driven bar is
        # the affine field, so the homogenized stress and tangent are the micro
        # material's own — the nested element reduces to the element it nests.
        material = RelaxingBarParameters()
        nested = nested_case(strategy; moduli = Float64[], material)
        (; op, dh, qrc, u, ctx) = nested

        ref_op = setup_operator(strategy, SimpleRelaxingBar(material, qrc, :u, :q), dh;
                                slots = (:u, :q, :qprev))
        uref = zeros(unknown_size(ref_op))
        view(uref, 1:residual_size(ref_op)) .= view(u, 1:residual_size(op))
        sref = condensed_states(uref, zeros(length(uref)))

        condense_internal!(op, nested.states, nothing, ctx)
        condense_internal!(ref_op, sref, nothing, ctx)

        r = zeros(residual_size(op)); rref = zeros(residual_size(ref_op))
        update_linearization!(op, r, nested.states, nothing, ctx)
        update_linearization!(ref_op, rref, sref, nothing, ctx)

        @test r ≈ rref rtol = 1.0e-10
        @test op.J.nzval ≈ ref_op.J.nzval rtol = 1.0e-10
        @test norm(rref) > 1.0e-3          # the agreement is not agreement on zero
    end
end
