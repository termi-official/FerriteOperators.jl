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

    @testset "local Newton hits the root and the assembly is exact" begin
        tb  = relaxation_testbed(strategy, qrc; material = mat)
        op, dh, grid = tb.op, tb.dh, tb.grid
        γ̃   = 0.5
        ctx = TimeIntegrationContext(0.0, γ̃, γ̃)

        # A constant field has the same value at every quadrature point, so
        # every local problem has the same known root.
        u = zeros(unknown_size(op)); view(u, 1:ndofs(dh)) .= 1.0
        uprev = zeros(unknown_size(op))
        r = zeros(residual_size(op))
        evaluate!(op, r, (u = u, uprev = uprev), nothing, ctx)

        qref = reference_internal_state(mat, 1.0, 0.0, γ̃)
        q    = view(u, (ndofs(dh)+1):unknown_size(op))   # trial write-back
        @test all(qi -> isapprox(qi, qref; atol = 1e-11), q)
        # ∇u ≡ 0, so the residual is the exchange term only and sums to α(u−q)|Ω|.
        @test sum(r) ≈ mat.α * (1.0 - qref) * 4.0 rtol = 1e-12
    end

    @testset "consistent tangent through the local solve" begin
        tb = relaxation_testbed(strategy, qrc; material = mat)
        op = tb.op
        n  = unknown_size(tb.op)
        u  = 0.3 .* sin.(0.7 .* (1:n))
        states = (u = u, uprev = zeros(n))
        ctx = TimeIntegrationContext(0.0, 0.4, 0.4)

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
        u     = zeros(n); view(u, 1:ndofs(tb.dh)) .= 2.0
        states = (u = u, uprev = zeros(n))
        base   = TimeIntegrationContext(0.0, 1.0e3, 1.0e3)
        r      = zeros(residual_size(op))

        reset_local_solve_statistics!(op)
        evaluate!(op, r, states, nothing, base)
        tight = local_solve_statistics(op)

        u .= 0.0; view(u, 1:ndofs(tb.dh)) .= 2.0
        reset_local_solve_statistics!(op)
        evaluate!(op, r, states, nothing, InexactLocalSolveContext(base, 1.0e-2))
        loose = local_solve_statistics(op)

        @test tight.solves == loose.solves
        @test loose.iterations < tight.iterations
        @test loose.worst_iterations < tight.worst_iterations

        # A requested tolerance tighter than the element's own floor is ignored.
        u .= 0.0; view(u, 1:ndofs(tb.dh)) .= 2.0
        reset_local_solve_statistics!(op)
        evaluate!(op, r, states, nothing, InexactLocalSolveContext(base, 1.0e-30))
        @test local_solve_statistics(op).iterations == tight.iterations

        # The decoration survives the framework's context handling.
        @test evaluation_time(InexactLocalSolveContext(base, 1.0e-2)) == 0.0
        @test local_solve_tolerance(FerriteOperators.with_time(InexactLocalSolveContext(base, 1.0e-2), 3.0)) == 1.0e-2
        @test local_solve_tolerance(base) === nothing
        # ∂F/∂t of a time-independent element, through a custom context type.
        g = zeros(residual_size(op))
        time_sensitivity!(g, op, states, nothing, InexactLocalSolveContext(base, 1.0e-2);
                          method = FiniteDifferenceSensitivity())
        @test norm(g) < 1e-8
    end

    @testset "inner → outer: statistics accumulate, merge over workers and reset" begin
        tb  = relaxation_testbed(strategy, qrc; material = mat)
        op  = tb.op
        n   = unknown_size(op)
        u   = 0.5 .* sin.(0.9 .* (1:n))
        states = (u = u, uprev = zeros(n))
        ctx = TimeIntegrationContext(0.0, 1.0, 1.0)

        nqp    = getnquadpoints(first_element_cache(op).cv)
        ncells = getncells(tb.grid)

        reset_local_solve_statistics!(op)
        evaluate!(op, zeros(residual_size(op)), states, nothing, ctx)
        stats = local_solve_statistics(op)
        @test stats.solves == ncells * nqp
        @test stats.iterations ≥ stats.solves
        @test stats.worst_iterations ≥ 1
        @test stats.worst_cell ∈ 1:ncells
        @test stats.worst_qp ∈ 1:nqp

        reset_local_solve_statistics!(op)
        @test local_solve_statistics(op).solves == 0
        @test local_solve_statistics(op).worst_iterations == 0

        # Per-worker accumulators merge to the same totals.
        ptb = relaxation_testbed(PerColorAssemblyStrategy(PolyesterDevice(2)), qrc; material = mat)
        pu  = 0.5 .* sin.(0.9 .* (1:unknown_size(ptb.op)))
        reset_local_solve_statistics!(ptb.op)
        evaluate!(ptb.op, zeros(residual_size(ptb.op)), (u = pu, uprev = zeros(length(pu))), nothing, ctx)
        @test local_solve_statistics(ptb.op).solves == stats.solves
    end

    @testset "non-convergence fails loudly" begin
        budget = LocalNewtonSettings(max_iterations = 2, tolerance = 1e-12)
        tb  = relaxation_testbed(strategy, qrc; material = mat, local_solver = budget)
        n   = unknown_size(tb.op)
        u   = zeros(n); view(u, 1:ndofs(tb.dh)) .= 2.0
        ctx = TimeIntegrationContext(0.0, 1.0e3, 1.0e3)
        @test_throws LocalSolveNotConvergedError evaluate!(
            tb.op, zeros(residual_size(tb.op)), (u = u, uprev = zeros(n)), nothing, ctx)

        # The same problem within the default budget converges.
        tb2 = relaxation_testbed(strategy, qrc; material = mat)
        u2  = zeros(n); view(u2, 1:ndofs(tb2.dh)) .= 2.0
        evaluate!(tb2.op, zeros(residual_size(tb2.op)), (u = u2, uprev = zeros(n)), nothing, ctx)
        @test local_solve_statistics(tb2.op).worst_iterations > 2
    end
end
