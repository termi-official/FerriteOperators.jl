using NonlinearSolve, LinearSolve, FerriteOperators, Test
using ConcreteStructs

@concrete struct CondensedLinearSolveWrapper <: LinearSolve.SciMLLinearSolveAlgorithm
    alg
    internal_offset
end
function LinearSolve.init_cacheval(alg::CondensedLinearSolveWrapper, A, b, u, args...; kwargs...)
    sz = length(u)
    resize!(u, alg.internal_offset-1)
    inner_cache = LinearSolve.init_cacheval(alg.alg, A, b, u, args...; kwargs...)
    resize!(u, sz)
    return inner_cache
end
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CondensedLinearSolveWrapper; kwargs...)
    sz = length(cache.u)
    resize!(cache.u, alg.internal_offset-1)
    sol = solve!(cache, alg.alg; kwargs...)
    resize!(cache.u, sz)
    cache.u[alg.internal_offset:end] .= 0.0
    return sol
end

@concrete struct CondensedConstrainedLinearSolveWrapper <: LinearSolve.SciMLLinearSolveAlgorithm
    alg
    internal_offset
    ch
end
function LinearSolve.init_cacheval(alg::CondensedConstrainedLinearSolveWrapper, A, b, u, args...; kwargs...)
    sz = length(u)
    resize!(u, alg.internal_offset-1)
    inner_cache = LinearSolve.init_cacheval(alg.alg, A, b, u, args...; kwargs...)
    resize!(u, sz)
    return inner_cache
end
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CondensedConstrainedLinearSolveWrapper; kwargs...)
    sz = length(cache.u)
    resize!(cache.u, alg.internal_offset-1)
    apply_zero!(cache.A, cache.b, alg.ch)
    sol = solve!(cache, alg.alg; kwargs...)
    apply_zero!(cache.u, alg.ch)
    resize!(cache.u, sz)
    cache.u[alg.internal_offset:end] .= 0.0
    return sol
end

@testset "NonlinearSolve Integration" begin
    integrator = FerriteOperators.SimpleCondensedLinearViscoelasticity(
        FerriteOperators.MaxwellParameters(),
        QuadratureRuleCollection(2),
        :u,
        :εᵛ,
    )

    grid = generate_grid(Hexahedron, (3,3,3))
    Ferrite.transform_coordinates!(grid, x->Vec{3}(sign.(x.-0.5) .* (x.-0.5).^2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)

    residual = zeros(ndofs(dh))
    u        = zeros(ndofs(dh) + 6*8*(3*3*3)) # TODO get via interface
    uprev    = zeros(ndofs(dh) + 6*8*(3*3*3)) # TODO get via interface
    apply_analytical!(u, dh, :u, x->0.01x.^2)

    function f!(r, u, (nlop, ch, p))
        nlop(r, u, p)
        apply_zero!(r, ch)
    end
    function jac!(J, u, (nlop, ch, p))
        update_linearization!(nlop, u, p)
        J .= nlop.J
        apply!(J, ch)
    end

    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    nlop = setup_operator(strategy, integrator, dh)

    ch = ConstraintHandler(dh);
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> (0,0,0)));
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> (0.01,0,0)));
    close!(ch)

    p = (nlop, ch, FerriteOperators.ImplicitEulerInfo(uprev, π, 0.0))

    jac_prototype = nlop.J
    resid_prototype = residual
    nlf = NonlinearFunction(f!; jac=jac!, jac_prototype, resid_prototype)

    apply!(u, ch)
    prob = NonlinearProblem(nlf, u, p)
    # iter = init(prob, NewtonRaphson(; concrete_jac=Val{true}, linsolve=CondensedConstrainedLinearSolveWrapper(LinearSolve.LUFactorization(), ndofs(dh)+1, ch)))
    iter = init(prob, NewtonRaphson(; concrete_jac=Val{true}, linsolve=CondensedLinearSolveWrapper(LinearSolve.LUFactorization(), ndofs(dh)+1)))
    solve!(iter)

    # Precomputed
    @test norm(iter.u[1:ndofs(dh)]) ≈ 0.059623465672897884
    @test norm(iter.u[ndofs(dh)+1:end]) ≈ 0.062203435313135984
    @test norm(uprev) ≈ 0.0
end
