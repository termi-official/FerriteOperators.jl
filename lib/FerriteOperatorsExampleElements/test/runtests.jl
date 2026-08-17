using FerriteOperators
using FerriteOperatorsExampleElements
using LinearAlgebra: norm
using Test

# Smoke test only — the behavioural coverage of these elements lives in
# FerriteOperators' own suite, which uses them as its fixtures.
@testset "FerriteOperatorsExampleElements" begin
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)

    op = setup_operator(
        SequentialAssemblyStrategy(SequentialCPUDevice()),
        SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u),
        dh,
    )
    update_operator!(op, nothing)
    K = op.A
    # The diffusion stiffness is symmetric with a constant nullspace.
    @test K ≈ K'
    @test norm(K * ones(ndofs(dh))) < 1.0e-10
end
