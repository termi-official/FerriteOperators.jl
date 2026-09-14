# FerriteOperators is unregistered and lives one level up in this repo. A
# [sources] section would wire it in by path, but Pkg only understands
# [sources] from Julia 1.11 on while this package's compat floor is 1.10 — so
# the unresolvable path dep is dev'ed into the active test environment before
# anything loads.
import Pkg
let specs = Pkg.PackageSpec[]
    resolvable(name, uuid) = Base.locate_package(Base.PkgId(Base.UUID(uuid), name)) !== nothing
    resolvable("FerriteOperators", "27d9367a-5072-424e-9c5f-fe582399bac3") ||
        push!(specs, Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..")))
    isempty(specs) || Pkg.develop(specs)
end

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
        AssemblyStrategy(SequentialCPUDevice()),
        SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u),
        dh,
    )
    update_operator!(op, nothing)
    K = op.A
    # The diffusion stiffness is symmetric with a constant nullspace.
    @test K ≈ K'
    @test norm(K * ones(ndofs(dh))) < 1.0e-10

    # The two-sided element assembles over a DISCONTINUOUS space, and its
    # interior-facet coupling is declared rather than inferred.
    dg_dh = let g = generate_grid(Quadrilateral, (2, 2))
        handler = DofHandler(g)
        add!(handler, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
        close!(handler)
    end
    dg_op = setup_operator(
        AssemblyStrategy(SequentialCPUDevice();
            form = FullAssembly(StandardOperatorSpecification(;
                sparsity_entries = interior_facet_entries!))),
        SIPGDiffusionIntegrator(1.0, 4.0, QuadratureRuleCollection(2),
                                FerriteOperators.FacetQuadratureRuleCollection(2), :u),
        dg_dh,
    )
    update_operator!(dg_op, nothing)
    A = dg_op.A
    # SIPG is symmetric and annihilates constants: the jump of a constant is
    # zero across every interface, and so is its gradient inside every cell.
    @test A ≈ A'
    @test norm(A * ones(ndofs(dg_dh))) < 1.0e-10 * norm(A)
end
