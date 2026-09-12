# FerriteOperators is unregistered and lives two levels up in this repo. A
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
using FerriteOperatorsTensorProduct
using LinearAlgebra: mul!, norm
using Test

# Smoke test only — the behavioural coverage of this core and its two
# reference elements lives in FerriteOperators' own suite, which uses them as
# its matrix-free fixtures.
@testset "FerriteOperatorsTensorProduct" begin
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)

    strategy = AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction())
    op = setup_operator(strategy, SumFactorizedDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u), dh)
    y = zeros(ndofs(dh))
    u = ones(ndofs(dh))
    mul!(y, op, u)
    # The diffusion action on a constant field is zero everywhere (the constant
    # nullspace of the stiffness operator).
    @test norm(y) < 1.0e-10
end
