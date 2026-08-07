using FerriteOperators
using Test

struct NeoHookeanPatchTest
    E::Float64
    ν::Float64
end
function (mat::NeoHookeanPatchTest)(F)
    (; E, ν) = mat
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    C = tdot(F)
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

# References are assembled by direct kernel invocation over the same cells and
# dof maps — the test validates the patch loop and scatter mechanics.
function reference_patch_matrix(provider, i, cache, dh, u, p)
    np = patch_ndofs(provider, i)
    KP = zeros(np, np)
    dofmap = Dict(g => l for (l, g) in pairs(patch_dofs(provider, i)))
    cc = Ferrite.CellCache(dh)
    n = ndofs_per_cell(dh.subdofhandlers[1])
    Ke = zeros(n, n)
    uₑ = zeros(n)
    for cellid in sort(collect(provider.patches[i]))
        Ferrite.reinit!(cc, cellid)
        reinit_values!(cache, cc)
        uₑ .= u[celldofs(cc)]
        fill!(Ke, 0.0)
        FerriteOperators.assemble_cell!(JacobianRequest{:u}(Ke), cache,
            KernelArgs((u = uₑ,), cc, p, nothing, nothing))
        dofs = celldofs(cc)
        for (j, gj) in pairs(dofs), (i2, gi) in pairs(dofs)
            KP[dofmap[gi], dofmap[gj]] += Ke[i2, j]
        end
    end
    return KP
end

@testset "Patch assembly" begin
    grid = generate_grid(Quadrilateral, (4, 4))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(2)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

    # two overlapping interior patches plus one touching the boundary
    cellsets = [[1, 2, 5, 6], [6, 7, 10, 11], [13, 14]]
    sdh = dh.subdofhandlers[1]
    provider = PatchItems(sdh, cellsets)

    @testset "provider dof maps" begin
        @test patch_ndofs(provider, 1) == 9        # 2×2 Q1 patch
        @test patch_ndofs(provider, 3) == 6        # 1×2 Q1 patch
        @test issorted(patch_dofs(provider, 1))
        @test_throws ArgumentError PatchItems(sdh, [[1, 999]])
    end

    @testset "bilinear diffusion patches match direct kernel references" begin
        op = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.3, qrc, :u), dh)
        u = zeros(ndofs(dh))
        dest = [zeros(patch_ndofs(provider, i), patch_ndofs(provider, i)) for i in 1:3]
        assemble_patch_matrices!(dest, op, provider, u, nothing)
        cache = op.engine.subdomain_caches[1].domain.element
        for i in 1:3
            KPref = reference_patch_matrix(provider, i, cache, dh, u, nothing)
            @test dest[i] ≈ KPref rtol = 1e-14
            @test !iszero(dest[i])
        end
    end

    @testset "nonlinear (state-dependent) patches match direct kernel references" begin
        hgrid = generate_grid(Hexahedron, (2, 2, 2))
        hdh = DofHandler(hgrid)
        add!(hdh, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(hdh)
        hint = FerriteOperators.SimpleHyperelasticityIntegrator(NeoHookeanPatchTest(10.0, 0.3), qrc, :u)
        hop = setup_operator(strategy, hint, hdh)
        hu = 0.05 .* sin.(0.3 .* (1:ndofs(hdh)))
        hprovider = PatchItems(hdh.subdofhandlers[1], [[1, 2, 3, 4], [5, 6]])
        dest = [zeros(patch_ndofs(hprovider, i), patch_ndofs(hprovider, i)) for i in 1:2]
        assemble_patch_matrices!(dest, hop, hprovider, hu, 0.0)
        cache = hop.engine.subdomain_caches[1].domain.element
        for i in 1:2
            KPref = reference_patch_matrix(hprovider, i, cache, hdh, hu, 0.0)
            @test dest[i] ≈ KPref rtol = 1e-13
        end
        # re-running overwrites, not accumulates
        first_result = deepcopy(dest[1])
        assemble_patch_matrices!(dest, hop, hprovider, hu, 0.0)
        @test dest[1] == first_result
    end

    @testset "guards" begin
        op = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        u = zeros(ndofs(dh))
        @test_throws DimensionMismatch assemble_patch_matrices!([zeros(1, 1)], op, provider, u, nothing)
        odh = DofHandler(generate_grid(Quadrilateral, (2, 2)))
        add!(odh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(odh)
        foreign = PatchItems(odh.subdofhandlers[1], [[1]])
        dest = [zeros(patch_ndofs(foreign, 1), patch_ndofs(foreign, 1))]
        @test_throws ArgumentError assemble_patch_matrices!(dest, op, foreign, u, nothing)
    end
end
