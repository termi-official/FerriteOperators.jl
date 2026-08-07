using FerriteOperators
using SparseArrays
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

# A term payload: the element-specific data a patch request hands to
# `assemble_patch_cell!`. Weighted constant source, L(v) = w ∫ v dΩ.
struct WeightedSource
    w::Float64
end
function FerriteOperators.assemble_patch_cell!(
        req::ResidualRequest, cache::FerriteOperators.SimpleBilinearDiffusionElementCache,
        args::KernelArgs, data::WeightedSource
    )
    cv = cache.cellvalues
    for qp in 1:getnquadpoints(cv), i in 1:getnbasefunctions(cv)
        req.r[i] += data.w * shape_value(cv, qp, i) * getdetJdV(cv, qp)
    end
    return nothing
end

# Reference vector over a patch: the same term kernels, invoked directly, in
# the same cell order and the same per-cell term order.
function reference_patch_vector(provider, i, cache, dh, terms)
    v = zeros(patch_ndofs(provider, i))
    dofmap = Dict(g => l for (l, g) in pairs(patch_dofs(provider, i)))
    cc = Ferrite.CellCache(dh)
    re = zeros(ndofs_per_cell(dh.subdofhandlers[1]))
    for (k, cellid) in pairs(patch_cells(provider, i))
        group = patch_cell_groups(provider, i)[k]
        fill!(re, 0.0)
        Ferrite.reinit!(cc, cellid)
        reinit_values!(cache, cc)
        active = false
        for t in terms
            FerriteOperators.patch_term_active(t.restriction, group) || continue
            active = true
            FerriteOperators.assemble_patch_cell!(ResidualRequest(re), cache,
                KernelArgs((;), cc, nothing, nothing, nothing), t.data)
        end
        active || continue
        for (l, g) in pairs(celldofs(cc))
            v[dofmap[g]] += re[l]
        end
    end
    return v
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
        @test_throws DimensionMismatch PatchItems(sdh, [[1, 2]]; groups = [[1]])
        @test_throws ArgumentError PatchItems(sdh, [[1, 2]]; prescribed_facets = [[FacetIndex(9, 1)]])
    end

    @testset "term tuples with per-term domain restrictions" begin
        op = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        cache = op.engine.subdomain_caches[1].domain.element
        # cells 1,2 carry group 1; cells 5,6 carry group 2
        prov = PatchItems(sdh, [[1, 2, 5, 6]]; groups = [[1, 1, 2, 2]])
        terms = (PatchTerm(WholePatch(), WeightedSource(1.0)), PatchTerm(CellGroup(2), WeightedSource(10.0)))
        dest = [zeros(patch_ndofs(prov, 1))]
        assemble_patches!(PatchVectorKind(terms, PatchLocalSink(dest)), op, prov, (;), nothing)
        @test dest[1] == reference_patch_vector(prov, 1, cache, dh, terms)
        @test !iszero(dest[1])

        # A group-restricted term alone touches only its group's cells.
        only2 = (PatchTerm(CellGroup(2), WeightedSource(10.0)),)
        d2 = [zeros(patch_ndofs(prov, 1))]
        assemble_patches!(PatchVectorKind(only2, PatchLocalSink(d2)), op, prov, (;), nothing)
        @test d2[1] == reference_patch_vector(prov, 1, cache, dh, only2)
        # cell 1's exclusive dofs (not shared with cells 5,6) stay untouched
        shared = union(celldofs(dh, 5), celldofs(dh, 6))
        exclusive = [prov.dofmaps[1][g] for g in celldofs(dh, 1) if !(g in shared)]
        @test !isempty(exclusive)
        @test all(iszero, d2[1][exclusive])

        # ... and the whole-patch term alone is the difference
        onlyw = (PatchTerm(WholePatch(), WeightedSource(1.0)),)
        dw = [zeros(patch_ndofs(prov, 1))]
        assemble_patches!(PatchVectorKind(onlyw, PatchLocalSink(dw)), op, prov, (;), nothing)
        @test dest[1] ≈ dw[1] .+ d2[1] rtol = 1e-14

        @test whole_patch_terms(terms) === (terms[1],)
        @test whole_patch_terms(only2) === ()
    end

    @testset "sinks" begin
        op = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        cache = op.engine.subdomain_caches[1].domain.element
        terms = (PatchTerm(WholePatch(), WeightedSource(2.0)),)
        local_dest = [zeros(patch_ndofs(provider, i)) for i in 1:3]
        assemble_patches!(PatchVectorKind(terms, PatchLocalSink(local_dest)), op, provider, (;), nothing)

        @testset "additive global vector" begin
            g = zeros(ndofs(dh))
            assemble_patches!(PatchVectorKind(terms, PatchGlobalVectorSink(g)), op, provider, (;), nothing)
            ref = zeros(ndofs(dh))
            for i in 1:3, (l, gd) in pairs(patch_dofs(provider, i))
                ref[gd] += local_dest[i][l]
            end
            @test g == ref
            @test !iszero(g)
        end

        @testset "duplicate-summing triplets" begin
            # patches 1 and 2 share cell 6, so their columns overlap on rows
            sink = PatchTripletSink([1, 1, 2])
            assemble_patches!(PatchVectorKind(terms, sink), op, provider, (;), nothing)
            W = sparse(sink, ndofs(dh), 2)
            @test W[patch_dofs(provider, 3)[1], 2] == local_dest[3][1]
            shared = intersect(patch_dofs(provider, 1), patch_dofs(provider, 2))
            @test !isempty(shared)
            for gd in shared
                l1 = findfirst(==(gd), patch_dofs(provider, 1))
                l2 = findfirst(==(gd), patch_dofs(provider, 2))
                @test W[gd, 1] == local_dest[1][l1] + local_dest[2][l2]
            end
            # emission order is the item order; a caller merges its own chunks
            other = PatchTripletSink([1, 1, 2])
            emit_patch_column!(other, [3, 4], 7, [1.0, 2.0])
            @test (append!(other, sink); other.J == vcat([7, 7], sink.J))
        end

        @testset "matrix scatter modes agree" begin
            u = zeros(ndofs(dh))
            dense = [zeros(patch_ndofs(provider, i), patch_ndofs(provider, i)) for i in 1:3]
            assemble_patch_matrices!(dense, op, provider, u, nothing)
            patterned = map(1:3) do i
                n = patch_ndofs(provider, i)
                sparse(ones(n, n))
            end
            assemble_patches!(PatchMatrixKind(FerriteOperators.WHOLE_PATCH_TERMS, PatchAssemblerSink(patterned)),
                op, provider, (u = u,), nothing)
            for i in 1:3
                @test Matrix(patterned[i]) ≈ dense[i] rtol = 1e-14
            end
        end

        @testset "sink/kind guards" begin
            g = zeros(ndofs(dh))
            @test_throws ArgumentError assemble_patches!(
                PatchMatrixKind(terms, PatchGlobalVectorSink(g)), op, provider, (;), nothing)
            @test_throws ArgumentError assemble_patches!(
                PatchVectorKind(terms, PatchAssemblerSink([spzeros(1, 1)])), op, provider, (;), nothing)
        end
    end

    @testset "view-derived dof partition" begin
        cells = [1, 2, 5, 6]
        pf = [fi for fi in Ferrite.getfacetset(grid, "left") if fi[1] in cells]
        @test !isempty(pf)
        prov = PatchItems(sdh, [cells]; prescribed_facets = [pf])

        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, Set(pf), (x, t) -> 0.0))
        close!(ch)
        expected = sort([prov.dofmaps[1][g] for g in ch.prescribed_dofs])
        @test patch_prescribed_dofs(prov, 1) == expected
        @test patch_free_dofs(prov, 1) == setdiff(1:patch_ndofs(prov, 1), expected)

        # the classification is not closed: callers pin extra dofs themselves
        extra = first(patch_free_dofs(prov, 1))
        augment_prescribed_dofs!(prov, 1, [extra])
        @test patch_prescribed_dofs(prov, 1) == sort(vcat(expected, extra))
        @test !(extra in patch_free_dofs(prov, 1))
        @test_throws ArgumentError augment_prescribed_dofs!(prov, 1, [patch_ndofs(prov, 1) + 1])

        # no classification given ⇒ everything is free
        plain = PatchItems(sdh, [cells])
        @test isempty(patch_prescribed_dofs(plain, 1))
        @test patch_free_dofs(plain, 1) == collect(1:patch_ndofs(plain, 1))

        @testset "vertex ↔ dof correspondence" begin
            vd = patch_vertex_dofs(prov, 1)
            @test sort(collect(keys(vd))) == patch_vertices(prov, 1)
            @test length(patch_vertices(prov, 1)) == 9        # 2×2 Q1 block
            # scalar Q1: one dof per vertex, and the union is the whole patch
            @test all(length(v) == 1 for v in values(vd))
            @test sort(reduce(vcat, values(vd))) == collect(1:patch_ndofs(prov, 1))
        end

        @testset "vector field component expansion" begin
            vgrid = generate_grid(Quadrilateral, (2, 2))
            vdh = DofHandler(vgrid)
            add!(vdh, :u, Lagrange{RefQuadrilateral, 1}()^2)
            close!(vdh)
            vsdh = vdh.subdofhandlers[1]
            vpf = [fi for fi in Ferrite.getfacetset(vgrid, "left") if fi[1] in (1, 3)]
            vprov = PatchItems(vsdh, [[1, 3]]; prescribed_facets = [vpf])
            vch = ConstraintHandler(vdh)
            add!(vch, Dirichlet(:u, Set(vpf), (x, t) -> [0.0, 0.0]))
            close!(vch)
            @test patch_prescribed_dofs(vprov, 1) ==
                sort([vprov.dofmaps[1][g] for g in vch.prescribed_dofs])
            @test all(length(v) == 2 for v in values(patch_vertex_dofs(vprov, 1)))
        end
    end

    @testset "item-lifetime state" begin
        st = PatchItemStates{Vector{Float64}}(3)
        @test length(st) == 3
        @test !has_item_state(st, 2)
        @test_throws ArgumentError item_state(st, 2)
        set_item_state!(st, 2, [1.0, 2.0])
        @test has_item_state(st, 2)
        @test item_state(st, 2) == [1.0, 2.0]
        # slots are per item, so a second item is untouched by the first
        @test !has_item_state(st, 1)
        invalidate_item_state!(st, 2)
        @test !has_item_state(st, 2)
        set_item_state!(st, 1, [3.0])
        invalidate_item_states!(st)
        @test !any(has_item_state(st, i) for i in 1:3)
    end
end
