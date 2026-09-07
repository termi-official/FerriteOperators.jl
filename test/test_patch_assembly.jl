using FerriteOperators
using FerriteOperatorsExampleElements
using LinearAlgebra
using SparseArrays
using Test

include(joinpath(@__DIR__, "fixture_elements.jl"))

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
            CellArgs((u = uₑ,), cc, p, nothing))
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
        req::ResidualRequest, cache::FerriteOperatorsExampleElements.SimpleBilinearDiffusionElementCache,
        args, data::WeightedSource
    )
    cv = cache.cellvalues
    for qp in 1:getnquadpoints(cv), i in 1:getnbasefunctions(cv)
        req.r[i] += data.w * shape_value(cv, qp, i) * getdetJdV(cv, qp)
    end
    return nothing
end

# Reference vector over a patch: the same term kernels, invoked directly over a
# hand-built `PatchArgs`, in the same cell order and the same per-cell term
# order.
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
        ldofs = [dofmap[g] for g in celldofs(cc)]
        args = PatchArgs((;), cc, nothing, nothing, provider, i, group, ldofs)
        active = false
        for t in terms
            FerriteOperators.patch_term_active(t.restriction, group) || continue
            active = true
            FerriteOperators.assemble_patch_cell!(ResidualRequest(re), cache, args, t.data)
        end
        active || continue
        for (l, g) in pairs(celldofs(cc))
            v[dofmap[g]] += re[l]
        end
    end
    return v
end

# A term payload that reads the PATCH CONTEXT: it weights the constant source
# by a patch-local field indexed through the cell's dof window, and logs what
# the context reported on every cell.
struct PatchContextSource{V <: AbstractVector}
    w::V                 # patch-local, indexed by `args.ldofs`
    seen::Vector{Any}
end
function FerriteOperators.assemble_patch_cell!(
        req::ResidualRequest, cache::FerriteOperatorsExampleElements.SimpleBilinearDiffusionElementCache,
        args, data::PatchContextSource
    )
    push!(data.seen, (patch = args.patch, group = args.group, cell = cellid(args.cell),
            ldofs = copy(args.ldofs), provider = args.provider, p = args.p, ctx = args.ctx))
    cv = cache.cellvalues
    for qp in 1:getnquadpoints(cv), i in 1:getnbasefunctions(cv)
        req.r[i] += data.w[args.ldofs[i]] * shape_value(cv, qp, i) * getdetJdV(cv, qp)
    end
    return nothing
end

# Per-column term payload of a local BVP with several right-hand sides:
# L_j(v) = ∫ v gⱼ(x) dΩ with a gⱼ that genuinely differs per column, so no
# column is a rescaling of another.
struct ColumnSource
    j::Int
end
function FerriteOperators.assemble_patch_cell!(
        req::ResidualRequest, cache::FerriteOperatorsExampleElements.SimpleBilinearDiffusionElementCache,
        args, data::ColumnSource
    )
    cv = cache.cellvalues
    coords = getcoordinates(args.cell)
    for qp in 1:getnquadpoints(cv)
        x = spatial_coordinate(cv, qp, coords)
        g = sin(data.j * x[1]) + 0.5 * cos(data.j * x[2])
        for i in 1:getnbasefunctions(cv)
            req.r[i] += g * shape_value(cv, qp, i) * getdetJdV(cv, qp)
        end
    end
    return nothing
end

# The downstream shape these seams serve: per patch assemble the local matrix,
# factorize it once (retained in the item-state slot), then solve and emit one
# column per right-hand side.
const PATCH_BVP_TERMS = (PatchTerm(WholePatch()),)
const PatchLU = typeof(lu(Matrix{Float64}(I, 2, 2)))

function solve_patch_columns!(sink, facts, pws, pid, states, p, ncols)
    provider = patch_provider(pws)
    n = patch_ndofs(provider, pid)
    free = patch_free_dofs(provider, pid)
    if !has_item_state(facts, pid)
        K = zeros(n, n)
        assemble_patch_target!(K, PATCH_BVP_TERMS, pws, states, p)
        set_item_state!(facts, pid, lu(K[free, free]))
    end
    F = item_state(facts, pid)
    rows = patch_dofs(provider, pid)[free]
    rhs = zeros(n)
    for j in 1:ncols
        fill!(rhs, 0.0)
        assemble_patch_target!(rhs, (PatchTerm(WholePatch(), ColumnSource(j)),), pws, states, p)
        emit_patch_column!(sink, rows, (pid - 1) * ncols + j, F \ rhs[free])
    end
    return sink
end

# The same local BVP by direct kernel invocation, reusing the reference
# assembly above.
function reference_patch_columns!(sink, provider, pid, cache, dh, u, p, ncols)
    free = patch_free_dofs(provider, pid)
    K = reference_patch_matrix(provider, pid, cache, dh, u, p)
    F = lu(K[free, free])
    rows = patch_dofs(provider, pid)[free]
    for j in 1:ncols
        rhs = reference_patch_vector(provider, pid, cache, dh, (PatchTerm(WholePatch(), ColumnSource(j)),))
        emit_patch_column!(sink, rows, (pid - 1) * ncols + j, F \ rhs[free])
    end
    return sink
end

@testset "Patch assembly" begin
    (; grid, dh, qrc, strategy) = scalar_quad_testbed((4, 4))

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

    # The whole-patch Jacobian of every patch, one `assemble_patch_target!` call
    # per patch into a caller-owned dense matrix — the shape every local BVP
    # starts from.
    function patch_matrices(op, prov, states, p)
        dest = [zeros(patch_ndofs(prov, i), patch_ndofs(prov, i)) for i in 1:npatches(prov)]
        foreach_patch(op, prov, states, p) do pws, pid
            assemble_patch_target!(dest[pid], PATCH_BVP_TERMS, pws, states, p)
        end
        return dest
    end

    @testset "bilinear diffusion patches match direct kernel references" begin
        op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.3, qrc, :u), dh)
        u = zeros(ndofs(dh))
        dest = patch_matrices(op, provider, (u = u,), nothing)
        cache = first_element_cache(op)
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
        hint = SimpleHyperelasticityIntegrator(NeoHookean(10.0, 0.3), qrc, :u)
        hop = setup_operator(strategy, hint, hdh)
        hu = 0.05 .* sin.(0.3 .* (1:ndofs(hdh)))
        hprovider = PatchItems(hdh.subdofhandlers[1], [[1, 2, 3, 4], [5, 6]])
        dest = patch_matrices(hop, hprovider, (u = hu,), 0.0)
        cache = first_element_cache(hop)
        for i in 1:2
            KPref = reference_patch_matrix(hprovider, i, cache, hdh, hu, 0.0)
            @test dest[i] ≈ KPref rtol = 1e-13
        end
        # a patch sweep is pure evaluation: a second one over a target the
        # caller zeroed gives the same matrix
        @test patch_matrices(hop, hprovider, (u = hu,), 0.0)[1] == dest[1]
    end

    @testset "guards" begin
        op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        odh = scalar_quad_testbed((2, 2)).dh
        foreign = PatchItems(odh.subdofhandlers[1], [[1]])
        @test_throws ArgumentError patch_workspace(op, foreign)
        @test_throws DimensionMismatch PatchItems(sdh, [[1, 2]]; groups = [[1]])
        @test_throws ArgumentError PatchItems(sdh, [[1, 2]]; prescribed_facets = [[FacetIndex(9, 1)]])
    end

    @testset "term tuples with per-term domain restrictions" begin
        op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        cache = first_element_cache(op)
        # cells 1,2 carry group 1; cells 5,6 carry group 2
        prov = PatchItems(sdh, [[1, 2, 5, 6]]; groups = [[1, 1, 2, 2]])
        terms = (PatchTerm(WholePatch(), WeightedSource(1.0)), PatchTerm(CellGroup(2), WeightedSource(10.0)))
        only2 = (PatchTerm(CellGroup(2), WeightedSource(10.0)),)
        onlyw = (PatchTerm(WholePatch(), WeightedSource(1.0)),)

        both, d2, dw = zeros(patch_ndofs(prov, 1)), zeros(patch_ndofs(prov, 1)), zeros(patch_ndofs(prov, 1))
        foreach_patch(op, prov, (;), nothing) do pws, pid
            assemble_patch_target!(both, terms, pws, (;), nothing)
            assemble_patch_target!(d2, only2, pws, (;), nothing)
            assemble_patch_target!(dw, onlyw, pws, (;), nothing)
        end
        @test both == reference_patch_vector(prov, 1, cache, dh, terms)
        @test !iszero(both)

        # A group-restricted term alone touches only its group's cells.
        @test d2 == reference_patch_vector(prov, 1, cache, dh, only2)
        # cell 1's exclusive dofs (not shared with cells 5,6) stay untouched
        shared = union(celldofs(dh, 5), celldofs(dh, 6))
        exclusive = [prov.dofmaps[1][g] for g in celldofs(dh, 1) if !(g in shared)]
        @test !isempty(exclusive)
        @test all(iszero, d2[exclusive])

        # ... and the whole-patch term alone is the difference
        @test both ≈ dw .+ d2 rtol = 1e-14
    end

    @testset "term kernels receive the patch context" begin
        op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        cache = first_element_cache(op)
        # group tags that are neither cell ids nor 1-based, so a kernel reading
        # `args.group` cannot accidentally agree with an index
        prov = PatchItems(sdh, [[1, 2, 5, 6], [13, 14]]; groups = [[7, 7, 9, 9], [4, 4]])
        weights = [collect(1.0:patch_ndofs(prov, i)) for i in 1:2]

        seen = Any[]
        dest = [zeros(patch_ndofs(prov, i)) for i in 1:2]
        foreach_patch(op, prov, (;), 1.5, :mine) do pws, pid
            assemble_patch_target!(dest[pid],
                (PatchTerm(WholePatch(), PatchContextSource(weights[pid], seen)),), pws, (;), 1.5, :mine)
        end

        # the context names the patch, the cell's group tag and the cell's dofs
        # in patch-local numbering, and carries the sweep's parameters on
        expected = [(i, patch_cell_groups(prov, i)[k], c, [prov.dofmaps[i][g] for g in celldofs(dh, c)])
                    for i in 1:2 for (k, c) in pairs(patch_cells(prov, i))]
        @test [(e.patch, e.group, e.cell, e.ldofs) for e in seen] == expected
        @test all(e -> e.provider === prov, seen)
        @test all(e -> e.p == 1.5 && e.ctx === :mine, seen)

        # ... and the window is usable: indexing patch-local data through it
        # reproduces the direct-kernel reference
        for i in 1:2
            ref = reference_patch_vector(prov, i, cache, dh,
                (PatchTerm(WholePatch(), PatchContextSource(weights[i], Any[])),))
            @test dest[i] == ref
            @test !iszero(dest[i])
        end
    end

    @testset "sinks" begin
        op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        terms = (PatchTerm(WholePatch(), WeightedSource(2.0)),)
        local_dest = [zeros(patch_ndofs(provider, i)) for i in 1:3]
        foreach_patch(op, provider, (;), nothing) do pws, pid
            assemble_patch_target!(local_dest[pid], terms, pws, (;), nothing)
        end

        @testset "additive global vector" begin
            g = zeros(ndofs(dh))
            sink = PatchGlobalVectorSink(g)
            foreach_patch(op, provider, (;), nothing) do pws, pid
                v = zeros(patch_ndofs(provider, pid))
                assemble_patch_target!(v, terms, pws, (;), nothing)
                patch_emit!(sink, provider, pid, v)
            end
            ref = zeros(ndofs(dh))
            for i in 1:3, (l, gd) in pairs(patch_dofs(provider, i))
                ref[gd] += local_dest[i][l]
            end
            @test g == ref
            @test !iszero(g)
        end

        @testset "duplicate-summing triplets" begin
            # patches 1 and 2 emit into the same column and share cell 6, so
            # their contributions overlap on rows
            sink = PatchTripletSink()
            cols = [1, 1, 2]
            foreach_patch(op, provider, (;), nothing) do pws, pid
                v = zeros(patch_ndofs(provider, pid))
                assemble_patch_target!(v, terms, pws, (;), nothing)
                emit_patch_column!(sink, patch_dofs(provider, pid), cols[pid], v)
            end
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
            other = PatchTripletSink()
            emit_patch_column!(other, [3, 4], 7, [1.0, 2.0])
            @test (append!(other, sink); other.J == vcat([7, 7], sink.J))
        end

        @testset "matrix scatter modes agree" begin
            u = zeros(ndofs(dh))
            dense = patch_matrices(op, provider, (u = u,), nothing)
            # a Ferrite assembler over a patch-local pattern is the other target
            patterned = map(i -> sparse(ones(patch_ndofs(provider, i), patch_ndofs(provider, i))), 1:3)
            foreach_patch(op, provider, (u = u,), nothing) do pws, pid
                assemble_patch_target!(start_assemble(patterned[pid]), PATCH_BVP_TERMS, pws, (u = u,), nothing)
            end
            for i in 1:3
                @test Matrix(patterned[i]) ≈ dense[i] rtol = 1e-14
            end
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
        st = ItemStates{Vector{Float64}}(3)
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

    @testset "chunk helper" begin
        @test patch_chunks(provider, 1) == [1:3]
        @test patch_chunks(provider, 2) == [1:1, 2:3]
        @test patch_chunks(provider, 3) == [1:1, 2:2, 3:3]
        # more chunks than items drops the empty ones rather than returning them
        @test patch_chunks(provider, 7) == [1:1, 2:2, 3:3]
        @test reduce(vcat, patch_chunks(provider, 2)) == 1:npatches(provider)
        @test_throws ArgumentError patch_chunks(provider, 0)
        # coloring has no meaning without item adjacency, and says so
        @test_throws ArgumentError FerriteOperators.compute_partition(ColoredScheduling(), provider)

        @testset "over an item subset" begin
            big = PatchItems(sdh, [[c] for c in 1:6])
            @test patch_chunks(big, 2; items = [1, 3, 5]) == [[1], [3, 5]]
            @test patch_chunks(big, 3; items = [5, 1, 3]) == [[1], [3], [5]]   # ascending
            @test patch_chunks(big, 2; items = 2:5) == [2:3, 4:5]
            @test reduce(vcat, patch_chunks(big, 3; items = [1, 3, 5])) == [1, 3, 5]
            @test_throws ArgumentError patch_chunks(big, 2; items = [1, 7])
            @test_throws ArgumentError patch_chunks(big, 2; items = [1, 1])
        end
    end

    @testset "item-subset sweeps" begin
        op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        big = PatchItems(sdh, [[c] for c in 1:6])
        terms = (PatchTerm(WholePatch(), WeightedSource(2.0)),)
        subset = [2, 3, 5]

        visited = Int[]
        foreach_patch(op, big, (;), nothing; items = subset) do pws, pid
            push!(visited, pid)
            @test current_patch(pws) == pid
        end
        @test visited == subset

        # an unsorted request is visited ascending, and the empty one is a no-op
        visited2 = Int[]
        foreach_patch((pws, pid) -> push!(visited2, pid), op, big, (;), nothing; items = [5, 2, 3])
        @test visited2 == subset
        foreach_patch((pws, pid) -> push!(visited2, pid), op, big, (;), nothing; items = Int[])
        @test visited2 == subset

        @test_throws ArgumentError foreach_patch((pws, pid) -> nothing, op, big, (;), nothing; items = [7])
        @test_throws ArgumentError foreach_patch((pws, pid) -> nothing, op, big, (;), nothing; items = [2, 2])

        # the chunked subset merge reproduces the sequential subset stream
        seq = PatchTripletSink()
        foreach_patch(op, big, (;), nothing; items = subset) do pws, pid
            v = zeros(patch_ndofs(big, pid))
            assemble_patch_target!(v, terms, pws, (;), nothing)
            emit_patch_column!(seq, patch_dofs(big, pid), pid, v)
        end
        chunks = patch_chunks(big, 2; items = subset)
        @test chunks == [[2], [3, 5]]
        sinks = [PatchTripletSink() for _ in chunks]
        wss = [patch_workspace(op, big) for _ in chunks]
        @sync for c in eachindex(chunks)
            Threads.@spawn for pid in chunks[c]
                Ferrite.reinit!(wss[c], pid)
                v = zeros(patch_ndofs(big, pid))
                assemble_patch_target!(v, terms, wss[c], (;), nothing)
                emit_patch_column!(sinks[c], patch_dofs(big, pid), pid, v)
            end
        end
        merged = PatchTripletSink()
        foreach(s -> append!(merged, s), sinks)
        @test merged.I == seq.I
        @test merged.J == seq.J
        @test merged.V == seq.V
        @test !isempty(seq.V)
    end

    @testset "per-patch callback: local BVP with several right-hand sides" begin
        op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.7, qrc, :u), dh)
        cache = first_element_cache(op)
        u = zeros(ndofs(dh))
        ncols = 4

        # the Neumann patch matrix is singular; pinning one dof per patch is the
        # caller's job, exactly as the partition contract states
        prov = PatchItems(sdh, cellsets)
        for i in 1:npatches(prov)
            augment_prescribed_dofs!(prov, i, [1])
        end

        sink = PatchTripletSink()
        facts = ItemStates{PatchLU}(npatches(prov))
        foreach_patch(op, prov, (u = u,), nothing) do pws, pid
            @test current_patch(pws) == pid
            @test patch_provider(pws) === prov
            solve_patch_columns!(sink, facts, pws, pid, (u = u,), nothing, ncols)
        end

        ref = PatchTripletSink()
        for pid in 1:npatches(prov)
            reference_patch_columns!(ref, prov, pid, cache, dh, u, nothing, ncols)
        end
        @test sink.I == ref.I
        @test sink.J == ref.J
        @test sink.V ≈ ref.V rtol = 1e-12
        @test all(has_item_state(facts, i) for i in 1:npatches(prov))

        @testset "multi-column emission" begin
            W = sparse(sink, ndofs(dh), npatches(prov) * ncols)
            @test size(W) == (ndofs(dh), npatches(prov) * ncols)
            nfree = sum(pid -> length(patch_free_dofs(prov, pid)), 1:npatches(prov))
            @test length(sink.V) == ncols * nfree
            for pid in 1:npatches(prov)
                rows = patch_dofs(prov, pid)[patch_free_dofs(prov, pid)]
                cols = [Vector(W[:, (pid - 1) * ncols + j]) for j in 1:ncols]
                for c in cols
                    @test Set(findall(!iszero, c)) ⊆ Set(rows)
                    @test !iszero(c)
                end
                # the per-column term data really differs: no column of a patch
                # is a multiple of another
                for j in 1:ncols, k in (j + 1):ncols
                    @test abs(dot(cols[j], cols[k])) < (1 - 1.0e-6) * norm(cols[j]) * norm(cols[k])
                end
            end
        end

        @testset "chunked parallel sweep reproduces the sequential stream" begin
            # Test code playing the downstream role: contiguous chunks, one
            # workspace and one collector per worker, chunk-order merge. Only
            # public seams are used.
            chunks = patch_chunks(prov, 2)          # [1:1, 2:3]: a short and a long chunk
            sinks = [PatchTripletSink() for _ in chunks]
            wss = [patch_workspace(op, prov) for _ in chunks]
            parfacts = ItemStates{PatchLU}(npatches(prov))
            @sync for c in eachindex(chunks)
                Threads.@spawn begin
                    ws = wss[c]
                    for pid in chunks[c]
                        Ferrite.reinit!(ws, pid)
                        solve_patch_columns!(sinks[c], parfacts, ws, pid, (u = u,), nothing, ncols)
                    end
                end
            end
            merged = PatchTripletSink()
            for s in sinks
                append!(merged, s)
            end
            # `==` on floats is deliberate here: bit-identity of the triplet
            # stream IS the property under test, not approximate agreement.
            @test merged.I == sink.I
            @test merged.J == sink.J
            @test merged.V == sink.V

            # one workspace per worker, and duplication is the other route to one
            dup = FerriteOperators.duplicate_for_device(SequentialCPUDevice(), wss[1])
            @test dup !== wss[1]
            @test patch_provider(dup) === prov
            dsink = PatchTripletSink()
            Ferrite.reinit!(dup, 1)
            solve_patch_columns!(dsink, ItemStates{PatchLU}(npatches(prov)),
                dup, 1, (u = u,), nothing, ncols)
            @test dsink.V == sinks[1].V          # chunk 1 is exactly patch 1
        end
    end

    @testset "assemble_patch_target! accumulates and rejects unusable targets" begin
        op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        u = zeros(ndofs(dh))
        pws = patch_workspace(op, provider)
        Ferrite.reinit!(pws, 1)
        n = patch_ndofs(provider, 1)
        K = zeros(n, n)
        assemble_patch_target!(K, PATCH_BVP_TERMS, pws, (u = u,), nothing)
        once = copy(K)
        assemble_patch_target!(K, PATCH_BVP_TERMS, pws, (u = u,), nothing)
        @test K ≈ 2 .* once rtol = 1e-14
        @test !iszero(once)
        @test_throws ArgumentError assemble_patch_target!(
            "not a target", PATCH_BVP_TERMS, pws, (u = u,), nothing)
    end
end
