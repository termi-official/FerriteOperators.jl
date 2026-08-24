using FerriteOperators
using Test
using Polyester

include(joinpath(@__DIR__, "fixture_elements.jl"))

sequential_strategy() = SequentialAssemblyStrategy(SequentialCPUDevice())

# A scalar field on a quad grid plus the facet set two boundary facetsets make
# together — the corner cell owns TWO declared facets, which is what a facet
# item groups.
function corner_testbed(dims = (3, 3))
    grid = generate_grid(Quadrilateral, dims)
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    facets = union(Set(getfacetset(grid, "right")), Set(getfacetset(grid, "top")))
    return (; grid, dh, facets)
end

# Validation doubles, over a subdomain that assembles nothing.
struct NoCacheFacetProbe <: AbstractLinearIntegrator
    declared::Set{FacetIndex}
end
FerriteOperators.setup_element_cache(::NoCacheFacetProbe, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
FerriteOperators.facet_items(m::NoCacheFacetProbe, ::SubDofHandler) = m.declared

struct BareFacetCache <: FerriteOperators.AbstractSurfaceElementCache end

struct OverclaimingFacetCache <: FerriteOperators.AbstractSurfaceElementCache end
FerriteOperators.assemble_facet!(::ResidualRequest, ::OverclaimingFacetCache, args, lfi::Int) = nothing
FerriteOperators.provides_analytic(::Type{OverclaimingFacetCache}, ::JacobianKind{:u}) = true

@testset "Facet items" begin
    @testset "the same cache on either route" begin
        (; grid, dh, facets) = corner_testbed()
        strategy = sequential_strategy()
        t̄ = 2.5
        fused = setup_operator(strategy, LinearNeumannProbe(t̄, :u, facets), dh)
        items = setup_operator(strategy, LinearFacetItemProbe(LinearNeumannProbe(t̄, :u, facets)), dh)
        update_operator!(fused, nothing)
        update_operator!(items, nothing)
        # `≈`, never `==`: the routes visit the facets in different orders, so
        # the summation into a shared dof differs in the last bits.
        @test items.b ≈ fused.b
        @test sum(items.b) ≈ t̄ * 4.0            # |Γ| = 2 + 2 on the [-1, 1]² grid
        @test !all(iszero, items.b)

        # The declaration is the only difference: one engine grows a facet-item
        # subdomain, the other keeps the surface cache on the cell sweep.
        @test length(items.engine.subdomain_caches) == length(dh.subdofhandlers) + 1
        fd = last(items.engine.subdomain_caches).domain
        @test fd isa FerriteOperators.FacetItemDomain
        @test first(items.engine.subdomain_caches).domain.boundary_element isa
            FerriteOperators.EmptySurfaceElementCache
        @test length(fused.engine.subdomain_caches) == length(dh.subdofhandlers)

        # One item per OWNING CELL, carrying all of that cell's declared facets
        # — the corner cell's two are one item, not two.
        @test length(fd.items) == length(facets) - 1
        @test sum(item -> length(item.local_facets), fd.items) == length(facets)
        @test count(item -> length(item.local_facets) == 2, fd.items) == 1
        @test allunique(item.cellid for item in fd.items)
        # Resolution sorts, so the item order does not depend on how a
        # `Set{FacetIndex}` happens to iterate.
        @test issorted(item.cellid for item in fd.items)
        @test all(item -> issorted(item.local_facets), fd.items)
    end

    @testset "the declared set is the gate" begin
        (; grid, dh, facets) = corner_testbed()
        t̄ = 1.75
        # The cache's own `is_facet_in_cache` gate is empty; the declaration is
        # the full set.
        gated = LinearNeumannProbe(t̄, :u, Set(FacetIndex[]))
        op = setup_operator(sequential_strategy(), LinearFacetItemProbe(gated, facets), dh)
        update_operator!(op, nothing)
        @test sum(op.b) ≈ t̄ * 4.0

        # The SAME cache on the fused route, where the gate is all there is,
        # contributes nothing — which is what makes the line above a statement
        # about the gate and not about the cache.
        fused = setup_operator(sequential_strategy(), gated, dh)
        update_operator!(fused, nothing)
        @test all(iszero, fused.b)
    end

    @testset "scheduling" begin
        (; grid, dh, facets) = corner_testbed((4, 4))
        t̄ = 0.8
        integrator() = LinearFacetItemProbe(LinearNeumannProbe(t̄, :u, facets))
        ops = setup_operator(sequential_strategy(), integrator(), dh)
        update_operator!(ops, nothing)
        @test sum(ops.b) ≈ t̄ * 4.0

        # Atomic scatter: neighbouring cells' facets share dofs, and chunk size
        # 1 puts every item on its own worker.
        opp = setup_operator(AssemblyStrategy(FullAssembly(), SequentialScheduling(), PolyesterDevice(1)),
                             integrator(), dh)
        update_operator!(opp, nothing)
        @test opp.b ≈ ops.b

        # Coloring of the OWNING CELLS: the partition covers every declared item
        # exactly once, and needs more than one barrier to do it.
        opc = setup_operator(AssemblyStrategy(FullAssembly(), ColoredScheduling(), PolyesterDevice(1)),
                             integrator(), dh)
        update_operator!(opc, nothing)
        @test opc.b ≈ ops.b
        sc = last(opc.engine.subdomain_caches)
        @test sort!(reduce(vcat, sc.partition)) == collect(1:length(sc.domain.items))
        @test length(sc.partition) > 1
    end

    @testset "sensitivities enter the sweep" begin
        (; grid, dh, facets) = corner_testbed()
        θ = 1.4
        n = ndofs(dh)
        u = zeros(n)
        op = setup_operator(sequential_strategy(), TractionFacetProbe(:u, facets), dh;
                            requests = (ParameterJacobianKind,))

        B = zeros(n, 1)
        update_parameter_jacobian!(B, op, u, θ)

        # Central differences of the FULL residual, which on this operator is
        # the facet term alone.
        residual(θ̃) = (r = zeros(n); evaluate!(op, r, u, θ̃); r)
        h = 1.0e-6
        @test B[:, 1] ≈ (residual(θ + h) .- residual(θ - h)) ./ (2h) rtol = 1.0e-6
        @test !all(iszero, B)
        # r = θ ∫_Γ v dΓ, so ∂r/∂θ sums to |Γ|.
        @test sum(B[:, 1]) ≈ 4.0

        # This operator carries no fused-route boundary cache, so nothing warns
        # about an omission that does not apply to this route.
        @test_logs setup_operator(sequential_strategy(), TractionFacetProbe(:u, facets), dh;
                                  requests = (ParameterJacobianKind,))
    end

    @testset "loud errors" begin
        (; grid, dh, facets) = corner_testbed()
        strategy = sequential_strategy()
        probe(set) = LinearFacetItemProbe(LinearNeumannProbe(1.0, :u, set), set)

        # A facet whose cell this subdomain does not own.
        @test_throws ArgumentError setup_operator(
            strategy, probe(Set([FacetIndex(getncells(grid) + 1, 1)])), dh)
        # A local facet index that does not exist on the cell.
        @test_throws ArgumentError setup_operator(strategy, probe(Set([FacetIndex(1, 9)])), dh)

        # Declared items without the mandatory `setup_facet_item_cache`.
        err = @test_throws ArgumentError setup_operator(strategy, NoCacheFacetProbe(facets), dh)
        @test occursin("setup_facet_item_cache", err.value.msg)

        # A cache without the mandatory residual facet kernel.
        err = @test_throws ArgumentError FerriteOperators.validate_facet_item_cache(BareFacetCache())
        @test occursin("assemble_facet!", err.value.msg)
        @test occursin("ResidualRequest", err.value.msg)

        # A `provides_analytic` claim with no facet kernel behind it — the
        # trait ↔ kernel check over the facet entry point, trailing `::Int`
        # and all.
        err = @test_throws ArgumentError FerriteOperators.validate_facet_item_cache(OverclaimingFacetCache())
        @test occursin("provides_analytic", err.value.msg)
        @test occursin("::Int", err.value.msg)

        # Nothing stands behind a missing facet sensitivity kernel, so a
        # declared sensitivity kind the cache cannot serve is a setup error.
        err = @test_throws ArgumentError setup_operator(
            strategy, TractionFacetProbe(:u, facets; with_parameter_jacobian = false), dh;
            requests = (ParameterJacobianKind,))
        @test occursin("no automatic-differentiation", err.value.msg)
        # Undeclared, the same operator builds — the check is eager, not a ban.
        @test setup_operator(strategy, TractionFacetProbe(:u, facets; with_parameter_jacobian = false), dh) isa
            FerriteOperators.AbstractNonlinearOperator
    end

    @testset "multi-domain routes a facet set per subdomain" begin
        grid = generate_grid(Quadrilateral, (4, 4))
        addcellset!(grid, "right_cells", x -> x[1] ≥ 0.0)
        addcellset!(grid, "left_cells", x -> x[1] ≤ 0.0)
        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, getcellset(grid, "right_cells"))
        add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}())
        sdh2 = SubDofHandler(dh, getcellset(grid, "left_cells"))
        add!(sdh2, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)

        probe(scale, name) = LinearFacetItemProbe(
            LinearNeumannProbe(scale, :u, Set(getfacetset(grid, name))))
        op = setup_operator(sequential_strategy(), LinearMultiDomainIntegrator(Dict(
            "right_cells" => probe(1.5, "right"),
            "left_cells"  => probe(-0.5, "left"))), dh)
        update_operator!(op, nothing)

        # Each subdomain assembled its OWN facet set through its own cache.
        @test sum(op.b) ≈ (1.5 - 0.5) * 2.0
        facet_domains = [sc.domain for sc in op.engine.subdomain_caches
                         if sc.domain isa FerriteOperators.FacetItemDomain]
        @test length(facet_domains) == 2
        @test [length(d.items) for d in facet_domains] == [4, 4]
        @test all(d -> issubset(Set(item.cellid for item in d.items), d.sdh.cellset), facet_domains)

        # A facetset declared on the wrong subdomain is the same loud error the
        # single-domain route raises.
        @test_throws ArgumentError setup_operator(sequential_strategy(), LinearMultiDomainIntegrator(Dict(
            "right_cells" => probe(1.5, "left"),
            "left_cells"  => probe(-0.5, "left"))), dh)
    end
end

# A facet term coupling to a dof that lives in no cell needs a DofHandler that
# can carry one, which is Ferrite's mesh-free algebraic variables.
if !isdefined(Ferrite, :AlgebraicVariable)
    @info "Skipping the facet-item tying test: this Ferrite has no `AlgebraicVariable`, " *
          "so a DofHandler cannot carry dofs outside the mesh."
else
    @testset "Facet items coupling to a global dof (the tying shape)" begin
        testbed = tying_facet_testbed()
        (; dh, coupling, integrator, pdof) = testbed
        spec = StandardOperatorSpecification(; algebraic_couplings = (coupling,))
        n = ndofs(dh)
        u = 0.2 .* sin.(0.7 .* (1:n))
        u[pdof] = 0.35
        Kref, rref = tying_facet_reference(testbed, u)

        op = setup_operator(AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), SequentialCPUDevice()),
                            integrator, dh)
        r = zeros(n)
        update_linearization!(op, r, u, nothing)
        @test op.J ≈ Kref
        @test r ≈ rref

        r2 = zeros(n)
        evaluate!(op, r2, u, nothing)
        @test r2 ≈ rref

        # The coupling rows AND columns the augmented tail carries are what the
        # comparison is about, so they must be non-empty.
        @test maximum(abs, Kref[pdof, :]) > 0
        @test maximum(abs, Kref[:, pdof]) > 0
        @test abs(rref[pdof]) > 0

        # The local system is `[celldofs(cell); the pressure dof]`, and the
        # workspace addresses through it.
        ws = first(last(op.engine.subdomain_caches).device_cache)
        @test length(FerriteOperators.item_dofs(ws)) == ndofs_per_cell(dh) + 1
        @test last(FerriteOperators.item_dofs(ws)) == pdof

        # A declared global dof rules out coloring, so the parallel route is the
        # atomic scatter — and every worker's duplicated workspace has to
        # rebuild the same augmented dof vector.
        opp = setup_operator(AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), PolyesterDevice(1)),
                             integrator, dh)
        rp = zeros(n)
        update_linearization!(opp, rp, u, nothing)
        @test opp.J ≈ Kref
        @test rp ≈ rref
        wsp = last(last(opp.engine.subdomain_caches).device_cache)
        @test last(FerriteOperators.item_dofs(wsp)) == pdof
    end
end
