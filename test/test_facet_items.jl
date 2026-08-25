using FerriteOperators
using Test
using Polyester
using SparseArrays

include(joinpath(@__DIR__, "fixture_elements.jl"))

sequential_strategy() = SequentialAssemblyStrategy(SequentialCPUDevice())

# A scalar field on a quad grid plus the facet set two boundary facetsets make
# together — the corner cell owns TWO declared facets, which is what a facet
# item groups.
function corner_testbed(dims = (3, 3))
    (; grid, dh) = scalar_quad_testbed(dims)
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

# A facet-item cache declaring condensed internal state, so the family's own
# admissibility rejection becomes reachable.
struct CondensedFacetCache <: FerriteOperators.AbstractSurfaceElementCache end
FerriteOperators.assemble_facet!(::ResidualRequest, ::CondensedFacetCache, args, lfi::Int) = nothing
FerriteOperators.duplicate_for_device(device, c::CondensedFacetCache) = c
FerriteOperators.has_internal_state(::Type{CondensedFacetCache}) = true

struct CondensedFacetProbe <: AbstractNonlinearIntegrator
    facetset::Set{FacetIndex}
end
FerriteOperators.setup_element_cache(::CondensedFacetProbe, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
FerriteOperators.facet_items(m::CondensedFacetProbe, ::SubDofHandler) = m.facetset
FerriteOperators.setup_facet_item_cache(::CondensedFacetProbe, ::SubDofHandler) = CondensedFacetCache()

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
        @test length(fused.engine.subdomain_caches) == length(dh.subdofhandlers)
        fd = last(items.engine.subdomain_caches).domain

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

    @testset "the pullback and the time sensitivity enter the sweep too" begin
        # `parameter_vjp!` and `time_sensitivity!` have no AD fallback on
        # facets, so the analytic kernels are the only route — and a cache
        # without them must fail before a sweep reaches a missing method.
        (; grid, dh, facets) = corner_testbed()
        θ = 1.4
        n = ndofs(dh)
        u = zeros(n)
        ctx = TimeIntegrationContext(0.6, 1.0, 1.0)
        serving(kinds...) = setup_operator(sequential_strategy(),
            TractionFacetProbe(:u, facets; with_sensitivities = true), dh; requests = kinds)

        # r = θ(1 + t) ∫_Γ v dΓ, so λᵀ∂r/∂θ = (1 + t)·λᵀ∫_Γ v dΓ and
        # ∂r/∂t = θ ∫_Γ v dΓ — both against the parameter Jacobian as referee.
        op = serving(ParameterJacobianKind, ParameterVJPKind, TimeSensitivityKind)
        B = zeros(n, 1)
        update_parameter_jacobian!(B, op, (u = u,), θ, ctx)
        @test sum(B[:, 1]) ≈ (1 + evaluation_time(ctx)) * 4.0

        λ = cos.(0.7 .* (1:n))
        g = zeros(1)
        parameter_vjp!(g, op, λ, (u = u,), θ, ctx)
        @test g ≈ B' * λ rtol = 1e-12

        gt = zeros(n)
        time_sensitivity!(gt, op, (u = u,), θ, ctx)
        @test gt ≈ (θ / (1 + evaluation_time(ctx))) .* B[:, 1] rtol = 1e-12
        @test sum(gt) ≈ θ * 4.0

        # The same declarations over a cache that serves neither kind are a
        # setup error, one message per kind.
        bare = TractionFacetProbe(:u, facets)   # ∂r/∂θ only
        for kind in (ParameterVJPKind, TimeSensitivityKind)
            err = @test_throws ArgumentError setup_operator(sequential_strategy(), bare, dh;
                                                            requests = (kind,))
            @test occursin("no automatic-differentiation", err.value.msg)
        end
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

        # The internal-state rejection of the facet family names the kernel an
        # author has to write, trailing local facet index and all.
        cop = setup_operator(strategy, CondensedFacetProbe(facets), dh)
        err = @test_throws ArgumentError update_parameter_jacobian!(
            zeros(residual_size(cop), 1), cop, zeros(ndofs(dh)), 1.0)
        @test occursin("assemble_facet!", err.value.msg)
        @test occursin("::FacetArgs, ::Int", err.value.msg)
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
    # Records the width of the local system every volumetric kernel call is
    # handed, which is what discriminates the two global-dof declarations.
    struct CellWidthProbeCache <: AbstractVolumetricElementCache
        widths::Vector{Int}
    end
    FerriteOperators.duplicate_for_device(device, c::CellWidthProbeCache) = c
    FerriteOperators.reinit_values!(::CellWidthProbeCache, cell) = nothing
    function FerriteOperators.assemble_cell!(req::ResidualRequest, c::CellWidthProbeCache, args::CellArgs)
        push!(c.widths, length(args.states.u))
        return nothing
    end

    # The tying integrator with a volumetric term on the same subdomain.
    struct CellWidthProbe <: AbstractNonlinearIntegrator
        tying::TyingFacetIntegrator
        widths::Vector{Int}
    end
    CellWidthProbe(tying) = CellWidthProbe(tying, Int[])
    FerriteOperators.setup_element_cache(m::CellWidthProbe, ::SubDofHandler) = CellWidthProbeCache(m.widths)
    FerriteOperators.facet_item_global_dofs(m::CellWidthProbe, sdh::SubDofHandler) =
        facet_item_global_dofs(m.tying, sdh)
    FerriteOperators.facet_items(m::CellWidthProbe, sdh::SubDofHandler) = facet_items(m.tying, sdh)
    FerriteOperators.setup_facet_item_cache(m::CellWidthProbe, sdh::SubDofHandler) =
        setup_facet_item_cache(m.tying, sdh)

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
        sdh = dh.subdofhandlers[1]
        nc  = ndofs_per_cell(sdh)
        ws = first(last(op.engine.subdomain_caches).device_cache)
        @test length(FerriteOperators.item_dofs(ws)) == nc + 1
        @test last(FerriteOperators.item_dofs(ws)) == pdof
        @test size(ws.Ke) == (nc + 1, nc + 1)

        # The declaration belongs to the facet items alone: the same subdomain's
        # CELL sweep keeps the field-local system, buffers included.
        ws_cell = first(first(op.engine.subdomain_caches).device_cache)
        @test isempty(global_dofs(integrator, sdh))
        @test ws_cell.dofs === nothing
        @test length(FerriteOperators.item_dofs(ws_cell)) == nc
        @test size(ws_cell.Ke) == (nc, nc)
        @test length(ws_cell.re) == nc
        @test length(ws_cell.slot_buffers.u) == nc

        # ... and what the volumetric KERNEL is handed is that same system, on
        # every call of a linearization sweep (the AD passes included).
        probe = CellWidthProbe(integrator)
        opw = setup_operator(AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), SequentialCPUDevice()),
                             probe, dh)
        rw = zeros(n)
        update_linearization!(opw, rw, u, nothing)
        @test !isempty(probe.widths)
        @test all(==(nc), probe.widths)
        # A volumetric term contributing nothing changes no value either.
        @test opw.J ≈ Kref
        @test rw ≈ rref

        # The narrowed sparsity the un-augmented cell sweep permits: the same
        # numbers over the tying facets' entries alone.
        narrow = StandardOperatorSpecification(; algebraic_couplings = (testbed.facet_coupling,))
        opn = setup_operator(AssemblyStrategy(FullAssembly(narrow), SequentialScheduling(), SequentialCPUDevice()),
                             integrator, dh)
        rn = zeros(n)
        update_linearization!(opn, rn, u, nothing)
        @test opn.J ≈ Kref
        @test rn ≈ rref
        # Only the dofs of the cells owning a declared facet couple to the
        # pressure, plus its own diagonal.
        adjacent = unique!(reduce(vcat, [celldofs(dh, facet[1]) for facet in testbed.facets]))
        @test length(nzrange(opn.J, pdof)) == length(adjacent) + 1
        @test length(nzrange(opn.J, pdof)) < length(nzrange(op.J, pdof))
        @test nnz(opn.J) < nnz(op.J)

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

@testset "structural reduction answer of the facet family" begin
    # Facet domains decline reductions; an all-facet reduction must fail loudly.
    fd = FerriteOperators.FacetItemDomain(nothing, nothing, nothing)
    @test !FerriteOperators._may_contribute(fd, FunctionalKind(:probe))
    @test !FerriteOperators._may_contribute(fd, FerriteOperators.CondensationKind((u = 1.0,)))
    @test FerriteOperators._may_contribute(fd, JacobianKind())
end
