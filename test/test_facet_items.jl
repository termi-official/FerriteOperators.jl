using FerriteOperators
using Test
using Polyester
using SparseArrays

include(joinpath(@__DIR__, "fixture_elements.jl"))

sequential_strategy() = AssemblyStrategy(SequentialCPUDevice())

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
        opp = setup_operator(AssemblyStrategy(FullAssembly(), SequentialScheduling(), PolyesterDevice(min_items_per_worker = 1)),
                             integrator(), dh)
        update_operator!(opp, nothing)
        @test opp.b ≈ ops.b

        # Coloring of the OWNING CELLS: the partition covers every declared item
        # exactly once, and needs more than one barrier to do it.
        opc = setup_operator(AssemblyStrategy(FullAssembly(), ColoredScheduling(), PolyesterDevice(min_items_per_worker = 1)),
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
        # CELL sweep keeps the field-local system, buffers included. Its dof
        # vector is the geometry cache's, so the workspace is positioned by hand
        # — this subdomain carries neither a volumetric nor a boundary kernel,
        # and no assembly sweep traverses it.
        ws_cell = first(first(op.engine.subdomain_caches).device_cache)
        @test isempty(global_dofs(integrator, sdh))
        @test ws_cell.dofs === nothing
        reinit!(ws_cell, first(sdh.cellset))
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
        opp = setup_operator(AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), PolyesterDevice(min_items_per_worker = 1)),
                             integrator, dh)
        rp = zeros(n)
        update_linearization!(opp, rp, u, nothing)
        @test opp.J ≈ Kref
        @test rp ≈ rref
        wsp = last(last(opp.engine.subdomain_caches).device_cache)
        @test last(FerriteOperators.item_dofs(wsp)) == pdof
    end
end

####################################
## Reductions over declared facets
####################################
# A surface cache whose reason to exist is its reductions: the area of the
# declared set and the state- and position-dependent flux ∫ u (x⋅n) dΓ. The
# VOLUMETRIC cache is empty, so the facet items are the only family that can
# contribute — which is what makes the numbers below statements about this
# family's own traversal.

struct FacetFunctionalCache{FV <: FacetValues} <: FerriteOperators.AbstractSurfaceElementCache
    fv::FV
end

struct FacetFunctionalProbe <: AbstractNonlinearIntegrator
    field_name::Symbol
    facetset::Set{FacetIndex}
end

FerriteOperators.setup_element_cache(::FacetFunctionalProbe, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
FerriteOperators.facet_items(m::FacetFunctionalProbe, ::SubDofHandler) = m.facetset
function FerriteOperators.setup_facet_item_cache(m::FacetFunctionalProbe, sdh::SubDofHandler)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    fqr    = FacetQuadratureRule{Ferrite.getrefshape(ip)}(2)
    return FacetFunctionalCache(FacetValues(fqr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::FacetFunctionalCache) =
    FacetFunctionalCache(FerriteOperators.duplicate_for_device(device, c.fv))
# The mandatory residual kernel, contributing nothing.
FerriteOperators.assemble_facet!(::ResidualRequest, ::FacetFunctionalCache, args::FacetArgs, lfi::Int) = nothing

function FerriteOperators.evaluate_facet_functional(::FunctionalKind{:facet_area},
        c::FacetFunctionalCache, args::FacetArgs, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    a = 0.0
    for qp in 1:getnquadpoints(c.fv)
        a += getdetJdV(c.fv, qp)
    end
    return a
end
function FerriteOperators.evaluate_facet_functional(::FunctionalKind{:facet_flux},
        c::FacetFunctionalCache, args::FacetArgs, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    coords = getcoordinates(args.cell)
    uₑ = args.states.u
    f = 0.0
    for qp in 1:getnquadpoints(c.fv)
        x = spatial_coordinate(c.fv, qp, coords)
        f += function_value(c.fv, qp, uₑ) * (x ⋅ getnormal(c.fv, qp)) * getdetJdV(c.fv, qp)
    end
    return f
end

FerriteOperators.functional_value_type(::FunctionalKind{:facet_area}) = Float64
FerriteOperators.functional_value_type(::FunctionalKind{:facet_flux}) = Float64

# The same integrands under undeclared tags, so the accumulator-type scan stays
# covered and can be compared against the typed route.
FerriteOperators.evaluate_facet_functional(::FunctionalKind{:facet_area_undeclared},
        c::FacetFunctionalCache, args::FacetArgs, lfi::Int) =
    FerriteOperators.evaluate_facet_functional(FunctionalKind(:facet_area), c, args, lfi)
FerriteOperators.evaluate_facet_functional(::FunctionalKind{:facet_flux_undeclared},
        c::FacetFunctionalCache, args::FacetArgs, lfi::Int) =
    FerriteOperators.evaluate_facet_functional(FunctionalKind(:facet_flux), c, args, lfi)

# Kernels that run on every declared facet and contribute nothing: an empty sum,
# which only the declared tag can answer.
FerriteOperators.functional_value_type(::FunctionalKind{:facet_quiet}) = Float64
FerriteOperators.evaluate_facet_functional(::FunctionalKind{:facet_quiet},
        ::FacetFunctionalCache, args::FacetArgs, lfi::Int) = nothing
FerriteOperators.evaluate_facet_functional(::FunctionalKind{:facet_quiet_undeclared},
        ::FacetFunctionalCache, args::FacetArgs, lfi::Int) = nothing

# The area and the flux of the declared set, by a hand-rolled `FacetIterator`
# loop over the same facets.
function facet_functional_reference(dh, facets, u)
    sdh    = dh.subdofhandlers[1]
    ip     = Ferrite.getfieldinterpolation(sdh, :u)
    fv     = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), ip,
                         Ferrite.geometric_interpolation(Quadrilateral))
    uₑ     = zeros(ndofs_per_cell(sdh))
    area, flux = 0.0, 0.0
    for facet in FacetIterator(sdh, facets)
        reinit!(fv, facet)
        coords = getcoordinates(facet)
        uₑ .= @view u[celldofs(facet)]
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            area += dΓ
            flux += function_value(fv, qp, uₑ) *
                    (spatial_coordinate(fv, qp, coords) ⋅ getnormal(fv, qp)) * dΓ
        end
    end
    return (; area, flux)
end

@testset "Facet functionals" begin
    (; grid, dh, facets) = corner_testbed()
    u = sin.(0.4 .* (1:ndofs(dh)))
    probe = FacetFunctionalProbe(:u, facets)
    op = setup_operator(sequential_strategy(), probe, dh)
    reference = facet_functional_reference(dh, facets, u)

    @testset "the declared facets are the domain of integration" begin
        # No cell can contribute here, so these ARE the facet family's numbers.
        @test evaluate_functional(op, FunctionalKind(:facet_area), u, nothing) ≈ reference.area rtol = 1.0e-14
        @test reference.area ≈ 4.0                       # |Γ| = 2 + 2 on the [-1, 1]² grid
        @test evaluate_functional(op, FunctionalKind(:facet_flux), u, nothing) ≈ reference.flux rtol = 1.0e-13
        @test abs(reference.flux) > 0
        @test evaluate_functional(op, FunctionalKind(:facet_area), u, nothing) isa Float64
    end

    @testset "declared value type reproduces the scan path" begin
        @test evaluate_functional(op, FunctionalKind(:facet_area), u, nothing) ===
              evaluate_functional(op, FunctionalKind(:facet_area_undeclared), u, nothing)
        @test evaluate_functional(op, FunctionalKind(:facet_flux), u, nothing) ===
              evaluate_functional(op, FunctionalKind(:facet_flux_undeclared), u, nothing)
    end

    @testset "parallel workers reduce to the same value" begin
        # Chunk size 1 over the coloring, so the barriers are handed to genuine
        # per-worker folds rather than to one worker walking everything.
        pop = setup_operator(AssemblyStrategy(FullAssembly(), ColoredScheduling(), PolyesterDevice(min_items_per_worker = 1)),
                             FacetFunctionalProbe(:u, facets), dh)
        psc = last(pop.engine.subdomain_caches)
        @test length(psc.device_cache) == min(Threads.nthreads(), maximum(length, psc.partition))
        @test evaluate_functional(pop, FunctionalKind(:facet_area), u, nothing) ≈ reference.area rtol = 1.0e-12
        @test evaluate_functional(pop, FunctionalKind(:facet_flux), u, nothing) ≈ reference.flux rtol = 1.0e-12
        # The parallel route allocates its partials up front, so an undeclared
        # kind cannot run there — the facet family's answer is the cell family's.
        @test_throws ArgumentError evaluate_functional(pop, FunctionalKind(:facet_area_undeclared), u, nothing)
    end

    @testset "an all-quiet sweep is an empty sum" begin
        @test evaluate_functional(op, FunctionalKind(:facet_quiet), u, nothing) === 0.0
        @test_throws ArgumentError evaluate_functional(op, FunctionalKind(:facet_quiet_undeclared), u, nothing)
    end

    @testset "structural emptiness fails before any facet runs" begin
        # No items to reduce over, whatever the kind declares.
        sc = last(op.engine.subdomain_caches)
        empty_partition = (engine = FerriteOperators.AssemblyEngine(
            op.engine.strategy,
            [FerriteOperators.SubdomainCache(sc.domain, sc.device_cache, (Int[],))],
            op.engine.dh, op.engine.ivh, op.engine.protocol),)
        @test_throws ArgumentError evaluate_functional(empty_partition, FunctionalKind(:facet_area), u, nothing)
        @test_throws ArgumentError evaluate_functional(empty_partition, FunctionalKind(:facet_area_undeclared), u, nothing)
    end

    @testset "a missing facet kernel fails loudly" begin
        @test_throws MethodError evaluate_functional(op, FunctionalKind(:facet_enthalpy), u, nothing)
    end
end

####################################
## Composed facet-item terms
####################################
# Several boundary terms over ONE subdomain, each supported on its own facet
# set. The composite declares their union and the fan-out re-gates per inner —
# the per-inner `is_facet_in_cache` gate of the fused route, restated as the
# declaration this route reads it from.

# Declares a facet-item global-dof tail and nothing else: the composite's
# derivation rule is integrator-level, so no cache is needed to exercise it.
struct GlobalDofProbe <: AbstractLinearIntegrator
    dofs::Vector{Int}
end
FerriteOperators.facet_item_global_dofs(m::GlobalDofProbe, ::SubDofHandler) = m.dofs

# The same physics on either route: raw probes ride the cell sweep, the same
# probes wrapped in `LinearFacetItemProbe` get their own traversal.
itemized(probes...) = LinearCompositeIntegrator(map(LinearFacetItemProbe, probes))
fused_route(probes...) = LinearCompositeIntegrator(probes)

@testset "Composed facet items" begin
    (; grid, dh) = corner_testbed()
    strategy = sequential_strategy()
    sdh   = dh.subdofhandlers[1]
    right = Set(getfacetset(grid, "right"))
    top   = Set(getfacetset(grid, "top"))

    # Both routes exist in this release, so the fused assembly of the SAME
    # physics is the referee. `≈`, never `==`: neither route promises the other's
    # summation order into a dof several facets share, so the two may differ in
    # the last bits. The tolerance is measured — the relative difference is 0 on
    # these fixtures, which walk the facets in the same order.
    function agrees_with_fused(probes...; p = nothing)
        items = setup_operator(strategy, itemized(probes...), dh)
        fused = setup_operator(strategy, fused_route(probes...), dh)
        update_operator!(items, p)
        update_operator!(fused, p)
        return (; items, fused, agree = isapprox(items.b, fused.b; rtol = 1.0e-15))
    end

    @testset "the declaration is the union, the gate is per inner" begin
        a = LinearNeumannProbe(2.0, :u, right)
        b = LinearNeumannProbe(-3.0, :u, top)
        (; items, agree) = agrees_with_fused(a, b)
        @test agree
        # Each inner integrated ITS OWN half. Fanning out to every inner on
        # every declared facet would give (2.0 - 3.0)·|Γ| = -4.0 instead.
        @test sum(items.b) ≈ 2.0 * 2.0 + (-3.0) * 2.0
        @test sum(items.b) ≈ -2.0

        declared = facet_items(itemized(a, b), sdh)
        @test Set(declared) == union(right, top)
        @test issorted(declared; by = facet -> (facet[1], facet[2]))
        # The corner cell owns one facet from each inner, and they are ONE item.
        fd = last(items.engine.subdomain_caches).domain
        @test length(fd.items) == length(declared) - 1
        @test count(item -> length(item.local_facets) == 2, fd.items) == 1
    end

    @testset "inners may share a facet" begin
        c = LinearNeumannProbe(2.0, :u, right)
        d = LinearNeumannProbe(0.5, :u, right)
        (; items, agree) = agrees_with_fused(c, d)
        @test agree
        @test sum(items.b) ≈ (2.0 + 0.5) * 2.0
        # One declaration per facet: taking the union is what keeps the family's
        # declared-twice rejection off two terms supported on the same facet.
        @test length(facet_items(itemized(c, d), sdh)) == length(right)
    end

    @testset "each inner is handed its own parameter view" begin
        # `param_scaled` makes the load the inner's own
        # `query_facet_parameters` answer, so a fan-out re-seeding `args.p` per
        # inner is observable in the assembled vector.
        a = LinearNeumannProbe(2.0, :u, right; param_scaled = true)
        b = LinearNeumannProbe(-3.0, :u, top; param_scaled = true)
        p = 1.5
        (; items, agree) = agrees_with_fused(a, b; p)
        @test agree
        @test sum(items.b) ≈ (2.0 - 3.0) * 2.0 * p
    end

    @testset "collapse rules" begin
        a = LinearNeumannProbe(2.0, :u, right)
        quiet = LinearFacetItemProbe(LinearNeumannProbe(1.0, :u, right), Set(FacetIndex[]))
        # An inner declaring nothing contributes to neither the declaration nor
        # the cache, and one survivor is returned unwrapped.
        one_declarer = LinearCompositeIntegrator(LinearFacetItemProbe(a), quiet)
        @test Set(facet_items(one_declarer, sdh)) == right
        @test setup_facet_item_cache(one_declarer, sdh) isa NeumannProbeCache
        @test setup_facet_item_cache(itemized(a, LinearNeumannProbe(1.0, :u, top)), sdh) isa
            FerriteOperators.CompositeFacetItemCache

        # All-silent keeps the additive `()` default, so a composite of ordinary
        # boundary terms grows no facet-item subdomain.
        @test facet_items(fused_route(a, LinearNeumannProbe(1.0, :u, top)), sdh) == ()
    end

    @testset "facet_item_global_dofs is the inners' one declaration" begin
        quiet = LinearNeumannProbe(1.0, :u, right)
        @test facet_item_global_dofs(LinearCompositeIntegrator(GlobalDofProbe([7, 9]), quiet), sdh) == [7, 9]
        @test facet_item_global_dofs(LinearCompositeIntegrator(GlobalDofProbe([7, 9]), GlobalDofProbe([7, 9])), sdh) == [7, 9]
        # Each family answers from its own hook: a facet-item tail leaves the
        # cell sweep in the field space.
        @test global_dofs(LinearCompositeIntegrator(GlobalDofProbe([7, 9]), quiet), sdh) == ()
        # Two different declarations have no unambiguous tail, on this hook as
        # on `global_dofs`.
        err = @test_throws ArgumentError facet_item_global_dofs(
            LinearCompositeIntegrator(GlobalDofProbe([7, 9]), GlobalDofProbe([9, 7])), sdh)
        @test occursin("facet_item_global_dofs", err.value.msg)
        @test facet_item_global_dofs(LinearCompositeIntegrator(quiet, quiet), sdh) == ()
    end

    @testset "routing outside, composition inside" begin
        rgrid = generate_grid(Quadrilateral, (4, 4))
        addcellset!(rgrid, "right_cells", x -> x[1] ≥ 0.0)
        addcellset!(rgrid, "left_cells", x -> x[1] ≤ 0.0)
        rdh = DofHandler(rgrid)
        for name in ("right_cells", "left_cells")
            sub = SubDofHandler(rdh, getcellset(rgrid, name))
            add!(sub, :u, Lagrange{RefQuadrilateral, 1}())
        end
        close!(rdh)

        # A facet item's local system is its owning cell's, so each subdomain
        # declares only the facets its own cells own.
        owned(name, cellset) = Set(f for f in getfacetset(rgrid, name) if f[1] in cellset)
        item(scale, facets) = LinearFacetItemProbe(LinearNeumannProbe(scale, :u, facets))
        rc, lc = getcellset(rgrid, "right_cells"), getcellset(rgrid, "left_cells")

        op = setup_operator(sequential_strategy(), LinearMultiDomainIntegrator(Dict(
            "right_cells" => LinearCompositeIntegrator(item(1.5, owned("right", rc)), item(2.0, owned("top", rc))),
            "left_cells" => LinearCompositeIntegrator(item(-0.5, owned("left", lc)), item(0.25, owned("top", lc))))), rdh)
        update_operator!(op, nothing)
        @test sum(op.b) ≈ 1.5 * 2.0 + 2.0 * 1.0 + (-0.5) * 2.0 + 0.25 * 1.0

        facet_domains = [sc.domain for sc in op.engine.subdomain_caches
                             if sc.domain isa FerriteOperators.FacetItemDomain]
        @test length(facet_domains) == 2
        @test all(d -> d.element isa FerriteOperators.CompositeFacetItemCache, facet_domains)
        @test all(d -> issubset(Set(item.cellid for item in d.items), d.sdh.cellset), facet_domains)
    end
end

####################################
## Weighted Jacobians on facet items
####################################

@testset "Weighted Jacobians over facet items" begin
    (; grid, dh) = corner_testbed()
    strategy = sequential_strategy()
    right   = Set(getfacetset(grid, "right"))
    n       = ndofs(dh)
    Δt      = 0.25
    ctx     = TimeIntegrationContext(1.0, Δt, Δt)
    states  = (u = sin.(0.3 .* (1:n)), v = cos.(0.2 .* (1:n)))
    weights = (u = 1.0, v = 1 / (0.5 * Δt))

    spring  = NonlinearFacetItemProbe(RobinFacetIntegrator(2.5, :u, right; slot = :u, fused = true))
    dashpot = NonlinearFacetItemProbe(RobinFacetIntegrator(0.75, :u, right; slot = :v, fused = true))

    # ∂F/∂u of the spring and ∂F/∂v of the dashpot, each assembled on its own.
    sop = setup_operator(strategy, spring, dh; slots = (:u, :v))
    dop = setup_operator(strategy, dashpot, dh; slots = (:u, :v))
    Kf = FerriteOperators.create_system_matrix(sop.engine.strategy, dh)
    Df = share_pattern(Kf)
    assemble_slot_jacobian!(Kf, sop, JacobianKind{:u}(), states, nothing, ctx)
    assemble_slot_jacobian!(Df, dop, JacobianKind{:v}(), states, nothing, ctx)
    @test nnz(Kf) > 0 && nnz(Df) > 0

    @testset "a composite routes the weighted sweep per inner" begin
        pericardium = setup_operator(strategy, NonlinearCompositeIntegrator(spring, dashpot), dh; slots = (:u, :v))
        FUSED_FACET_W_CALLS[] = 0
        Wp = share_pattern(pericardium.J)
        assemble_weighted_jacobian!(Wp, pericardium, weights, states, nothing, ctx)
        # One fused kernel call per inner per facet: the composite claims
        # neither slot, which is what makes the spring/dashpot pair legal.
        @test FUSED_FACET_W_CALLS[] == 2 * length(right)
        @test Matrix(Wp) ≈ weights.u .* Matrix(Kf) .+ weights.v .* Matrix(Df) rtol = 1.0e-12
    end

    @testset "the fused kernel is the only route, and its absence is loud" begin
        # Per-slot facet Jacobians alone: enough for the fused boundary route's
        # fold, not for this one.
        unfused = NonlinearFacetItemProbe(RobinFacetIntegrator(2.5, :u, right; slot = :u))
        pinned = "only route a weighted Jacobian sweep takes on the facet-item route"

        err = @test_throws ArgumentError setup_operator(strategy, unfused, dh;
                                                        slots = (:u, :v), requests = (WeightedJacobianKind,))
        @test occursin(pinned, err.value.msg)
        @test occursin("WeightedJacobianRequest", err.value.msg)

        # Undeclared, the operator builds and the sweep raises the same message
        # instead of reaching a bare `MethodError`.
        op = setup_operator(strategy, unfused, dh; slots = (:u, :v))
        err = @test_throws ArgumentError assemble_weighted_jacobian!(
            share_pattern(op.J), op, weights, states, nothing, ctx)
        @test occursin(pinned, err.value.msg)
        @test occursin("provides_analytic", err.value.msg)

        # One unfused inner is enough — the election is per inner, at setup and
        # in the sweep alike.
        mixed = NonlinearCompositeIntegrator(spring, unfused)
        err = @test_throws ArgumentError setup_operator(strategy, mixed, dh;
                                                        slots = (:u, :v), requests = (WeightedJacobianKind,))
        @test occursin(pinned, err.value.msg)
        mop = setup_operator(strategy, mixed, dh; slots = (:u, :v))
        @test_throws ArgumentError assemble_weighted_jacobian!(
            share_pattern(mop.J), mop, weights, states, nothing, ctx)
    end
end

@testset "structural reduction answer of the facet family" begin
    # Facet domains serve reductions and decline condensation; the declared set
    # is the domain of integration, so no cache-type refinement narrows this.
    fd = FerriteOperators.FacetItemDomain(nothing, nothing, nothing)
    @test FerriteOperators._may_contribute(fd, FunctionalKind(:probe))
    @test !FerriteOperators._may_contribute(fd, FerriteOperators.CondensationKind((u = 1.0,)))
    @test FerriteOperators._may_contribute(fd, JacobianKind())
end
