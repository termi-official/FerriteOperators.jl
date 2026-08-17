using FerriteOperators
using FerriteOperatorsExampleElements
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester
using TimerOutputs

# A real facet kernel exercising the framework-owned boundary driver: a
# constant Neumann load t̄ on a facet set, with the analytic reference
# sum(b) = t̄ · |Γ|.
struct NeumannTestIntegrator <: AbstractLinearIntegrator
    t̄::Float64
    qrc::QuadratureRuleCollection
    field_name::Symbol
    facetset::Set{FacetIndex}
end
struct NeumannTestCache{FV <: FacetValues} <: FerriteOperators.AbstractSurfaceElementCache
    t̄::Float64
    fv::FV
    facetset::Set{FacetIndex}
end
function FerriteOperators.setup_element_cache(m::NeumannTestIntegrator, sdh::SubDofHandler)
    return FerriteOperators.EmptyVolumetricElementCache()
end
function FerriteOperators.setup_boundary_cache(m::NeumannTestIntegrator, sdh::SubDofHandler)
    fqr = FacetQuadratureRule{RefHexahedron}(2)
    ip  = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return NeumannTestCache(m.t̄, FacetValues(fqr, ip, ip_geo), m.facetset)
end
FerriteOperators.duplicate_for_device(device, c::NeumannTestCache) =
    NeumannTestCache(c.t̄, FerriteOperators.duplicate_for_device(device, c.fv), c.facetset)
FerriteOperators.is_facet_in_cache(idx::FacetIndex, cell, c::NeumannTestCache) = idx ∈ c.facetset
function FerriteOperators.assemble_facet!(req::ResidualRequest, c::NeumannTestCache, args, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    for qp in 1:getnquadpoints(c.fv)
        dΓ = getdetJdV(c.fv, qp)
        for i in 1:getnbasefunctions(c.fv)
            req.r[i] += c.t̄ * shape_value(c.fv, qp, i) * dΓ
        end
    end
end

@testset "Facet driver with a real Neumann kernel" begin
    grid = generate_grid(Hexahedron, (2, 2, 2))   # unit cube [-1,1]³ → right face area 4.0
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    t̄ = 3.25
    right = Set(getfacetset(grid, "right"))
    integrator = NeumannTestIntegrator(t̄, QuadratureRuleCollection(2), :u, right)

    op = setup_operator(SequentialAssemblyStrategy(SequentialCPUDevice()), integrator, dh)
    update_operator!(op, nothing)
    area = 4.0
    @test sum(op.b) ≈ t̄ * area rtol = 1e-12
    # only dofs on the loaded face carry entries
    @test count(!iszero, op.b) == 9

    # parallel path through the same driver
    opp = setup_operator(PerColorAssemblyStrategy(PolyesterDevice(2)), integrator, dh)
    update_operator!(opp, nothing)
    @test opp.b ≈ op.b rtol = 1e-13

    # composite surface fan-out re-gates per inner cache: two identical inner
    # caches double the load
    sdh = first(dh.subdofhandlers)
    inner = FerriteOperators.setup_boundary_cache(integrator, sdh)
    comp  = FerriteOperators.CompositeSurfaceElementCache((inner, inner))
    b2 = zeros(ndofs(dh))
    cc = Ferrite.CellCache(sdh)
    for cellid in 1:getncells(grid)
        reinit!(cc, cellid)
        rₑ = zeros(ndofs_per_cell(sdh))
        args = KernelArgs((;), cc, nothing, nothing, nothing)
        for lfi in 1:nfacets(cc)
            if FerriteOperators.is_facet_in_cache(FacetIndex(cellid, lfi), cc, comp)
                FerriteOperators.assemble_facet!(ResidualRequest(rₑ), comp, args, lfi)
            end
        end
        b2[celldofs(cc)] .+= rₑ
    end
    @test sum(b2) ≈ 2 * t̄ * area rtol = 1e-12
end

@testset "Element API" begin
    import FerriteOperators: assemble_cell!, assemble_facet!
    import FerriteOperators: setup_element_cache, setup_boundary_cache
    import FerriteOperators

    setup_test_cache(kwargs...) =
        FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
    function setup_test_composite_volume_cache(kwargs...)
        element_cache =
            FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeVolumetricElementCache((element_cache, element_cache)),
        )
    end
    function setup_test_composite_surface_cache(kwargs...)
        element_cache =
            FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_boundary_cache(kwargs...))
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeSurfaceElementCache((element_cache, element_cache)),
        )
    end

    grid = generate_grid(Hexahedron, (1, 1, 1))
    qrc  = QuadratureRuleCollection(3)
    qr   = QuadratureRule{RefHexahedron}(3)
    qrcf = QuadratureRuleCollection(3)
    qrf  = FacetQuadratureRule{RefHexahedron}(3)
    ip   = Lagrange{RefHexahedron, 1}()

    dhs = DofHandler(grid)
    add!(dhs, :u, ip)
    close!(dhs)
    sdhs = first(dhs.subdofhandlers)
    cell_cache_s = Ferrite.CellCache(sdhs)
    Ferrite.reinit!(cell_cache_s, 1)
    uₑs = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0] .* 1e-4

    ipv = ip^3
    dhv = DofHandler(grid)
    add!(dhv, :u, ipv)
    close!(dhv)
    sdhv = first(dhv.subdofhandlers)
    cell_cache_v = Ferrite.CellCache(sdhv)
    Ferrite.reinit!(cell_cache_v, 1)
    uₑv =
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ] .* 1e-4

    # We check for pairwise consistency of the assembly operations
    # First we check if the empty caches work correctly
    @testset "Empty caches" begin
        rₑ¹ = zeros(ndofs(dhs))
        rₑ² = zeros(ndofs(dhs))
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        args = KernelArgs((u = uₑs,), cell_cache_s, 0.0, nothing, nothing)

        # Volume
        assemble_cell!(JacobianResidualRequest(Kₑ¹, rₑ¹), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(Kₑ¹)
        @test iszero(rₑ¹)

        assemble_cell!(ResidualRequest(rₑ²), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(rₑ²)

        assemble_cell!(JacobianRequest{:u}(Kₑ²), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(Kₑ²)

        # Surface: the empty cache never claims a facet, and its kernels are no-ops
        for local_facet_index = 1:nfacets(cell_cache_s)
            @test !FerriteOperators.is_facet_in_cache(FacetIndex(1, local_facet_index), cell_cache_s, FerriteOperators.EmptySurfaceElementCache())
            assemble_facet!(JacobianResidualRequest(Kₑ¹, rₑ¹), FerriteOperators.EmptySurfaceElementCache(), args, local_facet_index)
            @test iszero(Kₑ¹)
            @test iszero(rₑ¹)

            assemble_facet!(ResidualRequest(rₑ²), FerriteOperators.EmptySurfaceElementCache(), args, local_facet_index)
            @test iszero(rₑ²)

            assemble_facet!(JacobianRequest{:u}(Kₑ²), FerriteOperators.EmptySurfaceElementCache(), args, local_facet_index)
            @test iszero(Kₑ²)
        end
    end

    @testset "Scalar volumetric bilinear composite elements: $model" for model in (
        SimpleBilinearMassIntegrator(1.0, qrc, :u),
        SimpleBilinearDiffusionIntegrator(1.0, qrc, :u),
    )
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        args = KernelArgs((;), cell_cache_s, 0.0, nothing, nothing)
        reinit_values!(element_cache, cell_cache_s)
        assemble_cell!(JacobianRequest{:u}(Kₑ¹), element_cache, args)
        @test !iszero(Kₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        reinit_values!(composite_element_cache, cell_cache_s)
        assemble_cell!(JacobianRequest{:u}(Kₑ²), composite_element_cache, args)
        @test 2Kₑ¹ ≈ Kₑ²
    end

    @testset "Scalar linear composite elements: $model" for model in (
        SimpleLinearIntegrator(1.0, qrc, :u),
    )
        bₑ¹ = zeros(ndofs(dhs))
        bₑ² = zeros(ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        args = KernelArgs((;), cell_cache_s, 0.0, nothing, nothing)
        reinit_values!(element_cache, cell_cache_s)
        assemble_cell!(ResidualRequest(bₑ¹), element_cache, args)
        @test !iszero(bₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        reinit_values!(composite_element_cache, cell_cache_s)
        assemble_cell!(ResidualRequest(bₑ²), composite_element_cache, args)
        @test 2bₑ¹ ≈ bₑ²
    end
end

# Probes for composition: each carries its own parameter scale, so a residual
# assembled through a composite reveals whether every inner got its OWN
# parameter view or the outer one was reused for all of them.
struct ParamProbeIntegrator <: AbstractLinearIntegrator
    scale::Float64
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct ParamProbeCache{CV <: CellValues} <: FerriteOperators.AbstractVolumetricElementCache
    scale::Float64
    cv::CV
end
function FerriteOperators.setup_element_cache(m::ParamProbeIntegrator, sdh::SubDofHandler)
    qr = getquadraturerule(m.qrc, sdh)
    ip = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return ParamProbeCache(m.scale, CellValues(qr, ip, ip_geo))
end
Ferrite.getnquadpoints(c::ParamProbeCache) = getnquadpoints(c.cv)
FerriteOperators.reinit_values!(c::ParamProbeCache, cell) = reinit!(c.cv, cell)
FerriteOperators.duplicate_for_device(device, c::ParamProbeCache) =
    ParamProbeCache(c.scale, FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.query_cell_parameters(c::ParamProbeCache, cell, p) = c.scale * p
function FerriteOperators.assemble_cell!(req::ResidualRequest, c::ParamProbeCache, args)
    for qp in 1:getnquadpoints(c.cv)
        dΩ = getdetJdV(c.cv, qp)
        for i in 1:getnbasefunctions(c.cv)
            req.r[i] += args.p * shape_value(c.cv, qp, i) * dΩ
        end
    end
end

struct FacetParamProbeIntegrator <: AbstractLinearIntegrator
    scale::Float64
    field_name::Symbol
    facetset::Set{FacetIndex}
end
struct FacetParamProbeCache{FV <: FacetValues} <: FerriteOperators.AbstractSurfaceElementCache
    scale::Float64
    fv::FV
    facetset::Set{FacetIndex}
end
FerriteOperators.setup_element_cache(::FacetParamProbeIntegrator, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
function FerriteOperators.setup_boundary_cache(m::FacetParamProbeIntegrator, sdh::SubDofHandler)
    fqr = FacetQuadratureRule{RefHexahedron}(2)
    ip = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return FacetParamProbeCache(m.scale, FacetValues(fqr, ip, ip_geo), m.facetset)
end
FerriteOperators.duplicate_for_device(device, c::FacetParamProbeCache) =
    FacetParamProbeCache(c.scale, FerriteOperators.duplicate_for_device(device, c.fv), c.facetset)
FerriteOperators.is_facet_in_cache(idx::FacetIndex, cell, c::FacetParamProbeCache) = idx ∈ c.facetset
FerriteOperators.query_facet_parameters(c::FacetParamProbeCache, cell, lfi, p) = c.scale * p
function FerriteOperators.assemble_facet!(req::ResidualRequest, c::FacetParamProbeCache, args, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    for qp in 1:getnquadpoints(c.fv)
        dΓ = getdetJdV(c.fv, qp)
        for i in 1:getnbasefunctions(c.fv)
            req.r[i] += args.p * shape_value(c.fv, qp, i) * dΓ
        end
    end
end

# A cache that only claims internal state, to reach the composite guard
# without standing up an InternalVariableHandler.
struct StatefulProbeCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.has_internal_state(::Type{StatefulProbeCache}) = true
FerriteOperators.assemble_cell!(::ResidualRequest, ::StatefulProbeCache, args) = nothing
FerriteOperators.reinit_values!(::StatefulProbeCache, cell) = nothing
Ferrite.getnquadpoints(::StatefulProbeCache) = 0

# A stateless cache serving one sensitivity kind analytically, so a composite's
# admissibility rejection has an inner it must NOT name.
struct AnalyticProbeCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.provides_analytic(::Type{AnalyticProbeCache}, ::ParameterJacobianKind) = true

@testset "Composition" begin
    grid = generate_grid(Hexahedron, (2, 2, 2))    # [-1,1]³ → volume 8, right face area 4
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    sdh = first(dh.subdofhandlers)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    qrc = QuadratureRuleCollection(2)
    n = ndofs(dh)

    @testset "per-inner parameter views" begin
        # Debt (iii): the driver queries parameters on the composite and bakes
        # ONE pₑ into args. Each inner must still receive its own view.
        p = 1.5
        op = setup_operator(strategy, LinearCompositeIntegrator(
            ParamProbeIntegrator(2.0, qrc, :u),
            ParamProbeIntegrator(5.0, qrc, :u),
        ), dh)
        update_operator!(op, p)
        @test sum(op.b) ≈ (2.0 + 5.0) * p * 8.0 rtol = 1e-12

        # Same for the facet path, whose parameters are queried per facet.
        right = Set(getfacetset(grid, "right"))
        fop = setup_operator(strategy, LinearCompositeIntegrator(
            FacetParamProbeIntegrator(2.0, :u, right),
            FacetParamProbeIntegrator(5.0, :u, right),
        ), dh)
        update_operator!(fop, p)
        @test sum(fop.b) ≈ (2.0 + 5.0) * p * 4.0 rtol = 1e-12

        # A hand-built args carrying a plain `p` (no composite query ran)
        # reaches every inner unchanged.
        composite = FerriteOperators.CompositeVolumetricElementCache((
            FerriteOperators.setup_element_cache(ParamProbeIntegrator(2.0, qrc, :u), sdh),
            FerriteOperators.setup_element_cache(ParamProbeIntegrator(5.0, qrc, :u), sdh),
        ))
        cc = Ferrite.CellCache(sdh)
        Ferrite.reinit!(cc, 1)
        reinit_values!(composite, cc)
        rraw = zeros(ndofs_per_cell(sdh))
        assemble_cell!(ResidualRequest(rraw), composite, KernelArgs((;), cc, p, nothing, nothing))
        rown = zeros(ndofs_per_cell(sdh))
        pₑ = FerriteOperators.query_cell_parameters(composite, cc, p)
        assemble_cell!(ResidualRequest(rown), composite, KernelArgs((;), cc, pₑ, nothing, nothing))
        @test sum(rraw) ≈ 2 * p * 1.0 rtol = 1e-12
        @test sum(rown) ≈ (2.0 + 5.0) * p * 1.0 rtol = 1e-12
    end

    @testset "constructor-built equals hand-built" begin
        D = SimpleBilinearDiffusionIntegrator(1.3, qrc, :u)
        M = SimpleBilinearMassIntegrator(0.7, qrc, :u)
        built = FerriteOperators.setup_element_cache(BilinearCompositeIntegrator(D, M), sdh)
        hand = FerriteOperators.CompositeVolumetricElementCache((
            FerriteOperators.setup_element_cache(D, sdh),
            FerriteOperators.setup_element_cache(M, sdh),
        ))
        cc = Ferrite.CellCache(sdh)
        Ferrite.reinit!(cc, 1)
        args = KernelArgs((;), cc, nothing, nothing, nothing)
        K1 = zeros(ndofs_per_cell(sdh), ndofs_per_cell(sdh))
        K2 = similar(K1); fill!(K2, 0.0)
        reinit_values!(built, cc); assemble_cell!(JacobianRequest{:u}(K1), built, args)
        reinit_values!(hand, cc);  assemble_cell!(JacobianRequest{:u}(K2), hand, args)
        @test K1 == K2                 # bit-level, same inputs
        @test !iszero(K1)
    end

    @testset "collapse rules" begin
        empty_v = FerriteOperators.EmptyVolumetricElementCache()
        empty_s = FerriteOperators.EmptySurfaceElementCache()
        c1 = FerriteOperators.setup_element_cache(SimpleLinearIntegrator(1.0, qrc, :u), sdh)
        c2 = FerriteOperators.setup_element_cache(SimpleBilinearMassIntegrator(1.0, qrc, :u), sdh)

        @test FerriteOperators.compose_element_caches((empty_v, empty_v)) === empty_v
        @test FerriteOperators.compose_element_caches((empty_v, c1)) === c1
        @test FerriteOperators.compose_element_caches((c1, c2)) isa FerriteOperators.CompositeVolumetricElementCache
        @test length(FerriteOperators.compose_element_caches((c1, empty_v, c2)).inner_caches) == 2
        @test FerriteOperators.compose_boundary_caches((empty_s, empty_s)) === empty_s

        # The empty-boundary fast path survives composition: an integrator with
        # no boundary term still yields the empty surface cache.
        model = LinearCompositeIntegrator(
            SimpleLinearIntegrator(1.0, qrc, :u),
            SimpleLinearIntegrator(2.0, qrc, :u),
        )
        @test FerriteOperators.setup_boundary_cache(model, sdh) isa FerriteOperators.EmptySurfaceElementCache
    end

    @testset "quadrature agreement" begin
        c2a = FerriteOperators.setup_element_cache(SimpleLinearIntegrator(1.0, QuadratureRuleCollection(2), :u), sdh)
        c2b = FerriteOperators.setup_element_cache(SimpleBilinearMassIntegrator(1.0, QuadratureRuleCollection(2), :u), sdh)
        c3 = FerriteOperators.setup_element_cache(SimpleLinearIntegrator(1.0, QuadratureRuleCollection(3), :u), sdh)
        agree = FerriteOperators.CompositeVolumetricElementCache((c2a, c2b))
        @test getnquadpoints(agree) == getnquadpoints(c2a)
        disagree = FerriteOperators.CompositeVolumetricElementCache((c2a, c3))
        @test_throws ArgumentError getnquadpoints(disagree)
    end

    @testset "loud rejections" begin
        D = SimpleBilinearDiffusionIntegrator(1.0, qrc, :u)
        L = SimpleLinearIntegrator(1.0, qrc, :u)
        visco = SimpleCondensedLinearViscoelasticity(MaxwellParameters(), qrc, :u, :εᵛ)

        @test_throws ArgumentError BilinearCompositeIntegrator(())
        @test_throws ArgumentError NonlinearCompositeIntegrator(())
        @test_throws ArgumentError LinearCompositeIntegrator(())
        @test_throws ArgumentError BilinearCompositeIntegrator(D, L)   # linear sink in bilinear
        @test_throws ArgumentError LinearCompositeIntegrator(L, D)     # bilinear sink in linear
        @test_throws ArgumentError NonlinearCompositeIntegrator(D, L)  # linear sink in nonlinear
        @test_throws ArgumentError NonlinearCompositeIntegrator(D, visco)  # condensed inner

        # A bilinear inner in a nonlinear composite is legitimate.
        @test NonlinearCompositeIntegrator(D, D) isa AbstractNonlinearIntegrator

        # Nested composites are flattened at construction.
        @test length(BilinearCompositeIntegrator(BilinearCompositeIntegrator(D, D), D).subintegrators) == 3

        # A hand-built composite with a condensed inner is rejected too — the
        # constructor is not the only way to build one.
        hand = FerriteOperators.CompositeVolumetricElementCache((
            StatefulProbeCache(),
            FerriteOperators.setup_element_cache(D, sdh),
        ))
        @test_throws ArgumentError FerriteOperators.validate_element_cache(hand)
    end

    @testset "internal state on a hand-built composite" begin
        # `validate_element_cache` rejects stateful inners, so a composite's
        # internal-state seams are reachable only on a cache built by hand.
        D = FerriteOperators.setup_element_cache(SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), sdh)
        stateless = FerriteOperators.CompositeVolumetricElementCache((AnalyticProbeCache(), D))
        stateful  = FerriteOperators.CompositeVolumetricElementCache((StatefulProbeCache(), AnalyticProbeCache()))

        @test !FerriteOperators.has_internal_state(typeof(stateless))
        @test FerriteOperators.has_internal_state(typeof(stateful))

        kind = ParameterJacobianKind()
        @test FerriteOperators.assert_sensitivity_admissible(typeof(stateless), kind) === nothing

        err = @test_throws ArgumentError FerriteOperators.assert_sensitivity_admissible(typeof(stateful), kind)
        @test occursin("StatefulProbeCache", err.value.msg)      # names the inner lacking the kind
        @test !occursin("AnalyticProbeCache", err.value.msg)     # and only that one
    end
end
