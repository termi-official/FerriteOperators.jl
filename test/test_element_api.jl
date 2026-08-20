using FerriteOperators
using FerriteOperatorsExampleElements
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester
using TimerOutputs

include(joinpath(@__DIR__, "fixture_elements.jl"))

# A real facet kernel exercising the framework-owned boundary driver: a
# constant Neumann load t̄ on a facet set, with the analytic reference
# sum(b) = t̄ · |Γ|.
@testset "Facet driver with a real Neumann kernel" begin
    grid = generate_grid(Hexahedron, (2, 2, 2))   # unit cube [-1,1]³ → right face area 4.0
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    t̄ = 3.25
    right = Set(getfacetset(grid, "right"))
    integrator = LinearNeumannProbe(t̄, :u, right)

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

        args = CellArgs((u = uₑs,), cell_cache_s, 0.0, nothing)

        # Volume
        assemble_cell!(JacobianResidualRequest(Kₑ¹, rₑ¹), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(Kₑ¹)
        @test iszero(rₑ¹)

        assemble_cell!(ResidualRequest(rₑ²), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(rₑ²)

        assemble_cell!(JacobianRequest{:u}(Kₑ²), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(Kₑ²)

        # Surface: the empty cache never claims a facet …
        for local_facet_index = 1:nfacets(cell_cache_s)
            @test !FerriteOperators.is_facet_in_cache(FacetIndex(1, local_facet_index), cell_cache_s, FerriteOperators.EmptySurfaceElementCache())
        end

        # … and its kernels are no-ops, whichever facet they are handed
        fargs = FacetArgs((u = uₑs,), cell_cache_s, 0.0, nothing)
        assemble_facet!(JacobianResidualRequest(Kₑ¹, rₑ¹), FerriteOperators.EmptySurfaceElementCache(), fargs, 1)
        @test iszero(Kₑ¹)
        @test iszero(rₑ¹)

        assemble_facet!(ResidualRequest(rₑ²), FerriteOperators.EmptySurfaceElementCache(), fargs, 1)
        @test iszero(rₑ²)

        assemble_facet!(JacobianRequest{:u}(Kₑ²), FerriteOperators.EmptySurfaceElementCache(), fargs, 1)
        @test iszero(Kₑ²)
    end

    @testset "Scalar volumetric bilinear composite elements: $model" for model in (
        SimpleBilinearMassIntegrator(1.0, qrc, :u),
        SimpleBilinearDiffusionIntegrator(1.0, qrc, :u),
    )
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        args = CellArgs((;), cell_cache_s, 0.0, nothing)
        reinit_values!(element_cache, cell_cache_s)
        assemble_cell!(JacobianRequest{:u}(Kₑ¹), element_cache, args)
        @test !iszero(Kₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        reinit_values!(composite_element_cache, cell_cache_s)
        assemble_cell!(JacobianRequest{:u}(Kₑ²), composite_element_cache, args)
        @test 2Kₑ¹ ≈ Kₑ²
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

# The facet counterpart: the load is the parameter the driver queried for this
# facet, so a composite reveals whether each inner got its own view.
FacetParamProbeIntegrator(scale, field_name, facetset) =
    LinearNeumannProbe(scale, field_name, facetset; param_scaled = true)

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
        # The driver queries parameters on the composite and bakes ONE pₑ into
        # args. Each inner must still receive its own view.
        p = 1.5
        # Distinct scales pin what the doubling property of identical inners
        # does not: every inner contributes once, with its own view. The
        # parallel strategy is the composite through `duplicate_for_device`.
        @testset "$composite_strategy" for composite_strategy in (
            SequentialAssemblyStrategy(SequentialCPUDevice()),
            PerColorAssemblyStrategy(PolyesterDevice(2)),
        )
            op = setup_operator(composite_strategy, LinearCompositeIntegrator(
                ParamProbeIntegrator(2.0, qrc, :u),
                ParamProbeIntegrator(5.0, qrc, :u),
            ), dh)
            update_operator!(op, p)
            @test sum(op.b) ≈ (2.0 + 5.0) * p * 8.0 rtol = 1e-12

            # Same for the facet path, whose parameters are queried per facet.
            right = Set(getfacetset(grid, "right"))
            fop = setup_operator(composite_strategy, LinearCompositeIntegrator(
                FacetParamProbeIntegrator(2.0, :u, right),
                FacetParamProbeIntegrator(5.0, :u, right),
            ), dh)
            update_operator!(fop, p)
            @test sum(fop.b) ≈ (2.0 + 5.0) * p * 4.0 rtol = 1e-12
        end

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
        assemble_cell!(ResidualRequest(rraw), composite, CellArgs((;), cc, p, nothing))
        rown = zeros(ndofs_per_cell(sdh))
        pₑ = FerriteOperators.query_cell_parameters(composite, cc, p)
        assemble_cell!(ResidualRequest(rown), composite, CellArgs((;), cc, pₑ, nothing))
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
        args = CellArgs((;), cc, nothing, nothing)
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

# `has_internal_state` caches without an analytic Jacobian/JacobianResidual
# kernel used to slip past `setup_operator`: JacobianKind/JacobianResidualKind
# were absent from `requires_admissibility_check`, so a stateful cache with no
# analytic kernel took the AD-from-residual fallback THROUGH its local solve
# instead of being rejected. That fallback differentiates correctly, but the
# driver's trial write-back only runs on the non-AD path, so the local state
# stored afterward is stale. `StatefulProbeCache` (defined above) is exactly
# such a cache; wiring it to an integrator reaches it through `setup_operator`.
struct StaleQIntegrator <: AbstractNonlinearIntegrator end
FerriteOperators.setup_element_cache(::StaleQIntegrator, ::SubDofHandler) = StatefulProbeCache()

struct InsensitiveStatefulProbeCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.has_internal_state(::Type{InsensitiveStatefulProbeCache}) = true
FerriteOperators.internal_state_insensitive(::Type{InsensitiveStatefulProbeCache}, ::Union{JacobianKind, JacobianResidualKind}) = true
FerriteOperators.assemble_cell!(::ResidualRequest, ::InsensitiveStatefulProbeCache, args) = nothing
FerriteOperators.reinit_values!(::InsensitiveStatefulProbeCache, cell) = nothing
Ferrite.getnquadpoints(::InsensitiveStatefulProbeCache) = 0

struct InsensitiveStaleQIntegrator <: AbstractNonlinearIntegrator end
FerriteOperators.setup_element_cache(::InsensitiveStaleQIntegrator, ::SubDofHandler) = InsensitiveStatefulProbeCache()

struct StatelessNoAnalyticCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.assemble_cell!(::ResidualRequest, ::StatelessNoAnalyticCache, args) = nothing
FerriteOperators.reinit_values!(::StatelessNoAnalyticCache, cell) = nothing
Ferrite.getnquadpoints(::StatelessNoAnalyticCache) = 0

struct StatelessNoAnalyticIntegrator <: AbstractNonlinearIntegrator end
FerriteOperators.setup_element_cache(::StatelessNoAnalyticIntegrator, ::SubDofHandler) = StatelessNoAnalyticCache()

@testset "Stale-q admissibility hole: JacobianKind/JacobianResidualKind" begin
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

    # No analytic kernel and no insensitivity declaration: rejected at setup,
    # naming the actual remedies (never `FiniteDifferenceSensitivity` — that
    # escape exists only for time sensitivities, not for Jacobian kinds).
    err = @test_throws ArgumentError setup_operator(strategy, StaleQIntegrator(), dh)
    @test occursin("StatefulProbeCache", err.value.msg)
    @test occursin("internal_state_insensitive", err.value.msg)
    @test occursin("provides_analytic", err.value.msg)
    @test !occursin("FiniteDifferenceSensitivity", err.value.msg)

    # `internal_state_insensitive` declared for both kinds: admissible.
    @test setup_operator(strategy, InsensitiveStaleQIntegrator(), dh) isa FerriteOperators.LinearizedFerriteOperator

    # Stateless cache without analytic kernels: the AD fallback is legitimate,
    # this fix must not gate anything but `has_internal_state` caches.
    @test setup_operator(strategy, StatelessNoAnalyticIntegrator(), dh) isa FerriteOperators.LinearizedFerriteOperator
end
