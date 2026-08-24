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
    ip   = Lagrange{RefHexahedron, 1}()

    dhs = DofHandler(grid)
    add!(dhs, :u, ip)
    close!(dhs)
    sdhs = first(dhs.subdofhandlers)
    cell_cache_s = Ferrite.CellCache(sdhs)
    Ferrite.reinit!(cell_cache_s, 1)
    uₑs = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0] .* 1e-4

    # We check for pairwise consistency of the assembly operations
    # First we check if the empty caches work correctly
    @testset "Empty caches" begin
        K = zeros(ndofs(dhs), ndofs(dhs))
        r = zeros(ndofs(dhs))
        args  = CellArgs((u = uₑs,), cell_cache_s, 0.0, nothing)
        fargs = FacetArgs((u = uₑs,), cell_cache_s, 0.0, nothing)

        # The empty cache never claims a facet …
        for local_facet_index = 1:nfacets(cell_cache_s)
            @test !FerriteOperators.is_facet_in_cache(FacetIndex(1, local_facet_index), cell_cache_s, FerriteOperators.EmptySurfaceElementCache())
        end

        # … and both empty caches are null objects: whichever request they are
        # handed, the buffers it names come back as they went in.
        @testset "$route" for (route, run!) in (
            ("volumetric", req -> assemble_cell!(req, FerriteOperators.EmptyVolumetricElementCache(), args)),
            ("surface",    req -> assemble_facet!(req, FerriteOperators.EmptySurfaceElementCache(), fargs, 1)),
        )
            for (req, buffers) in ((JacobianResidualRequest(K, r), (K, r)),
                                   (ResidualRequest(r), (r,)),
                                   (JacobianRequest{:u}(K), (K,)))
                fill!(K, 0.0); fill!(r, 0.0)
                run!(req)
                @test all(iszero, buffers)
            end
        end
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

# A `has_internal_state` cache serving neither JacobianKind nor
# JacobianResidualKind analytically is inadmissible: the AD-from-residual
# fallback would differentiate through the local solve, and the driver's trial
# write-back runs only on the non-AD path, so the state stored afterwards would
# be stale. `StatefulProbeCache` (defined above) is exactly such a cache; wiring
# it to an integrator reaches it through `setup_operator`.
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

@testset "Stateful caches need an analytic JacobianKind/JacobianResidualKind kernel" begin
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
    # so the gate closes on `has_internal_state` caches and on nothing else.
    @test setup_operator(strategy, StatelessNoAnalyticIntegrator(), dh) isa FerriteOperators.LinearizedFerriteOperator
end

# A condensed cache whose internal-dof count hook is written with every
# argument annotated — the shape an author naturally reaches for, and the one
# an `Any`-argument `hasmethod` probe does not match.
struct AnnotatedHookCache <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.has_internal_state(::Type{AnnotatedHookCache}) = true
FerriteOperators.internal_state_insensitive(::Type{AnnotatedHookCache}, kind) = true
FerriteOperators.assemble_cell!(::ResidualRequest, ::AnnotatedHookCache, args) = nothing
FerriteOperators.reinit_values!(::AnnotatedHookCache, cell) = nothing
Ferrite.getnquadpoints(::AnnotatedHookCache) = 0

struct AnnotatedHookIntegrator <: AbstractNonlinearIntegrator end
FerriteOperators.setup_element_cache(::AnnotatedHookIntegrator, ::SubDofHandler) = AnnotatedHookCache()
FerriteOperators.get_number_of_internal_dofs_per_element(
    ::AnnotatedHookIntegrator, ::AnnotatedHookCache, sdh::SubDofHandler) = [2 for _ in sdh.cellset]

@testset "An annotated internal-dof hook gets its internal block" begin
    grid = generate_grid(Hexahedron, (2, 1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

    op  = setup_operator(strategy, AnnotatedHookIntegrator(), dh)
    ivh = op.engine.ivh
    @test ndofs(ivh) == 2 * getncells(grid)
    @test internal_variable_range(ivh, 1) == (ndofs(dh) + 1):(ndofs(dh) + 2)
    @test unknown_size(op) == ndofs(dh) + 2 * getncells(grid)
end

####################################
## Boundary cache setup validation (fused route)
####################################
# `validate_boundary_cache` is `validate_facet_item_cache`'s counterpart for
# `setup_boundary_cache` — the 0.3→0.4 migration gap: a surface cache written
# against the old positional `assemble_facet!` signature builds fine and then
# either MethodErrors mid-sweep or, if `is_facet_in_cache` never matches,
# contributes nothing.

# The old 0.3 signature — no method matches the mandatory 0.4 one.
struct OldSignatureBoundaryCache <: FerriteOperators.AbstractSurfaceElementCache end
FerriteOperators.assemble_facet!(rₑ, uₑ, cell, lfi::Int, ::OldSignatureBoundaryCache, p) = nothing
FerriteOperators.is_facet_in_cache(::FacetIndex, cell, ::OldSignatureBoundaryCache) = false

struct OldSignatureBoundaryProbe <: AbstractLinearIntegrator end
FerriteOperators.setup_element_cache(::OldSignatureBoundaryProbe, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
FerriteOperators.setup_boundary_cache(::OldSignatureBoundaryProbe, ::SubDofHandler) = OldSignatureBoundaryCache()

# The 0.4 residual kernel ported; `is_facet_in_cache` did not.
struct NoGateBoundaryCache <: FerriteOperators.AbstractSurfaceElementCache end
FerriteOperators.assemble_facet!(::ResidualRequest, ::NoGateBoundaryCache, args, lfi::Int) = nothing

struct NoGateBoundaryProbe <: AbstractLinearIntegrator end
FerriteOperators.setup_element_cache(::NoGateBoundaryProbe, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
FerriteOperators.setup_boundary_cache(::NoGateBoundaryProbe, ::SubDofHandler) = NoGateBoundaryCache()

# A `provides_analytic` claim with no facet kernel behind it — the trait ↔
# kernel check shared with `validate_facet_item_cache`.
struct OverclaimingBoundaryCache <: FerriteOperators.AbstractSurfaceElementCache end
FerriteOperators.assemble_facet!(::ResidualRequest, ::OverclaimingBoundaryCache, args, lfi::Int) = nothing
FerriteOperators.is_facet_in_cache(::FacetIndex, cell, ::OverclaimingBoundaryCache) = false
FerriteOperators.provides_analytic(::Type{OverclaimingBoundaryCache}, ::JacobianKind{:u}) = true

struct OverclaimingBoundaryProbe <: AbstractLinearIntegrator end
FerriteOperators.setup_element_cache(::OverclaimingBoundaryProbe, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
FerriteOperators.setup_boundary_cache(::OverclaimingBoundaryProbe, ::SubDofHandler) = OverclaimingBoundaryCache()

@testset "Boundary cache setup validation" begin
    grid = generate_grid(Quadrilateral, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

    # The legitimate default validates trivially.
    @test FerriteOperators.validate_boundary_cache(FerriteOperators.EmptySurfaceElementCache()) === nothing

    # A 0.3-signature cache: no method matches the mandatory 0.4 residual kernel.
    err = @test_throws ArgumentError setup_operator(strategy, OldSignatureBoundaryProbe(), dh)
    @test occursin("assemble_facet!", err.value.msg)
    @test occursin("ResidualRequest", err.value.msg)
    @test occursin("FacetArgs", err.value.msg)

    # The residual kernel ported; the gate did not.
    err = @test_throws ArgumentError setup_operator(strategy, NoGateBoundaryProbe(), dh)
    @test occursin("is_facet_in_cache", err.value.msg)
    @test occursin("FacetIndex", err.value.msg)

    # A trait claim without the facet kernel behind it.
    err = @test_throws ArgumentError setup_operator(strategy, OverclaimingBoundaryProbe(), dh)
    @test occursin("provides_analytic", err.value.msg)
    @test occursin("::Int", err.value.msg)

    # Existing boundary fixtures — the fused Neumann route — pass unchanged.
    t̄ = 1.5
    right = Set(getfacetset(grid, "right"))
    op = setup_operator(strategy, LinearNeumannProbe(t̄, :u, right), dh)
    update_operator!(op, nothing)
    @test sum(op.b) ≈ t̄ * 2.0
end
