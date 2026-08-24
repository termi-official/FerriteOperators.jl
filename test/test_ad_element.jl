using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using LinearAlgebra: mul!

include(joinpath(@__DIR__, "fixture_elements.jl"))

# A residual-only nonlinear diffusion element — no analytic Jacobian, so it is
# wrapped in `ADElementCache` automatically. Its closed-form stiffness is the
# bundled bilinear diffusion element's `D·K`, the reference this file checks
# the decorator against.
struct WrapDiffusionIntegrator <: AbstractNonlinearIntegrator
    D::Float64
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct WrapDiffusionCache{CV <: CellValues} <: AbstractVolumetricElementCache
    D::Float64
    cv::CV
end
function FerriteOperators.setup_element_cache(m::WrapDiffusionIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return WrapDiffusionCache(m.D, CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::WrapDiffusionCache) =
    WrapDiffusionCache(c.D, FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.reinit_values!(c::WrapDiffusionCache, cell) = reinit!(c.cv, cell)
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::WrapDiffusionCache, args)
    (; cv, D) = cache
    uₑ = args.states.u
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += D * (shape_gradient(cv, qp, i) ⋅ ∇u) * dΩ
        end
    end
end

@testset "Construction-time wrapping" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh   = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
    qrc  = QuadratureRuleCollection(2)
    n    = ndofs(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

    @testset "auto-wrapped Jacobian equals the analytic reference" begin
        op = setup_operator(strategy, WrapDiffusionIntegrator(2.3, qrc, :u), dh)
        @test first_element_cache(op) isa ADElementCache

        Kop = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(2.3, qrc, :u), dh)
        update_operator!(Kop, nothing)

        u = sin.(0.3 .* (1:n))
        update_linearization!(op, u, nothing)
        @test op.J ≈ Kop.A rtol = 1e-12
    end

    @testset "explicit wrap of a plain cache matches the same reference" begin
        sdh = dh.subdofhandlers[1]
        cache = FerriteOperators.setup_element_cache(WrapDiffusionIntegrator(2.3, qrc, :u), sdh)
        ad = ADElementCache(cache, sdh)   # hand-constructed, not through setup_operator
        @test provides_analytic(typeof(ad), JacobianKind{:u}())

        cc = Ferrite.CellCache(dh); reinit!(cc, 1)
        FerriteOperators.reinit_values!(ad, cc)
        ndofs_cell = ndofs_per_cell(sdh)
        uₑ = sin.(0.2 .* (1:ndofs_cell))
        args = CellArgs((u = uₑ,), cc, nothing, nothing)

        K = zeros(ndofs_cell, ndofs_cell)
        assemble_cell!(JacobianRequest{:u}(K), ad, args)

        refcache = FerriteOperators.setup_element_cache(SimpleBilinearDiffusionIntegrator(2.3, qrc, :u), sdh)
        FerriteOperators.reinit_values!(refcache, cc)
        Kref = zeros(ndofs_cell, ndofs_cell)
        assemble_cell!(JacobianRequest{:u}(Kref), refcache, args)
        @test K ≈ Kref rtol = 1e-10

        # the residual passes straight through, undifferentiated
        r = zeros(ndofs_cell)
        assemble_cell!(ResidualRequest(r), ad, args)
        rplain = zeros(ndofs_cell)
        assemble_cell!(ResidualRequest(rplain), cache, args)
        @test r == rplain
    end

    @testset "ad_backend = nothing opts out of wrapping" begin
        op = setup_operator(strategy, WrapDiffusionIntegrator(1.0, qrc, :u), dh; ad_backend = nothing)
        @test !(first_element_cache(op) isa ADElementCache)
        @test first_element_cache(op) isa WrapDiffusionCache
        u = sin.(0.3 .* (1:n))
        @test_throws MethodError update_linearization!(op, u, nothing)
    end

    @testset "composite: non-analytic inners wrap as ONE sub-composite" begin
        # Structural check on `decorate_element_cache` directly: marker caches
        # whose only declared trait is `provides_analytic`, so the wrapping
        # policy is observable without a working kernel.
        struct NeedsADMarkerA <: FerriteOperators.AbstractVolumetricElementCache end
        struct NeedsADMarkerB <: FerriteOperators.AbstractVolumetricElementCache end
        struct AlreadyAnalyticMarker <: FerriteOperators.AbstractVolumetricElementCache end
        FerriteOperators.provides_analytic(::Type{AlreadyAnalyticMarker}, kind) = true

        sdh = dh.subdofhandlers[1]
        composite = FerriteOperators.CompositeVolumetricElementCache(
            (NeedsADMarkerA(), AlreadyAnalyticMarker(), NeedsADMarkerB()))
        wrapped = decorate_element_cache(composite, sdh, ForwardDiffAD())

        @test wrapped isa FerriteOperators.CompositeVolumetricElementCache
        @test length(wrapped.inner_caches) == 2   # the analytic one bare + ONE grouped wrap
        @test any(c -> c isa AlreadyAnalyticMarker, wrapped.inner_caches)
        grouped = only(filter(c -> c isa ADElementCache, wrapped.inner_caches))
        @test grouped.inner isa FerriteOperators.CompositeVolumetricElementCache
        @test length(grouped.inner.inner_caches) == 2   # A and B, ONE seeding pass shared

        # a single non-analytic inner wraps directly, no nested sub-composite
        single = FerriteOperators.CompositeVolumetricElementCache((NeedsADMarkerA(), AlreadyAnalyticMarker()))
        wrapped_single = decorate_element_cache(single, sdh, ForwardDiffAD())
        grouped_single = only(filter(c -> c isa ADElementCache, wrapped_single.inner_caches))
        @test grouped_single.inner isa NeedsADMarkerA
    end
end

@testset "Condensed generic Consistent bootstrap (power-law) vs its analytic kernel" begin
    # A wrapper around the power-law cache that hides its analytic Jacobian
    # (forcing the decorator's generic AD+corrector-block path) and completes
    # the Tier-2 `dq/dū` block (§5.1) from the inner's own Tier-1 scalar
    # store, `dq_qp/du_j = dq_qp/du_qp · φⱼ(qp)` — the element already
    # computes `dq_qp/du_qp`; this just finishes the chain rule to nodal dofs.
    struct GenericBootstrapCache{C} <: FerriteOperators.AbstractVolumetricElementCache
        inner::C
        blocks::Vector{Matrix{Float64}}   # per-cell nq × ndofs, filled by condense_cell!
    end
    FerriteOperators.query_cell_parameters(c::GenericBootstrapCache, cell, p) =
        FerriteOperators.query_cell_parameters(c.inner, cell, p)
    FerriteOperators.reinit_values!(c::GenericBootstrapCache, cell) = FerriteOperators.reinit_values!(c.inner, cell)
    FerriteOperators.reinit_values!(c::GenericBootstrapCache, cell, kind) = FerriteOperators.reinit_values!(c.inner, cell, kind)
    Ferrite.getnquadpoints(c::GenericBootstrapCache) = getnquadpoints(c.inner)
    FerriteOperators.has_internal_state(::Type{<:GenericBootstrapCache}) = true
    FerriteOperators.get_number_of_internal_dofs_per_element(model, c::GenericBootstrapCache, sdh) =
        FerriteOperators.get_number_of_internal_dofs_per_element(model, c.inner, sdh)
    FerriteOperators.duplicate_for_device(device, c::GenericBootstrapCache) =
        GenericBootstrapCache(FerriteOperators.duplicate_for_device(device, c.inner), c.blocks)
    FerriteOperators.assemble_cell!(req::ResidualRequest, c::GenericBootstrapCache, args) =
        FerriteOperators.assemble_cell!(req, c.inner, args)
    # Deliberately NO `provides_analytic` for JacobianKind/JacobianResidualKind:
    # the whole point is to force the decorator's generic bootstrap.
    function FerriteOperators.condense_cell!(c::GenericBootstrapCache, args, weights)
        report = FerriteOperators.condense_cell!(c.inner, args, weights)
        cv  = c.inner.cv
        id  = cellid(args.cell)
        nq  = getnquadpoints(cv)
        nd  = getnbasefunctions(cv)
        dqdu = FerriteOperators.item_state(c.inner.correctors, id)
        block = c.blocks[id]
        @inbounds for qp in 1:nq, j in 1:nd
            block[qp, j] = dqdu[qp] * shape_value(cv, qp, j)
        end
        return report
    end
    FerriteOperators.condensed_corrector(c::GenericBootstrapCache, id::Int) = c.blocks[id]

    mat  = NortonRelaxationParameters()
    qrc  = QuadratureRuleCollection(2)
    integ = SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q)
    grid = generate_grid(Quadrilateral, (1, 1))
    dh   = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
    sdh  = dh.subdofhandlers[1]

    reference_cache = FerriteOperators.setup_element_cache(integ, sdh)
    nd = ndofs_per_cell(sdh)
    nq = getnquadpoints(reference_cache.cv)

    bootstrap_cache = GenericBootstrapCache(
        FerriteOperators.setup_element_cache(integ, sdh),
        [zeros(nq, nd) for _ in 1:getncells(grid)],
    )
    ad = ADElementCache(bootstrap_cache, sdh)
    @test provides_analytic(FerriteOperatorsExampleElements.SimpleCondensedPowerLawRelaxationCache, JacobianKind{:u}())  # sanity: reference IS analytic
    @test !provides_analytic(GenericBootstrapCache, JacobianKind{:u, Consistent}())   # wrapper is NOT
    @test provides_analytic(typeof(ad), JacobianKind{:u, Consistent}())               # the DECORATOR covers it generically

    cc = Ferrite.CellCache(dh); reinit!(cc, 1)
    FerriteOperators.reinit_values!(reference_cache, cc)
    FerriteOperators.reinit_values!(ad, cc)

    uₑ = 0.3 .* sin.(1:nd)
    ctx = TimeIntegrationContext(0.0, 0.5, 0.5)

    qref = zeros(nq)
    FerriteOperators.condense_cell!(reference_cache, CellArgs((u = uₑ, q = qref, qprev = zeros(nq)), cc, nothing, ctx), (u = 1.0,))
    qad = zeros(nq)
    FerriteOperators.condense_cell!(bootstrap_cache, CellArgs((u = uₑ, q = qad, qprev = zeros(nq)), cc, nothing, ctx), (u = 1.0,))
    @test qref ≈ qad

    Kref = zeros(nd, nd)
    assemble_cell!(JacobianRequest{:u, Consistent}(Kref), reference_cache, CellArgs((u = uₑ, q = qref), cc, nothing, ctx))
    Kad = zeros(nd, nd)
    assemble_cell!(JacobianRequest{:u, Consistent}(Kad), ad, CellArgs((u = uₑ, q = qad), cc, nothing, ctx))
    @test Kad ≈ Kref rtol = 1e-8

    # the fused request agrees too, and leaves the primal residual correct
    Kfused = zeros(nd, nd); rfused = zeros(nd)
    assemble_cell!(JacobianResidualRequest{Consistent}(Kfused, rfused), ad, CellArgs((u = uₑ, q = qad), cc, nothing, ctx))
    rref = zeros(nd)
    assemble_cell!(ResidualRequest(rref), reference_cache, CellArgs((u = uₑ, q = qref), cc, nothing, ctx))
    @test Kfused ≈ Kref rtol = 1e-8
    @test rfused ≈ rref rtol = 1e-12
end

# Two deliberately incomplete caches: setup validation must reject them by
# probing the AUTHOR-written method set, not the decorator's forwarding
# surface (which answers `hasmethod` for any inner).
struct NoResidualIntegrator <: AbstractNonlinearIntegrator end
struct NoResidualCache <: AbstractVolumetricElementCache end
FerriteOperators.setup_element_cache(::NoResidualIntegrator, ::SubDofHandler) = NoResidualCache()
FerriteOperators.reinit_values!(::NoResidualCache, cell) = nothing

struct NoReinitIntegrator <: AbstractNonlinearIntegrator end
struct NoReinitCache <: AbstractVolumetricElementCache end
FerriteOperators.setup_element_cache(::NoReinitIntegrator, ::SubDofHandler) = NoReinitCache()
FerriteOperators.assemble_cell!(req::ResidualRequest, ::NoReinitCache, args::CellArgs) = nothing

@testset "setup validation reaches through the decorator" begin
    grid = generate_grid(Quadrilateral, (2, 2))
    dh   = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

    @test_throws "NoResidualCache implements no `assemble_cell!(::ResidualRequest" setup_operator(
        strategy, NoResidualIntegrator(), dh)
    @test_throws "NoReinitCache implements no `reinit_values!" setup_operator(
        strategy, NoReinitIntegrator(), dh)

    # A decorated composite recurses to its leaves: the bad inner is named
    # even when it sits inside the wrapped sub-composite.
    qrc = QuadratureRuleCollection(2)
    bad_composite = NonlinearCompositeIntegrator(WrapDiffusionIntegrator(1.0, qrc, :u), NoResidualIntegrator())
    @test_throws "NoResidualCache implements no `assemble_cell!(::ResidualRequest" setup_operator(
        strategy, bad_composite, dh)

    # The unwrap must not over-reject: the residual-only cache still sets up.
    op = setup_operator(strategy, WrapDiffusionIntegrator(1.0, qrc, :u), dh)
    @test first_element_cache(op) isa FerriteOperators.ADElementCache
end
