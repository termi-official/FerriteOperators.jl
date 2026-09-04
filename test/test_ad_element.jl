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
    (; dh, n, qrc, strategy) = scalar_quad_testbed((3, 3))

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
        # The decorator SERVES the kind without the inner having a kernel for it.
        @test FerriteOperators.serves_kind(typeof(ad), JacobianKind{:u}())
        @test !provides_analytic(typeof(ad), JacobianKind{:u}())

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
    # A wrapper around the power-law cache that hides its analytic Jacobian, so
    # the decorator's generic AD+corrector-block path is what serves the
    # `Consistent` tangent. Its `condensed_corrector` is the nodal-dof block
    # `dq_qp/du_j = dq_qp/du_qp · φⱼ(qp)`, finished here from the scalar
    # per-quadrature-point slope the inner element already stores.
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
    FerriteOperators.condensed_corrector(c::GenericBootstrapCache, args) = c.blocks[cellid(args.cell)]

    (; grid, dh, qrc) = scalar_quad_testbed((1, 1))
    integ = SimpleCondensedPowerLawRelaxation(NortonRelaxationParameters(), qrc, :u, :q)
    sdh  = dh.subdofhandlers[1]

    reference_cache = FerriteOperators.setup_element_cache(integ, sdh)
    nd = ndofs_per_cell(sdh)
    nq = getnquadpoints(reference_cache.cv)

    bootstrap_cache = GenericBootstrapCache(
        FerriteOperators.setup_element_cache(integ, sdh),
        [zeros(nq, nd) for _ in 1:getncells(grid)],
    )
    # One internal dof per quadrature point here, which the decorator still has
    # to be TOLD: `setup_operator` resolves it from the count hook.
    ad = ADElementCache(bootstrap_cache, sdh; n_internal_dofs = nq)
    @test !provides_analytic(GenericBootstrapCache, JacobianKind{:u, Consistent}())   # wrapper is NOT
    @test FerriteOperators.serves_kind(typeof(ad), JacobianKind{:u, Consistent}())    # the DECORATOR covers it generically
    @test !provides_analytic(typeof(ad), JacobianKind{:u, Consistent}())              # and does not claim a kernel for it

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

####################################
## The same bootstrap on a MULTI-dof-per-quadrature-point internal variable
####################################
# The viscoelastic element condenses a symmetric viscous strain, SIX internal
# dofs per quadrature point rather than one, so the `:q` seeds only fit when
# they are sized from the element's declared internal-dof count. The wrapper
# hides the analytic Jacobian exactly as `GenericBootstrapCache` does above;
# its `condensed_corrector` is the local solve's tensor slope
# `dεᵛ/dε = (𝕀/γ̃ + E₁/η₁ ℂ)⁻¹ : (E₁/η₁ ℂ)` contracted with each shape
# function's strain, laid out in the `[6 × nqp]` order the element's `q` is
# reshaped into.
struct SLSBootstrapIntegrator{I} <: AbstractCondensedNonlinearIntegrator
    inner::I
    ncells::Int
end
struct SLSBootstrapCache{C} <: FerriteOperators.AbstractVolumetricElementCache
    inner::C
    blocks::Vector{Matrix{Float64}}   # per-cell 6nqp × ndofs, filled by condense_cell!
end
function FerriteOperators.setup_element_cache(m::SLSBootstrapIntegrator, sdh::SubDofHandler)
    inner = FerriteOperators.setup_element_cache(m.inner, sdh)
    nq    = getnquadpoints(inner.cv)
    return SLSBootstrapCache(inner, [zeros(6nq, ndofs_per_cell(sdh)) for _ in 1:m.ncells])
end
FerriteOperators.reinit_values!(c::SLSBootstrapCache, cell) = FerriteOperators.reinit_values!(c.inner, cell)
FerriteOperators.reinit_values!(c::SLSBootstrapCache, cell, kind) = FerriteOperators.reinit_values!(c.inner, cell, kind)
Ferrite.getnquadpoints(c::SLSBootstrapCache) = getnquadpoints(c.inner)
FerriteOperators.has_internal_state(::Type{<:SLSBootstrapCache}) = true
FerriteOperators.get_number_of_internal_dofs_per_element(model, c::SLSBootstrapCache, sdh) =
    FerriteOperators.get_number_of_internal_dofs_per_element(model, c.inner, sdh)
# `blocks` is item-keyed, so workers share it; only the values object is copied.
FerriteOperators.duplicate_for_device(device, c::SLSBootstrapCache) =
    SLSBootstrapCache(FerriteOperators.duplicate_for_device(device, c.inner), c.blocks)
# Deliberately NO `provides_analytic`: the generic bootstrap is the whole point.
FerriteOperators.assemble_cell!(req::ResidualRequest, c::SLSBootstrapCache, args) =
    FerriteOperators.assemble_cell!(req, c.inner, args)
function FerriteOperators.condense_cell!(c::SLSBootstrapCache, args, weights)
    report = FerriteOperators.condense_cell!(c.inner, args, weights)
    (; E₁, η₁, ν) = c.inner.material_parameters
    cv    = c.inner.cv
    γ̃     = stage_scaling(args.ctx)
    block = c.blocks[cellid(args.cell)]
    for qp in 1:getnquadpoints(cv)
        ε    = symmetric(function_gradient(cv, qp, args.states.u))
        ℂ    = FerriteOperatorsExampleElements._sls_unit_stiffness(ε, ν)
        B    = (E₁ / η₁) * ℂ
        dqdε = inv(one(ℂ) / γ̃ + B) ⊡ B
        for j in 1:getnbasefunctions(cv)
            block[(6(qp - 1) + 1):(6qp), j] .= (dqdε ⊡ symmetric(shape_gradient(cv, qp, j))).data
        end
    end
    return report
end
FerriteOperators.condensed_corrector(c::SLSBootstrapCache, args) = c.blocks[cellid(args.cell)]

@testset "Condensed generic Consistent (six internal dofs per QP) vs its analytic kernel" begin
    strategy = AssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)
    (; op, dh, grid) = visco_testbed(strategy, qrc, (2, 1, 1))
    integ = SimpleCondensedLinearViscoelasticity(MaxwellParameters(), qrc, :u, :εᵛ)

    sdh = dh.subdofhandlers[1]
    nq  = getnquadpoints(FerriteOperators.setup_element_cache(integ, sdh).cv)
    gop = setup_operator(strategy, SLSBootstrapIntegrator(integ, getncells(grid)), dh;
                         slots = (:u, :q, :qprev))

    # The `:q` buffers follow the DECLARED count, six per quadrature point —
    # not the quadrature-point count the sizing used to assume.
    @test length(first_element_cache(gop).buffers.Lₑ) == 6nq
    @test FerriteOperators.serves_kind(typeof(first_element_cache(gop)), JacobianKind{:u, Consistent}())
    @test !provides_analytic(typeof(first_element_cache(gop)), JacobianKind{:u, Consistent}())

    u     = zeros(unknown_size(op)); u[1:ndofs(dh)] .= 0.02 .* sin.(1:ndofs(dh))
    uprev = zeros(unknown_size(op))
    ctx   = TimeIntegrationContext(0.0, 0.5, 0.5)

    uref, ugen = copy(u), copy(u)
    condense_internal!(op,  condensed_states(uref, uprev), nothing, ctx)
    condense_internal!(gop, condensed_states(ugen, uprev), nothing, ctx)
    @test uref ≈ ugen                                   # the same condensed εᵛ

    rref = zeros(residual_size(op));  update_linearization!(op,  rref, condensed_states(uref, uprev), nothing, ctx)
    rgen = zeros(residual_size(gop)); update_linearization!(gop, rgen, condensed_states(ugen, uprev), nothing, ctx)
    @test rgen ≈ rref rtol = 1e-12
    @test Matrix(gop.J) ≈ Matrix(op.J) rtol = 1e-7      # generic combination == analytic tangent

    # A decorator built without the declaration has no `:q` configuration and
    # refuses by name, rather than seeding a mis-sized sweep.
    cc = Ferrite.CellCache(dh); reinit!(cc, 1)
    undeclared = ADElementCache(FerriteOperators.setup_element_cache(
        SLSBootstrapIntegrator(integ, getncells(grid)), sdh), sdh)
    FerriteOperators.reinit_values!(undeclared, cc)
    nd = ndofs_per_cell(sdh)
    @test_throws "no ForwardDiff configuration for the `:q`" assemble_cell!(
        JacobianRequest{:u, Consistent}(zeros(nd, nd)), undeclared,
        CellArgs((u = zeros(nd), q = zeros(6nq)), cc, nothing, ctx))
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
    (; dh, qrc, strategy) = scalar_quad_testbed((2, 2))

    @test_throws "NoResidualCache implements no `assemble_cell!(::ResidualRequest" setup_operator(
        strategy, NoResidualIntegrator(), dh)
    @test_throws "NoReinitCache implements no `reinit_values!" setup_operator(
        strategy, NoReinitIntegrator(), dh)

    # A decorated composite recurses to its leaves: the bad inner is named
    # even when it sits inside the wrapped sub-composite.
    bad_composite = NonlinearCompositeIntegrator(WrapDiffusionIntegrator(1.0, qrc, :u), NoResidualIntegrator())
    @test_throws "NoResidualCache implements no `assemble_cell!(::ResidualRequest" setup_operator(
        strategy, bad_composite, dh)

    # The unwrap must not over-reject: the residual-only cache still sets up.
    op = setup_operator(strategy, WrapDiffusionIntegrator(1.0, qrc, :u), dh)
    @test first_element_cache(op) isa FerriteOperators.ADElementCache
end
