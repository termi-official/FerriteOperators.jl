using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "fixture_elements.jl"))

# Transient diffusion, r(u, u̇) = ∫ (u̇ v + ∇u⋅∇v) dΩ — ∂F/∂u is the stiffness
# and ∂F/∂du the mass matrix, so every weighted combination is known in closed
# form. `fused` selects the flavour whose cache serves the weighted Jacobian
# analytically; both flavours share the residual.
const DeclaredDiffusionCache = CVCache{:declared}
const FusedDiffusionCache    = CVCache{:declared_fused}
const AnyDiffusionCache      = Union{DeclaredDiffusionCache, FusedDiffusionCache}
DeclaredDiffusionIntegrator(qrc, field_name, fused = false) =
    fused ? CVIntegrator{:declared_fused}(qrc, field_name) : CVIntegrator{:declared}(qrc, field_name)

function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::AnyDiffusionCache, args)
    transient_diffusion_residual!(req.r, cache, args)
end
# The hand-fused SDIRK/BE scheme matrix, reading its scalars from the request.
const FUSED_W_SWEEPS = Ref(0)
FerriteOperators.provides_analytic(::Type{<:FusedDiffusionCache}, ::WeightedJacobianKind) = true
function FerriteOperators.assemble_cell!(req::WeightedJacobianRequest, cache::FusedDiffusionCache, args)
    FUSED_W_SWEEPS[] += 1
    analytic_weighted_jacobian!(req.K, cache.cv, req.weights)
end
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:mass}, cache::AnyDiffusionCache, args) =
    sum(qp -> getdetJdV(cache.cv, qp), 1:getnquadpoints(cache.cv))

# The worked SDIRK-W declaration: two slots, the weighted Jacobian the scheme
# solves with, and the residual. No coefficients — γ and Δt ride with the
# evaluation, not with the declaration.
function declared_testbed(; fused = false, slots = (:u, :du),
        requests = (WeightedJacobianKind, ResidualKind))
    (; grid, dh, qrc, strategy) = scalar_quad_testbed((3, 2))
    op  = setup_operator(strategy, DeclaredDiffusionIntegrator(qrc, :u, fused), dh; slots, requests)
    Mop = setup_operator(strategy, SimpleBilinearMassIntegrator(1.0, qrc, :u), dh)
    Kop = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
    update_operator!(Mop, nothing)
    update_operator!(Kop, nothing)
    return (; op, Mop, Kop, dh, grid, qrc, strategy, n = ndofs(dh))
end

@testset "Setup-time declarations" begin
    Δt  = 0.25
    γ   = 0.5
    ctx = TimeIntegrationContext(1.0, Δt, Δt)
    weights = (u = 1.0, du = 1 / (γ * Δt))

    # Declaring only declares; the weighted-Jacobian VALUES on both routes are
    # pinned against the bundled bilinear integrators in test_stage_block.jl.
    @testset "SDIRK-W witness: the declarations reach the engine" begin
        tb = declared_testbed(; fused = true)
        @test FerriteOperators._declared_slots(tb.op.engine) == (:u, :du)

        u  = sin.(0.3 .* (1:tb.n)); du = cos.(0.2 .* (1:tb.n))
        states = (u = u, du = du)

        FUSED_W_SWEEPS[] = 0
        W = share_pattern(tb.op.J)
        assemble_weighted_jacobian!(W, tb.op, weights, states, nothing, ctx)
        @test FUSED_W_SWEEPS[] == getncells(tb.grid)     # one kernel call per cell, one sweep

        # the residual the same operator declares still runs
        r = zeros(tb.n)
        evaluate!(tb.op, r, states, nothing, ctx)
        @test r ≈ tb.Mop.A * du .+ tb.Kop.A * u rtol = 1e-12
    end

    @testset "the same declarations over a non-analytic cache agree" begin
        tb = declared_testbed(; fused = false)
        u  = sin.(0.3 .* (1:tb.n)); du = cos.(0.2 .* (1:tb.n))
        states = (u = u, du = du)

        FUSED_W_SWEEPS[] = 0
        W = share_pattern(tb.op.J)
        assemble_weighted_jacobian!(W, tb.op, weights, states, nothing, ctx)
        @test FUSED_W_SWEEPS[] == 0                       # the analytic kernel never ran

        Wc = share_pattern(tb.op.J)
        FerriteOperators._weighted_jacobian_composed!(
            Wc, tb.op, FerriteOperators.WeightedJacobianKind(weights), states, nothing, ctx)
        @test Wc.nzval ≈ W.nzval rtol = 1e-12
    end

    @testset "sensitivity buffers are built structurally, by integrator family" begin
        tb = declared_testbed()
        qrc = QuadratureRuleCollection(2)

        # bilinear and linear operators never carry sensitivity machinery —
        # decided by `needs_ad_decoration(integrator)`, not by declarations
        @test !carries_sensitivity_buffers(tb.Mop)
        lop = setup_operator(tb.strategy, SimpleLinearIntegrator(1.0, qrc, :u), tb.dh)
        @test !carries_sensitivity_buffers(lop)

        # the nonlinear family always carries it, declared or not
        @test carries_sensitivity_buffers(tb.op)
        @test !FerriteOperators.needs_ad_decoration(tb.Mop.integrator)
        @test FerriteOperators.needs_ad_decoration(tb.op.integrator)
    end

    @testset "workspaces are immutable" begin
        tb = declared_testbed()
        @test !ismutable(first_workspace(tb.op))
        @test !ismutable(first_workspace(tb.Mop))
    end

    @testset "undeclared kinds stay usable" begin
        tb = declared_testbed()
        u = sin.(0.3 .* (1:tb.n)); du = cos.(0.2 .* (1:tb.n))
        states = (u = u, du = du)

        # the testbed declares neither the state JVP nor a functional
        Jv = zeros(tb.n); v = cos.(0.11 .* (1:tb.n))
        state_jvp!(Jv, tb.op, v, states, nothing, ctx)
        @test Jv ≈ tb.Kop.A * v rtol = 1e-10

        # a functional sweep reads no state, so an undeclared one just runs
        area = evaluate_functional(tb.op, FunctionalKind(:mass), states, nothing, ctx)
        @test area ≈ 4.0 rtol = 1e-12          # the [-1,1]² reference grid

        # …and declaring it builds nothing: the declaring operator answers the
        # same, and a bilinear one still carries no sensitivity buffers at all.
        fop = setup_operator(tb.strategy, DeclaredDiffusionIntegrator(tb.qrc, :u), tb.dh;
                             slots = (:u, :du), requests = (FunctionalKind,))
        @test evaluate_functional(fop, FunctionalKind(:mass), states, nothing, ctx) == area
        bfop = setup_operator(tb.strategy, SimpleBilinearMassIntegrator(1.0, tb.qrc, :u), tb.dh;
                              requests = (FunctionalKind,))
        @test !carries_sensitivity_buffers(bfop)
    end

    @testset "two operators from one declaration evaluate concurrently" begin
        (; dh, n, qrc, strategy) = scalar_quad_testbed((6, 5))
        integrator = DeclaredDiffusionIntegrator(qrc, :u)
        declarations = (slots = (:u, :du), requests = (WeightedJacobianKind, ResidualKind))

        op1 = setup_operator(strategy, integrator, dh; declarations...)
        op2 = setup_operator(strategy, integrator, dh; declarations...)

        # no mutable state is shared between the two caches
        @test op1.J !== op2.J
        @test first_workspace(op1) !== first_workspace(op2)

        s1 = (u = sin.(0.3 .* (1:n)), du = cos.(0.2 .* (1:n)))
        s2 = (u = cos.(0.7 .* (1:n)), du = sin.(0.5 .* (1:n)))
        r1 = zeros(n); r2 = zeros(n)
        update_linearization!(op1, r1, s1, nothing, ctx)
        update_linearization!(op2, r2, s2, nothing, ctx)
        seq = (copy(r1), copy(op1.J), copy(r2), copy(op2.J))

        fill!(r1, 0.0); fill!(r2, 0.0)
        t1 = Threads.@spawn update_linearization!(op1, r1, s1, nothing, ctx)
        t2 = Threads.@spawn update_linearization!(op2, r2, s2, nothing, ctx)
        wait(t1); wait(t2)
        @test r1 == seq[1]
        @test op1.J == seq[2]
        @test r2 == seq[3]
        @test op2.J == seq[4]
    end
end

####################################
## Declaration-hook signature drift
####################################
# `global_dofs`, `facet_items`, `facet_item_global_dofs` and `algebraic_items`
# all default to an EMPTY declaration, and `item_families` to the families those
# imply, so a method written against a signature the engine does not call is
# never reached and the operator assembles a silent subset. `DriftProbe{hook}`
# wears one drifted method for `hook`; `:none` wears none, `:correct` wears all
# five at the engine's own signature.

struct DriftProbe{hook} <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
end
FerriteOperators.setup_element_cache(m::DriftProbe, sdh::SubDofHandler) =
    FerriteOperators.setup_element_cache(CVIntegrator{:declared}(m.qrc, :u), sdh)

# The right name on the right integrator, the argument the engine does not pass.
# `SubDofHandler <: AbstractDofHandler`, so a drifted second argument has to name
# the concrete `DofHandler` to actually miss the per-subdomain call.
FerriteOperators.global_dofs(::DriftProbe{:global_dofs}, ::DofHandler) = (1,)
FerriteOperators.facet_items(::DriftProbe{:facet_items}, ::DofHandler) = (FacetIndex(1, 1),)
FerriteOperators.facet_item_global_dofs(::DriftProbe{:facet_item_global_dofs}, ::DofHandler) = (1,)
FerriteOperators.algebraic_items(::DriftProbe{:algebraic_items}, ::SubDofHandler) = ([1],)
FerriteOperators.item_families(::DriftProbe{:item_families}, ::SubDofHandler) = (CellFamily(),)
# A drifted ARITY misses the call the same way a drifted argument type does.
FerriteOperators.global_dofs(::DriftProbe{:arity}, ::SubDofHandler, ::Int) = (1,)

# A specialized method the engine's call resolves to passes, whatever it returns.
FerriteOperators.global_dofs(::DriftProbe{:correct}, ::SubDofHandler) = ()
FerriteOperators.facet_items(::DriftProbe{:correct}, ::SubDofHandler) = ()
FerriteOperators.facet_item_global_dofs(::DriftProbe{:correct}, ::SubDofHandler) = ()
FerriteOperators.algebraic_items(::DriftProbe{:correct}, ::DofHandler) = ()
FerriteOperators.item_families(::DriftProbe{:correct}, ::DofHandler) = (CellFamily(),)

@testset "Declaration-hook signature drift" begin
    (; dh, qrc, strategy) = scalar_quad_testbed((3, 2))
    probe(hook) = DriftProbe{hook}(qrc)
    check(integrator) = FerriteOperators.assert_declaration_signatures(integrator, dh)

    @testset "a drifted signature is rejected, naming hook and both signatures" begin
        for (hook, expected, drifted) in ((:global_dofs, "SubDofHandler", "DofHandler"),
                                          (:facet_items, "SubDofHandler", "DofHandler"),
                                          (:facet_item_global_dofs, "SubDofHandler", "DofHandler"),
                                          (:algebraic_items, "DofHandler", "SubDofHandler"),
                                          (:item_families, "DofHandler", "SubDofHandler"))
            err = @test_throws ArgumentError check(probe(hook))
            @test occursin("expected: $hook(::DriftProbe, ::$expected)", err.value.msg)
            @test occursin("DriftProbe{:$hook}, ::Ferrite.$drifted)", err.value.msg)
        end
        err = @test_throws ArgumentError check(probe(:arity))
        @test occursin("expected: global_dofs(::DriftProbe, ::SubDofHandler)", err.value.msg)
        @test occursin("DriftProbe{:arity}, ::Ferrite.SubDofHandler, ::Int64)", err.value.msg)
    end

    @testset "the rejection is a setup error, not a call-time one" begin
        err = @test_throws ArgumentError setup_operator(
            strategy, probe(:facet_items), dh; slots = (:u, :du))
        @test occursin("facet_items", err.value.msg)
        # The same element without the drifted method builds.
        op = setup_operator(strategy, probe(:none), dh; slots = (:u, :du))
        @test op isa FerriteOperators.LinearizedFerriteOperator
    end

    @testset "correct declarers and non-declarers pass" begin
        @test check(probe(:correct)) === nothing
        @test check(probe(:none)) === nothing
        @test check(DeclaredDiffusionIntegrator(qrc, :u)) === nothing
    end

    @testset "wrappers are probed through, not around" begin
        plain, drifted, correct = probe(:none), probe(:facet_items), probe(:correct)

        # Both wrappers forward `facet_items` at the engine's own signature, so
        # the wrapper's own method resolves and only the recursion into the
        # inners can see the drift standing behind it.
        routed(subs...) = NonlinearMultiDomainIntegrator(
            Dict{String, AbstractNonlinearIntegrator}(string(i) => sub for (i, sub) in enumerate(subs)))
        for wrapped in (NonlinearCompositeIntegrator(plain, drifted),
                        routed(plain, drifted),
                        routed(NonlinearCompositeIntegrator(plain, drifted)))
            err = @test_throws ArgumentError check(wrapped)
            @test occursin("DriftProbe{:facet_items} has a method", err.value.msg)
        end

        # …and the wrappers' own forwarding methods are not themselves drift.
        @test check(NonlinearCompositeIntegrator(plain, correct)) === nothing
        @test check(routed(plain, correct)) === nothing
        @test check(routed(NonlinearCompositeIntegrator(plain, correct))) === nothing
    end
end

####################################
## Item-family registration
####################################
# `item_families` decides WHICH families an engine carries and in what ORDER;
# `setup_family_caches` is the one dispatch that builds any of them. The subject
# here is registration and traversal order, not what the families assemble, so
# `FamilyProbe` declares all three with no-op kernels throughout.
# `registration` selects what the integrator declares: `:default` leaves the
# derived tuple alone, `:nocells` drops the cell family, `:reordered` puts the
# algebraic family first.

struct NoopAlgebraicItemCache end
FerriteOperators.assemble_algebraic!(::AbstractAssemblyRequest, ::NoopAlgebraicItemCache, args) = nothing
FerriteOperators.duplicate_for_device(device, c::NoopAlgebraicItemCache) = c

struct FamilyProbe{registration} <: AbstractBilinearIntegrator
    facetset::Set{FacetIndex}
    items::Vector{Vector{Int}}
end
FerriteOperators.setup_element_cache(::FamilyProbe, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
FerriteOperators.facet_items(m::FamilyProbe, ::SubDofHandler) = m.facetset
FerriteOperators.setup_facet_item_cache(::FamilyProbe, ::SubDofHandler) =
    FerriteOperators.EmptySurfaceElementCache()
FerriteOperators.algebraic_items(m::FamilyProbe, ::DofHandler) = m.items
FerriteOperators.setup_algebraic_cache(::FamilyProbe, ::DofHandler) = NoopAlgebraicItemCache()

# The registration itself, written exactly as a downstream integrator writes it.
FerriteOperators.item_families(::FamilyProbe{:nocells}, dh) = (FacetItemFamily(), AlgebraicItemFamily())
FerriteOperators.item_families(::FamilyProbe{:reordered}, dh) =
    (AlgebraicItemFamily(), FacetItemFamily(), CellFamily())

# The ITERATION seams are keyed on the element CACHE and not on the integrator,
# so their drift check has its own subject and runs once the caches exist.
struct IterationDriftCache{hook} <: FerriteOperators.AbstractVolumetricElementCache end
FerriteOperators.assemble_cell!(::AbstractAssemblyRequest, ::IterationDriftCache, args) = nothing
FerriteOperators.reinit_values!(::IterationDriftCache, cell) = nothing

# The right name on the right cache, against a KIND the engine does not pass for
# a `FullAssembly` operator — so the default answers and the sweep silently
# walks cells.
FerriteOperators.assembly_iterator(::MatrixFreeActionKind, ::IterationDriftCache{:assembly_iterator}, sdh) =
    Ferrite.CellCache(sdh)
# A drifted ARITY misses the call the same way.
FerriteOperators.item_provider(kind, ::IterationDriftCache{:item_provider}, sdh, ::Int) = CellItems(sdh)
# The engine's own signature: resolved, therefore no drift.
FerriteOperators.assembly_iterator(kind, ::IterationDriftCache{:correct}, sdh) = Ferrite.CellCache(sdh)
FerriteOperators.item_provider(kind, ::IterationDriftCache{:correct}, sdh) = CellItems(sdh)

struct IterationDriftIntegrator{hook} <: AbstractBilinearIntegrator end
FerriteOperators.setup_element_cache(::IterationDriftIntegrator{hook}, ::SubDofHandler) where {hook} =
    IterationDriftCache{hook}()

@testset "Item-family registration" begin
    (; dh, grid, qrc, strategy) = scalar_quad_testbed((3, 2))
    facetset = getfacetset(grid, "left")
    items    = [[1, 2], [3, 4]]
    probe(reg) = FamilyProbe{reg}(Set(facetset), items)
    domains(op) = [typeof(sc.domain).name.wrapper for sc in op.engine.subdomain_caches]

    @testset "the default declaration is what the other hooks imply" begin
        # A cells-only integrator registers the cell family alone, which is
        # what keeps its `subdomain_caches` element type concrete.
        plain = DeclaredDiffusionIntegrator(qrc, :u)
        @test item_families(plain, dh) === (CellFamily(),)
        @test isconcretetype(eltype(
            setup_operator(strategy, plain, dh; slots = (:u, :du)).engine.subdomain_caches))
        # Declaring facet items adds the second marker, algebraic items the third.
        @test item_families(probe(:default), dh) ===
            (CellFamily(), FacetItemFamily(), AlgebraicItemFamily())
        @test item_families(FamilyProbe{:default}(Set(facetset), Vector{Int}[]), dh) ===
            (CellFamily(), FacetItemFamily())
        @test item_families(FamilyProbe{:default}(Set{FacetIndex}(), items), dh) ===
            (CellFamily(), AlgebraicItemFamily())
    end

    @testset "all three families at once, in declaration order" begin
        op = setup_operator(strategy, probe(:default), dh)
        # Cells (one per subdomain), then the facet items, then the algebraic
        # items — the order a reduction's determinism rests on.
        @test domains(op) == vcat(
            fill(FerriteOperators.AssemblyDomain, length(dh.subdofhandlers)),
            [FerriteOperators.FacetItemDomain, FerriteOperators.AlgebraicDomain])
        update_operator!(op, nothing)   # the traversal runs, all three families
    end

    @testset "the traversal is the declaration's, with no privileged path" begin
        # Dropping the cell family really drops it: were the cell setup still
        # hand-appended, this operator would carry cell caches nobody registered.
        nocells = setup_operator(strategy, probe(:nocells), dh)
        @test domains(nocells) == [FerriteOperators.FacetItemDomain, FerriteOperators.AlgebraicDomain]
        # And the order is the tuple's, not a rule inside the engine.
        reordered = setup_operator(strategy, probe(:reordered), dh)
        @test domains(reordered) == vcat(
            [FerriteOperators.AlgebraicDomain, FerriteOperators.FacetItemDomain],
            fill(FerriteOperators.AssemblyDomain, length(dh.subdofhandlers)))
    end

    @testset "the shipped families answer the same dispatch as a downstream one" begin
        for family in (CellFamily(), FacetItemFamily(), AlgebraicItemFamily())
            @test hasmethod(setup_family_caches, Tuple{typeof(family), Any, Any, Any, Any})
        end
        # …and nothing else builds subdomain caches: the three names the engine
        # used to call one by one are gone.
        for gone in (:setup_subdomain_caches, :setup_facet_item_caches, :setup_algebraic_caches)
            @test !isdefined(FerriteOperators, gone)
        end
    end

    @testset "a drifted iteration seam is rejected at setup" begin
        check(cache) = FerriteOperators.assert_iteration_signatures(nothing, [cache], dh)
        for (hook, expected, drifted) in (
                (:assembly_iterator, "assembly_iterator(::Nothing, ::IterationDriftCache, ::SubDofHandler)",
                 "MatrixFreeActionKind"),
                (:item_provider, "item_provider(::Nothing, ::IterationDriftCache, ::SubDofHandler)",
                 "::Int64"))
            err = @test_throws ArgumentError check(IterationDriftCache{hook}())
            @test occursin("expected: $expected", err.value.msg)
            @test occursin(drifted, err.value.msg)
            @test occursin("iteration seam `$hook`", err.value.msg)
        end
        # A declaration the engine's call resolves to passes, and so does a
        # cache that declares neither seam.
        @test check(IterationDriftCache{:correct}()) === nothing
        @test check(IterationDriftCache{:none}()) === nothing
        @test check(FerriteOperators.EmptyVolumetricElementCache()) === nothing
        # The rejection is a setup error, not a call-time one.
        @test_throws ArgumentError setup_operator(strategy, IterationDriftIntegrator{:item_provider}(), dh)
        @test setup_operator(strategy, IterationDriftIntegrator{:correct}(), dh) isa
            FerriteOperators.BilinearFerriteOperator
    end
end

####################################
## Decorator forwarding of author declarations
####################################
# `AbstractElementCacheDecorator` forwards the seams that declare what the
# ELEMENT itself is/does — `assembly_iterator`, `device_assembly_iterator`,
# `item_provider`, `item_update_flags`, `element_local_length`,
# `element_action_row`, `element_matrix_symmetry` — so a cache wrapped in
# `ADElementCache`/`ElementAssemblyCache` keeps its own declarations instead of
# silently losing them to the cell default.
# `assert_iteration_signatures`'s drift probe runs on the `unwrap` fixpoint for
# the same reason `_assert_trait_backed` does — see `IterationDriftCache` above.

# A `CellCache` wrapper distinct in TYPE from the engine's own default, so a
# hook resolving to it proves the CUSTOM declaration ran and not the open one.
struct TaggedCellCache{C}
    cc::C
end
TaggedCellCache(sdh::SubDofHandler) = TaggedCellCache(Ferrite.CellCache(sdh))
Ferrite.reinit!(t::TaggedCellCache, item::Int) = (Ferrite.reinit!(t.cc, item); t)
Ferrite.cellid(t::TaggedCellCache) = Ferrite.cellid(t.cc)
FerriteOperators.iterator_dofs(t::TaggedCellCache) = FerriteOperators.iterator_dofs(t.cc)
FerriteOperators.iterator_handler(t::TaggedCellCache) = FerriteOperators.iterator_handler(t.cc)

# A distinct provider type, so `item_provider`'s forwarding is checked the same
# way; `compute_partition` delegates so a real sweep still partitions.
struct TaggedItems{P}
    provider::P
end
FerriteOperators.compute_partition(s::FerriteOperators.AssemblyStrategy, p::TaggedItems) =
    FerriteOperators.compute_partition(s, p.provider)

# A sentinel the DEFAULT `device_assembly_iterator` never answers with — it
# falls through to `assembly_iterator` (a `TaggedCellCache`) instead — so
# resolving to THIS proves the decorator forwards `device_assembly_iterator`
# itself and not just the seam it happens to default through.
struct DeviceSentinel end

# The declaring inner: residual-only, so `setup_operator` auto-wraps it in
# `ADElementCache`, plus every forwarded seam, each answering something the cell
# default never would.
struct DeclaringIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct DeclaringCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::DeclaringIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return DeclaringCache(CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::DeclaringCache) =
    DeclaringCache(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.reinit_values!(c::DeclaringCache, cell::TaggedCellCache) = reinit!(c.cv, cell.cc)
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::DeclaringCache, args)
    (; cv) = cache
    uₑ = args.states.u
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u) * dΩ
        end
    end
end

FerriteOperators.assembly_iterator(kind, ::DeclaringCache, sdh) = TaggedCellCache(sdh)
FerriteOperators.device_assembly_iterator(kind, ::DeclaringCache, sdh, device_sdh) = DeviceSentinel()
FerriteOperators.item_provider(kind, ::DeclaringCache, sdh) = TaggedItems(CellItems(sdh))
FerriteOperators.item_update_flags(kind, ::DeclaringCache) =
    Ferrite.UpdateFlags(nodes = false, coords = false, dofs = true)
FerriteOperators.element_local_length(::DeclaringCache) = Val(7)
FerriteOperators.element_action_row(::DeclaringCache, uₑ, args::CellArgs, i::Int) = 99.0 + i

@testset "Decorator forwarding of author declarations" begin
    (; dh, qrc, strategy) = scalar_quad_testbed((3, 2))
    sdh = dh.subdofhandlers[1]

    @testset "a wrapped cache's declarations reach setup, and the resolved seams are the inner's" begin
        op = setup_operator(strategy, DeclaringIntegrator(qrc, :u), dh)
        wrapped = first_element_cache(op)
        @test wrapped isa ADElementCache
        inner = FerriteOperators.unwrap(wrapped)
        @test inner isa DeclaringCache

        # Reached through SETUP itself: `setup_family_caches` partitions over
        # `item_provider(kind, wrapped, sdh)` and positions the workspace on
        # `assembly_iterator(kind, wrapped, sdh)`, then `validate_element_cache`
        # probes `reinit_values!` against that resolved iterator type —
        # `DeclaringCache` implements it only for `TaggedCellCache`, so a broken
        # forward would have thrown before `setup_operator` returned.
        @test assembly_iterator(nothing, wrapped, sdh) isa TaggedCellCache
        @test item_provider(nothing, wrapped, sdh) isa TaggedItems
        @test device_assembly_iterator(nothing, wrapped, sdh, sdh) isa DeviceSentinel
        @test item_update_flags(nothing, wrapped) == item_update_flags(nothing, inner)
        @test item_update_flags(nothing, wrapped) == Ferrite.UpdateFlags(nodes = false, coords = false, dofs = true)
        @test FerriteOperators.element_local_length(wrapped) == Val(7)

        n = ndofs_per_cell(sdh)
        cc = Ferrite.CellCache(sdh); reinit!(cc, 1)
        args = CellArgs((u = zeros(n),), cc, nothing, nothing)
        @test element_action_row(wrapped, zeros(n), args, 2) == element_action_row(inner, zeros(n), args, 2)
    end

    @testset "a decorator's own explicit method wins over the forwarded default" begin
        bilinear = FerriteOperators.setup_element_cache(SimpleBilinearDiffusionIntegrator(2.0, qrc, :u), sdh)
        eac      = with_action_storage(bilinear, ElementAssembly(), sdh)
        @test eac isa ElementAssemblyCache

        # `SimpleBilinearDiffusionElementCache` declares neither iteration seam,
        # so both still resolve to the plain cell default THROUGH the decorator:
        # the forwarding changes nothing for a cache that declares nothing.
        @test assembly_iterator(nothing, eac, sdh) isa Ferrite.CellCache
        @test item_provider(nothing, eac, sdh) isa CellItems

        # `ElementAssemblyCache` OWNS extent/row-action/matrix-free flags —
        # its own explicit methods answer, not the (undeclared) forwarded
        # default.
        @test FerriteOperators.element_local_length(bilinear) === nothing
        @test FerriteOperators.element_local_length(eac) == eac.local_size
        @test item_update_flags(MatrixFreeActionKind(), eac) ==
            Ferrite.UpdateFlags(nodes = false, coords = false, dofs = true)

        # Further wrapped in `ADElementCache`, the SAME override still wins —
        # the outer decorator's forwarding lands on `ElementAssemblyCache`'s
        # own method by ordinary dispatch, never on `eac.inner`'s default —
        # and the seams `ElementAssemblyCache` does NOT override still forward
        # straight through both layers.
        ad = ADElementCache(eac, sdh)
        @test FerriteOperators.element_local_length(ad) == eac.local_size
        @test item_update_flags(MatrixFreeActionKind(), ad) == item_update_flags(MatrixFreeActionKind(), eac)
        @test assembly_iterator(nothing, ad, sdh) isa Ferrite.CellCache
    end

    @testset "the drift probe sees a declaration the decorated path would otherwise mask" begin
        check(cache) = FerriteOperators.assert_iteration_signatures(nothing, [cache], dh)
        wrapped_drift   = ADElementCache(IterationDriftCache{:assembly_iterator}(), sdh)
        wrapped_correct = ADElementCache(IterationDriftCache{:correct}(), sdh)
        wrapped_none    = ADElementCache(IterationDriftCache{:none}(), sdh)

        err = @test_throws ArgumentError check(wrapped_drift)
        @test occursin("iteration seam `assembly_iterator`", err.value.msg)
        @test occursin("IterationDriftCache{:assembly_iterator}", err.value.msg)
        @test check(wrapped_correct) === nothing
        @test check(wrapped_none) === nothing
    end
end

####################################
## Downstream-style custom kinds
####################################
# Everything below is what a downstream package writes: kind + request +
# request_type/materialize_request + traits + execute_kind! — all from outside
# the package.

# 1. A matrix-assembly kind riding the built-in primal driver body. It scales
#    the stiffness by a factor carried on the REQUEST, so the assembled result
#    has a closed form against the plain stiffness operator.
struct ScaledStiffnessKind end
struct ScaledStiffnessRequest{M <: AbstractMatrix} <: FerriteOperators.AbstractAssemblyRequest
    K::M
    scale::Float64
end
FerriteOperators.request_type(::ScaledStiffnessKind) = ScaledStiffnessRequest
FerriteOperators.materialize_request(::ScaledStiffnessKind, ws) = ScaledStiffnessRequest(ws.Ke, 2.5)
FerriteOperators.assembles_matrix(::ScaledStiffnessKind) = true
FerriteOperators.execute_kind!(kind::ScaledStiffnessKind, task, ws) =
    FerriteOperators.primal_cell_sweep!(kind, task, ws)
FerriteOperators.provides_analytic(::Type{<:AnyDiffusionCache}, ::ScaledStiffnessKind) = true
function FerriteOperators.assemble_cell!(req::ScaledStiffnessRequest, cache::AnyDiffusionCache, args)
    (; cv) = cache
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        for i in 1:getnbasefunctions(cv), j in 1:getnbasefunctions(cv)
            req.K[i, j] += req.scale * (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) * dΩ
        end
    end
end

# 2. A kind riding the sensitivity driver body, reading the engine's
#    SensitivityBuffers directly — no family declaration needed, since they
#    are structurally present on any nonlinear operator (`needs_ad_decoration`)
#    and absent otherwise. `materialize_request`/`scatter_request!` alone are
#    the whole recipe; `execute_kind!` reuses `sensitivity_cell_sweep!` as-is.
struct ResidualProbeKind end
FerriteOperators.has_cell_request(::Type{<:ResidualProbeKind}) = false
FerriteOperators.execute_kind!(kind::ResidualProbeKind, task, ws) =
    FerriteOperators.sensitivity_cell_sweep!(kind, task, ws)
function FerriteOperators.materialize_request(::ResidualProbeKind, ws, task)
    fill!(ws.sensitivity.gu, 0.0)
    return ResidualRequest(ws.sensitivity.gu)
end
FerriteOperators.scatter_request!(req::ResidualRequest, assembler, cell) = assemble!(assembler, cell, req.r)

# 3. A kind claiming an analytic kernel it does not implement.
struct OrphanKind end
struct OrphanRequest{M <: AbstractMatrix} <: FerriteOperators.AbstractAssemblyRequest
    K::M
end
FerriteOperators.request_type(::OrphanKind) = OrphanRequest
FerriteOperators.materialize_request(::OrphanKind, ws) = OrphanRequest(ws.Ke)
FerriteOperators.assembles_matrix(::OrphanKind) = true
FerriteOperators.provides_analytic(::Type{<:AnyDiffusionCache}, ::OrphanKind) = true

# Measured inside a function: at testset scope `A` and the operator are
# captured variables, and on Julia 1.10 the boxing of those captures is charged
# to the call being measured rather than to the sweep.
function scaled_stiffness_allocations(A, op)
    FerriteOperators.assemble_into!(ScaledStiffnessKind(), (A,), op, (;), nothing, nothing)
    return @allocated FerriteOperators.assemble_into!(
        ScaledStiffnessKind(), (A,), op, (;), nothing, nothing)
end

@testset "Custom request kinds" begin
    @testset "matrix kind on the primal driver body" begin
        tb = declared_testbed(; requests = (ScaledStiffnessKind,))
        A = allocate_matrix(tb.dh)
        FerriteOperators.assemble_into!(ScaledStiffnessKind(), (A,), tb.op, (;), nothing, nothing)
        @test A ≈ 2.5 * tb.Kop.A rtol = 1e-13

        # The declaration reached setup validation, and the sweep allocates
        # nothing per pass beyond the assembler's own bookkeeping.
        fill!(A.nzval, 0.0)
        @test scaled_stiffness_allocations(A, tb.op) < 1024
    end

    @testset "trait claimed without a kernel errors at setup" begin
        (; dh, qrc, strategy) = scalar_quad_testbed((3, 2))
        @test_throws ArgumentError setup_operator(
            strategy, DeclaredDiffusionIntegrator(qrc, :u), dh; requests = (OrphanKind,))
    end

    @testset "sensitivity-shaped downstream kind reads ws.sensitivity directly" begin
        plain = declared_testbed(; requests = (ResidualProbeKind,))

        # A bilinear operator carries no sensitivity buffers whatever it
        # declares — structural, by integrator family, not by declaration.
        declared = setup_operator(plain.strategy, SimpleBilinearMassIntegrator(1.0, plain.qrc, :u),
                                  plain.dh; requests = (ResidualProbeKind,))
        @test !carries_sensitivity_buffers(declared)

        n = plain.n
        u  = sin.(0.3 .* (1:n))
        du = cos.(0.2 .* (1:n))
        ctx = TimeIntegrationContext(1.0, 0.25, 0.25)
        probe = zeros(n)
        FerriteOperators.assemble_into!(ResidualProbeKind(), (probe,), plain.op,
                                        (u = u, du = du), nothing, ctx)
        reference = zeros(n)
        evaluate!(plain.op, reference, (u = u, du = du), nothing, ctx)
        @test probe ≈ reference rtol = 1e-13
    end

    @testset "kind dispatch stays constant-folded" begin
        # Only branches the compiler could NOT decide count: a `GotoIfNot` whose
        # condition is already a literal `Bool` is dead IR, which Julia ≥ 1.12
        # deletes and 1.10 leaves behind. A trait that stopped folding would
        # leave its condition as an SSAValue, which is still counted.
        count_branches(ci) = count(x -> x isa Core.GotoIfNot && !(x.cond isa Bool), ci.code)
        # Built-in path: the predicate-driven scatter collapses to one call.
        for K in (JacobianKind{:u}, ResidualKind, JacobianResidualKind,
                  FerriteOperators.BilinearKind, FerriteOperators.LinearKind)
            ci = code_typed(FerriteOperators.scatter_local!, Tuple{K, Any, Any})[1][1]
            @test count_branches(ci) == 0
        end
        # A downstream kind folds the same way — literal traits, no branch.
        ci = code_typed(FerriteOperators.scatter_local!, Tuple{ScaledStiffnessKind, Any, Any})[1][1]
        @test count_branches(ci) == 0
        # Family resolution folds to the singleton, so declarations are static.
        @test FerriteOperators.sweep_family(ScaledStiffnessKind) === FerriteOperators.NoFamily()
    end
end
