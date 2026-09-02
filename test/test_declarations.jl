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
        @test FerriteOperators._declared_kinds(tb.op.engine) == (WeightedJacobianKind, ResidualKind)

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
        @test !(StateJVPKind in FerriteOperators._declared_kinds(tb.op.engine))
        Jv = zeros(tb.n); v = cos.(0.11 .* (1:tb.n))
        state_jvp!(Jv, tb.op, v, states, nothing, ctx)
        @test Jv ≈ tb.Kop.A * v rtol = 1e-10

        # a functional sweep reads no state, so an undeclared one just runs
        @test !(FunctionalKind in FerriteOperators._declared_kinds(tb.op.engine))
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
# all default to an EMPTY declaration, so a method written against a signature
# the engine does not call is never reached and the operator assembles a silent
# subset. `DriftProbe{hook}` wears one drifted method for `hook`; `:none` wears
# none, `:correct` wears all four at the engine's own signature.

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
# A drifted ARITY misses the call the same way a drifted argument type does.
FerriteOperators.global_dofs(::DriftProbe{:arity}, ::SubDofHandler, ::Int) = (1,)

# A specialized method the engine's call resolves to passes, whatever it returns.
FerriteOperators.global_dofs(::DriftProbe{:correct}, ::SubDofHandler) = ()
FerriteOperators.facet_items(::DriftProbe{:correct}, ::SubDofHandler) = ()
FerriteOperators.facet_item_global_dofs(::DriftProbe{:correct}, ::SubDofHandler) = ()
FerriteOperators.algebraic_items(::DriftProbe{:correct}, ::DofHandler) = ()

@testset "Declaration-hook signature drift" begin
    (; dh, qrc, strategy) = scalar_quad_testbed((3, 2))
    probe(hook) = DriftProbe{hook}(qrc)
    check(integrator) = FerriteOperators.assert_declaration_signatures(integrator, dh)

    @testset "a drifted signature is rejected, naming hook and both signatures" begin
        for (hook, expected, drifted) in ((:global_dofs, "SubDofHandler", "DofHandler"),
                                          (:facet_items, "SubDofHandler", "DofHandler"),
                                          (:facet_item_global_dofs, "SubDofHandler", "DofHandler"),
                                          (:algebraic_items, "DofHandler", "SubDofHandler"))
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
