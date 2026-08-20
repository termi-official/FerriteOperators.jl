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
const ProtocolDiffusionCache = CVCache{:protocol}
const FusedDiffusionCache    = CVCache{:protocol_fused}
const AnyDiffusionCache      = Union{ProtocolDiffusionCache, FusedDiffusionCache}
ProtocolDiffusionIntegrator(qrc, field_name, fused = false) =
    fused ? CVIntegrator{:protocol_fused}(qrc, field_name) : CVIntegrator{:protocol}(qrc, field_name)

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

# The worked SDIRK-W scheme protocol: two slots, the weighted Jacobian it
# solves with, and the residual. No coefficients — γ and Δt ride with the
# evaluation, not with the declaration.
struct SDIRKWProtocol <: AbstractSchemeProtocol end
FerriteOperators.get_declared_slots(::SDIRKWProtocol) = (:u, :du)
FerriteOperators.get_declared_kinds(::SDIRKWProtocol) = (WeightedJacobianKind, ResidualKind)

function protocol_testbed(; fused = false, protocol = SDIRKWProtocol())
    grid = generate_grid(Quadrilateral, (3, 2))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(2)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op  = setup_operator(strategy, ProtocolDiffusionIntegrator(qrc, :u, fused), dh, protocol)
    Mop = setup_operator(strategy, SimpleBilinearMassIntegrator(1.0, qrc, :u), dh)
    Kop = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
    update_operator!(Mop, nothing)
    update_operator!(Kop, nothing)
    return (; op, Mop, Kop, dh, grid, qrc, strategy, n = ndofs(dh))
end

@testset "Scheme protocols" begin
    Δt  = 0.25
    γ   = 0.5
    ctx = TimeIntegrationContext(1.0, Δt, Δt)
    weights = (u = 1.0, du = 1 / (γ * Δt))

    @testset "declarations-only surface" begin
        p = SDIRKWProtocol()
        @test get_declared_slots(p) == (:u, :du)
        @test get_declared_kinds(p) == (WeightedJacobianKind, ResidualKind)
    end

    # A protocol only declares; the weighted-Jacobian VALUES on both routes are
    # pinned against the bundled bilinear integrators in test_stage_block.jl.
    @testset "SDIRK-W witness: the declarations reach the engine" begin
        tb = protocol_testbed(; fused = true)
        @test get_declared_kinds(tb.op.engine.protocol) == (WeightedJacobianKind, ResidualKind)

        u  = sin.(0.3 .* (1:tb.n)); du = cos.(0.2 .* (1:tb.n))
        states = (u = u, du = du)

        FUSED_W_SWEEPS[] = 0
        W = share_pattern(tb.op.J)
        assemble_weighted_jacobian!(W, tb.op, weights, states, nothing, ctx)
        @test FUSED_W_SWEEPS[] == getncells(tb.grid)     # one kernel call per cell, one sweep

        # the residual the same protocol declares still runs
        r = zeros(tb.n)
        evaluate!(tb.op, r, states, nothing, ctx)
        @test r ≈ tb.Mop.A * du .+ tb.Kop.A * u rtol = 1e-12
    end

    @testset "the same protocol over a non-analytic cache agrees" begin
        tb = protocol_testbed(; fused = false)
        cache = first_element_cache(tb.op)
        @test !FerriteOperators.provides_analytic(typeof(cache), WeightedJacobianKind(weights))

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

    @testset "the keyword form is sugar for DefaultProtocol" begin
        grid = generate_grid(Quadrilateral, (3, 2))
        dh   = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
        qrc = QuadratureRuleCollection(2); n = ndofs(dh)
        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
        integrator = ProtocolDiffusionIntegrator(qrc, :u)

        kw  = setup_operator(strategy, integrator, dh; slots = (:u, :du), requests = (StateJVPKind,))
        pos = setup_operator(strategy, integrator, dh,
                             DefaultProtocol(; slots = (:u, :du), requests = (StateJVPKind,)))
        @test typeof(kw.engine.protocol) === typeof(pos.engine.protocol)
        @test get_declared_slots(kw.engine.protocol) == get_declared_slots(pos.engine.protocol)
        @test get_declared_kinds(kw.engine.protocol) == get_declared_kinds(pos.engine.protocol)

        u = sin.(0.3 .* (1:n)); du = cos.(0.2 .* (1:n))
        states = (u = u, du = du)
        rkw = zeros(n); update_linearization!(kw, rkw, states, nothing, ctx)
        rpos = zeros(n); update_linearization!(pos, rpos, states, nothing, ctx)
        @test rkw == rpos
        @test kw.J == pos.J
    end

    @testset "families are built only where declarations call for them" begin
        tb = protocol_testbed()
        qrc = QuadratureRuleCollection(2)

        # bilinear and linear operators never differentiate
        bws = first_workspace(tb.Mop)
        @test bws.ad === nothing
        lop = setup_operator(tb.strategy, SimpleLinearIntegrator(1.0, qrc, :u), tb.dh)
        @test first_workspace(lop).ad === nothing

        # the nonlinear family always differentiates, declared or not
        @test first_workspace(tb.op).ad !== nothing
        @test mandatory_kinds(tb.Mop.integrator) == (FerriteOperators.BilinearKind, ResidualKind)

        # a bilinear operator asked to differentiate says so instead of failing obscurely
        @test_throws ArgumentError FerriteOperators.sweep_state(bws, StateJVPKind(nothing))
        # a functional kind returns its value, so it has no per-worker state to read
        @test_throws ArgumentError FerriteOperators.sweep_state(bws, FunctionalKind(:mass))
    end

    @testset "workspaces are immutable" begin
        tb = protocol_testbed()
        @test !ismutable(first_workspace(tb.op))
        @test !ismutable(first_workspace(tb.Mop))
    end

    @testset "undeclared kinds stay usable" begin
        tb = protocol_testbed()
        u = sin.(0.3 .* (1:tb.n)); du = cos.(0.2 .* (1:tb.n))
        states = (u = u, du = du)

        # SDIRKWProtocol declares neither the state JVP nor a functional
        @test !(StateJVPKind in get_declared_kinds(tb.op.engine.protocol))
        Jv = zeros(tb.n); v = cos.(0.11 .* (1:tb.n))
        state_jvp!(Jv, tb.op, v, states, nothing, ctx)
        @test Jv ≈ tb.Kop.A * v rtol = 1e-10

        # a functional sweep reads no state, so an undeclared one just runs
        @test !(FunctionalKind in get_declared_kinds(tb.op.engine.protocol))
        area = evaluate_functional(tb.op, FunctionalKind(:mass), states, nothing, ctx)
        @test area ≈ 4.0 rtol = 1e-12          # the [-1,1]² reference grid

        # …and declaring it builds nothing: the declaring operator answers the
        # same, and a bilinear one still carries no sweep-state family at all.
        fop = setup_operator(tb.strategy, ProtocolDiffusionIntegrator(tb.qrc, :u), tb.dh,
                             DefaultProtocol(; slots = (:u, :du), requests = (FunctionalKind,)))
        @test evaluate_functional(fop, FunctionalKind(:mass), states, nothing, ctx) == area
        bfop = setup_operator(tb.strategy, SimpleBilinearMassIntegrator(1.0, tb.qrc, :u), tb.dh,
                              DefaultProtocol(; requests = (FunctionalKind,)))
        @test first_workspace(bfop).ad === nothing
    end

    @testset "two operators from one protocol evaluate concurrently" begin
        grid = generate_grid(Quadrilateral, (6, 5))
        dh   = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
        qrc = QuadratureRuleCollection(2); n = ndofs(dh)
        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
        integrator = ProtocolDiffusionIntegrator(qrc, :u)
        protocol = SDIRKWProtocol()

        op1 = setup_operator(strategy, integrator, dh, protocol)
        op2 = setup_operator(strategy, integrator, dh, protocol)

        # no mutable state is shared between the two caches
        @test op1.J !== op2.J
        @test first_workspace(op1) !== first_workspace(op2)
        @test first_workspace(op1).Ke !== first_workspace(op2).Ke
        @test first_workspace(op1).ad !== first_workspace(op2).ad
        @test first_workspace(op1).slot_buffers.u !== first_workspace(op2).slot_buffers.u

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

# 2. A derivative-family kind riding the sensitivity driver body. Declaring it
#    must build the ADWorkspace; its kernel reads the family through
#    `sweep_state`, which only resolves if that happened.
struct ResidualProbeKind end
FerriteOperators.sweep_family(::Type{<:ResidualProbeKind}) = FerriteOperators.DerivativeFamily()
FerriteOperators.has_cell_request(::Type{<:ResidualProbeKind}) = false
FerriteOperators.execute_kind!(kind::ResidualProbeKind, task, ws) =
    FerriteOperators.sensitivity_cell_sweep!(kind, task, ws)
function FerriteOperators.sensitivity_kernel!(kind::ResidualProbeKind, task, ws, args)
    gₑ = FerriteOperators.sweep_state(ws, kind).gu     # the declared family
    fill!(gₑ, 0.0)
    FerriteOperators.assemble_cell!(ResidualRequest(gₑ), ws.element, args)
    assemble!(task.inner_assembler, ws.cell, gₑ)
end

# 3. A kind claiming an analytic kernel it does not implement.
struct OrphanKind end
struct OrphanRequest{M <: AbstractMatrix} <: FerriteOperators.AbstractAssemblyRequest
    K::M
end
FerriteOperators.request_type(::OrphanKind) = OrphanRequest
FerriteOperators.materialize_request(::OrphanKind, ws) = OrphanRequest(ws.Ke)
FerriteOperators.assembles_matrix(::OrphanKind) = true
FerriteOperators.provides_analytic(::Type{<:AnyDiffusionCache}, ::OrphanKind) = true

struct CustomKindProtocol{K <: Tuple} <: AbstractSchemeProtocol
    kinds::K
end
FerriteOperators.get_declared_slots(::CustomKindProtocol)  = (:u, :du)
FerriteOperators.get_declared_kinds(p::CustomKindProtocol) = p.kinds

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
        tb = protocol_testbed(protocol = CustomKindProtocol((ScaledStiffnessKind,)))
        A = allocate_matrix(tb.dh)
        FerriteOperators.assemble_into!(ScaledStiffnessKind(), (A,), tb.op, (;), nothing, nothing)
        @test A ≈ 2.5 * tb.Kop.A rtol = 1e-13

        # The declaration reached setup validation, and the sweep allocates
        # nothing per pass beyond the assembler's own bookkeeping.
        fill!(A.nzval, 0.0)
        @test scaled_stiffness_allocations(A, tb.op) < 1024
    end

    @testset "trait claimed without a kernel errors at setup" begin
        grid = generate_grid(Quadrilateral, (3, 2))
        dh = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
        integrator = ProtocolDiffusionIntegrator(QuadratureRuleCollection(2), :u)
        @test_throws ArgumentError setup_operator(
            SequentialAssemblyStrategy(SequentialCPUDevice()), integrator, dh,
            CustomKindProtocol((OrphanKind,)))
    end

    @testset "derivative-family kind builds the ADWorkspace" begin
        plain = protocol_testbed(protocol = CustomKindProtocol((ResidualProbeKind,)))

        # An operator whose mandatory kinds never differentiate carries no
        # derivative family; declaring a derivative-family kind builds one.
        @test first_workspace(plain.Mop).ad === nothing
        declared = setup_operator(plain.strategy, SimpleBilinearMassIntegrator(1.0, plain.qrc, :u),
                                  plain.dh, DefaultProtocol(; requests = (ResidualProbeKind,)))
        @test first_workspace(declared).ad !== nothing

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
        @test FerriteOperators.sweep_family(ResidualProbeKind) === FerriteOperators.DerivativeFamily()
        @test FerriteOperators.sweep_family(ScaledStiffnessKind) === FerriteOperators.NoFamily()
    end
end
