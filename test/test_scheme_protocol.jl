using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using LinearAlgebra
using SparseArrays

# Transient diffusion, r(u, u̇) = ∫ (u̇ v + ∇u⋅∇v) dΩ — ∂F/∂u is the stiffness
# and ∂F/∂du the mass matrix, so every weighted combination is known in closed
# form. `fused` selects the analytic W provider.
struct ProtocolDiffusionIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
    fused::Bool
end
ProtocolDiffusionIntegrator(qrc, field_name) = ProtocolDiffusionIntegrator(qrc, field_name, false)
struct ProtocolDiffusionCache{fused, CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::ProtocolDiffusionIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    cv     = CellValues(qr, ip, ip_geo)
    return ProtocolDiffusionCache{m.fused, typeof(cv)}(cv)
end
FerriteOperators.duplicate_for_device(device, c::ProtocolDiffusionCache{f}) where {f} =
    ProtocolDiffusionCache{f, typeof(c.cv)}(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.reinit_values!(c::ProtocolDiffusionCache, cell) = reinit!(c.cv, cell)
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::ProtocolDiffusionCache, args)
    (; cv) = cache
    uₑ, duₑ = args.states.u, args.states.du
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        u̇  = function_value(cv, qp, duₑ)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (u̇ * shape_value(cv, qp, i) + ∇u ⋅ shape_gradient(cv, qp, i)) * dΩ
        end
    end
end
# The hand-fused SDIRK/BE scheme matrix, reading its scalars from the request.
const FUSED_W_SWEEPS = Ref(0)
FerriteOperators.provides_analytic(::Type{<:ProtocolDiffusionCache{true}}, ::WeightedJacobianKind) = true
function FerriteOperators.assemble_cell!(req::WeightedJacobianRequest, cache::ProtocolDiffusionCache{true}, args)
    FUSED_W_SWEEPS[] += 1
    (; cv) = cache
    wu, wdu = req.weights.u, req.weights.du
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        for i in 1:getnbasefunctions(cv), j in 1:getnbasefunctions(cv)
            req.K[i, j] += (wu * (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) +
                            wdu * shape_value(cv, qp, i) * shape_value(cv, qp, j)) * dΩ
        end
    end
end
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:mass}, cache::ProtocolDiffusionCache, args) =
    sum(qp -> getdetJdV(cache.cv, qp), 1:getnquadpoints(cache.cv))

# The worked SDIRK-W scheme protocol: two slots, the weighted Jacobian it
# solves with, and the residual. No coefficients — γ and Δt ride with the
# evaluation, not with the declaration.
struct SDIRKWProtocol <: AbstractSchemeProtocol end
FerriteOperators.declared_slots(::SDIRKWProtocol)     = (:u, :du)
FerriteOperators.declared_kinds(::SDIRKWProtocol)     = (WeightedJacobianKind, ResidualKind)
FerriteOperators.declared_scratch(::SDIRKWProtocol)   = (;)
FerriteOperators.declared_args_type(::SDIRKWProtocol) = KernelArgs

first_workspace(op) = first(first(op.engine.subdomain_caches).device_cache)

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
        @test declared_slots(p) == (:u, :du)
        @test declared_kinds(p) == (WeightedJacobianKind, ResidualKind)
        @test declared_scratch(p) == (;)
        @test declared_args_type(p) === KernelArgs
        # the element-side scratch hook must not silently swallow a protocol
        @test_throws ArgumentError FerriteOperators.declare_scratch(p)
    end

    @testset "SDIRK-W witness: analytic provider gives the one-sweep W" begin
        tb = protocol_testbed(; fused = true)
        u  = sin.(0.3 .* (1:tb.n)); du = cos.(0.2 .* (1:tb.n))
        states = (u = u, du = du)
        ref = Matrix(tb.Kop.A) .+ (1 / (γ * Δt)) .* Matrix(tb.Mop.A)

        FUSED_W_SWEEPS[] = 0
        W = share_pattern(tb.op.J)
        assemble_weighted_jacobian!(W, tb.op, weights, states, nothing, ctx)
        @test FUSED_W_SWEEPS[] == getncells(tb.grid)     # one kernel call per cell, one sweep
        @test Matrix(W) ≈ ref rtol = 1e-12

        # the residual the same protocol declares still runs
        r = zeros(tb.n)
        evaluate!(tb.op, r, states, nothing, ctx)
        @test r ≈ tb.Mop.A * du .+ tb.Kop.A * u rtol = 1e-12
    end

    @testset "the same protocol over a non-analytic cache agrees" begin
        tb = protocol_testbed(; fused = false)
        cache = first(tb.op.engine.subdomain_caches).domain.element
        @test !FerriteOperators.provides_analytic(typeof(cache), WeightedJacobianKind(weights))

        u  = sin.(0.3 .* (1:tb.n)); du = cos.(0.2 .* (1:tb.n))
        states = (u = u, du = du)
        ref = Matrix(tb.Kop.A) .+ (1 / (γ * Δt)) .* Matrix(tb.Mop.A)

        FUSED_W_SWEEPS[] = 0
        W = share_pattern(tb.op.J)
        assemble_weighted_jacobian!(W, tb.op, weights, states, nothing, ctx)
        @test FUSED_W_SWEEPS[] == 0                       # the analytic kernel never ran
        @test Matrix(W) ≈ ref rtol = 1e-12

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
        @test declared_slots(kw.engine.protocol) == declared_slots(pos.engine.protocol)
        @test declared_kinds(kw.engine.protocol) == declared_kinds(pos.engine.protocol)
        # kind instances normalize to the UnionAll base, like the kwarg form always did
        @test declared_kinds(DefaultProtocol(; requests = (StateJVPKind(zeros(n)),))) == (StateJVPKind,)

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

        # bilinear and linear operators never differentiate and never reduce
        bws = first_workspace(tb.Mop)
        @test bws.ad === nothing
        @test bws.functional === nothing
        lop = setup_operator(tb.strategy, SimpleLinearIntegrator(1.0, qrc, :u), tb.dh)
        lws = first_workspace(lop)
        @test lws.ad === nothing
        @test lws.functional === nothing

        # the nonlinear family always differentiates, declared or not
        @test first_workspace(tb.op).ad !== nothing
        @test mandatory_kinds(tb.Mop.integrator) == (FerriteOperators.BilinearKind, ResidualKind)

        # a bilinear operator asked to differentiate says so instead of failing obscurely
        @test_throws ArgumentError FerriteOperators.sweep_state(bws, StateJVPKind(nothing))
    end

    @testset "undeclared kinds stay usable" begin
        tb = protocol_testbed()
        u = sin.(0.3 .* (1:tb.n)); du = cos.(0.2 .* (1:tb.n))
        states = (u = u, du = du)

        # SDIRKWProtocol declares neither the state JVP nor a functional
        @test !(StateJVPKind in declared_kinds(tb.op.engine.protocol))
        Jv = zeros(tb.n); v = cos.(0.11 .* (1:tb.n))
        state_jvp!(Jv, tb.op, v, states, nothing, ctx)
        @test Jv ≈ tb.Kop.A * v rtol = 1e-10

        # the functional family is materialized by the sweep that needs it
        ws = first_workspace(tb.op)
        @test ws.functional === nothing
        area = evaluate_functional(tb.op, FunctionalKind(:mass), states, nothing, ctx)
        @test ws.functional !== nothing
        @test area ≈ 4.0 rtol = 1e-12          # the [-1,1]² reference grid
        # …and a declaring protocol has it from the start
        fop = setup_operator(tb.strategy, ProtocolDiffusionIntegrator(tb.qrc, :u), tb.dh,
                             DefaultProtocol(; slots = (:u, :du), requests = (FunctionalKind,)))
        @test first_workspace(fop).functional !== nothing
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
# request_type/materialize_request + traits + execute_kind!. No src edits.

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
FerriteOperators.provides_analytic(::Type{<:ProtocolDiffusionCache}, ::ScaledStiffnessKind) = true
function FerriteOperators.assemble_cell!(req::ScaledStiffnessRequest, cache::ProtocolDiffusionCache, args)
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
FerriteOperators.provides_analytic(::Type{<:ProtocolDiffusionCache}, ::OrphanKind) = true

struct CustomKindProtocol{K <: Tuple} <: AbstractSchemeProtocol
    kinds::K
end
FerriteOperators.declared_slots(::CustomKindProtocol)     = (:u, :du)
FerriteOperators.declared_kinds(p::CustomKindProtocol)    = p.kinds
FerriteOperators.declared_scratch(::CustomKindProtocol)   = (;)
FerriteOperators.declared_args_type(::CustomKindProtocol) = KernelArgs

@testset "Custom request kinds" begin
    @testset "matrix kind on the primal driver body" begin
        tb = protocol_testbed(protocol = CustomKindProtocol((ScaledStiffnessKind,)))
        A = allocate_matrix(tb.dh)
        FerriteOperators.assemble_into!(ScaledStiffnessKind(), (A,), tb.op, (;), nothing, nothing)
        @test A ≈ 2.5 * tb.Kop.A rtol = 1e-13

        # The declaration reached setup validation, and the sweep allocates
        # nothing per pass beyond the assembler's own bookkeeping.
        fill!(A.nzval, 0.0)
        @test @allocated(FerriteOperators.assemble_into!(
            ScaledStiffnessKind(), (A,), tb.op, (;), nothing, nothing)) < 1024
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
        # Without the declaration the operator carries no derivative family.
        plain = protocol_testbed(protocol = CustomKindProtocol((ResidualProbeKind,)))
        @test first_workspace(plain.op).ad !== nothing

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
        count_branches(ci) = count(x -> x isa Core.GotoIfNot, ci.code)
        # Built-in path: the predicate-driven scatter collapses to one call.
        for K in (JacobianKind{:u}, ResidualKind, JacobianResidualKind,
                  FerriteOperators.BilinearKind, FerriteOperators.LinearKind)
            ci = code_typed(FerriteOperators.scatter_local!, Tuple{K, Any, Any})[1][1]
            @test count_branches(ci) == 0
        end
        # A downstream kind folds the same way — literal traits, no branch.
        ci = code_typed(FerriteOperators.scatter_local!, Tuple{ScaledStiffnessKind, Any, Any})[1][1]
        @test count_branches(ci) == 0
        # A trait call site reduces to a single literal return, which is what
        # lets the driver bodies drop the branches guarding it.
        folded_body(f) = string.(code_typed(f, Tuple{})[1][1].code)
        @test folded_body(() -> FerriteOperators.assembles_matrix(ScaledStiffnessKind())) == ["return true"]
        @test folded_body(() -> FerriteOperators.depends_on_unknowns(ScaledStiffnessKind())) == ["return false"]
        @test folded_body(() -> FerriteOperators.assembles_matrix(JacobianKind{:u}())) == ["return true"]
        @test folded_body(() -> FerriteOperators.assembles_vector(ResidualKind())) == ["return true"]
        # Family resolution folds to the singleton, so declarations are static.
        @test FerriteOperators.sweep_family(ResidualProbeKind) === FerriteOperators.DerivativeFamily()
        @test FerriteOperators.sweep_family(ScaledStiffnessKind) === FerriteOperators.NoFamily()
    end
end
