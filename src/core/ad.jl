####################################
## AD fallbacks (ForwardDiff over the residual kernel)
####################################
#
# Limitations:
# - AD sweeps cover the volumetric kernel only; boundary contributions flow
#   through the analytic facet path, so sensitivities of parameter- or
#   time-dependent boundary terms are NOT captured.
# - State sweeps (∂F/∂u Jacobian, JVP, VJP, ∂F/∂t) run over per-worker
#   preallocated ForwardDiff configs. The parameter sweeps build their
#   configs per call: their seed dimension nθ is call-time knowledge, and a
#   cached config would be abstractly typed across nθ changes.
#
# These helpers are the derivative family's internals and read `ws.ad`
# directly; the family's presence is the operator's setup-time guarantee (see
# `sweep_state` and `mandatory_kinds`).

"Tag for the package-owned ForwardDiff configs, so per-worker configs outlive the per-cell closures."
struct FerriteOperatorsADTag end

"""
    ADWorkspace

Per-worker buffers and ForwardDiff machinery for derivative sweeps, allocated
once per worker at setup (analytic sensitivity kernels use the same output
buffers). Element-sized members are eager; parameter-sized members (`θ`,
`Bₑ`, `gθ`) are (re)allocated on first use via
[`parameter_sweep_buffers!`](@ref) because nθ arrives with `p` at call time.
"""
@concrete mutable struct ADWorkspace
    θ         # flat primal parameter copy (nθ)
    Bₑ        # local parameter Jacobian block (residual × nθ)
    gθ        # parameter pullback output (nθ)
    λₑ        # residual-sized adjoint gather
    vₑ        # unknown-sized JVP direction gather
    Jvₑ       # residual-sized JVP output
    gu        # unknown-sized state-VJP output
    gₜ        # residual-sized time-sensitivity output
    jac_cfg   # ∂F/∂u JacobianConfig (fixed chunk, package tag)
    deriv_cfg # scalar-seed DerivativeConfig — JVP directional sweep and ∂F/∂t
    grad_cfg  # state-VJP GradientConfig over the unknown buffer
    u_dual    # single-partial Dual unknown buffer for the JVP direction
    re_dual   # Dual residual buffer for the state-VJP closure
    wseed     # zeroed unknown-sized seed point of the weighted-Jacobian sweep
    wdual     # Dual slot buffers of the weighted sweep, grown to the slot count on first use
end

function create_ad_workspace(element, sdh)
    vₑ  = allocate_element_unknown_vector(element, sdh)
    gu  = allocate_element_unknown_vector(element, sdh)
    λₑ  = allocate_element_residual_vector(element, sdh)
    Jvₑ = allocate_element_residual_vector(element, sdh)
    gₜ  = allocate_element_residual_vector(element, sdh)
    T   = eltype(Jvₑ)
    tag       = ForwardDiff.Tag{FerriteOperatorsADTag, T}()
    chunk     = ForwardDiff.Chunk(vₑ)
    jac_cfg   = ForwardDiff.JacobianConfig(nothing, Jvₑ, vₑ, chunk, tag)
    deriv_cfg = ForwardDiff.DerivativeConfig(nothing, Jvₑ, zero(T), tag)
    grad_cfg  = ForwardDiff.GradientConfig(nothing, vₑ, chunk, tag)
    u_dual    = similar(vₑ, ForwardDiff.Dual{typeof(tag), T, 1})
    re_dual   = similar(Jvₑ, eltype(grad_cfg.duals))
    wseed     = zero(vₑ)
    wdual     = Vector{Vector{eltype(jac_cfg.duals[2])}}()
    return ADWorkspace(
        Vector{T}(), Matrix{T}(undef, length(Jvₑ), 0), Vector{T}(),
        λₑ, vₑ, Jvₑ, gu, gₜ,
        jac_cfg, deriv_cfg, grad_cfg, u_dual, re_dual, wseed, wdual,
    )
end

"""
    weighted_seed_buffers!(ad::ADWorkspace, nslots) -> Vector of Dual buffers

Grow the weighted sweep's Dual slot buffers to `nslots` entries, one per
participating slot. Sized once per worker on the first sweep of a given slot
count; the Dual type is fixed by the sweep's ∂F/∂u configuration.
"""
function weighted_seed_buffers!(ad::ADWorkspace, nslots::Int)
    while length(ad.wdual) < nslots
        push!(ad.wdual, similar(ad.wseed, eltype(ad.jac_cfg.duals[2])))
    end
    return ad.wdual
end

"""
    parameter_sweep_buffers!(ad::ADWorkspace, nθ) -> ADWorkspace

Size the parameter-sweep members (`θ`, `Bₑ`, `gθ`) for `nθ` flat parameters,
reallocating only when nθ changed since the last sweep on this worker.
"""
function parameter_sweep_buffers!(ad::ADWorkspace, nθ::Int)
    if length(ad.θ) != nθ
        T = eltype(ad.θ)
        ad.θ  = Vector{T}(undef, nθ)
        ad.gθ = Vector{T}(undef, nθ)
        ad.Bₑ = Matrix{T}(undef, size(ad.Bₑ, 1), nθ)
    end
    return ad
end

# Evaluate the volumetric local residual for `args`, overwriting `r`.
function evaluate_cell_residual!(r, cache, args)
    fill!(r, zero(eltype(r)))
    assemble_cell!(ResidualRequest(r), cache, args)
    return r
end

# ∂F/∂slot — writes the Jacobian into K and the primal residual into ws.re.
# Only the named slot is seeded; every other slot stays at its primal value —
# including `AffineRate`-reconstructed slots, which are formed at gather time.
# All slot buffers share one size, so the per-worker config serves every slot.
# Tag checking is off: the config carries the package tag, not the closure's.
function ad_state_jacobian!(K, ws, args, ::Val{slot} = Val(:u)) where {slot}
    f! = (r, x) -> evaluate_cell_residual!(
        r, ws.element, with_states(args, merge(args.states, NamedTuple{(slot,)}((x,)))))
    ForwardDiff.jacobian!(K, f!, ws.re, args.states[slot], ws.ad.jac_cfg, Val{false}())
    return K
end

# Σₛ wₛ ∂F/∂s in ONE sweep: the seed variable `x` is the weighted variation
# itself, so every participating slot enters as `sₑ + wₛ·x` and the derivative
# w.r.t. `x` at `x = 0` is exactly the weighted combination — with the same
# seed dimension, config, and chunking as a single-slot ∂F/∂s sweep. Slots
# outside `weights` (including `AffineRate` reconstructions) stay at their
# primal value, matching the frozen-slot contract of `JacobianKind`.
function ad_weighted_jacobian!(K, ws, args, weights::NamedTuple{slots}) where {slots}
    bufs  = weighted_seed_buffers!(ws.ad, length(slots))
    prim  = NamedTuple{slots}(args.states)
    duals = NamedTuple{slots}(ntuple(i -> bufs[i], Val(length(slots))))
    # The Dual-carrying args are built once per cell; the differentiated
    # closure only refreshes the buffers behind them.
    dargs = with_states(args, merge(args.states, duals))
    cache = ws.element
    f! = (r, x) -> begin
        map((buf, sₑ, w) -> (@. buf = sₑ + w * x), duals, prim, weights)
        evaluate_cell_residual!(r, cache, dargs)
    end
    ForwardDiff.jacobian!(K, f!, ws.re, ws.ad.wseed, ws.ad.jac_cfg, Val{false}())
    return K
end

# ∂F/∂θ — dense local parameter Jacobian. The parameter sweep re-queries the
# element parameters from the Dual-rebuilt global `p` so wrappers forward
# Duals transparently.
function ad_parameter_jacobian!(Bₑ, ws, args, p)
    θ = copyto!(ws.ad.θ, parameter_vector(p))
    f! = (r, θᵢ) -> begin
        pₑ = query_cell_parameters(ws.element, ws.cell, rebuild_parameters(p, θᵢ))
        evaluate_cell_residual!(r, ws.element, with_parameters(args, pₑ))
    end
    ForwardDiff.jacobian!(Bₑ, f!, ws.re, θ)
    return Bₑ
end

# (∂F/∂θ)ᵀλₑ — adjoint pullback as the gradient of the scalar λₑ·rₑ(θ).
function ad_parameter_vjp!(gₑ, λₑ, ws, args, p)
    θ = copyto!(ws.ad.θ, parameter_vector(p))
    fscalar = θᵢ -> begin
        pₑ = query_cell_parameters(ws.element, ws.cell, rebuild_parameters(p, θᵢ))
        r = zeros(eltype(θᵢ), length(λₑ))
        evaluate_cell_residual!(r, ws.element, with_parameters(args, pₑ))
        return dot(λₑ, r)
    end
    ForwardDiff.gradient!(gₑ, fscalar, θ)
    return gₑ
end

# ∂F/∂t — explicit time dependence, seeded through the context channel: the
# sweep's ctx is rebuilt with a Dual evaluation time, so an element reading
# `evaluation_time(args.ctx)` differentiates exactly. The preallocated config
# is typed for the residual eltype; exotic time types fall back to a per-call
# config.
function ad_time_sensitivity!(gₑ, ws, args)
    ctx = args.ctx
    t = evaluation_time(ctx)
    f! = (r, t̃) -> evaluate_cell_residual!(r, ws.element, with_context(args, with_time(ctx, t̃)))
    if t isa eltype(ws.re)
        ForwardDiff.derivative!(gₑ, f!, ws.re, t, ws.ad.deriv_cfg, Val{false}())
    else
        ForwardDiff.derivative!(gₑ, f!, ws.re, t)
    end
    return gₑ
end

# (∂F/∂u)·v — one directional-Dual sweep through the residual kernel: the
# MFEM NONE level, no matrices anywhere. The perturbed state is written into
# the per-worker Dual buffer instead of allocating per cell.
function ad_state_jvp!(Jvₑ, ws, args, vₑ)
    ud = ws.ad.u_dual
    f! = (r, s) -> begin
        @. ud = args.states.u + s * vₑ
        evaluate_cell_residual!(r, ws.element, with_states(args, merge(args.states, (u = ud,))))
    end
    ForwardDiff.derivative!(Jvₑ, f!, ws.re, zero(eltype(vₑ)), ws.ad.deriv_cfg, Val{false}())
    return Jvₑ
end

# (∂F/∂u)ᵀλₑ — gradient of the scalar λₑ·rₑ(u) w.r.t. the element state, with
# the Dual residual evaluated into the per-worker buffer.
function ad_state_vjp!(gₑ, λₑ, ws, args)
    rd = ws.ad.re_dual
    fscalar = u -> begin
        evaluate_cell_residual!(rd, ws.element, with_states(args, merge(args.states, (u = u,))))
        return dot(λₑ, rd)
    end
    ForwardDiff.gradient!(gₑ, fscalar, args.states.u, ws.ad.grad_cfg, Val{false}())
    return gₑ
end
