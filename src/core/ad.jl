####################################
## AD fallbacks (ForwardDiff over the residual kernel)
####################################
#
# Slice limitations, lifted with the facet-driver rework and the parameter
# layout (ParameterSpace) work:
# - AD sweeps cover the volumetric kernel only; boundary contributions keep
#   flowing through the analytic facet path. Sensitivities of parameter- or
#   time-dependent boundary terms are NOT captured yet.
# - State sweeps (∂F/∂u Jacobian, JVP, VJP, ∂F/∂t) run over per-worker
#   preallocated ForwardDiff configs. The PARAMETER sweeps still build their
#   configs per call: their seed dimension nθ is call-time knowledge until
#   parameter layouts land, and caching a config across nθ changes would make
#   the field abstractly typed for little gain on these non-hot paths.

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
    return ADWorkspace(
        Vector{T}(), Matrix{T}(undef, length(Jvₑ), 0), Vector{T}(),
        λₑ, vₑ, Jvₑ, gu, gₜ,
        jac_cfg, deriv_cfg, grad_cfg, u_dual, re_dual,
    )
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

# Evaluate the volumetric local residual at (statesₑ, pₑ), overwriting `r`.
function evaluate_cell_residual!(r, statesₑ, ws, pₑ, ctx)
    fill!(r, zero(eltype(r)))
    assemble_cell!(ResidualRequest(r), ws.element, KernelArgs(statesₑ, ws.cell, pₑ, ws.scratch, ctx))
    return r
end

# ∂F/∂u — writes the Jacobian into K and the primal residual into ws.re. The
# `:u` slot is seeded; every other slot stays at its primal value. Tag
# checking is off: the config carries the package tag, not the closure's.
function ad_state_jacobian!(K, ws, statesₑ, pₑ, ctx)
    f! = (r, u) -> evaluate_cell_residual!(r, merge(statesₑ, (u = u,)), ws, pₑ, ctx)
    ForwardDiff.jacobian!(K, f!, ws.re, statesₑ.u, ws.ad.jac_cfg, Val{false}())
    return K
end

# ∂F/∂θ — dense local parameter Jacobian. The parameter sweep re-queries the
# element parameters from the Dual-rebuilt global `p` so wrappers forward
# Duals transparently.
function ad_parameter_jacobian!(Bₑ, ws, statesₑ, p, ctx)
    θ = copyto!(ws.ad.θ, parameter_vector(p))
    f! = (r, θᵢ) -> begin
        pₑ = query_cell_parameters(ws.element, ws.cell, rebuild_parameters(p, θᵢ))
        evaluate_cell_residual!(r, statesₑ, ws, pₑ, ctx)
    end
    ForwardDiff.jacobian!(Bₑ, f!, ws.re, θ)
    return Bₑ
end

# (∂F/∂θ)ᵀλₑ — adjoint pullback as the gradient of the scalar λₑ·rₑ(θ).
function ad_parameter_vjp!(gₑ, λₑ, ws, statesₑ, p, ctx)
    θ = copyto!(ws.ad.θ, parameter_vector(p))
    fscalar = θᵢ -> begin
        pₑ = query_cell_parameters(ws.element, ws.cell, rebuild_parameters(p, θᵢ))
        r = zeros(eltype(θᵢ), length(λₑ))
        evaluate_cell_residual!(r, statesₑ, ws, pₑ, ctx)
        return dot(λₑ, r)
    end
    ForwardDiff.gradient!(gₑ, fscalar, θ)
    return gₑ
end

# ∂F/∂t — explicit time dependence, seeded through the bare-time parameter
# channel. The ctx-based seed (`with_time`) replaces this in phase 2. The
# preallocated config is Float64-typed; exotic time types fall back to the
# per-call config.
function ad_time_sensitivity!(gₑ, ws, statesₑ, t::Real, ctx)
    f! = (r, t̃) -> begin
        pₑ = query_cell_parameters(ws.element, ws.cell, t̃)
        evaluate_cell_residual!(r, statesₑ, ws, pₑ, ctx)
    end
    if t isa eltype(ws.re)
        ForwardDiff.derivative!(gₑ, f!, ws.re, t, ws.ad.deriv_cfg, Val{false}())
    else
        ForwardDiff.derivative!(gₑ, f!, ws.re, t)
    end
    return gₑ
end

function ad_time_sensitivity!(gₑ, ws, statesₑ, t, ctx)
    throw(ArgumentError(
        "Time sensitivities currently require the bare evaluation time as the parameter " *
        "object (got $(typeof(t))). Context-based seeding lands with the phase-2 API."))
end

# (∂F/∂u)·v — one directional-Dual sweep through the residual kernel: the
# MFEM NONE level, no matrices anywhere. The perturbed state is written into
# the per-worker Dual buffer instead of allocating per cell.
function ad_state_jvp!(Jvₑ, ws, statesₑ, vₑ, pₑ, ctx)
    ud = ws.ad.u_dual
    f! = (r, s) -> begin
        @. ud = statesₑ.u + s * vₑ
        evaluate_cell_residual!(r, merge(statesₑ, (u = ud,)), ws, pₑ, ctx)
    end
    ForwardDiff.derivative!(Jvₑ, f!, ws.re, zero(eltype(vₑ)), ws.ad.deriv_cfg, Val{false}())
    return Jvₑ
end

# (∂F/∂u)ᵀλₑ — gradient of the scalar λₑ·rₑ(u) w.r.t. the element state, with
# the Dual residual evaluated into the per-worker buffer.
function ad_state_vjp!(gₑ, λₑ, ws, statesₑ, pₑ, ctx)
    rd = ws.ad.re_dual
    fscalar = u -> begin
        evaluate_cell_residual!(rd, merge(statesₑ, (u = u,)), ws, pₑ, ctx)
        return dot(λₑ, rd)
    end
    ForwardDiff.gradient!(gₑ, fscalar, statesₑ.u, ws.ad.grad_cfg, Val{false}())
    return gₑ
end
