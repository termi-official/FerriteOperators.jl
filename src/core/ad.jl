####################################
## AD fallbacks (ForwardDiff over the residual kernel)
####################################
#
# Slice limitations, lifted with the facet-driver rework and the setup-declared
# request kinds:
# - AD sweeps cover the volumetric kernel only; boundary contributions keep
#   flowing through the analytic facet path. Sensitivities of parameter- or
#   time-dependent boundary terms are NOT captured yet.
# - ForwardDiff configs are rebuilt per call. FIXME preallocate per worker.

# Evaluate the volumetric local residual at (statesₑ, pₑ), overwriting `r`.
function evaluate_cell_residual!(r, statesₑ, ws, pₑ, ctx)
    fill!(r, zero(eltype(r)))
    assemble_cell!(ResidualRequest(r), ws.element, KernelArgs(statesₑ, ws.cell, pₑ, ws.scratch, ctx))
    return r
end

# ∂F/∂u — writes the Jacobian into K and the primal residual into ws.re. The
# `:u` slot is seeded; every other slot stays at its primal value.
function ad_state_jacobian!(K, ws, statesₑ, pₑ, ctx)
    f! = (r, u) -> evaluate_cell_residual!(r, merge(statesₑ, (u = u,)), ws, pₑ, ctx)
    ForwardDiff.jacobian!(K, f!, ws.re, statesₑ.u)
    return K
end

# ∂F/∂θ — dense local parameter Jacobian. The parameter sweep re-queries the
# element parameters from the Dual-rebuilt global `p` so wrappers forward
# Duals transparently.
function ad_parameter_jacobian!(Bₑ, ws, statesₑ, p, ctx)
    θ = Vector(parameter_vector(p))
    f! = (r, θᵢ) -> begin
        pₑ = query_cell_parameters(ws.element, ws.cell, rebuild_parameters(p, θᵢ))
        evaluate_cell_residual!(r, statesₑ, ws, pₑ, ctx)
    end
    ForwardDiff.jacobian!(Bₑ, f!, ws.re, θ)
    return Bₑ
end

# (∂F/∂θ)ᵀλₑ — adjoint pullback as the gradient of the scalar λₑ·rₑ(θ).
function ad_parameter_vjp!(gₑ, λₑ, ws, statesₑ, p, ctx)
    θ = Vector(parameter_vector(p))
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
# channel. The ctx-based seed (`with_time`) replaces this in phase 2.
function ad_time_sensitivity!(gₑ, ws, statesₑ, t::Real, ctx)
    f! = (r, t̃) -> begin
        pₑ = query_cell_parameters(ws.element, ws.cell, t̃)
        evaluate_cell_residual!(r, statesₑ, ws, pₑ, ctx)
    end
    ForwardDiff.derivative!(gₑ, f!, ws.re, t)
    return gₑ
end

function ad_time_sensitivity!(gₑ, ws, statesₑ, t, ctx)
    throw(ArgumentError(
        "Time sensitivities currently require the bare evaluation time as the parameter " *
        "object (got $(typeof(t))). Context-based seeding lands with the phase-2 API."))
end
