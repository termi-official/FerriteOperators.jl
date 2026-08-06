####################################
## AD fallbacks (ForwardDiff over the residual kernel)
####################################
#
# Slice-1 limitations, lifted with the facet-driver rework:
# - AD sweeps cover the volumetric kernel only; boundary contributions keep
#   flowing through the analytic legacy facet path. Sensitivities of
#   parameter- or time-dependent boundary terms are NOT captured yet.
# - ForwardDiff configs are rebuilt per call. FIXME preallocate per worker.

# Evaluate the volumetric local residual at (uₑ, pₑ), overwriting `r`.
function evaluate_cell_residual!(r, uₑ, ws, pₑ)
    fill!(r, zero(eltype(r)))
    cache = ws.element
    if implements_v2_kernels(typeof(cache))
        assemble_cell!(ResidualRequest(r), cache, KernelArgs((u = uₑ,), ws.cell, pₑ, nothing, nothing))
    else
        assemble_element!(r, uₑ, ws.cell, cache, pₑ)
    end
    return r
end

# ∂F/∂u — writes the Jacobian into K and the primal residual into ws.re.
function ad_state_jacobian!(K, ws, uₑ, pₑ)
    f! = (r, u) -> evaluate_cell_residual!(r, u, ws, pₑ)
    ForwardDiff.jacobian!(K, f!, ws.re, uₑ)
    return K
end

# ∂F/∂θ — dense local parameter Jacobian. The parameter sweep re-queries the
# element parameters from the Dual-rebuilt global `p` so wrappers forward
# Duals transparently.
function ad_parameter_jacobian!(Bₑ, ws, uₑ, p)
    θ = Vector(parameter_vector(p))
    f! = (r, θᵢ) -> begin
        pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, rebuild_parameters(p, θᵢ))
        evaluate_cell_residual!(r, uₑ, ws, pₑ)
    end
    ForwardDiff.jacobian!(Bₑ, f!, ws.re, θ)
    return Bₑ
end

# (∂F/∂θ)ᵀλₑ — adjoint pullback as the gradient of the scalar λₑ·rₑ(θ).
function ad_parameter_vjp!(gₑ, λₑ, ws, uₑ, p)
    θ = Vector(parameter_vector(p))
    fscalar = θᵢ -> begin
        pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, rebuild_parameters(p, θᵢ))
        r = zeros(eltype(θᵢ), length(λₑ))
        evaluate_cell_residual!(r, uₑ, ws, pₑ)
        return dot(λₑ, r)
    end
    ForwardDiff.gradient!(gₑ, fscalar, θ)
    return gₑ
end

# ∂F/∂t — explicit time dependence, seeded through the bare-time parameter
# channel. The ctx-based seed (`with_time`) replaces this in phase 2.
function ad_time_sensitivity!(gₑ, ws, uₑ, t::Real)
    f! = (r, t̃) -> begin
        pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, t̃)
        evaluate_cell_residual!(r, uₑ, ws, pₑ)
    end
    ForwardDiff.derivative!(gₑ, f!, ws.re, t)
    return gₑ
end

function ad_time_sensitivity!(gₑ, ws, uₑ, t)
    throw(ArgumentError(
        "Time sensitivities currently require the bare evaluation time as the parameter " *
        "object (got $(typeof(t))). Context-based seeding lands with the phase-2 API."))
end
