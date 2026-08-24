"""
    LinearizedFerriteOperator(J, engine, integrator)

A model for a function with its fully assembled linearization. Entry points
route request kinds through the shared assembly engine; element kernels are
the request-typed `assemble_cell!`/`assemble_facet!` methods.
"""
@concrete struct LinearizedFerriteOperator <: AbstractNonlinearOperator
    J
    engine
    integrator
    slot_components   # lazily built per-slot matrices of the composed weighted route
end
# `slot_components` is operator-owned scratch, not payload: it holds one matrix
# per slot ever combined through `assemble_weighted_jacobian!`, each sharing
# `J`'s sparsity pattern, so a repeated evaluation allocates nothing.
LinearizedFerriteOperator(J, engine, integrator) =
    LinearizedFerriteOperator(J, engine, integrator, Dict{Symbol, typeof(J)}())

# Interface. The states/ctx forms are canonical; the u-vector forms are
# conveniences for stationary problems (states = (u = u,), no context).
update_linearization!(op::LinearizedFerriteOperator, states::NamedTuple, p, ctx) =
    assemble_into!(JacobianKind(), (op.J,), op, states, p, ctx)
update_linearization!(op::LinearizedFerriteOperator, u::AbstractVector, p) =
    update_linearization!(op, (u = u,), p, nothing)
update_linearization!(op::LinearizedFerriteOperator, residual::AbstractVector, states::NamedTuple, p, ctx) =
    assemble_into!(JacobianResidualKind(), (op.J, residual), op, states, p, ctx)
update_linearization!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    update_linearization!(op, residual, (u = u,), p, nothing)
evaluate!(op::LinearizedFerriteOperator, residual::AbstractVector, states::NamedTuple, p, ctx) =
    assemble_into!(ResidualKind(), (residual,), op, states, p, ctx)
evaluate!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    evaluate!(op, residual, (u = u,), p, nothing)

"""
    assemble_slot_jacobian!(J, op, kind::JacobianKind, states, p, ctx)

Assemble ∂F/∂slot into `J`, where `kind` names the slot
([`JacobianKind`](@ref)). `J` is any matrix the operator's assembler accepts —
in particular a member of a component bag from
[`allocate_components`](@ref) — so the multi-slot linearization
`Σ wₛ ∂F/∂s` is assembled once per slot and folded with [`combine!`](@ref).
`op.J` is untouched unless it is passed as `J`.
"""
assemble_slot_jacobian!(J::AbstractMatrix, op::LinearizedFerriteOperator, kind::JacobianKind, states::NamedTuple, p, ctx) =
    assemble_into!(kind, (J,), op, states, p, ctx)

"""
    assemble_weighted_jacobian!(W, op, weights::NamedTuple, states, p, ctx)

Assemble the weighted Jacobian `W = Σₛ weights[s] · ∂F/∂s` — the matrix a
scheme solves with — over the slots `weights` names, at frozen values of every
other slot. `weights` are the solver's chain-rule scalars
(`(u = 1.0, du = 1/(γΔt))` for SDIRK/backward Euler).

Two routes produce the same matrix from the same weights, and which one runs
is a capability of the operator's element caches, not a caller choice:

- **fused** — one sweep of [`WeightedJacobianKind`](@ref): the element's
  analytic `WeightedJacobianRequest` kernel where declared, otherwise the
  residual kernel with every participating slot seeded by its weight-scaled
  Duals.
- **composed** — one [`assemble_slot_jacobian!`](@ref) sweep per slot into
  operator-held components sharing `W`'s pattern, folded by [`combine!`](@ref)
  with the very same `weights`.

The composed route runs for complex weights (the element matrix and the Dual
machinery are real — this is what transformed Radau needs) and for caches whose
condensed internal state makes the AD-seeded fused route inadmissible; there
the per-slot sweeps apply their own guards, so the weighted kind is servable
exactly when every participating [`JacobianKind`](@ref) is. Both routes agree
to round-off, which [`check_derivatives`](@ref) verifies.

`W` must share the operator's sparsity pattern — use `op.J`, a member of
[`allocate_components`](@ref), or [`share_pattern`](@ref) (with `ComplexF64`
for a complex combination).
"""
function assemble_weighted_jacobian!(W::AbstractMatrix, op::LinearizedFerriteOperator, weights::NamedTuple, states::NamedTuple, p, ctx)
    kind = WeightedJacobianKind(weights)
    return _fused_weighted_route(op, kind) ?
        _weighted_jacobian_fused!(W, op, kind, states, p, ctx) :
        _weighted_jacobian_composed!(W, op, kind, states, p, ctx)
end

# The fused sweep needs real weights, and needs every cache to either serve the
# kind analytically or be safe to differentiate: a `Consistent` AD fallback
# would silently drop a condensed cache's ∂F/∂q·dq/d· correction, and the
# fused sweep seeds ALL participating slots at once, so a condensed cache
# without the analytic weighted kernel has no admissible fused route
# (`assert_sensitivity_admissible`'s rule, applied to this kind). A `FrozenQ`
# election needs no such guard: the AD fallback IS the requested partial (the
# kernel it differentiates is pure at frozen `q`), same as for `JacobianKind`.
function _fused_weighted_route(op, kind::WeightedJacobianKind{slots, C}) where {slots, C}
    all(w -> w isa Real, values(kind.weights)) || return false
    C === FrozenQ && return true
    return all(op.engine.subdomain_caches) do sc
        T = typeof(sc.domain.element)
        provides_analytic(T, kind) || !has_internal_state(T) || internal_state_insensitive(T, kind)
    end
end

_weighted_jacobian_fused!(W, op, kind::WeightedJacobianKind, states, p, ctx) =
    assemble_into!(kind, (W,), op, states, p, ctx)

function _weighted_jacobian_composed!(W, op, kind::WeightedJacobianKind{slots}, states, p, ctx) where {slots}
    comps = _slot_component_bag!(op, slots)
    ntuple(i -> assemble_slot_jacobian!(comps[i], op, JacobianKind{slots[i]}(), states, p, ctx), Val(length(slots)))
    combine!(W, comps, kind.weights)
    return W
end

function _slot_component_bag!(op::LinearizedFerriteOperator, slots::NTuple{N, Symbol}) where {N}
    store = op.slot_components
    for slot in slots
        haskey(store, slot) || (store[slot] = share_pattern(op.J))
    end
    return NamedTuple{slots}(ntuple(i -> store[slots[i]], Val(N)))
end

# Call-time admissibility over all subdomain caches; the same per-cache check
# runs at setup for kinds declared via `setup_operator(...; requests)` — see
# `assert_sensitivity_admissible` for the rationale. Family-dispatched through
# `_assert_domain_sensitivity_admissible` so an algebraic subdomain's error
# names `assemble_algebraic!`, not `assemble_cell!`.
function _check_sensitivity_supported(op, kind)
    for sc in op.engine.subdomain_caches
        _assert_domain_sensitivity_admissible(sc.domain, kind)
    end
    return nothing
end

"""
    update_parameter_jacobian!(B, op, u, p)

Assemble the parameter Jacobian ∂F/∂θ into `B` (`residual_size(op) × nθ`),
evaluated at the trial state `u`. θ is the flat parameter view defined by
[`parameter_vector`](@ref)/[`rebuild_parameters`](@ref); elements provide
analytic [`ParameterJacobianRequest`](@ref) kernels or fall back to AD of
their residual. Never writes back into `u`.

!!! note
    The u-vector convenience forms evaluate at the stationary point
    `states = (u = u,)` with no time-integration context. Elements reading
    further slots (`uprev`, …) or `args.ctx` must use the states/ctx forms.
"""
function update_parameter_jacobian!(B::AbstractMatrix, op::LinearizedFerriteOperator, states::NamedTuple, p, ctx)
    _check_sensitivity_supported(op, ParameterJacobianKind())
    nθ = length(parameter_vector(p))
    size(B) == (residual_size(op), nθ) || throw(DimensionMismatch(
        "expected B of size $((residual_size(op), nθ)), got $(size(B))"))
    fill!(B, zero(eltype(B)))
    assembler = ParameterJacobianAssembler{eltype(B), typeof(B), dof_scatter_needs_atomic(op.engine.strategy)}(B)
    run_sweep!(ParameterJacobianKind(), assembler, op, states, p, ctx)
    return B
end
update_parameter_jacobian!(B::AbstractMatrix, op::LinearizedFerriteOperator, u::AbstractVector, p) =
    update_parameter_jacobian!(B, op, (u = u,), p, nothing)

"""
    parameter_vjp!(g, op, λ, u, p)

Accumulate the adjoint pullback `g = (∂F/∂θ)ᵀ λ` (length nθ) at the trial
state `u` without materializing ∂F/∂θ. Never writes back into `u`.
"""
function parameter_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, states::NamedTuple, p, ctx)
    _check_sensitivity_supported(op, ParameterVJPKind(λ))
    length(g) == length(parameter_vector(p)) || throw(DimensionMismatch(
        "expected g of length $(length(parameter_vector(p))), got $(length(g))"))
    length(λ) == residual_size(op) || throw(DimensionMismatch(
        "expected λ of length $(residual_size(op)), got $(length(λ))"))
    fill!(g, zero(eltype(g)))
    atomic = parameter_scatter_needs_atomic(op.engine.strategy)
    if atomic && op.engine.strategy.scheduling isa ColoredScheduling
        @warn "PerColorAssemblyStrategy provides no isolation for parameter-space " *
              "accumulation; the VJP scatter falls back to atomic adds." maxlog = 1
    end
    assembler = ParameterVJPAssembler{eltype(g), typeof(g), atomic}(g)
    run_sweep!(ParameterVJPKind(λ), assembler, op, states, p, ctx)
    return g
end
parameter_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, u::AbstractVector, p) =
    parameter_vjp!(g, op, λ, (u = u,), p, nothing)

"""
    state_jvp!(Jv, op, v, states, p, ctx)
    state_jvp!(Jv, op, v, u, p)

Matrix-free action of the state Jacobian: `Jv = (∂F/∂u)·v` at the trial
state, computed kernel-level (one directional-Dual sweep per cell for the AD
fallback; analytic [`StateJVPRequest`](@ref) kernels win per cache) — no
matrix is materialized anywhere. Never writes back into the caller's state.
Restricted to operators without condensed unknowns for now
(`unknown_size == residual_size`).
"""
function state_jvp!(Jv::AbstractVector, op::LinearizedFerriteOperator, v::AbstractVector, states::NamedTuple, p, ctx)
    unknown_size(op) == residual_size(op) || throw(ArgumentError(
        "state_jvp!/state_vjp! are not yet supported for operators with condensed unknowns."))
    _check_sensitivity_supported(op, StateJVPKind(v))
    length(Jv) == residual_size(op) || throw(DimensionMismatch(
        "expected Jv of length $(residual_size(op)), got $(length(Jv))"))
    length(v) == unknown_size(op) || throw(DimensionMismatch(
        "expected v of length $(unknown_size(op)), got $(length(v))"))
    assemble_into!(StateJVPKind(v), (Jv,), op, states, p, ctx)
    return Jv
end
state_jvp!(Jv::AbstractVector, op::LinearizedFerriteOperator, v::AbstractVector, u::AbstractVector, p) =
    state_jvp!(Jv, op, v, (u = u,), p, nothing)

"""
    state_vjp!(g, op, λ, states, p, ctx)
    state_vjp!(g, op, λ, u, p)

Matrix-free pullback of the state Jacobian: `g = (∂F/∂u)ᵀλ` at the trial
state — the action adjoint time stepping applies. Kernel-level (per-cell
gradient of `λₑ·rₑ` for the AD fallback; analytic [`StateVJPRequest`](@ref)
kernels win per cache). Never writes back into the caller's state. Restricted
to operators without condensed unknowns for now.
"""
function state_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, states::NamedTuple, p, ctx)
    unknown_size(op) == residual_size(op) || throw(ArgumentError(
        "state_jvp!/state_vjp! are not yet supported for operators with condensed unknowns."))
    _check_sensitivity_supported(op, StateVJPKind(λ))
    length(g) == unknown_size(op) || throw(DimensionMismatch(
        "expected g of length $(unknown_size(op)), got $(length(g))"))
    length(λ) == residual_size(op) || throw(DimensionMismatch(
        "expected λ of length $(residual_size(op)), got $(length(λ))"))
    assemble_into!(StateVJPKind(λ), (g,), op, states, p, ctx)
    return g
end
state_vjp!(g::AbstractVector, op::LinearizedFerriteOperator, λ::AbstractVector, u::AbstractVector, p) =
    state_vjp!(g, op, λ, (u = u,), p, nothing)

"""
    ADSensitivity()

Default derivative method: ForwardDiff seeding through the residual kernel.
"""
struct ADSensitivity end

"""
    FiniteDifferenceSensitivity(h = cbrt(eps(Float64)))

Central-difference derivative method: evaluates the PRIMAL residual at
contexts carrying perturbed evaluation times, on a protected copy of `u` (so
trial write-back never leaks), and forms `(F(t+h) − F(t−h)) / 2h`. Exact local
solves, no Dual propagation — admissible for condensed elements, where it
yields the total t-derivative at fixed `u` including the element-local state's
response (as does AD, when admissible). Accuracy O(h²).
"""
struct FiniteDifferenceSensitivity{T}
    h::T
end
FiniteDifferenceSensitivity() = FiniteDifferenceSensitivity(cbrt(eps(Float64)))

"""
    time_sensitivity!(g, op, states, p, ctx; method = ADSensitivity())

Assemble the time sensitivity ∂F/∂t into `g` (`residual_size(op)`) at the
trial state, never writing back into the caller's state. The evaluation time
is `evaluation_time(ctx)` — time reaches elements through the context channel
only, so `ctx` is mandatory here and `p` carries user parameters as in every
other entry point.

Method hierarchy: with [`ADSensitivity`](@ref) (default), each element cache
that declares an analytic [`TimeSensitivityRequest`](@ref) kernel is used
directly, and every other cache falls back to ForwardDiff of its residual
kernel over a Dual-timed context — analytic kernels always win per cache.
[`FiniteDifferenceSensitivity`](@ref) is an operator-level override that
differences primal residual evaluations and therefore BYPASSES analytic
sensitivity kernels; prefer it only where AD is inadmissible (condensed
internal state without analytic kernels or insensitivity declarations).
"""
function time_sensitivity!(g::AbstractVector, op::LinearizedFerriteOperator, states::NamedTuple, p, ctx; method = ADSensitivity())
    length(g) == residual_size(op) || throw(DimensionMismatch(
        "expected g of length $(residual_size(op)), got $(length(g))"))
    ctx === nothing && throw(ArgumentError(
        "∂F/∂t seeds through the context channel, so `time_sensitivity!` needs a context " *
        "and got `nothing`. Pass a `TimeIntegrationContext(t, Δt, γ̃)` (or a custom context " *
        "type implementing `evaluation_time`/`with_time`) as the last positional argument."))
    return _time_sensitivity!(method, g, op, states, p, ctx)
end

function _time_sensitivity!(::ADSensitivity, g, op, states, p, ctx)
    _check_sensitivity_supported(op, TimeSensitivityKind())
    assemble_into!(TimeSensitivityKind(), (g,), op, states, p, ctx)
    return g
end

function _time_sensitivity!(method::FiniteDifferenceSensitivity, g, op, states, p, ctx)
    # Primal evaluations at perturbed contexts — a pure evaluation sweep
    # writes nothing back, so `u` (and, once condensed, `q`) stay fixed across
    # both calls; `uw` only protects the caller's `states.u` from aliasing.
    # `u` itself never changes (only the context time does), so one
    # condensation ahead of both evaluations covers both.
    t  = evaluation_time(ctx)
    h  = method.h * max(one(t), abs(t))
    uw = copy(states.u)
    statesw = merge(states, (u = uw,))
    if unknown_size(op) > residual_size(op) && haskey(states, :q) && states.q isa InternalSource
        statesw = merge(statesw, (q = InternalSource(uw),))
        condense_internal!(op, statesw, p, ctx)
    end
    rp = similar(g); evaluate!(op, rp, statesw, p, with_time(ctx, t + h))
    rm = similar(g); evaluate!(op, rm, statesw, p, with_time(ctx, t - h))
    g .= (rp .- rm) ./ (2h)
    return g
end

"""
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear operator to the vector `in`.
"""
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)
(op::LinearizedFerriteOperator)(residual, u, p) = evaluate!(op, residual, u, p)
Base.eltype(op::LinearizedFerriteOperator) = eltype(op.J)
Base.size(op::LinearizedFerriteOperator) = size(op.J)
Base.size(op::LinearizedFerriteOperator, axis) = size(op.J, axis)

residual_size(op::LinearizedFerriteOperator) = ndofs(op.engine.dh)
unknown_size(op::LinearizedFerriteOperator)  = ndofs(op.engine.dh) + ndofs(op.engine.ivh)
