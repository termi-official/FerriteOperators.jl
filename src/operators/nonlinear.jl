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
# `slot_components` is operator-owned scratch, not payload: one `J`-patterned
# matrix per slot ever combined through `assemble_weighted_jacobian!`, so a
# repeated evaluation allocates nothing.
LinearizedFerriteOperator(J, engine, integrator) =
    LinearizedFerriteOperator(J, engine, integrator, Dict{Symbol, typeof(J)}())

# Interface; the u-vector forms are the stationary conveniences described in
# `AbstractNonlinearOperator`.
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
in particular a member of an [`allocate_components`](@ref) bag, so the
multi-slot linearization `Σ wₛ ∂F/∂s` is assembled slot by slot and folded with
[`combine!`](@ref). `op.J` is untouched unless passed as `J`.

`:q` is not a slot here — ∂F/∂q is field × internal-shaped, not square, and has
its own target and entry point ([`update_internal_jacobian!`](@ref)).
"""
assemble_slot_jacobian!(J::AbstractMatrix, op::LinearizedFerriteOperator, kind::JacobianKind, states::NamedTuple, p, ctx) =
    assemble_into!(kind, (J,), op, states, p, ctx)
assemble_slot_jacobian!(J::AbstractMatrix, op::LinearizedFerriteOperator, ::JacobianKind{:q}, states::NamedTuple, p, ctx) =
    throw(ArgumentError(
        "∂F/∂q is not a square slot Jacobian: `q` lives in the condensed `[ū; q]` tail, so the " *
        "block is `residual_size(op) × ndofs(op.engine.ivh)` and cannot be assembled into a " *
        "matrix sharing the operator's pattern. Use `update_internal_jacobian!(Kq, op, states, " *
        "p, ctx)` with a target from `allocate_internal_jacobian(op)`."))

"""
    assemble_weighted_jacobian!(W, op, weights::NamedTuple, states, p, ctx)

Assemble the weighted Jacobian `W = Σₛ weights[s] · ∂F/∂s` — the matrix a
scheme solves with — over the slots `weights` names, at frozen values of every
other slot. `weights` are chain-rule scalars (`(u = 1.0, du = 1/(γΔt))` for
SDIRK/backward Euler). `W` must share the operator's sparsity pattern: `op.J`,
a member of [`allocate_components`](@ref), or [`share_pattern`](@ref)
(`ComplexF64` for a complex combination).

Two routes give the same matrix to round-off ([`check_derivatives`](@ref)
verifies); which one runs is a capability of the element caches, not a caller
choice:

- **fused** — one [`WeightedJacobianKind`](@ref) sweep: the analytic
  `WeightedJacobianRequest` kernel where declared, else the residual kernel
  with every participating slot seeded by weight-scaled Duals.
- **composed** — one [`assemble_slot_jacobian!`](@ref) sweep per slot into
  components sharing `W`'s pattern, folded by [`combine!`](@ref). Runs for
  complex weights (the Dual machinery is real — transformed Radau needs this)
  and where condensed internal state makes the fused route inadmissible; its
  per-slot sweeps carry their own guards, so the weighted kind is servable
  exactly when every participating [`JacobianKind`](@ref) is.

A [`facet_items`](@ref) term is its own traversal and takes neither route: it
serves the weighted kind through its own fused
`assemble_facet!(::WeightedJacobianRequest, …)` kernel or not at all
([`assert_facet_item_route`](@ref)).
"""
function assemble_weighted_jacobian!(W::AbstractMatrix, op::LinearizedFerriteOperator, weights::NamedTuple, states::NamedTuple, p, ctx)
    kind = WeightedJacobianKind(weights)
    return _fused_weighted_route(op, kind) ?
        _weighted_jacobian_fused!(W, op, kind, states, p, ctx) :
        _weighted_jacobian_composed!(W, op, kind, states, p, ctx)
end

# The fused sweep needs real weights, and needs every cache to either serve the
# kind analytically or be safe to differentiate: a `Consistent` AD fallback
# would silently drop a condensed cache's ∂F/∂q·dq/d· correction, and the fused
# sweep seeds ALL participating slots at once (`assert_sensitivity_admissible`'s
# rule, applied to this kind). A `FrozenQ` election needs no such guard: the AD
# fallback IS the requested partial, same as for `JacobianKind`.
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

# Call-time admissibility over all subdomain caches; a declared kind runs the
# same per-cache check at setup (rationale: `assert_sensitivity_admissible`).
# Family-dispatched, so an algebraic subdomain's error names
# `assemble_algebraic!`, not `assemble_cell!`.
function _check_sensitivity_supported(op, kind)
    for sc in op.engine.subdomain_caches
        _assert_domain_sensitivity_admissible(sc.domain, kind)
    end
    return nothing
end

"""
    update_parameter_jacobian!(B, op, u, p)

Assemble the parameter Jacobian ∂F/∂θ into `B` (`residual_size(op) × nθ`) at
the trial state `u`, never writing back into `u`. θ is the flat parameter view
defined by [`parameter_vector`](@ref)/[`rebuild_parameters`](@ref); elements
provide analytic [`ParameterJacobianRequest`](@ref) kernels or fall back to AD
of their residual.

!!! note
    The u-vector forms evaluate at `states = (u = u,)` with no
    time-integration context. Elements reading further slots (`uprev`, …) or
    `args.ctx` must use the states/ctx forms.
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
        @warn "ColoredScheduling provides no isolation for parameter-space " *
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

Matrix-free action of the state Jacobian: `Jv = (∂F/∂u)·v` at the trial state,
computed kernel-level (one directional-Dual sweep per cell for the AD fallback;
analytic [`StateJVPRequest`](@ref) kernels win per cache). Never writes back
into the caller's state. Requires `unknown_size(op) == residual_size(op)`, i.e.
no condensed unknowns; anything else is an `ArgumentError`.
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
kernels win per cache). Never writes back into the caller's state, and
requires no condensed unknowns, like [`state_jvp!`](@ref).
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

Central-difference derivative method, accuracy O(h²): `(F(t+h) − F(t−h)) / 2h`
from PRIMAL residual evaluations at contexts carrying perturbed evaluation
times, on a protected copy of `u` so trial write-back never leaks. No Dual
propagation, hence admissible for condensed elements: it condenses AT each
perturbed context and so yields the TOTAL t-derivative at fixed `u`, the
element-local state's response included. The operator's corrector stores are
left holding the last perturbed point's condensation.

The operator-level counterpart of the [`ADElementCache`](@ref) decorator, and
the split is final: this is the only BOUNDARY-INCLUSIVE route (it differences
`evaluate!`, facet terms included, rather than an element kernel) and the only
Dual-free one, and being an operator-level OVERRIDE it bypasses analytic
sensitivity kernels everywhere.
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
only, so `ctx` is mandatory here.

With [`ADSensitivity`](@ref) (default) an analytic
[`TimeSensitivityRequest`](@ref) kernel wins per cache, and every other cache
falls back to ForwardDiff of its residual kernel over a Dual-timed context.
[`FiniteDifferenceSensitivity`](@ref) is an operator-level override that
BYPASSES analytic sensitivity kernels; prefer it only where AD is inadmissible
(condensed internal state without analytic kernels or insensitivity
declarations).
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
    # A condensed operator is re-condensed AT each perturbed context, not once
    # ahead of both: a local model reading `evaluation_time` has its own
    # t-dependence, and differencing at a frozen `q` would silently return the
    # partial where this method's contract is the total. `u` never changes —
    # only the context time and the `q` each condensation writes for it.
    t  = evaluation_time(ctx)
    h  = method.h * max(one(t), abs(t))
    uw = copy(states.u)
    statesw = merge(states, (u = uw,))
    condensed = unknown_size(op) > residual_size(op) && haskey(states, :q) && states.q isa InternalSource
    condensed && (statesw = merge(statesw, (q = InternalSource(uw),)))
    _evaluate_at_time!(r, ctxt) = begin
        condensed && condense_internal!(op, statesw, p, ctxt)
        evaluate!(op, r, statesw, p, ctxt)
    end
    rp = similar(g); _evaluate_at_time!(rp, with_time(ctx, t + h))
    rm = similar(g); _evaluate_at_time!(rm, with_time(ctx, t - h))
    g .= (rp .- rm) ./ (2h)
    return g
end

"""
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β)

Apply the (scaled) action of the assembled linearization to the vector `in`.
"""
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)
operator_payload(op::LinearizedFerriteOperator) = op.J

"""
    residual_size(op) -> Int
    unknown_size(op) -> Int

The two lengths an operator's entry points size their arguments by:
`residual_size` is the number of FE dofs (rows of `F`, length of a residual or
adjoint vector); `unknown_size` adds the condensed internal tail (`[ū; q]`,
length of a solution vector). They coincide exactly when the operator carries
no condensed element.
"""
residual_size(op::LinearizedFerriteOperator) = ndofs(op.engine.dh)

@doc (@doc residual_size)
unknown_size(op::LinearizedFerriteOperator)  = ndofs(op.engine.dh) + ndofs(op.engine.ivh)
