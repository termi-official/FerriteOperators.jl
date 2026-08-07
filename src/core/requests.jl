####################################
## Assembly requests (element interface v2)
####################################

"""
    TimeIntegrationContext(t, Δt, γ̃)

Solver-controlled scalars the framework must understand. `t` is the evaluation
time (Dual-typed during ∂F/∂t sweeps), `Δt` the physical step size for
reference, and `γ̃` the effective local stage interval of the element-local
internal-variable problem. Its NORMALIZATION is fixed by the canonical form

    q = q_ref + γ̃ · g(·, q)

— i.e. a solver passing `γ̃` means "an implicit-Euler local integrator would
solve exactly this" (the reference state `q_ref` is dof-shaped and flows
through a slot, solver-folded for multistep schemes). The canonical form
normalizes the *number*, it does not prescribe the element's local rule: the
element owns its local integrator and may realize the update with any
consistent rule over `γ̃` — implicit Euler, exact exponential
(Rush-Larsen/EME-type `q = q∞ + (q_ref − q∞)·exp(−γ̃/τ)`), or local
substepping.

!!! warning
    `γ̃` is NOT the rate-reconstruction slope of any state slot. Under
    backward Euler the two happen to be reciprocals (`slope = 1/Δt`,
    `γ̃ = Δt`), so writing `1/γ̃` for a rate slope is accidentally right under
    BE and silently wrong under every other scheme (Newmark:
    `slope = γ/(βΔt)` while `γ̃ = Δt`). Rate slopes belong to the slot that
    carries the reconstruction, never to the context.
"""
struct TimeIntegrationContext{T}
    t::T
    Δt::T
    γ̃::T
end
function TimeIntegrationContext(t, Δt, γ̃)
    tp, Δtp, γ̃p = promote(t, Δt, γ̃)
    return TimeIntegrationContext{typeof(tp)}(tp, Δtp, γ̃p)
end
# Deliberately no 1-arg convenience constructor: a defaulted γ̃ (e.g. zero)
# would silently break local internal-variable stage problems that scale by
# it. All three scalars are the solver's explicit statement.

# Framework code touches contexts only through these accessors, so schemes
# with richer per-sweep scalars can pass their own context types.
"Evaluation time of the sweep this context belongs to."
evaluation_time(ctx::TimeIntegrationContext) = ctx.t
"Rebuild `ctx` with the evaluation time replaced by `t̃` (Dual-typed in ∂F/∂t sweeps)."
with_time(ctx::TimeIntegrationContext, t̃) = TimeIntegrationContext(t̃, oftype(t̃, ctx.Δt), oftype(t̃, ctx.γ̃))

"""
    AbstractAssemblyRequest

What a kernel is asked to compute for one cell. Kernels dispatch on the
request type, never on argument shapes, so kernel sets for different state
slots can coexist without ambiguity.
"""
abstract type AbstractAssemblyRequest end

"Accumulate the local residual into `r`."
struct ResidualRequest{V <: AbstractVector} <: AbstractAssemblyRequest
    r::V
end

"Accumulate ∂F/∂slot into `K`, slot ∈ (:u, :du, :v, :a, …)."
struct JacobianRequest{slot, M <: AbstractMatrix} <: AbstractAssemblyRequest
    K::M
end
JacobianRequest{slot}(K::M) where {slot, M <: AbstractMatrix} = JacobianRequest{slot, M}(K)

"Accumulate ∂F/∂u and the residual in one sweep (the Newton hot path)."
struct JacobianResidualRequest{M <: AbstractMatrix, V <: AbstractVector} <: AbstractAssemblyRequest
    K::M
    r::V
end

"Accumulate the dense local parameter Jacobian ∂Fₑ/∂θ into `B` (ndofsₑ × nθ)."
struct ParameterJacobianRequest{M <: AbstractMatrix} <: AbstractAssemblyRequest
    B::M
end

"Accumulate the local adjoint pullback (∂Fₑ/∂θ)ᵀλₑ into `g` (length nθ)."
struct ParameterVJPRequest{V <: AbstractVector, L <: AbstractVector} <: AbstractAssemblyRequest
    g::V
    λₑ::L
end

"Accumulate the explicit time sensitivity ∂Fₑ/∂t into `g`."
struct TimeSensitivityRequest{V <: AbstractVector} <: AbstractAssemblyRequest
    g::V
end

"Accumulate the state Jacobian-vector product (∂Fₑ/∂u)·vₑ into `Jv` (matrix-free J action)."
struct StateJVPRequest{W <: AbstractVector, V <: AbstractVector} <: AbstractAssemblyRequest
    Jv::W
    vₑ::V
end

"Accumulate the state pullback (∂Fₑ/∂u)ᵀλₑ into `g` (matrix-free Jᵀ action)."
struct StateVJPRequest{G <: AbstractVector, L <: AbstractVector} <: AbstractAssemblyRequest
    g::G
    λₑ::L
end

"""
    AffineRate(slope, anchor)

Slot *source* reconstructing a rate-like slot from the primary unknown: the
slot's cell-local value is `slope · (u − anchor)`, formed at gather time from
the `:u` slot. Solvers pass it in place of a plain vector, e.g. backward
Euler `states = (u = u, du = AffineRate(1/Δt, uprev))` or Newmark
`states = (u = u, v = AffineRate(γ/(β*Δt), uᵥ))`. The slot is declared like
any other (`setup_operator(...; slots = (:u, :du))`).

A `:u` slot must exist and PRECEDE the reconstructed slot in the states
`NamedTuple` — the sweep throws otherwise. Kernels read the reconstructed
values through `args.states.<name>` and the slope through
[`slot_slope`](@ref), so an element never encodes a time-integration scheme.

!!! warning "Reconstructed slots are frozen under AD"
    Reconstruction happens at gather time, before any Dual seeding, and the
    ∂F/∂u sweep seeds only the `:u` buffer. A kernel reading a reconstructed
    slot therefore sees it at its primal value throughout the sweep, and the
    assembled Jacobian is ∂F/∂u at frozen slot values. The chain-rule term
    through the reconstruction (`slope · ∂F/∂slot`) belongs to the solver,
    which contributes it through its per-slot weights.

!!! note "Condensed elements"
    The reconstruction applies uniformly to ALL entries of the element
    buffer, because `u` and `anchor` are both `[ū; q]`-shaped gathers.
    Internal variables are never rate-reconstructed by the framework
    contract: the condensed tail of a reconstructed slot is an artifact of
    the uniform formula and elements must not interpret it.
"""
struct AffineRate{T, V <: AbstractVector}
    slope::T
    anchor::V
end

"""
    KernelArgs(states, cell, p, scratch, ctx)
    KernelArgs(states, cell, p, scratch, ctx, slot_meta)

The argument bundle passed to v2 element kernels. `states` is a `NamedTuple`
of cell-local slot buffers (e.g. `(u = uₑ,)`), `cell` the read-only geometry
cache, `p` the user parameter bag, `scratch` per-worker scratch space, `ctx`
the [`TimeIntegrationContext`](@ref) (or `nothing` while the legacy parameter
channel still carries time), and `slot_meta` the per-slot reconstruction
metadata queried through [`slot_slope`](@ref) (empty when no slot is
reconstructed).
"""
struct KernelArgs{States <: NamedTuple, Cell, P, Scratch, Ctx, Meta <: NamedTuple}
    states::States
    cell::Cell
    p::P
    scratch::Scratch
    ctx::Ctx
    slot_meta::Meta
end
KernelArgs(states, cell, p, scratch, ctx) = KernelArgs(states, cell, p, scratch, ctx, (;))

"""
    slot_slope(args::KernelArgs, name::Symbol)

The reconstruction linearization of the slot `name`: the [`AffineRate`](@ref)
slope for a reconstructed slot, `nothing` for plain vector slots and for
names the sweep does not carry. This is slot metadata, not scheme knowledge —
a condensed corrector that needs `∂v/∂u` inside its local tangent queries it
here instead of being told which time integrator is running.
"""
slot_slope(args::KernelArgs, name::Symbol) = get(args.slot_meta, name, nothing)

"Rebuild `args` with the slot buffers replaced — the AD seeding seam. Slot metadata is preserved."
with_states(args::KernelArgs, states::NamedTuple) =
    KernelArgs(states, args.cell, args.p, args.scratch, args.ctx, args.slot_meta)

"Rebuild `args` with the element-local parameter view replaced — the parameter/time seeding seam."
with_parameters(args::KernelArgs, p) =
    KernelArgs(args.states, args.cell, p, args.scratch, args.ctx, args.slot_meta)

"""
    assemble_cell!(req::AbstractAssemblyRequest, cache, args::KernelArgs)

The v2 volumetric kernel entry point. Elements must at least provide the
[`ResidualRequest`](@ref) method (validated at setup); every other request falls back to automatic
differentiation of the residual kernel unless [`provides_analytic`](@ref)
declares an analytic method.
"""
function assemble_cell! end

####################################
## Query protocol
####################################

"""
    unwrap_parameters(p) -> p′

Solver-side wrapper trait: a solver that must wrap the user parameter bag
defines one unwrapping rule here instead of every element handling the
wrapper. Defaults to identity.
"""
unwrap_parameters(p) = p

"""
    query_cell_parameters(cache, cell, p)

Element-overridable query producing the element-local parameter view `pₑ`
handed to volumetric kernels. The default applies [`unwrap_parameters`](@ref)
and passes the bag through. Parameter layouts (parameter fields) gather their
per-element views through this seam.
"""
query_cell_parameters(cache, cell, p) = unwrap_parameters(p)

"""
    query_facet_parameters(cache, cell, local_facet_index, p)

Facet analogue of [`query_cell_parameters`](@ref) — boundary caches get their
own parameter query per facet instead of reusing the volumetric object.
"""
query_facet_parameters(cache, cell, local_facet_index, p) = unwrap_parameters(p)

"""
    provides_analytic(::Type{CacheType}, kind) -> Bool

`true` iff the v2 element cache implements `assemble_cell!` analytically for
the given request *kind* singleton (`JacobianKind()`, `ParameterJacobianKind()`,
…). Everything except the mandatory residual defaults to `false`, i.e.
AD-from-residual.

There is deliberately exactly ONE root method (with the residual default as a
runtime branch): specializations are therefore always strictly more specific,
so blanket declarations like
`provides_analytic(::Type{<:MyCache}, kind) = true` cannot create dispatch
ambiguities, and there is no request-type-parameter matching to get subtly
wrong. Kernel/trait consistency is validated once at operator setup
([`validate_element_cache`](@ref)).
"""
provides_analytic(::Type, kind) = kind isa ResidualKind

"""
    validate_element_cache(cache, declared_requests = ())

Setup-time consistency check for v2 element caches: a cache that opts into the
request protocol must implement the mandatory [`ResidualRequest`](@ref)
kernel, and every request kind the [`provides_analytic`](@ref) trait claims
must have a matching kernel method. Runs once per subdomain at
`setup_operator` time — a typo'd port fails loudly here instead of silently
assembling through the wrong path.

The trait ↔ kernel check always covers the primal kinds (the operator will
issue them). Sensitivity kinds are checked only when declared via
`setup_operator(...; requests = (ParameterVJPKind, …))`; declared sensitivity
kinds additionally run the internal-state admissibility check
([`has_internal_state`](@ref)) at setup instead of on first use. Undeclared
kinds stay usable — their checks simply run at the call-time entry points.
"""
function validate_element_cache(cache, declared_requests::Tuple = ())
    T = typeof(cache)
    hasmethod(assemble_cell!, Tuple{ResidualRequest, T, KernelArgs}) || throw(ArgumentError(
        "$(T) implements no `assemble_cell!(::ResidualRequest, ::$(nameof(T)), ::KernelArgs)` " *
        "method. The residual kernel is mandatory: it is the basis for AD-derived Jacobians " *
        "and sensitivities."))
    hasmethod(reinit_values!, Tuple{T, CellCache}) || throw(ArgumentError(
        "$(T) implements no `reinit_values!(::$(nameof(T)), cell)` method. The engine " *
        "reinitializes element values once per cell and sweep through this hook; " *
        "kernels are pure evaluation and must not rely on reinit inside them."))
    for (kind, ReqT) in _primal_validatable_kinds()
        _assert_trait_backed(T, kind, ReqT)
    end
    for (kind, ReqT) in _sensitivity_validatable_kinds()
        any(D -> kind isa D, declared_requests) || continue
        _assert_trait_backed(T, kind, ReqT)
        # Declaring moves the admissibility failure from first use to setup.
        # Time sensitivities are exempt: their FD escape is chosen per call.
        kind isa TimeSensitivityKind || assert_sensitivity_admissible(T, kind)
    end
    return nothing
end

function _assert_trait_backed(T, kind, ReqT)
    if provides_analytic(T, kind) && !hasmethod(assemble_cell!, Tuple{ReqT, T, KernelArgs})
        throw(ArgumentError(
            "$(T) declares `provides_analytic` for $(typeof(kind)) but implements no " *
            "matching `assemble_cell!(::$(ReqT), ::$(nameof(T)), ::KernelArgs)` method."))
    end
    return nothing
end

# Kind instances paired with the request types their analytic kernels take.
# Payload-carrying kinds get placeholder payloads — only the type matters for
# the trait query.
_primal_validatable_kinds() = (
    (JacobianKind{:u}(), JacobianRequest{:u}),
    (JacobianResidualKind(), JacobianResidualRequest),
)
_sensitivity_validatable_kinds() = (
    (ParameterJacobianKind(), ParameterJacobianRequest),
    (ParameterVJPKind(nothing), ParameterVJPRequest),
    (TimeSensitivityKind(nothing), TimeSensitivityRequest),
    (StateJVPKind(nothing), StateJVPRequest),
    (StateVJPKind(nothing), StateVJPRequest),
)

# AD-from-residual through an element-local solve is wrong in principle
# (implicit function; see references/implicit-ad-plasti.jl), so the rejection
# is PER CACHE and PER KIND: a cache with internal state is admissible when
# the requested kind is served analytically (the author carries dq/∂seed,
# like the consistent tangent), or when the author asserts the local
# equations are insensitive to the seeded quantity (`dq/∂seed ≡ 0`, making
# plain AD exact). Only the would-be AD fallback is rejected.
function assert_sensitivity_admissible(T::Type, kind)
    if has_internal_state(T) && !provides_analytic(T, kind) && !internal_state_insensitive(T, kind)
        throw(ArgumentError(
            "$(nameof(T)) carries condensed internal state, and AD-from-residual through " *
            "its local solve would be silently wrong. Either implement the analytic " *
            "`assemble_cell!` kernel for $(typeof(kind)) (declared via `provides_analytic`), " *
            "declare `internal_state_insensitive` if the local equations do not depend on " *
            "the seeded quantity, or (for time sensitivities) use " *
            "`FiniteDifferenceSensitivity`."))
    end
    return nothing
end

"""
    has_internal_state(::Type{CacheType}) -> Bool

`true` iff the element cache carries condensed per-item internal state whose
residual contains a local solve. Governs the sensitivity admissibility check:
AD-from-residual through a local solve is wrong in principle, so such caches
need an analytic kernel, an [`internal_state_insensitive`](@ref) declaration,
or a finite-difference method for the requested sensitivity.
"""
has_internal_state(::Type) = false

"""
    internal_state_insensitive(::Type{CacheType}, kind) -> Bool

Author-asserted declaration that the element-local internal-state equations
do NOT depend on the quantity the sensitivity `kind` seeds (`∂L/∂seed ≡ 0`).
When true, `dq/∂seed = 0` and plain AD-from-residual is exactly correct even
through the local solve — zero seed partials propagate exactly. The framework
CANNOT verify this claim; a wrong assertion produces a silently wrong
sensitivity. Same trust model as [`provides_analytic`](@ref).
"""
internal_state_insensitive(::Type, kind) = false

"""
    declare_scratch(cache) -> NamedTuple of nullary constructors

Element-side scratch declaration: each entry is instantiated once per worker
and reaches kernels via `args.scratch`. Solvers declare their own entries via
`setup_operator(...; scratch = (name = () -> ...,))`; both land merged in the
workspace. This replaces smuggling solver state through model trees or
parameter bags.
"""
declare_scratch(cache) = (;)

####################################
## Reserved item shapes (contract space only — see the plan §2.1b)
####################################

"""
    InterfaceKernelArgs

RESERVED: the N-participant argument bundle for interface/pair items (DG
facet pairs as the `N = 2` case, contact pairs, non-local one-to-many
couplings with an indexed collection of secondary sides). Each participant
carries its own states/geometry/restriction; `p`, `scratch`, and
per-participant contexts ride alongside. Implemented with the phase-4
interface work — the name and shape are reserved now so nothing forecloses
them.
"""
abstract type InterfaceKernelArgs end

####################################
## Differentiable parameter protocol
####################################

"""
    parameter_vector(p) -> AbstractVector

Flat vector view θ of the differentiable parameters in `p`. Together with
[`rebuild_parameters`](@ref) this is the seam through which parameter
sensitivities are seeded. Implement both for custom parameter types.

θ need not cover all of `p`: entries not exposed here are static — held
fixed by every sensitivity sweep — and all parameter-sensitivity costs
(seed dimension, local Jacobian columns) scale with `length(θ)`.
"""
parameter_vector(p::Real) = SVector(p)
parameter_vector(p::AbstractVector) = p
parameter_vector(::T) where T = throw(ArgumentError(
    "No `parameter_vector` method for parameter type $T. Implement " *
    "`parameter_vector`/`rebuild_parameters` to enable parameter sensitivities."))

"""
    rebuild_parameters(p, θ) -> typeof-compatible parameters

Reconstruct a parameter object structurally equal to `p` with the
differentiable entries replaced by `θ` (possibly Dual-valued).
"""
rebuild_parameters(::Real, θ) = θ[1]
rebuild_parameters(::AbstractVector, θ) = θ
rebuild_parameters(::T, θ) where T = throw(ArgumentError(
    "No `rebuild_parameters` method for parameter type $T."))
