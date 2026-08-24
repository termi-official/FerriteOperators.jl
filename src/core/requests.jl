####################################
## Assembly requests (element interface)
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
"Effective local stage interval γ̃ of the element-local internal-variable problem (see `TimeIntegrationContext`); custom/wrapper context types must forward it, same as `evaluation_time`."
stage_scaling(ctx::TimeIntegrationContext) = ctx.γ̃

"""
    AbstractAssemblyRequest

What a kernel is asked to compute for one cell. Kernels dispatch on the
request type, never on argument shapes, so kernel sets for different state
slots can coexist without ambiguity.
"""
abstract type AbstractAssemblyRequest end

"""
    CorrectionMode
    Consistent <: CorrectionMode
    FrozenQ    <: CorrectionMode

Whether a Jacobian-shaped request over a condensed element is the TOTAL
derivative or the PARTIAL at frozen internal state `q`:

    Consistent:  ∂F/∂·|_q + ∂F/∂q · dq/d·     (the total — DEFAULT)
    FrozenQ:     ∂F/∂·|_q only                (the partial)

`Consistent` is the default everywhere a `CorrectionMode` type parameter is
left unspecified — the unsafe direction is a silently missing correction, so
`FrozenQ` must always be spelled explicitly. For a stateless element (no `q`)
the two coincide. `FrozenQ` is a legitimate election for an iteration matrix
(modified Newton, MLN outer loops), where a wrong tangent costs convergence
rate and nothing else; it is never legitimate for a gradient, so the
sensitivity request kinds carry no `CorrectionMode` parameter at all — there
is no way to construct a `FrozenQ` election for them.
"""
abstract type CorrectionMode end
@doc (@doc CorrectionMode) struct Consistent <: CorrectionMode end
@doc (@doc CorrectionMode) struct FrozenQ <: CorrectionMode end

"Accumulate the local residual into `r`."
struct ResidualRequest{V <: AbstractVector} <: AbstractAssemblyRequest
    r::V
end

"""
    JacobianRequest{slot, C <: CorrectionMode}(K)

Accumulate ∂F/∂slot into `K`, slot ∈ (:u, :du, :v, :a, :q, …), under
correction mode `C` (see [`CorrectionMode`](@ref)). `JacobianRequest{slot}(K)`
defaults `C` to [`Consistent`](@ref).
"""
struct JacobianRequest{slot, C <: CorrectionMode, M <: AbstractMatrix} <: AbstractAssemblyRequest
    K::M
end
JacobianRequest{slot}(K::M) where {slot, M <: AbstractMatrix} = JacobianRequest{slot, Consistent, M}(K)
JacobianRequest{slot, C}(K::M) where {slot, C <: CorrectionMode, M <: AbstractMatrix} = JacobianRequest{slot, C, M}(K)

"""
    JacobianResidualRequest{C <: CorrectionMode}(K, r)

Accumulate ∂F/∂u and the residual in one sweep (the Newton hot path), under
correction mode `C`. `JacobianResidualRequest(K, r)` defaults `C` to
[`Consistent`](@ref).
"""
struct JacobianResidualRequest{C <: CorrectionMode, M <: AbstractMatrix, V <: AbstractVector} <: AbstractAssemblyRequest
    K::M
    r::V
end
JacobianResidualRequest(K::M, r::V) where {M <: AbstractMatrix, V <: AbstractVector} = JacobianResidualRequest{Consistent, M, V}(K, r)
JacobianResidualRequest{C}(K::M, r::V) where {C <: CorrectionMode, M <: AbstractMatrix, V <: AbstractVector} = JacobianResidualRequest{C, M, V}(K, r)

"""
    WeightedJacobianRequest(K, weights)

Accumulate the weighted Jacobian `Σₛ wₛ ∂F/∂s` into `K`, over the slots named
by `weights` and at frozen values of every other slot. `weights` is the
caller's per-slot `NamedTuple` — the scheme's chain-rule scalars ride as
request payload, so a fused kernel and the composed per-slot fallback draw
from one source (see [`WeightedJacobianKind`](@ref)).

A hand-fused scheme matrix (SDIRK/backward Euler `W = M/(γΔt) + K`) is an
analytic provider of THIS request, not of [`JacobianRequest`](@ref): it
computes the combination internally, which no single-slot Jacobian does.
"""
struct WeightedJacobianRequest{C <: CorrectionMode, M <: AbstractMatrix, W <: NamedTuple} <: AbstractAssemblyRequest
    K::M
    weights::W
end
WeightedJacobianRequest(K::M, weights::W) where {M <: AbstractMatrix, W <: NamedTuple} = WeightedJacobianRequest{Consistent, M, W}(K, weights)
WeightedJacobianRequest{C}(K::M, weights::W) where {C <: CorrectionMode, M <: AbstractMatrix, W <: NamedTuple} = WeightedJacobianRequest{C, M, W}(K, weights)

"""
Accumulate the dense local parameter Jacobian ∂Fₑ/∂θ into `B` (ndofsₑ × nθ).
`p` is the GLOBAL parameter bag (not `args.p`, the element-local view) — the
AD fallback re-queries [`query_cell_parameters`](@ref) from a Dual-rebuilt `p`
per seed direction, so wrappers and per-element parameter views forward Duals
transparently; an analytic kernel reads `B` only.
"""
struct ParameterJacobianRequest{M <: AbstractMatrix, P} <: AbstractAssemblyRequest
    B::M
    p::P
end

@doc (@doc ParameterJacobianRequest)
struct ParameterVJPRequest{V <: AbstractVector, L <: AbstractVector, P} <: AbstractAssemblyRequest
    g::V
    λₑ::L
    p::P
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
values through `args.states.<name>` and nothing else, so an element never
encodes a time-integration scheme.

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
    InternalSource(u::AbstractVector)

Slot *source* restricting the gather to the condensed internal-variable block
of `u` (the `q` tail of `[ū; q]`, [`internal_variable_range`](@ref)) instead of
`celldofs(cell)`. This is what makes `q` an ordinary slot (`states = (u = u,
q = InternalSource(u), …)`): the source carries its own restriction, exactly
like [`AffineRate`](@ref) carries reconstruction. A slot sourced this way is
sized per cell by the number of internal dofs the cell owns, not by
`ndofs_per_cell` — the element-local buffer is resized to fit on every gather.

[`condense_internal!`](@ref) is the only writer of an `InternalSource`-backed
slot's underlying vector: every evaluation sweep only reads through it.
"""
struct InternalSource{V <: AbstractVector}
    u::V
end

"""
    CellArgs(states, cell, p, ctx)

The argument bundle a cell kernel's third parameter receives.

| field | owner / lifetime |
|---|---|
| `states` | the engine; one cell-local slot buffer per slot declared at setup (`(u = uₑ, uprev = uₑprev, …)`), refreshed every sweep |
| `cell` | the engine; the geometry cache of the current item — READ-ONLY for kernels |
| `p` | the element; the cell-local parameter view from [`query_cell_parameters`](@ref) — configuration only, never time or history |
| `ctx` | the solver; the per-sweep scalars — the [`TimeIntegrationContext`](@ref) `(t, Δt, γ̃)`, or `nothing` for stationary sweeps. This is the one open channel: a scheme with richer per-sweep scalars passes its own context type, read through [`evaluation_time`](@ref)/[`with_time`](@ref)/[`stage_scaling`](@ref) instead of field access. |

Kernels select on the `(request, cache)` pair, never on `args`, so annotating
the parameter (`args::CellArgs`) is permitted.

Hand-constructing an instance is the supported way to unit-test a kernel
without an operator:

    args = CellArgs((u = uₑ,), cell_cache, p, nothing)
    assemble_cell!(ResidualRequest(rₑ), cache, args)
"""
struct CellArgs{States <: NamedTuple, Cell, P, Ctx}
    states::States
    cell::Cell
    p::P
    ctx::Ctx
end

"""
    FacetArgs(states, cell, p, ctx)

The argument bundle a facet kernel's third parameter receives — the same four
fields as [`CellArgs`](@ref) (see its docstring), built by the framework's
facet driver. `CellArgs` and `FacetArgs` share no supertype: a cell kernel and
a facet kernel never meet at the same dispatch site.
"""
struct FacetArgs{States <: NamedTuple, Cell, P, Ctx}
    states::States
    cell::Cell
    p::P
    ctx::Ctx
end

"""
    AlgebraicItem(index, dofs)

The item an algebraic kernel is positioned on: its `index` into the
[`algebraic_items`](@ref) declaration, and the global `dofs` its local system
maps onto. This is the whole geometry an item of this family has — a cache
serving several 0D rows selects its model by `args.item.index`.
"""
struct AlgebraicItem{D <: AbstractVector{Int}}
    index::Int
    dofs::D
end

"""
    AlgebraicArgs(states, item, p, ctx)

The argument bundle an algebraic kernel's third parameter receives — the
[`CellArgs`](@ref) analogue with the [`AlgebraicItem`](@ref) where the geometry
cache sits.

| field | owner / lifetime |
|---|---|
| `states` | the engine; one item-local slot buffer per slot declared at setup, refreshed every sweep |
| `item` | the engine; the [`AlgebraicItem`](@ref) the sweep stands on — READ-ONLY for kernels |
| `p` | the cache; the item-local parameter view from [`query_cell_parameters`](@ref), queried on the item |
| `ctx` | the solver; the per-sweep scalars — the [`TimeIntegrationContext`](@ref), or `nothing` for stationary sweeps |

Kernels select on the `(request, cache)` pair, never on `args`, so annotating
the parameter (`args::AlgebraicArgs`) is permitted. This record shares no
supertype with `CellArgs`/`FacetArgs`: a cell kernel and an algebraic kernel
never meet at the same dispatch site.

Hand-constructing an instance is the supported way to unit-test a kernel
without an operator:

    args = AlgebraicArgs((u = uₑ,), AlgebraicItem(1, [17]), p, nothing)
    assemble_algebraic!(ResidualRequest(rₑ), cache, args)
"""
struct AlgebraicArgs{States <: NamedTuple, I, P, Ctx}
    states::States
    item::I
    p::P
    ctx::Ctx
end

"Rebuild `args` with the slot buffers replaced."
with_states(args::CellArgs, states::NamedTuple) = CellArgs(states, args.cell, args.p, args.ctx)
with_states(args::FacetArgs, states::NamedTuple) = FacetArgs(states, args.cell, args.p, args.ctx)
with_states(args::AlgebraicArgs, states::NamedTuple) = AlgebraicArgs(states, args.item, args.p, args.ctx)

"Rebuild `args` with the element-local parameter view replaced."
with_parameters(args::CellArgs, p) = CellArgs(args.states, args.cell, p, args.ctx)
with_parameters(args::FacetArgs, p) = FacetArgs(args.states, args.cell, p, args.ctx)
with_parameters(args::AlgebraicArgs, p) = AlgebraicArgs(args.states, args.item, p, args.ctx)

"Rebuild `args` with the per-sweep context replaced — the ∂F/∂t seeding seam."
with_context(args::CellArgs, ctx) = CellArgs(args.states, args.cell, args.p, ctx)
with_context(args::FacetArgs, ctx) = FacetArgs(args.states, args.cell, args.p, ctx)
with_context(args::AlgebraicArgs, ctx) = AlgebraicArgs(args.states, args.item, args.p, ctx)

"""
    assemble_cell!(req::AbstractAssemblyRequest, cache, args)

The volumetric kernel entry point. Elements must at least provide the
[`ResidualRequest`](@ref) method (validated at setup); every other request falls back to automatic
differentiation of the residual kernel unless [`provides_analytic`](@ref)
declares an analytic method.

`args` is a [`CellArgs`](@ref); annotating the parameter is permitted.
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

`true` iff the element cache implements `assemble_cell!` analytically for
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

Setup-time consistency check for element caches: a cache that opts into the
request protocol must implement the mandatory [`ResidualRequest`](@ref)
kernel, and every request kind the [`provides_analytic`](@ref) trait claims
must have a matching kernel method. Runs once per subdomain at
`setup_operator` time — a typo'd port fails loudly here instead of silently
assembling through the wrong path.

Decorators ([`ADElementCache`](@ref), [`FusedFromSplit`](@ref)) unwrap to
their inner cache, and composites recurse into theirs: the mandatory-method
probes must reach the author-written method set, which a decorator's
forwarding methods would otherwise answer for unconditionally.

The trait ↔ kernel check always covers the primal kinds (the operator will
issue them). Every kind [`requires_admissibility_check`](@ref) names
additionally runs the internal-state admissibility check
([`has_internal_state`](@ref)) here instead of on first use — unconditionally
for the primal kinds it names, since those are always covered, and for any
other kind only when declared via
`setup_operator(...; requests = (ParameterVJPKind, …))`. Undeclared,
non-primal kinds stay usable — their checks simply run at the call-time entry
points.
"""
function validate_element_cache(cache, declared_requests::Tuple = ())
    T = typeof(cache)
    hasmethod(assemble_cell!, Tuple{ResidualRequest, T, CellArgs}) || throw(ArgumentError(
        "$(T) implements no `assemble_cell!(::ResidualRequest, ::$(nameof(T)), ::CellArgs)` " *
        "method. The residual kernel is mandatory: it is the basis for AD-derived Jacobians " *
        "and sensitivities."))
    hasmethod(reinit_values!, Tuple{T, CellCache}) || throw(ArgumentError(
        "$(T) implements no `reinit_values!(::$(nameof(T)), cell)` method. The engine " *
        "reinitializes element values once per cell and sweep through this hook; " *
        "kernels are pure evaluation and must not rely on reinit inside them."))
    for kind in _primal_validatable_kinds()
        _assert_trait_backed(T, kind)
        requires_admissibility_check(kind) && assert_sensitivity_admissible(T, kind)
    end
    for K in declared_requests
        has_cell_request(K) || continue
        kind = validation_instance(K)
        _assert_trait_backed(T, kind)
        requires_admissibility_check(kind) && assert_sensitivity_admissible(T, kind)
    end
    return nothing
end

"""
    has_cell_request(::Type{K}) -> Bool

Whether kind `K` materializes an [`assemble_cell!`](@ref) request, i.e. whether
[`request_type`](@ref) answers for it. Setup validates the trait ↔ kernel
backing of declared kinds that do.

Kinds served by a different element hook declare `false`:
[`FunctionalKind`](@ref) reaches the element through
[`evaluate_cell_functional`](@ref) and *returns* its contribution instead of
filling a request, so it has no request to check.
"""
has_cell_request(::Type{K}) where {K} = true

"""
    validation_instance(::Type{K}) -> kind instance

The placeholder instance setup-time validation queries `K`'s traits on.
Declarations carry kind TYPES normalized to their `UnionAll` base, while
[`request_type`](@ref) and [`provides_analytic`](@ref) are queried on
instances; this is the bridge.

The default calls `K()`, which serves every kind constructible without
payload. A kind whose payload is a type parameter overloads it with a
placeholder — only the type is read, never the value:

    FerriteOperators.validation_instance(::Type{<:MyVJPKind}) = MyVJPKind(nothing)

There is no fallback that skips: a declared kind that cannot be instantiated
raises at setup rather than silently missing its validation.
"""
validation_instance(::Type{K}) where {K} = K()

"""
    requires_admissibility_check(kind) -> Bool

Whether declaring `kind` runs the internal-state admissibility rule
(`assert_sensitivity_admissible`) at setup instead of on first use.

True for the kinds whose AD fallback differentiates THROUGH an element's local
solve. Time sensitivities and weighted Jacobians are exempt although they
differentiate: their escape (finite differences, the composed route) is chosen
per call, so setup cannot know whether the AD path will be taken. The default
is `false`, so a downstream kind opts in.
"""
requires_admissibility_check(kind) = false

# The trait ↔ kernel check, over the kernel entry point and args record of the
# item family the cache belongs to: `assemble_cell!`/`CellArgs` for a
# volumetric cache, `assemble_algebraic!`/`AlgebraicArgs` for an algebraic one,
# `assemble_facet!`/`FacetArgs` plus the trailing local facet index for a
# surface one.
_assert_trait_backed(T, kind) = _assert_trait_backed(T, kind, assemble_cell!, CellArgs)
function _assert_trait_backed(T, kind, entry, ::Type{Args}, trailing::Tuple = ()) where {Args}
    ReqT = request_type(kind)
    if provides_analytic(T, kind) && !hasmethod(entry, Tuple{ReqT, T, Args, trailing...})
        throw(ArgumentError(
            "$(T) declares `provides_analytic` for $(typeof(kind)) but implements no " *
            "matching `$(nameof(entry))(::$(ReqT), ::$(nameof(T)), ::$(nameof(Args))" *
            "$(_trailing_signature(trailing)))` method."))
    end
    return nothing
end

# The kernel-signature tail an entry point carries beyond `(req, cache, args)`,
# spelled the way an author writes it.
_trailing_signature(trailing::Tuple) = mapreduce(T -> ", ::$(T)", *, trailing; init = "")

# The kinds whose trait ↔ kernel consistency is checked at setup; the request
# each analytic kernel takes comes from [`request_type`](@ref), the single
# kind → request association. Payload-carrying kinds get placeholder payloads —
# only the type matters for the trait query.
_primal_validatable_kinds() = (JacobianKind{:u}(), JacobianResidualKind())

# Post-phase (condense_internal!/CondensationReport) a condensed cache's
# residual kernel is PURE — it reads the frozen `q` a prior condensation
# wrote, no local solve inside it. AD-from-residual is therefore no longer
# wrong in principle; what it computes is the FROZEN-q PARTIAL. The rejection
# survives with a different subject: these kinds carry no `CorrectionMode`
# (they are always the total, see `CorrectionMode`), so a partial-only AD
# fallback would be a silently MISSING ∂F/∂q·dq/d· correction, not an invalid
# derivative through an iteration. The rejection is PER CACHE and PER KIND: a
# cache with internal state is admissible when the requested kind is served
# analytically (the author carries the correction, like the consistent
# tangent), or when the author asserts the local equations are insensitive to
# the seeded quantity (`dq/∂seed ≡ 0`, making the partial exact — there is
# nothing to correct). Only the would-be AD fallback is rejected.
"""
    _display_cache_type(T) -> Type

The cache type an admissibility error should NAME: `T` itself, or the
innermost cache a decorator ([`ADElementCache`](@ref), [`FusedFromSplit`](@ref))
wraps — an author never wrote `T` when `T` is a decorator, so naming it would
send them looking at the wrong type. One root method; decorators override it.
"""
_display_cache_type(T::Type) = T

"""
    assert_sensitivity_admissible(T::Type, kind)
    assert_sensitivity_admissible(T::Type, kind, entry, ::Type{Args})

The internal-state admissibility check itself: throws unless a `has_internal_state`
cache `T` serves `kind` analytically, declares it
[`internal_state_insensitive`](@ref), or — for the parameter and time kinds of
a CELL cache — declares [`local_conditions!`](@ref), which lets the decorator
derive the missing `dq/dseed` itself. `entry`/`Args` name which item family's
kernel entry point and args record the error message should point authors at
— the 2-arg form defaults to `assemble_cell!`/`CellArgs` (a volumetric cache);
[`validate_algebraic_cache`](@ref) passes `assemble_algebraic!`/`AlgebraicArgs`
for the algebraic-item family, whose remedies differ (see below).
"""
function assert_sensitivity_admissible(T::Type, kind, entry = assemble_cell!, ::Type{Args} = CellArgs) where {Args}
    if has_internal_state(T) && !provides_analytic(T, kind) && !internal_state_insensitive(T, kind)
        name = nameof(_display_cache_type(T))
        wrapping_note = _display_cache_type(T) === T ? "" :
            " — automatically wrapped in `$(nameof(T))` because it lacks analytic coverage of " *
            "some request kind, which does not by itself supply this correction"
        # FiniteDifferenceSensitivity, `local_conditions!` and
        # `condensed_corrector` are remedies only where they actually apply:
        # the first is a call-time override that exists for time sensitivities
        # alone; the second and third are what ADElementCache's generic
        # combinations read, both CELL-shaped (the `∂F/∂q` block they multiply
        # is sized from `getnquadpoints`) and therefore meaningless for the
        # algebraic-item family, which has no generic bootstrap at all (see
        # `condense_algebraic!`).
        local_conditions_remedy = Args === CellArgs ?
            " Implementing `local_conditions!` admits the generic route instead, which derives " *
            "`dq/dseed` from the element's own local equations." : ""
        remedy = if kind isa TimeSensitivityKind
            "declare `internal_state_insensitive` if the local equations do not depend on " *
            "the seeded quantity, or use `FiniteDifferenceSensitivity`." * local_conditions_remedy
        elseif kind isa Union{ParameterJacobianKind, ParameterVJPKind}
            "declare `internal_state_insensitive` if the local equations do not depend on " *
            "the seeded quantity." * local_conditions_remedy
        elseif kind isa Union{JacobianKind{:u, Consistent}, JacobianResidualKind{Consistent}} && Args === CellArgs
            "declare `internal_state_insensitive` if the local equations do not depend on " *
            "the seeded quantity, or implement `condensed_corrector` to admit the generic " *
            "`Consistent` combination."
        elseif kind isa Union{JacobianKind{:u, Consistent}, JacobianResidualKind{Consistent}}
            "declare `internal_state_insensitive` if the local equations do not depend on " *
            "the seeded quantity. There is no generic `Consistent` bootstrap for the " *
            "algebraic-item family (an item has no cellid to key a corrector store by), so " *
            "this is the only remedy besides the analytic kernel."
        else
            "declare `internal_state_insensitive` if the local equations do not depend on " *
            "the seeded quantity."
        end
        throw(ArgumentError(
            "$(name) carries condensed internal state$(wrapping_note), and AD-from-residual " *
            "through its (now pure) residual kernel would compute only the frozen-q partial, " *
            "missing the ∂F/∂q·dq/d· correction this kind's total needs. Either implement the " *
            "analytic `$(nameof(entry))` kernel for $(typeof(kind)) (declared via `provides_analytic`), or " *
            remedy))
    end
    return nothing
end

"""
    has_internal_state(::Type{CacheType}) -> Bool

`true` iff the element cache carries condensed per-item internal state `q`
with a corrector store ([`condense_internal!`](@ref)). Governs the
sensitivity admissibility check: a kind with no [`CorrectionMode`](@ref) is
always the total, so a plain AD-from-residual fallback — which computes only
the frozen-q partial now that the kernel is pure — is missing the correction
unless the cache serves the kind analytically or declares it
[`internal_state_insensitive`](@ref); time sensitivities alone admit a
finite-difference method as a further escape.
"""
has_internal_state(::Type) = false

"""
    internal_state_insensitive(::Type{CacheType}, kind) -> Bool

Author-asserted declaration that the element-local internal-state equations
do NOT depend on the quantity the sensitivity `kind` seeds (`∂L/∂seed ≡ 0`).
When true, `dq/∂seed = 0`, so the total collapses to the frozen-q partial
plain AD-from-residual already computes on the (now pure) residual kernel —
there is nothing left for the ∂F/∂q·dq/∂seed correction to add. The framework
CANNOT verify this claim; a wrong assertion produces a silently wrong
sensitivity. Same trust model as [`provides_analytic`](@ref).
"""
internal_state_insensitive(::Type, kind) = false

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
