####################################
## Assembly requests (element interface)
####################################

"""
    TimeIntegrationContext(t, Δt, γ̃)

Solver-controlled scalars: evaluation time `t` (Dual-typed during ∂F/∂t
sweeps), physical step size `Δt`, and the effective local stage interval `γ̃`
of the element-local internal-variable problem, normalized by the canonical
form

    q = q_ref + γ̃ · g(·, q)

— passing `γ̃` means "an implicit-Euler local integrator would solve exactly
this" (`q_ref` is dof-shaped and flows through a slot, solver-folded for
multistep schemes). This normalizes the *number*, not the element's local
rule: the element may realize the update with any consistent rule over `γ̃` —
implicit Euler, exact exponential (Rush-Larsen-type
`q = q∞ + (q_ref − q∞)·exp(−γ̃/τ)`), or local substepping.

!!! warning
    `γ̃` is NOT the rate-reconstruction slope of any state slot. Under
    backward Euler the two are reciprocals (`slope = 1/Δt`, `γ̃ = Δt`), so
    `1/γ̃` as a rate slope is accidentally right under BE and silently wrong
    under every other scheme (Newmark: `slope = γ/(βΔt)` while `γ̃ = Δt`).
    Rate slopes belong to the slot that carries the reconstruction.
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
# No 1-arg constructor: a defaulted γ̃ (e.g. zero) would silently break local
# stage problems that scale by it.

# Framework code touches contexts only through these accessors, so schemes
# with richer per-sweep scalars can pass their own context types.
"Evaluation time of the sweep this context belongs to."
evaluation_time(ctx::TimeIntegrationContext) = ctx.t
"Rebuild `ctx` with the evaluation time replaced by `t̃` (Dual-typed in ∂F/∂t sweeps)."
with_time(ctx::TimeIntegrationContext, t̃) = TimeIntegrationContext(t̃, oftype(t̃, ctx.Δt), oftype(t̃, ctx.γ̃))
"Effective local stage interval γ̃ (see `TimeIntegrationContext`); wrapper context types must forward it, same as `evaluation_time`."
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

`Consistent` is the default wherever the type parameter is left unspecified:
the unsafe direction is a silently missing correction, so `FrozenQ` must be
spelled explicitly. For a stateless element (no `q`) the two coincide.
`FrozenQ` is a legitimate election for an iteration matrix (modified Newton,
MLN outer loops), where a wrong tangent costs convergence rate and nothing
else, and never for a gradient — the sensitivity request kinds therefore
carry no `CorrectionMode` parameter at all.
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
by the per-slot `NamedTuple` `weights` and at frozen values of every other
slot. The scheme's chain-rule scalars ride as request payload, so a fused
kernel and the composed per-slot fallback draw from one source (see
[`WeightedJacobianKind`](@ref)).

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
`p` is the GLOBAL parameter bag, not the element-local `args.p`: the AD
fallback re-queries [`query_cell_parameters`](@ref) from a Dual-rebuilt `p`
per seed direction, so wrappers and per-element views forward Duals
transparently. An analytic kernel reads `B` only.
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
`states = (u = u, v = AffineRate(γ/(β*Δt), uᵥ))`; the slot is declared like
any other (`setup_operator(...; slots = (:u, :du))`).

A `:u` slot must exist and PRECEDE the reconstructed slot in the states
`NamedTuple` — the sweep throws otherwise. Kernels read the reconstructed
values through `args.states.<name>` and nothing else, so an element never
encodes a time-integration scheme.

!!! warning "Reconstructed slots are frozen under AD"
    Reconstruction happens at gather time, before any Dual seeding, and the
    ∂F/∂u sweep seeds only the `:u` buffer. A kernel reading a reconstructed
    slot sees its primal value throughout, so the assembled Jacobian is
    ∂F/∂u at frozen slot values. The chain-rule term (`slope · ∂F/∂slot`)
    belongs to the solver, which contributes it through its per-slot weights.

!!! note "Condensed elements"
    The reconstruction applies uniformly to ALL entries of the element
    buffer, because `u` and `anchor` are both `[ū; q]`-shaped gathers.
    Internal variables are never rate-reconstructed by contract: the
    condensed tail of a reconstructed slot is an artifact of the uniform
    formula and elements must not interpret it.
"""
struct AffineRate{T, V <: AbstractVector}
    slope::T
    anchor::V
end

"""
    InternalSource(u::AbstractVector)

Slot *source* restricting the gather to the condensed internal-variable block
of `u` (the `q` tail of `[ū; q]`, [`internal_variable_range`](@ref)) instead
of `celldofs(cell)` — this is what makes `q` an ordinary slot
(`states = (u = u, q = InternalSource(u), …)`), the source carrying its own
restriction exactly as [`AffineRate`](@ref) carries reconstruction. Such a
slot is sized per cell by the cell's internal dof count, not by
`ndofs_per_cell`, so its buffer is resized on every gather.

[`condense_internal!`](@ref) is the only writer of the underlying vector;
evaluation sweeps only read through it.
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
| `ctx` | the solver; the per-sweep scalars — a [`TimeIntegrationContext`](@ref) `(t, Δt, γ̃)`, or `nothing` for stationary sweeps. The one open channel: a scheme with richer scalars passes its own context type, read through [`evaluation_time`](@ref)/[`with_time`](@ref)/[`stage_scaling`](@ref) instead of field access. |

Kernels select on the `(request, cache)` pair, never on `args`, so annotating
the parameter (`args::CellArgs`) is permitted. Hand-constructing an instance
unit-tests a kernel without an operator:

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
fields as [`CellArgs`](@ref), built by the framework's facet driver.
`CellArgs` and `FacetArgs` share no supertype: a cell kernel and a facet
kernel never meet at the same dispatch site.
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
[`CellArgs`](@ref) analogue (see its table for `states` and `ctx`) with the
[`AlgebraicItem`](@ref) where the geometry cache sits:

| field | owner / lifetime |
|---|---|
| `item` | the engine; the [`AlgebraicItem`](@ref) the sweep stands on — READ-ONLY for kernels |
| `p` | the cache; the item-local parameter view from [`query_cell_parameters`](@ref), queried on the item |

Kernels select on the `(request, cache)` pair, so annotating the parameter
(`args::AlgebraicArgs`) is permitted. This record shares no supertype with
`CellArgs`/`FacetArgs`: a cell kernel and an algebraic kernel never meet at
the same dispatch site. Hand-constructing an instance unit-tests a kernel
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

The volumetric kernel entry point. Elements must provide at least the
[`ResidualRequest`](@ref) method (validated at setup); every other request
falls back to automatic differentiation of the residual kernel unless
[`provides_analytic`](@ref) declares an analytic method. `args` is a
[`CellArgs`](@ref).
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
handed to volumetric kernels; parameter fields gather their per-element views
through this seam. Defaults to [`unwrap_parameters`](@ref) on the bag.
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

`true` iff the cache has an AUTHOR-WRITTEN analytic kernel for the given
request *kind* singleton (`JacobianKind()`, `ParameterJacobianKind()`, …) —
that question and no other; whether the cache the engine calls serves the kind
at all is [`serves_kind`](@ref). Everything except the mandatory residual
defaults to `false`, i.e. AD-from-residual, and decorators forward the trait
unchanged.

There is deliberately exactly ONE root method (the residual default is a
runtime branch), so specializations are always strictly more specific:
blanket declarations like `provides_analytic(::Type{<:MyCache}, kind) = true`
cannot create ambiguities, and there is no request-type-parameter matching to
get wrong. Kernel/trait consistency is validated once at operator setup
([`validate_element_cache`](@ref)).
"""
provides_analytic(::Type, kind) = kind isa ResidualKind

"""
    serves_kind(::Type{CacheType}, kind) -> Bool

`true` iff the RESOLVED cache — the possibly decorated type the engine calls —
answers `kind` with the quantity that kind names, whether from an
author-written kernel or from a decorator's generic route.
[`provides_analytic`](@ref) implies this; the converse does not, since
[`ADElementCache`](@ref) serves kinds its inner has no kernel for.

Same single-root shape as `provides_analytic`, so the literal returns constant
fold. Deliberately NOT exported: the methods belong to the wrapping layer
([`ADElementCache`](@ref), [`CompositeVolumetricElementCache`](@ref)), while an
element author declares `provides_analytic` and this default composes.
"""
serves_kind(T::Type, kind) = provides_analytic(T, kind)

"""
    validate_element_cache(cache, declared_requests = ())

Setup-time consistency check for element caches: a cache opting into the
request protocol must implement the mandatory [`ResidualRequest`](@ref)
kernel, and every request kind [`provides_analytic`](@ref) claims must have a
matching kernel method. Runs once per subdomain at `setup_operator` time, so
a typo'd port fails loudly instead of silently assembling through the wrong
path.

The two halves take different subjects, per the
[`AbstractElementCacheDecorator`](@ref) convention. The KERNEL half — the
mandatory-method probes and the trait ↔ kernel check — runs on the
[`unwrap`](@ref) fixpoint (a composite recurses into its inners from there),
since a decorator's forwarding methods would answer those probes
unconditionally. The ADMISSIBILITY half runs on `cache` as the engine calls
it, decoration included, so a decorator's generic routes count as the
coverage they are.

The trait ↔ kernel check always covers the primal kinds. Every kind
[`requires_admissibility_check`](@ref) names additionally runs the
internal-state admissibility check ([`has_internal_state`](@ref)) here
instead of on first use — unconditionally for the primal kinds, and for any
other kind only when declared via
`setup_operator(...; requests = (ParameterVJPKind, …))`. Undeclared,
non-primal kinds stay usable; their checks run at the call-time entry points.
"""
function validate_element_cache(cache, declared_requests::Tuple = ())
    _validate_element_kernels(unwrap(cache), declared_requests)
    _assert_admissible_kinds(typeof(cache), declared_requests)
    return nothing
end

function _validate_element_kernels(cache, declared_requests::Tuple)
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
    end
    for K in declared_requests
        has_cell_request(K) || continue
        _assert_trait_backed(T, validation_instance(K))
    end
    return nothing
end

function _assert_admissible_kinds(T::Type, declared_requests::Tuple)
    for kind in _primal_validatable_kinds()
        requires_admissibility_check(kind) && assert_sensitivity_admissible(T, kind)
    end
    for K in declared_requests
        has_cell_request(K) || continue
        kind = validation_instance(K)
        requires_admissibility_check(kind) && assert_sensitivity_admissible(T, kind)
    end
    return nothing
end

"""
    has_cell_request(::Type{K}) -> Bool

Whether kind `K` materializes an [`assemble_cell!`](@ref) request, i.e.
whether [`request_type`](@ref) answers for it. Setup validates the trait ↔
kernel backing of declared kinds that do.

Kinds served by a different element hook declare `false`:
[`FunctionalKind`](@ref) reaches the element through
[`evaluate_cell_functional`](@ref) and *returns* its contribution instead of
filling a request.
"""
has_cell_request(::Type{K}) where {K} = true

"""
    validation_instance(::Type{K}) -> kind instance

The placeholder instance setup-time validation queries `K`'s traits on:
declarations carry kind TYPES normalized to their `UnionAll` base, while
[`request_type`](@ref) and [`provides_analytic`](@ref) are queried on
instances.

The default calls `K()`. A kind whose payload is a type parameter overloads
it with a placeholder — only the type is read, never the value:

    FerriteOperators.validation_instance(::Type{<:MyVJPKind}) = MyVJPKind(nothing)

There is no fallback that skips: a declared kind that cannot be instantiated
raises at setup rather than silently missing its validation.
"""
validation_instance(::Type{K}) where {K} = K()

"""
    requires_admissibility_check(kind) -> Bool

Whether declaring `kind` runs the internal-state admissibility rule
(`assert_sensitivity_admissible`) at setup instead of on first use. Defaults
to `false`, so a downstream kind opts in.

True for the kinds whose AD fallback differentiates THROUGH an element's
local solve. Time sensitivities and weighted Jacobians are exempt although
they differentiate: their escape (finite differences, the composed route) is
chosen per call, so setup cannot know whether the AD path will be taken.
"""
requires_admissibility_check(kind) = false

# The trait ↔ kernel check, over the kernel entry point and args record of the
# item family the cache belongs to: `assemble_cell!`/`CellArgs` (volumetric),
# `assemble_algebraic!`/`AlgebraicArgs`, or `assemble_facet!`/`FacetArgs` plus
# the trailing local facet index (surface).
#
# A claim is backed by an AUTHOR-WRITTEN method, so the subject is the
# `unwrap` fixpoint: a decorator's blanket kernel methods answer `hasmethod`
# for every inner and would make the check pass vacuously.
_assert_trait_backed(T, kind) = _assert_trait_backed(T, kind, assemble_cell!, CellArgs)
function _assert_trait_backed(T, kind, entry, ::Type{Args}, trailing::Tuple = ()) where {Args}
    A = unwrap(T)
    ReqT = request_type(kind)
    if provides_analytic(A, kind) && !hasmethod(entry, Tuple{ReqT, A, Args, trailing...})
        throw(ArgumentError(
            "$(A) declares `provides_analytic` for $(typeof(kind)) but implements no " *
            "matching `$(nameof(entry))(::$(ReqT), ::$(nameof(A)), ::$(nameof(Args))" *
            "$(_trailing_signature(trailing)))` method."))
    end
    return nothing
end

# The kernel-signature tail an entry point carries beyond `(req, cache, args)`,
# spelled the way an author writes it.
_trailing_signature(trailing::Tuple) = mapreduce(T -> ", ::$(T)", *, trailing; init = "")

# The kinds whose trait ↔ kernel consistency is checked at setup; the request
# each analytic kernel takes comes from `request_type`, the single
# kind → request association.
_primal_validatable_kinds() = (JacobianKind{:u}(), JacobianResidualKind())

# The rejection below is PER CACHE and PER KIND, and hits only the would-be AD
# fallback: an analytic kernel carries the ∂F/∂q·dq/d· correction, and
# `internal_state_insensitive` asserts there is none to carry.
"""
    assert_sensitivity_admissible(T::Type, kind)
    assert_sensitivity_admissible(T::Type, kind, entry, ::Type{Args}, trailing = ())

The internal-state admissibility check: throws unless a `has_internal_state`
cache `T` [`serves_kind`](@ref) or declares it
[`internal_state_insensitive`](@ref). `entry`/`Args`/`trailing` name the item
family whose kernel entry point, args record and signature tail the error
message should point authors at; the 2-arg form defaults to the volumetric
`assemble_cell!`/`CellArgs`. [`validate_algebraic_cache`](@ref) passes
`assemble_algebraic!`/`AlgebraicArgs`, whose remedies differ, and the
facet-item family its trailing `::Int` local facet index.

`T` is a SERVED-CAPABILITY subject, so it is the cache the engine calls,
decorated where the engine decorated it. That is what admits the generic
routes [`ADElementCache`](@ref) builds from [`condensed_corrector`](@ref) and
[`local_conditions!`](@ref): [`serves_kind`](@ref) reports them as the coverage
they are, and an undecorated cache has no such route.
"""
function assert_sensitivity_admissible(T::Type, kind, entry = assemble_cell!, ::Type{Args} = CellArgs,
        trailing::Tuple = ()) where {Args}
    if has_internal_state(T) && !serves_kind(T, kind) && !internal_state_insensitive(T, kind)
        name = nameof(unwrap(T))
        wrapping_note = unwrap(T) === T ? "" :
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
            "through its pure residual kernel would compute only the frozen-q partial, " *
            "missing the ∂F/∂q·dq/d· correction this kind's total needs. Either implement the " *
            "analytic `$(nameof(entry))` kernel for $(typeof(kind)) — " *
            "`$(nameof(entry))(::$(request_type(kind)), ::$(name), ::$(nameof(Args))" *
            "$(_trailing_signature(trailing)))`, declared via `provides_analytic` — or " *
            remedy))
    end
    return nothing
end

"""
    has_internal_state(::Type{CacheType}) -> Bool

`true` iff the element cache carries condensed per-item internal state `q`
with a corrector store ([`condense_internal!`](@ref)). Governs the
sensitivity admissibility check: a kind with no [`CorrectionMode`](@ref) is
always the total, so AD-from-residual through the pure kernel — which
computes only the frozen-q partial — is missing the correction unless the
cache [`serves_kind`](@ref) or declares it
[`internal_state_insensitive`](@ref); time sensitivities alone admit a
finite-difference escape.
"""
has_internal_state(::Type) = false

"""
    internal_state_insensitive(::Type{CacheType}, kind) -> Bool

Author-asserted declaration that the element-local internal-state equations
do NOT depend on the quantity the sensitivity `kind` seeds (`∂L/∂seed ≡ 0`),
so `dq/∂seed = 0` and the total collapses to the frozen-q partial plain
AD-from-residual already computes. The framework CANNOT verify this claim; a
wrong assertion produces a silently wrong sensitivity. Same trust model as
[`provides_analytic`](@ref).
"""
internal_state_insensitive(::Type, kind) = false

####################################
## Differentiable parameter protocol
####################################

"""
    parameter_vector(p) -> AbstractVector

Flat vector view θ of the differentiable parameters in `p`. With
[`rebuild_parameters`](@ref) this is the seam through which parameter
sensitivities are seeded; implement both for custom parameter types.

θ need not cover all of `p`: entries not exposed here are held fixed by every
sensitivity sweep, and all parameter-sensitivity costs (seed dimension, local
Jacobian columns) scale with `length(θ)`.
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
