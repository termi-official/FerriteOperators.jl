####################################
## Element cache decorators — the shared forwarding layer
####################################

"""
    AbstractElementCacheDecorator{Inner} <: AbstractVolumetricElementCache

Supertype of the caches this package wraps around a user's element cache
([`ADElementCache`](@ref), [`FusedFromSplit`](@ref)). A decorator holds the
wrapped cache in a field named `inner`, and the MECHANICAL half of the element
contract — everything whose decorated behaviour simply IS the inner's — is
forwarded once here rather than per decorator. Traits describing what a
decorator SERVES ([`provides_analytic`](@ref)) genuinely differ between
decorators and stay with them.

Probes against a decorated cache take one of two subjects, and which one is
the whole convention:

- a probe about an AUTHOR-WRITTEN method — the mandatory-kernel checks, the
  trait ↔ kernel check, the `condensed_corrector`/`local_conditions!` hook
  probes — runs on the [`unwrap`](@ref) fixpoint. The forwarding methods below
  answer `hasmethod` for every inner, so probing the wrapper would pass a
  cache that implements nothing.
- a probe about SERVED CAPABILITY — [`provides_analytic`](@ref),
  [`assert_sensitivity_admissible`](@ref) — runs on the DECORATED type. That is
  the cache the engine calls, and a decorator serves kinds its inner does not.
"""
abstract type AbstractElementCacheDecorator{Inner} <: AbstractVolumetricElementCache end

"""
    unwrap(cache) -> cache
    unwrap(T::Type) -> Type

The cache an author wrote: `cache` itself, or the innermost cache a chain of
[`AbstractElementCacheDecorator`](@ref)s wraps. Defined on values and on types,
since the probes needing it come in both shapes.
"""
unwrap(cache) = cache
unwrap(d::AbstractElementCacheDecorator) = unwrap(d.inner)
unwrap(::Type{<:AbstractElementCacheDecorator{Inner}}) where {Inner} = unwrap(Inner)

query_cell_parameters(d::AbstractElementCacheDecorator, cell, p) = query_cell_parameters(d.inner, cell, p)
query_facet_parameters(d::AbstractElementCacheDecorator, cell, local_facet_index, p) =
    query_facet_parameters(d.inner, cell, local_facet_index, p)
Ferrite.getnquadpoints(d::AbstractElementCacheDecorator) = getnquadpoints(d.inner)
reinit_values!(d::AbstractElementCacheDecorator, cell) = reinit_values!(d.inner, cell)
reinit_values!(d::AbstractElementCacheDecorator, cell, kind) = reinit_values!(d.inner, cell, kind)
allocate_element_matrix(d::AbstractElementCacheDecorator, sdh) = allocate_element_matrix(d.inner, sdh)
allocate_element_unknown_vector(d::AbstractElementCacheDecorator, sdh) = allocate_element_unknown_vector(d.inner, sdh)
allocate_element_residual_vector(d::AbstractElementCacheDecorator, sdh) = allocate_element_residual_vector(d.inner, sdh)
evaluate_cell_functional(kind, d::AbstractElementCacheDecorator, args) = evaluate_cell_functional(kind, d.inner, args)
evaluate_algebraic_functional(kind, d::AbstractElementCacheDecorator, args) =
    evaluate_algebraic_functional(kind, d.inner, args)
query_element_quadrature_data(d::AbstractElementCacheDecorator, cell, ivh, q::QVector) =
    query_element_quadrature_data(d.inner, cell, ivh, q)
store_quadrature_data!(q::QVector, qe, cell, ivh, d::AbstractElementCacheDecorator) =
    store_quadrature_data!(q, qe, cell, ivh, d.inner)
has_internal_state(::Type{<:AbstractElementCacheDecorator{Inner}}) where {Inner} = has_internal_state(Inner)
internal_state_insensitive(::Type{<:AbstractElementCacheDecorator{Inner}}, kind) where {Inner} =
    internal_state_insensitive(Inner, kind)
get_number_of_internal_dofs_per_element(model, d::AbstractElementCacheDecorator, sdh) =
    get_number_of_internal_dofs_per_element(model, d.inner, sdh)
get_number_of_internal_dofs_per_algebraic_item(model, d::AbstractElementCacheDecorator, items) =
    get_number_of_internal_dofs_per_algebraic_item(model, d.inner, items)
condense_cell!(d::AbstractElementCacheDecorator, args, weights) = condense_cell!(d.inner, args, weights)
condense_algebraic!(d::AbstractElementCacheDecorator, args, weights) = condense_algebraic!(d.inner, args, weights)
condensed_corrector(d::AbstractElementCacheDecorator, args) = condensed_corrector(d.inner, args)
local_conditions!(L, d::AbstractElementCacheDecorator, args) = local_conditions!(L, d.inner, args)
invalidate_correctors!(d::AbstractElementCacheDecorator) = invalidate_correctors!(d.inner)
# Patch assembly (experimental) is a separate protocol with no AD fallback of
# its own — pass through to whatever the inner implements.
assemble_patch_cell!(req, d::AbstractElementCacheDecorator, args, data) =
    assemble_patch_cell!(req, d.inner, args, data)

####################################
## AD decorator (ForwardDiff over the residual kernel)
####################################
#
# Limitations:
# - AD sweeps cover the volumetric kernel only. A boundary term riding the cell
#   sweep is therefore NOT captured by a sensitivity sweep; a `facet_items`
#   term is, through its own traversal's analytic facet kernel.
# - State sweeps (∂F/∂u Jacobian, JVP, VJP, ∂F/∂t) run over per-worker
#   preallocated ForwardDiff configs, owned by this decorator. The parameter
#   sweeps build their configs per call: their seed dimension nθ is call-time
#   knowledge, and a cached config would be abstractly typed across nθ
#   changes. The `local_conditions!` route allocates per item on top of that —
#   a configuration-free seed of the hook, plus the local operator's pivots —
#   so a condensed θ/t sweep taking that route is not allocation-free per cell
#   the way the plain state and time sweeps are.

"Tag for the package-owned ForwardDiff configs, so per-worker configs outlive the per-cell closures."
struct FerriteOperatorsADTag end

"""
    ForwardDiffAD()

The default `ADElementCache` backend: ForwardDiff over the residual kernel.
The AD backend is a hub seam — a downstream extension implements its own
buffer struct and `assemble_cell!` methods for its own marker type.
"""
struct ForwardDiffAD end

"""
    ADElementBuffers

Per-worker seeds and ForwardDiff configs for `ADElementCache`. The
output/payload half of the buffer split lives on `SensitivityBuffers`
instead, since a request's destination is needed whether the resolved cache
ends up analytic or decorated.
"""
@concrete mutable struct ADElementBuffers
    re         # ndofs-sized scratch primal-residual output for AD closures
    jac_cfg    # ∂F/∂u JacobianConfig (fixed chunk, package tag), ndofs × ndofs
    deriv_cfg  # scalar-seed DerivativeConfig — JVP directional sweep and ∂F/∂t
    grad_cfg   # state-VJP GradientConfig over the unknown buffer
    u_dual     # single-partial Dual unknown buffer for the JVP direction
    re_dual    # Dual residual buffer for the state-VJP closure
    wseed      # zeroed unknown-sized seed point of the weighted-Jacobian sweep
    wdual      # Dual slot buffers of the weighted sweep, grown to the slot count on first use
    θ          # flat primal parameter copy (nθ), grown on first use
    jac_cfg_q  # ∂F/∂q JacobianConfig, ndofs × nqp — `nothing` unless the inner carries internal state
    Kq         # ndofs × nqp scratch for the condensed generic Consistent combination, or `nothing`
    L_cfg      # ∂L/∂q JacobianConfig, nqp × nqp (`local_conditions!`), or `nothing`
    Lₑ         # nqp local-conditions output of the differentiated calls, or `nothing`
    Lq         # nqp × nqp ∂L/∂q, consumed by its own factorization, or `nothing`
    Lθ         # nqp × nθ ∂L/∂θ, overwritten by dq/dθ; grown on first use
    qsc        # nqp scratch: ∂L/∂t, and the parameter VJP's (∂F/∂q)ᵀλ
end

function create_ad_element_buffers(inner, sdh, n_global_dofs::Int = 0)
    vₑ = pad_element_vector(allocate_element_unknown_vector(inner, sdh), n_global_dofs)
    re = pad_element_vector(allocate_element_residual_vector(inner, sdh), n_global_dofs)
    return _create_ad_element_buffers(inner, vₑ, re, true)
end

# The size-based path: an item family whose local system is described by a dof
# count alone (an algebraic item) has no `SubDofHandler` to allocate against,
# and no `getnquadpoints` to size the `:q`-Jacobian config the generic
# `Consistent` bootstrap needs — that combination is cell-only (see
# `condense_algebraic!`), so this path never builds it, whatever `inner`
# declares.
create_ad_element_buffers(inner, ndofs::Int, ::Type{T}) where {T} =
    _create_ad_element_buffers(inner, zeros(T, ndofs), zeros(T, ndofs), false)

# `supports_q_bootstrap` is the cell-vs-algebraic-item structural distinction
# the two constructors above already carry: `getnquadpoints` is a cell-cache
# contract, so an algebraic-sized inner must never be queried for it even when
# `has_internal_state` — its `jac_cfg_q`/`Kq` stay `nothing` unconditionally.
function _create_ad_element_buffers(inner, vₑ, re, supports_q_bootstrap::Bool)
    T   = eltype(re)
    tag       = ForwardDiff.Tag{FerriteOperatorsADTag, T}()
    chunk     = ForwardDiff.Chunk(vₑ)
    jac_cfg   = ForwardDiff.JacobianConfig(nothing, re, vₑ, chunk, tag)
    deriv_cfg = ForwardDiff.DerivativeConfig(nothing, re, zero(T), tag)
    grad_cfg  = ForwardDiff.GradientConfig(nothing, vₑ, chunk, tag)
    u_dual    = similar(vₑ, ForwardDiff.Dual{typeof(tag), T, 1})
    re_dual   = similar(re, eltype(grad_cfg.duals))
    wseed     = zero(vₑ)
    wdual     = Vector{Vector{eltype(jac_cfg.duals[2])}}()
    ndofs     = length(re)
    jac_cfg_q = nothing
    Kq        = nothing
    L_cfg     = nothing
    Lₑ        = nothing
    Lq        = nothing
    if supports_q_bootstrap && has_internal_state(typeof(inner))
        nqp       = getnquadpoints(inner)
        qseed     = zeros(T, nqp)
        chunk_q   = ForwardDiff.Chunk(qseed)
        jac_cfg_q = ForwardDiff.JacobianConfig(nothing, re, qseed, chunk_q, tag)
        Kq        = zeros(T, ndofs, nqp)
        # The local conditions are nqp-valued, so their ∂/∂q configuration is
        # square where `jac_cfg_q` is ndofs × nqp and cannot be shared.
        Lₑ        = zeros(T, nqp)
        L_cfg     = ForwardDiff.JacobianConfig(nothing, Lₑ, qseed, chunk_q, tag)
        Lq        = zeros(T, nqp, nqp)
    end
    nq = Lₑ === nothing ? 0 : length(Lₑ)
    return ADElementBuffers(re, jac_cfg, deriv_cfg, grad_cfg, u_dual, re_dual, wseed, wdual, Vector{T}(),
                            jac_cfg_q, Kq, L_cfg, Lₑ, Lq, Matrix{T}(undef, nq, 0), _copy_or_nothing(Lₑ))
end

_copy_or_nothing(::Nothing) = nothing
_copy_or_nothing(x) = copy(x)

function duplicate_for_device(device, b::ADElementBuffers)
    return ADElementBuffers(
        copy(b.re), b.jac_cfg, b.deriv_cfg, b.grad_cfg,
        copy(b.u_dual), copy(b.re_dual), copy(b.wseed), copy.(b.wdual), copy(b.θ),
        b.jac_cfg_q, _copy_or_nothing(b.Kq),
        b.L_cfg, _copy_or_nothing(b.Lₑ), _copy_or_nothing(b.Lq), copy(b.Lθ), _copy_or_nothing(b.qsc),
    )
end

"""
    weighted_seed_buffers!(buf::ADElementBuffers, nslots) -> Vector of Dual buffers

Grow the weighted sweep's Dual slot buffers to `nslots` entries, one per
participating slot. Sized once per worker on the first sweep of a given slot
count; the Dual type is fixed by the sweep's ∂F/∂u configuration.
"""
function weighted_seed_buffers!(buf::ADElementBuffers, nslots::Int)
    while length(buf.wdual) < nslots
        push!(buf.wdual, similar(buf.wseed, eltype(buf.jac_cfg.duals[2])))
    end
    return buf.wdual
end

# The flat parameter copy: input to the AD closure, hence decorator-owned
# (unlike Bₑ/gθ, its outputs, which live on the engine's SensitivityBuffers).
function _theta!(buf::ADElementBuffers, nθ::Int)
    length(buf.θ) == nθ || (buf.θ = Vector{eltype(buf.θ)}(undef, nθ))
    return buf.θ
end

# ∂L/∂θ, sized like every other parameter-sized member: nθ is call-time
# knowledge, so the block is grown on first use of a given seed dimension.
function _local_theta_block!(buf::ADElementBuffers, nθ::Int)
    size(buf.Lθ, 2) == nθ || (buf.Lθ = Matrix{eltype(buf.Lθ)}(undef, size(buf.Lθ, 1), nθ))
    return buf.Lθ
end

# slot-sized differentiation config: :q needs its own (nqp generally ≠ ndofs);
# every other slot shares the :u-sized config, matching today's assumption
# that non-condensed state slots are all field-dof-shaped.
_jac_config_for(buf::ADElementBuffers, ::Val{:q}) = buf.jac_cfg_q
_jac_config_for(buf::ADElementBuffers, ::Val{slot}) where {slot} = buf.jac_cfg

"""
    ADElementCache{Inner, Backend, Buffers} <: AbstractVolumetricElementCache
    ADElementCache(inner, sdh; backend = ForwardDiffAD(), n_global_dofs = 0)
    ADElementCache(inner, ndofs::Int, ::Type{T} = Float64; backend = ForwardDiffAD())

Decorates `inner`'s mandatory residual kernel with automatic differentiation,
serving every request `inner` does not provide analytically — per request,
`inner`'s own kernel where declared ([`provides_analytic`](@ref) forwards) or
ForwardDiff otherwise; the engine never forks between the two itself. Which
kernel is differentiated follows from the args record: `assemble_cell!` for a
[`CellArgs`](@ref), `assemble_algebraic!` for an [`AlgebraicArgs`](@ref), so one
decorator serves both item families.

For a condensed `inner` ([`has_internal_state`](@ref)) the `Consistent`
Jacobian/JacobianResidual has a generic path: `∂F/∂ū|_q` and `∂F/∂q` by AD,
combined with the `dq/dū` block from [`condensed_corrector`](@ref) —
`Jₑ = ∂F/∂ū|_q + ∂F/∂q · dq/dū`. Without an analytic kernel or
`condensed_corrector`, admissibility rejects it, naming the missing correction.

The parameter and time kinds have the same shape of generic path, from
[`local_conditions!`](@ref) instead of a stored block: `∂L/∂q` (factorized once
per item), `∂L/∂θ` and `∂L/∂t` by AD of the hook give `dq/dθ` and `dq/dt`
through the implicit function theorem, and the same `∂F/∂q` block completes
the total. Without the hook those kinds stay rejected for a condensed inner.

`n_global_dofs` pads the seeds and configs by the subdomain's
[`global_dofs`](@ref) count, so the differentiated system is the FULL augmented
one — `setup_operator` passes it. The `ndofs` form sizes the buffers from a dof
count alone, for an item family that has no `SubDofHandler` to allocate against.

`setup_operator` wraps automatically (`ad_backend = nothing` opts out);
hand-constructing an instance wraps a specific cache or tests the decorator.
"""
struct ADElementCache{Inner, Backend, Buffers} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
    backend::Backend
    buffers::Buffers
end
function ADElementCache(inner, sdh; backend = ForwardDiffAD(), n_global_dofs::Int = 0)
    _reject_condensed_global_dofs(inner, n_global_dofs)
    return ADElementCache(inner, backend, create_ad_element_buffers(inner, sdh, n_global_dofs))
end
ADElementCache(inner, ndofs::Int, ::Type{T} = Float64; backend = ForwardDiffAD()) where {T} =
    ADElementCache(inner, backend, create_ad_element_buffers(inner, ndofs, T))

# The generic `Consistent` combination multiplies the padded ∂F/∂q block by the
# `nq × ndofs_per_cell` corrector `condensed_corrector` returns — the FIELD
# space, while the padded partials span the augmented system, so the product is
# not even conformable. Rejected where the decorator is built rather than as a
# `DimensionMismatch` deep inside a sweep.
function _reject_condensed_global_dofs(inner, n_global_dofs::Int)
    (n_global_dofs == 0 || !has_internal_state(typeof(inner))) && return nothing
    provides_analytic(typeof(inner), JacobianKind{:u, Consistent}()) && return nothing
    throw(ArgumentError(
        "$(nameof(unwrap(typeof(inner)))) carries condensed internal state and sits " *
        "on a subdomain declaring $(n_global_dofs) `global_dofs`. Its `Consistent` Jacobian " *
        "would go through the generic combination `∂F/∂ū|_q + ∂F/∂q · dq/dū`, whose corrector " *
        "block spans the FIELD space while the AD partials span the augmented system — the " *
        "combination is not defined for this pair. Implement the analytic " *
        "`assemble_cell!(::JacobianRequest{:u, Consistent}, …)` kernel (declared through " *
        "`provides_analytic`), or drop the `global_dofs` declaration on this subdomain."))
end

duplicate_for_device(device, ad::ADElementCache) =
    ADElementCache(duplicate_for_device(device, ad.inner), ad.backend, duplicate_for_device(device, ad.buffers))

"""
    condensed_corrector(cache, args) -> AbstractMatrix

The completed `nq × ndofs` `dq/dū` block a condensed cache exposes for
`ADElementCache`'s generic `Consistent` combination. Only needed to admit the
generic bootstrap; a cache serving `Consistent` analytically never needs this.
No default.

This is the decorator's corrector ACCESS POINT, and it takes the item's
[`CellArgs`](@ref) rather than an item id so that both
[`CorrectorElection`](@ref)s can be served through it: a `Stored()` cache reads
its store keyed by `cellid(args.cell)`, a [`Recompute`](@ref) one re-derives
the block from `args.states`. The decorator does not know which, and the
combination `∂F/∂ū|_q + ∂F/∂q · dq/dū` is the same either way.
"""
function condensed_corrector end

# Capability: plain AD-from-residual is exact for every kind whose fallback
# does not differentiate through a condensed inner's local state —
# same rule `requires_admissibility_check` names. The ONE kind this decorator
# ALSO covers generically for a condensed inner is the state Consistent
# Jacobian, given the inner's stored corrector block.
#
# `requires_admissibility_check` is false for `TimeSensitivityKind` because
# setup cannot know whether the AD path even runs (FiniteDifferenceSensitivity
# is a call-time escape) — it does NOT mean the plain AD fallback is
# admissible on a condensed inner, so it gets an explicit override instead of
# the shortcut every other exempt kind uses.
_ad_covers(Inner, kind) = !requires_admissibility_check(kind) || !has_internal_state(Inner) || internal_state_insensitive(Inner, kind)
# The θ/t kinds gain the THIRD acceptance branch: a declared `local_conditions!`
# lets the decorator derive `dq/dθ`/`dq/dt` and complete the total itself.
_ad_covers(Inner, ::TimeSensitivityKind) =
    !has_internal_state(Inner) || internal_state_insensitive(Inner, TimeSensitivityKind()) || _has_local_conditions(Inner)
_ad_covers(Inner, ::ParameterJacobianKind) =
    !has_internal_state(Inner) || internal_state_insensitive(Inner, ParameterJacobianKind()) || _has_local_conditions(Inner)
_ad_covers(Inner, kind::ParameterVJPKind) =
    !has_internal_state(Inner) || internal_state_insensitive(Inner, kind) || _has_local_conditions(Inner)
_ad_covers(Inner, ::JacobianKind{:u, Consistent}) =
    !has_internal_state(Inner) || internal_state_insensitive(Inner, JacobianKind{:u, Consistent}()) || _has_condensed_corrector(Inner)
_ad_covers(Inner, ::JacobianResidualKind{Consistent}) =
    !has_internal_state(Inner) || internal_state_insensitive(Inner, JacobianResidualKind{Consistent}()) || _has_condensed_corrector(Inner)

# Author-written-method probes, so the subject is the `unwrap` fixpoint —
# `Inner` is itself a decorator whenever a split-analytic cache was fused
# first, and the shared forwarding layer answers for any inner.
#
# Probed against `CellArgs` specifically: the generic combination needs the
# `:q` Jacobian config, which only the cell-sized buffer constructor builds
# (`supports_q_bootstrap`), so an algebraic cache's method — written against
# `AlgebraicArgs` — must not be read as coverage. Same reasoning for the
# local-conditions hook, whose `L` argument is Dual-valued under the sweeps
# that differentiate it and so can be typed no tighter than `AbstractVector`.
_has_condensed_corrector(Inner) = hasmethod(condensed_corrector, Tuple{unwrap(Inner), CellArgs})
_has_local_conditions(Inner) = hasmethod(local_conditions!, Tuple{AbstractVector, unwrap(Inner), CellArgs})

# `true` for every kind `inner` serves analytically (forwarded), plus every
# kind this decorator's own AD path covers — every kind except the
# `Consistent` state Jacobian/JacobianResidual on a condensed `inner` lacking
# both the analytic kernel and `condensed_corrector`/`internal_state_insensitive`.
provides_analytic(::Type{<:ADElementCache{Inner}}, kind) where {Inner} =
    provides_analytic(Inner, kind) || _ad_covers(Inner, kind)
# `WeightedJacobianKind` is exempt from the AD-admissibility broadening: two
# call sites (`_check_differentiated_slot`'s AffineRate-with-AD rejection,
# `_fused_weighted_route`) read `provides_analytic` as "has a REAL analytic
# kernel", not "is AD admissible" — `_fused_weighted_route` already carries
# its own `!has_internal_state`/`internal_state_insensitive` fallback, so
# nothing needs this decorator to also claim coverage here.
provides_analytic(::Type{<:ADElementCache{Inner}}, kind::WeightedJacobianKind) where {Inner} =
    provides_analytic(Inner, kind)

####################################
## The seeding entries
####################################

# The decorator differentiates a residual kernel, and WHICH kernel that is
# follows from the args record: a cell item reaches `inner` through
# `assemble_cell!`, an algebraic item through `assemble_algebraic!`. Routing the
# inner call here is what makes one set of AD paths serve both item families.
_inner_kernel!(req, cache, args::CellArgs) = assemble_cell!(req, cache, args)
_inner_kernel!(req, cache, args::AlgebraicArgs) = assemble_algebraic!(req, cache, args)

# What `query_cell_parameters` is keyed on: the geometry cache of a cell item,
# the item record of an algebraic one.
_parameter_subject(args::CellArgs) = args.cell
_parameter_subject(args::AlgebraicArgs) = args.item

# The two public entry points of the decorator, one per item family; both
# resolve into the same seeding bodies below.
assemble_cell!(req::AbstractAssemblyRequest, ad::ADElementCache, args) = _ad_assemble!(req, ad, args)
assemble_algebraic!(req::AbstractAssemblyRequest, ad::ADElementCache, args) = _ad_assemble!(req, ad, args)

# A downstream request type outside the ones this decorator knows how to
# differentiate (e.g. a custom kind riding `primal_cell_sweep!`) has no AD
# fallback here — forward to `inner`, exactly like the mandatory residual
# kernel. `Inner`'s own `provides_analytic` is what setup-time validation
# checks (`_assert_trait_backed` redirects to `Inner`), so an unbacked claim
# still fails loudly there rather than as an opaque `MethodError` here.
_ad_assemble!(req::AbstractAssemblyRequest, ad::ADElementCache, args) = _inner_kernel!(req, ad.inner, args)

# Evaluate the item's local residual for `args`, overwriting `r`.
function evaluate_item_residual!(r, cache, args)
    fill!(r, zero(eltype(r)))
    _inner_kernel!(ResidualRequest(r), cache, args)
    return r
end

# `:q` is the one slot whose config is conditional — only the cell-sized buffer
# constructor builds it (`supports_q_bootstrap`) — so an algebraic cache
# reaches an AD `:q` sweep with `nothing` and is told so here instead of
# failing inside ForwardDiff. The branch is on a field TYPE and folds away.
_require_slot_config(cfg, ::Val, ad) = cfg
_require_slot_config(::Nothing, ::Val{slot}, ad::ADElementCache{Inner}) where {slot, Inner} = throw(ArgumentError(
    "$(nameof(unwrap(Inner))) has no ForwardDiff configuration for the `:$slot` " *
    "slot. `:q` configurations are sized from `getnquadpoints`, a CELL-cache contract, so an " *
    "item family described by a dof count alone (an algebraic item) has none: serve " *
    "`JacobianKind{:$slot}` with the analytic `assemble_algebraic!` kernel instead."))

# ∂F/∂slot — writes the Jacobian into K, overwriting `y` with the primal
# residual as a byproduct. Only the named slot is seeded; every other slot
# stays at its primal value — including `AffineRate`-reconstructed slots,
# which are formed at gather time. Tag checking is off: the config carries the
# package tag, not the closure's.
function ad_state_jacobian!(K, y, ad::ADElementCache, args, ::Val{slot} = Val(:u)) where {slot}
    cfg = _require_slot_config(_jac_config_for(ad.buffers, Val(slot)), Val(slot), ad)
    inner = ad.inner
    f! = (r, x) -> evaluate_item_residual!(
        r, inner, with_states(args, merge(args.states, NamedTuple{(slot,)}((x,)))))
    ForwardDiff.jacobian!(K, f!, y, args.states[slot], cfg, Val{false}())
    return K
end

# The generic Consistent combination for a condensed inner: AD gives both
# partials, `condensed_corrector` gives the dq/dū block — read from a store or
# re-derived from `args`, which is the inner's election to make.
function _condensed_consistent_jacobian!(K, y, ad::ADElementCache, args)
    buf = ad.buffers
    ad_state_jacobian!(K, y, ad, args, Val(:u))
    ad_state_jacobian!(buf.Kq, buf.re, ad, args, Val(:q))
    corr = condensed_corrector(ad.inner, args)
    mul!(K, buf.Kq, corr, true, true)
    return K
end

function _ad_assemble!(req::JacobianRequest{:u, Consistent}, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, JacobianKind{:u, Consistent}())
        _inner_kernel!(req, ad.inner, args)
    elseif has_internal_state(Inner)
        _condensed_consistent_jacobian!(req.K, ad.buffers.re, ad, args)
    else
        ad_state_jacobian!(req.K, ad.buffers.re, ad, args, Val(:u))
    end
    return req.K
end
function _ad_assemble!(req::JacobianRequest{slot, C}, ad::ADElementCache{Inner}, args) where {Inner, slot, C <: CorrectionMode}
    if provides_analytic(Inner, JacobianKind{slot, C}())
        _inner_kernel!(req, ad.inner, args)
    else
        ad_state_jacobian!(req.K, ad.buffers.re, ad, args, Val(slot))
    end
    return req.K
end

function _ad_assemble!(req::JacobianResidualRequest{Consistent}, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, JacobianResidualKind{Consistent}())
        _inner_kernel!(req, ad.inner, args)
    elseif has_internal_state(Inner)
        _condensed_consistent_jacobian!(req.K, req.r, ad, args)
    else
        ad_state_jacobian!(req.K, req.r, ad, args, Val(:u))
    end
    return req
end
function _ad_assemble!(req::JacobianResidualRequest{C}, ad::ADElementCache{Inner}, args) where {Inner, C <: CorrectionMode}
    if provides_analytic(Inner, JacobianResidualKind{C}())
        _inner_kernel!(req, ad.inner, args)
    else
        ad_state_jacobian!(req.K, req.r, ad, args, Val(:u))
    end
    return req
end

# Σₛ wₛ ∂F/∂s in ONE sweep: the seed variable `x` is the weighted variation
# itself, so every participating slot enters as `sₑ + wₛ·x` and the derivative
# w.r.t. `x` at `x = 0` is exactly the weighted combination. Slots outside
# `weights` (including `AffineRate` reconstructions) stay at their primal
# value, matching the frozen-slot contract of `JacobianKind`.
function _ad_assemble!(req::WeightedJacobianRequest{C}, ad::ADElementCache{Inner}, args) where {C, Inner}
    kind = WeightedJacobianKind{keys(req.weights), C}(req.weights)
    if provides_analytic(Inner, kind)
        return _inner_kernel!(req, ad.inner, args)
    end
    buf   = ad.buffers
    slots = keys(req.weights)
    bufs  = weighted_seed_buffers!(buf, length(slots))
    prim  = NamedTuple{slots}(args.states)
    duals = NamedTuple{slots}(ntuple(i -> bufs[i], Val(length(slots))))
    dargs = with_states(args, merge(args.states, duals))
    inner = ad.inner
    weights = req.weights
    f! = (r, x) -> begin
        map((b, sₑ, w) -> (@. b = sₑ + w * x), duals, prim, weights)
        evaluate_item_residual!(r, inner, dargs)
    end
    ForwardDiff.jacobian!(req.K, f!, buf.re, buf.wseed, buf.jac_cfg, Val{false}())
    return req.K
end

# ∂F/∂θ — dense local parameter Jacobian. The parameter sweep re-queries the
# element parameters from the Dual-rebuilt global `p` (`req.p`, NOT `args.p` —
# already the element-local view) so wrappers forward Duals transparently.
# For a condensed inner this is the FROZEN-q partial; the total needs the
# `∂F/∂q · dq/dθ` block a declared `local_conditions!` supplies.
function _ad_assemble!(req::ParameterJacobianRequest, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, ParameterJacobianKind())
        return _inner_kernel!(req, ad.inner, args)
    end
    _ad_parameter_jacobian!(req.B, ad, args, req.p)
    _has_local_conditions(Inner) && _add_local_parameter_correction!(req.B, ad, args, req.p)
    return req.B
end

function _ad_parameter_jacobian!(B, ad::ADElementCache, args, p)
    buf = ad.buffers
    inner = ad.inner
    θ = copyto!(_theta!(buf, size(B, 2)), parameter_vector(p))
    f! = (r, θᵢ) -> begin
        pₑ = query_cell_parameters(inner, _parameter_subject(args), rebuild_parameters(p, θᵢ))
        evaluate_item_residual!(r, inner, with_parameters(args, pₑ))
    end
    ForwardDiff.jacobian!(B, f!, buf.re, θ)
    return B
end

# (∂F/∂θ)ᵀλₑ — adjoint pullback as the gradient of the scalar λₑ·rₑ(θ).
function _ad_assemble!(req::ParameterVJPRequest, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, ParameterVJPKind(nothing))
        return _inner_kernel!(req, ad.inner, args)
    end
    _ad_parameter_vjp!(req.g, ad, args, req.λₑ, req.p)
    _has_local_conditions(Inner) && _add_local_parameter_vjp!(req.g, ad, args, req.λₑ, req.p)
    return req.g
end

function _ad_parameter_vjp!(g, ad::ADElementCache, args, λₑ, p)
    buf = ad.buffers
    inner = ad.inner
    θ = copyto!(_theta!(buf, length(parameter_vector(p))), parameter_vector(p))
    fscalar = θᵢ -> begin
        pₑ = query_cell_parameters(inner, _parameter_subject(args), rebuild_parameters(p, θᵢ))
        r = zeros(eltype(θᵢ), length(λₑ))
        evaluate_item_residual!(r, inner, with_parameters(args, pₑ))
        return dot(λₑ, r)
    end
    ForwardDiff.gradient!(g, fscalar, θ)
    return g
end

# ∂F/∂t — explicit time dependence, seeded through the context channel: the
# sweep's ctx is rebuilt with a Dual evaluation time, so an element reading
# `evaluation_time(args.ctx)` differentiates exactly. The preallocated config
# is typed for the residual eltype; exotic time types fall back to a per-call
# config.
function _ad_assemble!(req::TimeSensitivityRequest, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, TimeSensitivityKind())
        return _inner_kernel!(req, ad.inner, args)
    end
    _ad_time_sensitivity!(req.g, ad, args)
    _has_local_conditions(Inner) && _add_local_time_correction!(req.g, ad, args)
    return req.g
end

function _ad_time_sensitivity!(g, ad::ADElementCache, args)
    buf = ad.buffers
    inner = ad.inner
    ctx = args.ctx
    t = evaluation_time(ctx)
    f! = (r, t̃) -> evaluate_item_residual!(r, inner, with_context(args, with_time(ctx, t̃)))
    if t isa eltype(buf.re)
        ForwardDiff.derivative!(g, f!, buf.re, t, buf.deriv_cfg, Val{false}())
    else
        ForwardDiff.derivative!(g, f!, buf.re, t)
    end
    return g
end

####################################
## The local-conditions route — dq/dθ and dq/dt from `local_conditions!`
####################################
#
# Post-condensation the residual kernel is pure, so the AD paths above compute
# frozen-q PARTIALS. The kinds carrying no `CorrectionMode` are always totals,
# and the missing block is `∂F/∂q · dq/dseed` with `dq/dseed` given by the
# implicit function theorem on the element's declared local conditions. Both
# factors already exist: `∂F/∂q` is the `:q` slot Jacobian the Kq machinery
# assembles, and `∂L/∂q`/`∂L/∂seed` come from differentiating the hook.

# ∂L/∂q at the condensed state, factorized in place — the local inverse every
# seed of the sweep goes through, and the reason the hook is worth its cost:
# ONE factorization per item serves all nθ parameter directions. It is redone
# per sweep, not retained across sweeps: retaining it is corrector storage
# again, which is what `CorrectorElection` exists to let a user refuse.
function _local_operator_factorization(ad::ADElementCache, args)
    buf = ad.buffers
    inner = ad.inner
    f! = (L, x) -> local_conditions!(L, inner, with_states(args, merge(args.states, (q = x,))))
    ForwardDiff.jacobian!(buf.Lq, f!, buf.Lₑ, args.states.q, buf.L_cfg, Val{false}())
    return lu!(buf.Lq)
end

# `(∂L/∂q)⁻¹ ∂L/∂θ`, in place in the θ-sized block — `dq/dθ` up to the sign the
# callers fold into their accumulation.
function _local_parameter_slopes!(ad::ADElementCache, args, p, nθ::Int)
    buf = ad.buffers
    inner = ad.inner
    Lθ = _local_theta_block!(buf, nθ)
    θ = copyto!(_theta!(buf, nθ), parameter_vector(p))
    f! = (L, θᵢ) -> begin
        pₑ = query_cell_parameters(inner, _parameter_subject(args), rebuild_parameters(p, θᵢ))
        local_conditions!(L, inner, with_parameters(args, pₑ))
    end
    ForwardDiff.jacobian!(Lθ, f!, buf.Lₑ, θ)
    ldiv!(_local_operator_factorization(ad, args), Lθ)
    return Lθ
end

# B += ∂F/∂q · dq/dθ
function _add_local_parameter_correction!(B, ad::ADElementCache, args, p)
    buf = ad.buffers
    ad_state_jacobian!(buf.Kq, buf.re, ad, args, Val(:q))
    Lθ = _local_parameter_slopes!(ad, args, p, size(B, 2))
    mul!(B, buf.Kq, Lθ, -one(eltype(B)), one(eltype(B)))
    return B
end

# g += (dq/dθ)ᵀ (∂F/∂q)ᵀ λₑ — the same two factors, contracted right to left so
# no `nres × nθ` block is materialized.
function _add_local_parameter_vjp!(g, ad::ADElementCache, args, λₑ, p)
    buf = ad.buffers
    ad_state_jacobian!(buf.Kq, buf.re, ad, args, Val(:q))
    mul!(buf.qsc, transpose(buf.Kq), λₑ)
    Lθ = _local_parameter_slopes!(ad, args, p, length(g))
    mul!(g, transpose(Lθ), buf.qsc, -one(eltype(g)), one(eltype(g)))
    return g
end

# g += ∂F/∂q · dq/dt, the time seed against the same local operator.
function _add_local_time_correction!(g, ad::ADElementCache, args)
    buf = ad.buffers
    inner = ad.inner
    ad_state_jacobian!(buf.Kq, buf.re, ad, args, Val(:q))
    ctx = args.ctx
    f! = (L, t̃) -> local_conditions!(L, inner, with_context(args, with_time(ctx, t̃)))
    ForwardDiff.derivative!(buf.qsc, f!, buf.Lₑ, evaluation_time(ctx))
    ldiv!(_local_operator_factorization(ad, args), buf.qsc)
    mul!(g, buf.Kq, buf.qsc, -one(eltype(g)), one(eltype(g)))
    return g
end

# (∂F/∂u)·v — one directional-Dual sweep through the residual kernel: the
# MFEM NONE level, no matrices anywhere. The perturbed state is written into
# the per-worker Dual buffer instead of allocating per cell.
function _ad_assemble!(req::StateJVPRequest, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, StateJVPKind(nothing))
        return _inner_kernel!(req, ad.inner, args)
    end
    buf = ad.buffers
    inner = ad.inner
    ud = buf.u_dual
    vₑ = req.vₑ
    f! = (r, s) -> begin
        @. ud = args.states.u + s * vₑ
        evaluate_item_residual!(r, inner, with_states(args, merge(args.states, (u = ud,))))
    end
    ForwardDiff.derivative!(req.Jv, f!, buf.re, zero(eltype(vₑ)), buf.deriv_cfg, Val{false}())
    return req.Jv
end

# (∂F/∂u)ᵀλₑ — gradient of the scalar λₑ·rₑ(u) w.r.t. the element state, with
# the Dual residual evaluated into the per-worker buffer.
function _ad_assemble!(req::StateVJPRequest, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, StateVJPKind(nothing))
        return _inner_kernel!(req, ad.inner, args)
    end
    buf = ad.buffers
    inner = ad.inner
    λₑ = req.λₑ
    rd = buf.re_dual
    fscalar = u -> begin
        evaluate_item_residual!(rd, inner, with_states(args, merge(args.states, (u = u,))))
        return dot(λₑ, rd)
    end
    ForwardDiff.gradient!(req.g, fscalar, args.states.u, buf.grad_cfg, Val{false}())
    return req.g
end

####################################
## FusedFromSplit — the JacobianResidualKind "issue split kernels back to back" case
####################################

"""
    FusedFromSplit(inner)

Mini-decorator for a cache that serves `JacobianKind` and the mandatory
residual analytically but not the fused `JacobianResidualKind`: issues the two
split analytic kernels back to back instead of falling back to AD. Chosen
once, at construction (`decorate_element_cache`).
"""
struct FusedFromSplit{Inner} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
end

duplicate_for_device(device, f::FusedFromSplit) = FusedFromSplit(duplicate_for_device(device, f.inner))

assemble_cell!(req::AbstractAssemblyRequest, f::FusedFromSplit, args) = assemble_cell!(req, f.inner, args)
assemble_algebraic!(req::AbstractAssemblyRequest, f::FusedFromSplit, args) = assemble_algebraic!(req, f.inner, args)
function assemble_cell!(req::JacobianResidualRequest{C}, f::FusedFromSplit, args) where {C <: CorrectionMode}
    _split_into_fused!(req, f.inner, args)
    return req
end
function assemble_algebraic!(req::JacobianResidualRequest{C}, f::FusedFromSplit, args) where {C <: CorrectionMode}
    _split_into_fused!(req, f.inner, args)
    return req
end
function _split_into_fused!(req::JacobianResidualRequest{C}, inner, args) where {C <: CorrectionMode}
    _inner_kernel!(JacobianRequest{:u, C}(req.K), inner, args)
    _inner_kernel!(ResidualRequest(req.r), inner, args)
    return req
end
provides_analytic(::Type{<:FusedFromSplit{Inner}}, kind) where {Inner} = provides_analytic(Inner, kind)
provides_analytic(::Type{<:FusedFromSplit{Inner}}, ::JacobianResidualKind{C}) where {Inner, C <: CorrectionMode} = true

# Construction-time wrapping (`decorate_element_cache`, `needs_ad_decoration`,
# `fully_analytic`) lives in operators/ad_decoration.jl — it is setup_operator's
# decision, not part of the decorator types themselves.
