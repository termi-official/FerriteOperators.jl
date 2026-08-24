####################################
## Element cache decorators — the shared forwarding layer
####################################

"""
    AbstractElementCacheDecorator{Inner} <: AbstractVolumetricElementCache

Supertype of the caches this package wraps around a user's element cache
([`ADElementCache`](@ref), [`FusedFromSplit`](@ref)). The wrapped cache sits in
a field `inner`, and everything a decorator simply inherits is forwarded once
here; traits describing what a decorator SERVES ([`provides_analytic`](@ref))
stay with the decorators.

Which subject a probe takes is the whole convention:

- AUTHOR-WRITTEN-METHOD probes (mandatory kernels, trait ↔ kernel,
  `condensed_corrector`/`local_conditions!`) run on the [`unwrap`](@ref)
  fixpoint — the forwarding methods below answer `hasmethod` for every inner,
  so probing the wrapper would pass a cache that implements nothing.
- SERVED-CAPABILITY probes ([`provides_analytic`](@ref),
  [`assert_sensitivity_admissible`](@ref)) run on the DECORATED type — that is
  the cache the engine calls, and a decorator serves kinds its inner does not.
"""
abstract type AbstractElementCacheDecorator{Inner} <: AbstractVolumetricElementCache end

"""
    unwrap(cache) -> cache
    unwrap(T::Type) -> Type

The cache an author wrote: `cache` itself, or the innermost cache a chain of
[`AbstractElementCacheDecorator`](@ref)s wraps. Defined on values and on types,
since the probes come in both shapes.
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
# Patch assembly (experimental) has no AD fallback — pass through to the inner.
assemble_patch_cell!(req, d::AbstractElementCacheDecorator, args, data) =
    assemble_patch_cell!(req, d.inner, args, data)

####################################
## AD decorator (ForwardDiff over the residual kernel)
####################################
#
# Limitations:
# - AD sweeps cover the volumetric kernel only, so a boundary term riding the
#   cell sweep is NOT captured; a `facet_items` term is, through its own
#   traversal's analytic facet kernel.
# - State sweeps (∂F/∂u, JVP, VJP, ∂F/∂t) run over per-worker preallocated
#   configs. Parameter sweeps build theirs per call — nθ is call-time knowledge
#   and a cached config would be abstractly typed across nθ changes. The
#   `local_conditions!` route allocates per item on top of that, so a condensed
#   θ/t sweep is not allocation-free per cell the way state and time sweeps are.

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

Per-worker seeds and ForwardDiff configs for `ADElementCache`. Outputs live on
`SensitivityBuffers` instead: a request's destination is needed whether the
resolved cache ends up analytic or decorated.
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

create_ad_element_buffers(inner, ndofs::Int, ::Type{T}) where {T} =
    _create_ad_element_buffers(inner, zeros(T, ndofs), zeros(T, ndofs), false)

# `supports_q_bootstrap`: the generic `Consistent` combination is cell-only (see
# `condense_algebraic!`) and its `:q` config is sized from `getnquadpoints`, a
# cell-cache contract — so an algebraic inner keeps `jac_cfg_q`/`Kq` at `nothing`
# whatever it declares.
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
        # ∂L/∂q is nqp × nqp, so the ndofs × nqp `jac_cfg_q` cannot be shared.
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
participating slot. The Dual type is fixed by the sweep's ∂F/∂u configuration.
"""
function weighted_seed_buffers!(buf::ADElementBuffers, nslots::Int)
    while length(buf.wdual) < nslots
        push!(buf.wdual, similar(buf.wseed, eltype(buf.jac_cfg.duals[2])))
    end
    return buf.wdual
end

# The flat parameter copy: an AD-closure input, hence decorator-owned.
function _theta!(buf::ADElementBuffers, nθ::Int)
    length(buf.θ) == nθ || (buf.θ = Vector{eltype(buf.θ)}(undef, nθ))
    return buf.θ
end

# ∂L/∂θ: nθ is call-time knowledge, so the block is grown on first use.
function _local_theta_block!(buf::ADElementBuffers, nθ::Int)
    size(buf.Lθ, 2) == nθ || (buf.Lθ = Matrix{eltype(buf.Lθ)}(undef, size(buf.Lθ, 1), nθ))
    return buf.Lθ
end

# Slot-sized config: `:q` needs its own (nqp generally ≠ ndofs); every other
# slot shares the `:u`-sized one, assuming non-condensed state slots are all
# field-dof-shaped.
_jac_config_for(buf::ADElementBuffers, ::Val{:q}) = buf.jac_cfg_q
_jac_config_for(buf::ADElementBuffers, ::Val{slot}) where {slot} = buf.jac_cfg

"""
    ADElementCache{Inner, Backend, Buffers} <: AbstractVolumetricElementCache
    ADElementCache(inner, sdh; backend = ForwardDiffAD(), n_global_dofs = 0)
    ADElementCache(inner, ndofs::Int, ::Type{T} = Float64; backend = ForwardDiffAD())

Decorates `inner`'s mandatory residual kernel with automatic differentiation,
serving per request whatever `inner` does not provide analytically
([`provides_analytic`](@ref) forwards); the engine never forks between the two
itself. The differentiated kernel follows from the args record —
`assemble_cell!` for a [`CellArgs`](@ref), `assemble_algebraic!` for an
[`AlgebraicArgs`](@ref) — so one decorator serves both item families.

AD-from-residual on a condensed `inner` ([`has_internal_state`](@ref)) computes
the frozen-q PARTIAL, so the totals are completed generically: `Consistent`
Jacobian/JacobianResidual as `Jₑ = ∂F/∂ū|_q + ∂F/∂q · dq/dū` with the block
[`condensed_corrector`](@ref) supplies; the parameter and time kinds with
`dq/dθ`/`dq/dt` derived from [`local_conditions!`](@ref) through the implicit
function theorem. Lacking corrector or hook, admissibility rejects the kind,
naming the missing correction.

`n_global_dofs` pads the seeds and configs by the subdomain's
[`global_dofs`](@ref) count, so the differentiated system is the FULL augmented
one — `setup_operator` passes it. The `ndofs` form sizes the buffers from a dof
count alone, for an item family that has no `SubDofHandler` to allocate against.

`setup_operator` wraps automatically (`ad_backend = nothing` opts out).
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

# Rejected where the decorator is built, rather than as a `DimensionMismatch`
# deep inside a sweep.
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
`ADElementCache`'s generic `Consistent` combination. No default; a cache
serving `Consistent` analytically never needs it.

Takes the item's [`CellArgs`](@ref) rather than an item id so that both
[`CorrectorElection`](@ref)s can be served through it: a `Stored()` cache reads
its store keyed by `cellid(args.cell)`, a [`Recompute`](@ref) one re-derives
the block from `args.states`. The combination is the same either way.
"""
function condensed_corrector end

# Capability: plain AD-from-residual is exact for every kind whose fallback does
# not differentiate through a condensed inner's local state — the rule
# `requires_admissibility_check` names.
#
# TRAP: that flag is false for `TimeSensitivityKind` only because setup cannot
# know whether the AD path even runs (FiniteDifferenceSensitivity is a call-time
# escape); it does NOT mean plain AD is admissible on a condensed inner, hence
# the explicit override below instead of the shortcut.
_ad_covers(Inner, kind) = !requires_admissibility_check(kind) || !has_internal_state(Inner) || internal_state_insensitive(Inner, kind)
# The θ/t kinds gain a THIRD branch: a declared `local_conditions!` lets the
# decorator derive `dq/dθ`/`dq/dt` and complete the total itself.
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
# `Inner` is itself a decorator whenever a split-analytic cache was fused first.
# Probed against `CellArgs` specifically: the generic combination needs the `:q`
# config, which only the cell-sized constructor builds, so an algebraic cache's
# method must not be read as coverage. `L` is typed `AbstractVector` because it
# is Dual-valued under the sweeps that differentiate the hook.
_has_condensed_corrector(Inner) = hasmethod(condensed_corrector, Tuple{unwrap(Inner), CellArgs})
_has_local_conditions(Inner) = hasmethod(local_conditions!, Tuple{AbstractVector, unwrap(Inner), CellArgs})

provides_analytic(::Type{<:ADElementCache{Inner}}, kind) where {Inner} =
    provides_analytic(Inner, kind) || _ad_covers(Inner, kind)
# `WeightedJacobianKind` is exempt from the AD-admissibility broadening:
# `_check_differentiated_slot` and `_fused_weighted_route` read
# `provides_analytic` as "has a REAL analytic kernel", not "is AD admissible",
# and the latter carries its own
# `!has_internal_state`/`internal_state_insensitive` fallback.
provides_analytic(::Type{<:ADElementCache{Inner}}, kind::WeightedJacobianKind) where {Inner} =
    provides_analytic(Inner, kind)

####################################
## The seeding entries
####################################

# Routing the inner call on the args record is what makes one set of AD paths
# serve both item families.
_inner_kernel!(req, cache, args::CellArgs) = assemble_cell!(req, cache, args)
_inner_kernel!(req, cache, args::AlgebraicArgs) = assemble_algebraic!(req, cache, args)

# The subject `query_cell_parameters` is keyed on, per item family.
_parameter_subject(args::CellArgs) = args.cell
_parameter_subject(args::AlgebraicArgs) = args.item

# Both public entry points resolve into the same seeding bodies below.
assemble_cell!(req::AbstractAssemblyRequest, ad::ADElementCache, args) = _ad_assemble!(req, ad, args)
assemble_algebraic!(req::AbstractAssemblyRequest, ad::ADElementCache, args) = _ad_assemble!(req, ad, args)

# A request type this decorator cannot differentiate (a downstream kind riding
# `primal_cell_sweep!`) forwards to `inner` like the residual kernel. An
# unbacked `provides_analytic` claim still fails loudly at setup, where
# `_assert_trait_backed` redirects to `Inner`, not as a `MethodError` here.
_ad_assemble!(req::AbstractAssemblyRequest, ad::ADElementCache, args) = _inner_kernel!(req, ad.inner, args)

# Evaluate the item's local residual for `args`, overwriting `r`.
function evaluate_item_residual!(r, cache, args)
    fill!(r, zero(eltype(r)))
    _inner_kernel!(ResidualRequest(r), cache, args)
    return r
end

# An algebraic cache reaches an AD `:q` sweep with no config and is told so here
# instead of failing inside ForwardDiff. The branch is on a field TYPE and folds
# away.
_require_slot_config(cfg, ::Val, ad) = cfg
_require_slot_config(::Nothing, ::Val{slot}, ad::ADElementCache{Inner}) where {slot, Inner} = throw(ArgumentError(
    "$(nameof(unwrap(Inner))) has no ForwardDiff configuration for the `:$slot` " *
    "slot. `:q` configurations are sized from `getnquadpoints`, a CELL-cache contract, so an " *
    "item family described by a dof count alone (an algebraic item) has none: serve " *
    "`JacobianKind{:$slot}` with the analytic `assemble_algebraic!` kernel instead."))

# ∂F/∂slot — writes K, overwriting `y` with the primal residual. Only the named
# slot is seeded; every other stays primal, including `AffineRate`-reconstructed
# slots, which are formed at gather time. Tag checking is off because the config
# carries the package tag, not the closure's.
function ad_state_jacobian!(K, y, ad::ADElementCache, args, ::Val{slot} = Val(:u)) where {slot}
    cfg = _require_slot_config(_jac_config_for(ad.buffers, Val(slot)), Val(slot), ad)
    inner = ad.inner
    f! = (r, x) -> evaluate_item_residual!(
        r, inner, with_states(args, merge(args.states, NamedTuple{(slot,)}((x,)))))
    ForwardDiff.jacobian!(K, f!, y, args.states[slot], cfg, Val{false}())
    return K
end

# The generic Consistent combination for a condensed inner: AD gives both
# partials, `condensed_corrector` the dq/dū block.
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

# Σₛ wₛ ∂F/∂s in ONE sweep: the seed `x` is the weighted variation itself, so
# each participating slot enters as `sₑ + wₛ·x` and d/dx at `x = 0` is exactly
# the weighted combination. Slots outside `weights` (including `AffineRate`
# reconstructions) stay primal, matching `JacobianKind`'s frozen-slot contract.
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

# ∂F/∂θ — dense local parameter Jacobian, re-queried from the Dual-rebuilt
# global `p` (`req.p`, NOT the already element-local `args.p`) so wrappers
# forward Duals transparently. For a condensed inner this is the FROZEN-q
# partial; the total needs the `∂F/∂q · dq/dθ` block `local_conditions!` supplies.
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

# ∂F/∂t — the ctx is rebuilt with a Dual evaluation time, so an element reading
# `evaluation_time(args.ctx)` differentiates exactly. The preallocated config is
# typed for the residual eltype; exotic time types fall back to a per-call one.
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
# frozen-q PARTIALS, while the kinds carrying no `CorrectionMode` are always
# totals. The missing block is `∂F/∂q · dq/dseed`, with `dq/dseed` given by the
# implicit function theorem on the element's declared local conditions.

# ∂L/∂q at the condensed state, factorized in place — the reason the hook is
# worth its cost: ONE factorization per item serves all nθ directions. Redone
# per sweep rather than retained, since retaining it is corrector storage again,
# which is what `CorrectorElection` exists to let a user refuse.
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

# (∂F/∂u)·v — one directional-Dual sweep, no matrices anywhere (the MFEM NONE
# level). The perturbed state goes into the per-worker Dual buffer.
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

# (∂F/∂u)ᵀλₑ — gradient of the scalar λₑ·rₑ(u), with the Dual residual evaluated
# into the per-worker buffer.
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
split analytic kernels back to back instead of falling back to AD. Chosen at
construction (`decorate_element_cache`).
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
# `fully_analytic`) is setup_operator's decision, not part of the decorator
# types themselves; it lives in operators/ad_decoration.jl.
