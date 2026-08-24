####################################
## AD decorator (ForwardDiff over the residual kernel)
####################################
#
# Limitations:
# - AD sweeps cover the volumetric kernel only; boundary contributions flow
#   through the analytic facet path, so sensitivities of parameter- or
#   time-dependent boundary terms are NOT captured.
# - State sweeps (∂F/∂u Jacobian, JVP, VJP, ∂F/∂t) run over per-worker
#   preallocated ForwardDiff configs, owned by this decorator. The parameter
#   sweeps build their configs per call: their seed dimension nθ is call-time
#   knowledge, and a cached config would be abstractly typed across nθ
#   changes.

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
    if supports_q_bootstrap && has_internal_state(typeof(inner))
        nqp       = getnquadpoints(inner)
        qseed     = zeros(T, nqp)
        chunk_q   = ForwardDiff.Chunk(qseed)
        jac_cfg_q = ForwardDiff.JacobianConfig(nothing, re, qseed, chunk_q, tag)
        Kq        = zeros(T, ndofs, nqp)
    end
    return ADElementBuffers(re, jac_cfg, deriv_cfg, grad_cfg, u_dual, re_dual, wseed, wdual, Vector{T}(), jac_cfg_q, Kq)
end

function duplicate_for_device(device, b::ADElementBuffers)
    return ADElementBuffers(
        copy(b.re), b.jac_cfg, b.deriv_cfg, b.grad_cfg,
        copy(b.u_dual), copy(b.re_dual), copy(b.wseed), copy.(b.wdual), copy(b.θ),
        b.jac_cfg_q, b.Kq === nothing ? nothing : copy(b.Kq),
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
combined with the stored `dq/dū` block from [`condensed_corrector`](@ref) —
`Jₑ = ∂F/∂ū|_q + ∂F/∂q · dq/dū`. Without an analytic kernel or
`condensed_corrector`, admissibility rejects it, naming the missing correction.

`n_global_dofs` pads the seeds and configs by the subdomain's
[`global_dofs`](@ref) count, so the differentiated system is the FULL augmented
one — `setup_operator` passes it. The `ndofs` form sizes the buffers from a dof
count alone, for an item family that has no `SubDofHandler` to allocate against.

`setup_operator` wraps automatically (`ad_backend = nothing` opts out);
hand-constructing an instance wraps a specific cache or tests the decorator.
"""
struct ADElementCache{Inner, Backend, Buffers} <: AbstractVolumetricElementCache
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
        "$(nameof(_display_cache_type(typeof(inner)))) carries condensed internal state and sits " *
        "on a subdomain declaring $(n_global_dofs) `global_dofs`. Its `Consistent` Jacobian " *
        "would go through the generic combination `∂F/∂ū|_q + ∂F/∂q · dq/dū`, whose corrector " *
        "block spans the FIELD space while the AD partials span the augmented system — the " *
        "combination is not defined for this pair. Implement the analytic " *
        "`assemble_cell!(::JacobianRequest{:u, Consistent}, …)` kernel (declared through " *
        "`provides_analytic`), or drop the `global_dofs` declaration on this subdomain."))
end

duplicate_for_device(device, ad::ADElementCache) =
    ADElementCache(duplicate_for_device(device, ad.inner), ad.backend, duplicate_for_device(device, ad.buffers))

query_cell_parameters(ad::ADElementCache, cell, p) = query_cell_parameters(ad.inner, cell, p)
query_facet_parameters(ad::ADElementCache, cell, local_facet_index, p) =
    query_facet_parameters(ad.inner, cell, local_facet_index, p)
Ferrite.getnquadpoints(ad::ADElementCache) = getnquadpoints(ad.inner)
reinit_values!(ad::ADElementCache, cell) = reinit_values!(ad.inner, cell)
reinit_values!(ad::ADElementCache, cell, kind) = reinit_values!(ad.inner, cell, kind)
evaluate_cell_functional(kind, ad::ADElementCache, args) = evaluate_cell_functional(kind, ad.inner, args)

has_internal_state(::Type{<:ADElementCache{Inner}}) where {Inner} = has_internal_state(Inner)
internal_state_insensitive(::Type{<:ADElementCache{Inner}}, kind) where {Inner} = internal_state_insensitive(Inner, kind)
get_number_of_internal_dofs_per_element(model, ad::ADElementCache, sdh) =
    get_number_of_internal_dofs_per_element(model, ad.inner, sdh)
condense_cell!(ad::ADElementCache, args, weights) = condense_cell!(ad.inner, args, weights)
condense_algebraic!(ad::ADElementCache, args, weights) = condense_algebraic!(ad.inner, args, weights)
invalidate_correctors!(ad::ADElementCache) = invalidate_correctors!(ad.inner)
# Patch assembly (experimental) is a separate protocol with no AD fallback of
# its own — pass through to whatever `inner` implements.
assemble_patch_cell!(req, ad::ADElementCache, args, data) = assemble_patch_cell!(req, ad.inner, args, data)

"""
    condensed_corrector(cache, id) -> AbstractMatrix

The completed `nq × ndofs` `dq/dū` block a condensed cache exposes for
`ADElementCache`'s generic `Consistent` combination. Only needed to admit the
generic bootstrap; a cache serving `Consistent` analytically never needs this.
No default.
"""
function condensed_corrector end

# Capability: plain AD-from-residual is exact for every kind whose fallback
# does not differentiate through a condensed inner's (now pure) local state —
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
_ad_covers(Inner, ::TimeSensitivityKind) = !has_internal_state(Inner) || internal_state_insensitive(Inner, TimeSensitivityKind())
_ad_covers(Inner, ::JacobianKind{:u, Consistent}) =
    !has_internal_state(Inner) || internal_state_insensitive(Inner, JacobianKind{:u, Consistent}()) || hasmethod(condensed_corrector, Tuple{Inner, Int})
_ad_covers(Inner, ::JacobianResidualKind{Consistent}) =
    !has_internal_state(Inner) || internal_state_insensitive(Inner, JacobianResidualKind{Consistent}()) || hasmethod(condensed_corrector, Tuple{Inner, Int})

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

# The decorator ALWAYS has a matching `assemble_cell!` method (generic over
# every slot/request), so `hasmethod` on the wrapped type can never catch an
# author's `provides_analytic` claim without a kernel — the check that
# validates a claim must therefore run against `Inner`, the only type whose
# method set is author-written. Framework-provided decorator methods need no
# such check: they are what this package verifies by construction.
_assert_trait_backed(::Type{<:ADElementCache{Inner}}, kind, entry, ::Type{Args}, trailing::Tuple = ()) where {Inner, Args} =
    _assert_trait_backed(Inner, kind, entry, Args, trailing)
_display_cache_type(::Type{<:ADElementCache{Inner}}) where {Inner} = _display_cache_type(Inner)

# Validation unwraps for the same reason: the decorator's forwarding methods
# (`assemble_cell!` over every request, `reinit_values!`) answer `hasmethod`
# for ANY inner, so probing the wrapper would pass a cache that implements
# nothing. Recursing on `inner` also reaches the leaves of a wrapped
# sub-composite through the composite method.
validate_element_cache(ad::ADElementCache, declared_requests::Tuple = ()) =
    validate_element_cache(ad.inner, declared_requests)

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

# ∂F/∂slot — writes the Jacobian into K, overwriting `y` with the primal
# residual as a byproduct. Only the named slot is seeded; every other slot
# stays at its primal value — including `AffineRate`-reconstructed slots,
# which are formed at gather time. Tag checking is off: the config carries the
# package tag, not the closure's.
function ad_state_jacobian!(K, y, ad::ADElementCache, args, ::Val{slot} = Val(:u)) where {slot}
    cfg = _jac_config_for(ad.buffers, Val(slot))
    inner = ad.inner
    f! = (r, x) -> evaluate_item_residual!(
        r, inner, with_states(args, merge(args.states, NamedTuple{(slot,)}((x,)))))
    ForwardDiff.jacobian!(K, f!, y, args.states[slot], cfg, Val{false}())
    return K
end

# The generic Consistent combination for a condensed inner: AD gives both
# partials, `condensed_corrector` gives the stored dq/dū block.
function _condensed_consistent_jacobian!(K, y, ad::ADElementCache, args)
    buf = ad.buffers
    ad_state_jacobian!(K, y, ad, args, Val(:u))
    ad_state_jacobian!(buf.Kq, buf.re, ad, args, Val(:q))
    corr = condensed_corrector(ad.inner, cellid(args.cell))
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
function _ad_assemble!(req::ParameterJacobianRequest, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, ParameterJacobianKind())
        return _inner_kernel!(req, ad.inner, args)
    end
    buf = ad.buffers
    inner = ad.inner
    p = req.p
    θ = copyto!(_theta!(buf, size(req.B, 2)), parameter_vector(p))
    f! = (r, θᵢ) -> begin
        pₑ = query_cell_parameters(inner, _parameter_subject(args), rebuild_parameters(p, θᵢ))
        evaluate_item_residual!(r, inner, with_parameters(args, pₑ))
    end
    ForwardDiff.jacobian!(req.B, f!, buf.re, θ)
    return req.B
end

# (∂F/∂θ)ᵀλₑ — adjoint pullback as the gradient of the scalar λₑ·rₑ(θ).
function _ad_assemble!(req::ParameterVJPRequest, ad::ADElementCache{Inner}, args) where {Inner}
    if provides_analytic(Inner, ParameterVJPKind(nothing))
        return _inner_kernel!(req, ad.inner, args)
    end
    buf = ad.buffers
    inner = ad.inner
    p = req.p
    λₑ = req.λₑ
    θ = copyto!(_theta!(buf, length(parameter_vector(p))), parameter_vector(p))
    fscalar = θᵢ -> begin
        pₑ = query_cell_parameters(inner, _parameter_subject(args), rebuild_parameters(p, θᵢ))
        r = zeros(eltype(θᵢ), length(λₑ))
        evaluate_item_residual!(r, inner, with_parameters(args, pₑ))
        return dot(λₑ, r)
    end
    ForwardDiff.gradient!(req.g, fscalar, θ)
    return req.g
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
    buf = ad.buffers
    inner = ad.inner
    ctx = args.ctx
    t = evaluation_time(ctx)
    f! = (r, t̃) -> evaluate_item_residual!(r, inner, with_context(args, with_time(ctx, t̃)))
    if t isa eltype(buf.re)
        ForwardDiff.derivative!(req.g, f!, buf.re, t, buf.deriv_cfg, Val{false}())
    else
        ForwardDiff.derivative!(req.g, f!, buf.re, t)
    end
    return req.g
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
struct FusedFromSplit{Inner} <: AbstractVolumetricElementCache
    inner::Inner
end

duplicate_for_device(device, f::FusedFromSplit) = FusedFromSplit(duplicate_for_device(device, f.inner))
query_cell_parameters(f::FusedFromSplit, cell, p) = query_cell_parameters(f.inner, cell, p)
query_facet_parameters(f::FusedFromSplit, cell, local_facet_index, p) =
    query_facet_parameters(f.inner, cell, local_facet_index, p)
Ferrite.getnquadpoints(f::FusedFromSplit) = getnquadpoints(f.inner)
reinit_values!(f::FusedFromSplit, cell) = reinit_values!(f.inner, cell)
reinit_values!(f::FusedFromSplit, cell, kind) = reinit_values!(f.inner, cell, kind)
evaluate_cell_functional(kind, f::FusedFromSplit, args) = evaluate_cell_functional(kind, f.inner, args)
has_internal_state(::Type{<:FusedFromSplit{Inner}}) where {Inner} = has_internal_state(Inner)
internal_state_insensitive(::Type{<:FusedFromSplit{Inner}}, kind) where {Inner} = internal_state_insensitive(Inner, kind)
get_number_of_internal_dofs_per_element(model, f::FusedFromSplit, sdh) =
    get_number_of_internal_dofs_per_element(model, f.inner, sdh)
condense_cell!(f::FusedFromSplit, args, weights) = condense_cell!(f.inner, args, weights)
condense_algebraic!(f::FusedFromSplit, args, weights) = condense_algebraic!(f.inner, args, weights)
invalidate_correctors!(f::FusedFromSplit) = invalidate_correctors!(f.inner)
assemble_patch_cell!(req, f::FusedFromSplit, args, data) = assemble_patch_cell!(req, f.inner, args, data)

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
# Same reasoning as `ADElementCache`: the blanket catch-all method means
# `hasmethod` on the wrapper can never distinguish a backed claim from an
# author's overclaim, so the check runs against `Inner`.
_assert_trait_backed(::Type{<:FusedFromSplit{Inner}}, kind, entry, ::Type{Args}, trailing::Tuple = ()) where {Inner, Args} =
    _assert_trait_backed(Inner, kind, entry, Args, trailing)
_display_cache_type(::Type{<:FusedFromSplit{Inner}}) where {Inner} = _display_cache_type(Inner)
validate_element_cache(f::FusedFromSplit, declared_requests::Tuple = ()) =
    validate_element_cache(f.inner, declared_requests)

# Construction-time wrapping (`decorate_element_cache`, `needs_ad_decoration`,
# `fully_analytic`) lives in operators/ad_decoration.jl — it is setup_operator's
# decision, not part of the decorator types themselves.
