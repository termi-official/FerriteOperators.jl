####################################
## Construction-time wrapping
####################################
#
# Which element caches need ADElementCache/FusedFromSplit is setup_operator's
# decision, not part of the decorator types themselves (elements/ad_element.jl).

"""
    needs_ad_decoration(integrator) -> Bool

Whether an operator built for `integrator` may issue a kind
[`ADElementCache`](@ref) covers — STRUCTURAL, so a bilinear or linear operator
carries no AD/sensitivity machinery whatever an element cache implements
analytically. `true` for `AbstractNonlinearIntegrator`.
"""
needs_ad_decoration(integrator) = integrator isa AbstractNonlinearIntegrator

# The kind instances `fully_analytic` probes: every decorator-covered request
# kind, at a placeholder payload (only the TYPE is read).
const _AD_COVERED_KINDS = (
    JacobianKind{:u, Consistent}(), JacobianKind{:u, FrozenQ}(),
    JacobianResidualKind{Consistent}(), JacobianResidualKind{FrozenQ}(),
    WeightedJacobianKind((u = 1.0,)),
    ParameterJacobianKind(), ParameterVJPKind(nothing), TimeSensitivityKind(),
    StateJVPKind(nothing), StateVJPKind(nothing),
)

"true iff `T` serves every AD-decorator-covered kind analytically — no wrapping needed."
fully_analytic(::Type{T}) where {T} = all(kind -> provides_analytic(T, kind), _AD_COVERED_KINDS)

"true iff `T` provides `JacobianKind{:u}` and the mandatory residual analytically but not the fused `JacobianResidualKind` — the [`FusedFromSplit`](@ref) case."
_needs_fused_from_split(::Type{T}) where {T} =
    provides_analytic(T, JacobianKind{:u, Consistent}()) && !provides_analytic(T, JacobianResidualKind{Consistent}())

_maybe_fuse_split(cache) = _needs_fused_from_split(typeof(cache)) ? FusedFromSplit(cache) : cache

"""
    decorate_element_cache(cache, sdh, ad_backend, n_global_dofs = 0)

Resolve `cache` into the form the engine calls unconditionally:
[`FusedFromSplit`](@ref) where it provides split analytic kernels but not the
fused one, then [`ADElementCache`](@ref) where it still lacks analytic coverage
of some AD-decorator kind. `ad_backend === nothing` opts out of the
`ADElementCache` step only. A [`CompositeVolumetricElementCache`](@ref) wraps
its non-analytic inners as ONE sub-composite (the maximal-sub-composite
policy), since wrapping each individually costs a full seeding pass per inner.

`n_global_dofs` is the subdomain's [`global_dofs`](@ref) count and pads the AD
buffers, so an AD fallback differentiates the augmented local system rather
than its field-space head.
"""
function decorate_element_cache(cache, sdh, ad_backend, n_global_dofs::Int = 0)
    fused = _maybe_fuse_split(cache)
    ad_backend === nothing && return fused
    return fully_analytic(typeof(fused)) ? fused :
        ADElementCache(fused, sdh; backend = ad_backend, n_global_dofs)
end

function decorate_element_cache(cache::CompositeVolumetricElementCache, sdh, ad_backend, n_global_dofs::Int = 0)
    inners = map(_maybe_fuse_split, cache.inner_caches)
    ad_backend === nothing && return CompositeVolumetricElementCache(inners)
    analytic = filter(inner -> fully_analytic(typeof(inner)), inners)
    needs_ad = filter(inner -> !fully_analytic(typeof(inner)), inners)
    isempty(needs_ad) && return CompositeVolumetricElementCache(inners)
    wrapped = length(needs_ad) == 1 ?
        ADElementCache(only(needs_ad), sdh; backend = ad_backend, n_global_dofs) :
        ADElementCache(CompositeVolumetricElementCache(needs_ad), sdh; backend = ad_backend, n_global_dofs)
    return isempty(analytic) ? wrapped : CompositeVolumetricElementCache((analytic..., wrapped))
end
