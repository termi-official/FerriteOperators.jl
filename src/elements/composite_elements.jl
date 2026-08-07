"""
This cache allows to combine multiple elements over the same volume.

v2 composition is pure iteration: the request carries the buffers, so ONE
generic fan-out serves every request type. The composite is analytic for a
kind iff every inner cache is; otherwise the whole composite falls back to AD
over the fused residual (one Dual sweep, one geometry pass for the fused
Jacobian).
"""
struct CompositeVolumetricElementCache{CacheTupleType <: Tuple} <: AbstractVolumetricElementCache
    inner_caches::CacheTupleType
end

assemble_cell!(req::AbstractAssemblyRequest, composite::CompositeVolumetricElementCache, args::KernelArgs) =
    _composite_assemble_cell!(req, composite.inner_caches, args)
@unroll function _composite_assemble_cell!(req, inner_caches, args)
    @unroll for inner in inner_caches
        assemble_cell!(req, inner, args)
    end
end

provides_analytic(::Type{CompositeVolumetricElementCache{CT}}, kind) where {CT <: Tuple} = _all_provide(CT, kind)
# The composite's blanket fan-out method would satisfy any hasmethod check,
# so validation must recurse into the inner caches.
validate_element_cache(composite::CompositeVolumetricElementCache) =
    foreach(validate_element_cache, composite.inner_caches)
_all_provide(::Type{Tuple{}}, kind) = true
_all_provide(::Type{T}, kind) where {T <: Tuple} =
    provides_analytic(Base.tuple_type_head(T), kind) && _all_provide(Base.tuple_type_tail(T), kind)
has_internal_state(::Type{CompositeVolumetricElementCache{CT}}) where {CT <: Tuple} = _any_internal(CT)
_any_internal(::Type{Tuple{}}) = false
_any_internal(::Type{T}) where {T <: Tuple} =
    has_internal_state(Base.tuple_type_head(T)) || _any_internal(Base.tuple_type_tail(T))

function duplicate_for_device(device, cache::CompositeVolumetricElementCache)
    return CompositeVolumetricElementCache(
        map(inner_cache -> duplicate_for_device(device, inner_cache), cache.inner_caches),
    )
end

"""
This cache allows to combine multiple elements over the same surface. The
boundary driver gates on the composite (`is_facet_in_cache` = any inner covers
the facet); the fan-out re-gates per inner cache, since inner caches may cover
different facet sets.
"""
struct CompositeSurfaceElementCache{CacheTupleType <: Tuple} <: AbstractSurfaceElementCache
    inner_caches::CacheTupleType
end

function duplicate_for_device(device, cache::CompositeSurfaceElementCache)
    return CompositeSurfaceElementCache(
        map(inner_cache -> duplicate_for_device(device, inner_cache), cache.inner_caches),
    )
end

assemble_facet!(req::AbstractAssemblyRequest, composite::CompositeSurfaceElementCache, args::KernelArgs, local_facet_index::Int) =
    _composite_assemble_facet!(req, composite.inner_caches, args, local_facet_index)
@unroll function _composite_assemble_facet!(req, inner_caches, args, local_facet_index)
    @unroll for inner in inner_caches
        if is_facet_in_cache(FacetIndex(cellid(args.cell), local_facet_index), args.cell, inner)
            assemble_facet!(req, inner, args, local_facet_index)
        end
    end
end

is_facet_in_cache(idx::FacetIndex, cell, composite::CompositeSurfaceElementCache) =
    _any_facet_in_cache(idx, cell, composite.inner_caches)
@unroll function _any_facet_in_cache(idx, cell, inner_caches)
    @unroll for inner in inner_caches
        is_facet_in_cache(idx, cell, inner) && return true
    end
    return false
end

"""
This cache allows to combine multiple elements over the same interface.
Interface kernels are reserved for the DG work (phase 4); composition will
follow the same request fan-out pattern as cells and facets.
"""
struct CompositeInterfaceElementCache{CacheTupleType <: Tuple} <: AbstractInterfaceElementCache
    inner_caches::CacheTupleType
end
function duplicate_for_device(device, cache::CompositeInterfaceElementCache)
    return CompositeInterfaceElementCache(
        map(inner_cache -> duplicate_for_device(device, inner_cache), cache.inner_caches),
    )
end
