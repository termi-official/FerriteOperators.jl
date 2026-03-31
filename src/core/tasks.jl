# Element-level query functions (overridable by element caches)
query_element_parameters(element, geometry_cache, ivh, p) = p
query_element_unknown_buffer(element, ue) = ue

# Item extraction from subdomain cache
# Note: SubdomainCache is defined later in operators/general.jl — duck-typed access is fine.
get_items(sc) = _get_items(sc.sdh, sc.strategy_cache)
_get_items(sdh, ::SequentialAssemblyStrategyCache) = (sdh.cellset,)
_get_items(sdh, sc::PerColorAssemblyStrategyCache) = sc.colors
_get_items(sdh, ::ElementAssemblyStrategyCache) = (sdh.cellset,)
