Adapt.adapt_structure(to, c::BilinearLocalCacheContainer) =
    BilinearLocalCacheContainer(Adapt.adapt(to, c.Ke_pool))

Adapt.adapt_structure(to, c::NonlinearLocalCacheContainer) =
    NonlinearLocalCacheContainer(Adapt.adapt(to, c.Ke_pool), Adapt.adapt(to, c.ue_pool), Adapt.adapt(to, c.re_pool))

Adapt.adapt_structure(to, c::LinearLocalCacheContainer) =
    LinearLocalCacheContainer(Adapt.adapt(to, c.re_pool))

Adapt.adapt_structure(to, dc::GPUAssemblyCache) =
    GPUAssemblyCache(
        Adapt.adapt(to, dc.local_cache_container),
        Adapt.adapt(to, dc.cell_cache_container),
        Adapt.adapt(to, dc.ivh),
        Adapt.adapt(to, dc.element_cache_container),
    )
