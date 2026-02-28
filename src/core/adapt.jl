## Adapt.adapt_structure for GPU local cache factories ##
# @concrete structs need explicit adapt_structure so KA can convert
# ROCArray/CuArray → ROCDeviceArray/CuDeviceArray at kernel launch.

Adapt.adapt_structure(to, f::BilinearGlobalLocalCacheFactory) =
    BilinearGlobalLocalCacheFactory(Adapt.adapt(to, f.Ke_pool))

Adapt.adapt_structure(to, f::NonlinearGlobalLocalCacheFactory) =
    NonlinearGlobalLocalCacheFactory(Adapt.adapt(to, f.Ke_pool), Adapt.adapt(to, f.ue_pool), Adapt.adapt(to, f.re_pool))

Adapt.adapt_structure(to, f::LinearGlobalLocalCacheFactory) =
    LinearGlobalLocalCacheFactory(Adapt.adapt(to, f.re_pool))

Adapt.adapt_structure(to, dc::GPUAssemblyCache) =
    GPUAssemblyCache(
        Adapt.adapt(to, dc.local_cache_factory),
        Adapt.adapt(to, dc.cell_cache_factory),
        Adapt.adapt(to, dc.ivh),
        Adapt.adapt(to, dc.element_cache_factory),
    )
