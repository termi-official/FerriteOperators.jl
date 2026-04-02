#TODO: remove this
# Elements using @device_element define their own Adapt + per_thread in the struct definition.
# Only non-@device_element types need manual entries here:
Adapt.@adapt_structure CompositeVolumetricElementCache
@per_thread_structure CompositeVolumetricElementCache

Adapt.@adapt_structure CompositeSurfaceElementCache
@per_thread_structure CompositeSurfaceElementCache

Adapt.@adapt_structure CompositeInterfaceElementCache
@per_thread_structure CompositeInterfaceElementCache
