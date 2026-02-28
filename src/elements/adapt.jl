## Adapt.jl integration for element caches ##
# Element caches holding CellValues need Adapt so KA can convert
# CuArray/ROCArray → CuDeviceArray/ROCDeviceArray at kernel launch.

Adapt.@adapt_structure SimpleBilinearDiffusionElementCache
Adapt.@adapt_structure SimpleBilinearMassElementCache
Adapt.@adapt_structure SimpleHyperelasticityElementCache
Adapt.@adapt_structure SimpleCondensedLinearViscoelasticityCache
Adapt.@adapt_structure CompositeVolumetricElementCache
Adapt.@adapt_structure CompositeSurfaceElementCache
Adapt.@adapt_structure CompositeInterfaceElementCache
