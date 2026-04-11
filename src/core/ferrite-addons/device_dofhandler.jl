#TODO: remove
const ImmutableCellCache = Base.get_extension(Ferrite, :FerriteKAExt).ImmutableCellCache

# reinit!, getcoordinates, cellid, celldofs for ImmutableCellCache are now provided by Ferrite's FerriteKAExt

# CPU no-op: don't wrap DofHandler into HostDofHandler on CPU
Adapt.adapt_structure(::KA.CPU, dh::Ferrite.DofHandler) = dh
