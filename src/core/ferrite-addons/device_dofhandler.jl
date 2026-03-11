#TODO: remove
const ImmutableCellCache = Base.get_extension(Ferrite, :FerriteKAExt).ImmutableCellCache

# TODO: upstream to Ferrite.jl — these are type piracy until Ferrite adds them.
@inline Ferrite.reinit!(cv::Ferrite.AbstractCellValues, cc::ImmutableCellCache) = reinit!(cv, cc.coords)
@inline Ferrite.getcoordinates(cc::ImmutableCellCache) = cc.coords
@inline Ferrite.cellid(cc::ImmutableCellCache) = cc.cellid
