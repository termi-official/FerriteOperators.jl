
"""
    QuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct QuadratureRuleCollection{order}
end

QuadratureRuleCollection(order::Int) = QuadratureRuleCollection{order}()

getquadraturerule(qrc::QuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = QuadratureRule{ref_shape}(order)
getquadraturerule(qrc::QuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))


"""
    FacetQuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct FacetQuadratureRuleCollection{order}
end

FacetQuadratureRuleCollection(order::Int) = FacetQuadratureRuleCollection{order}()

getquadraturerule(qrc::FacetQuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = FacetQuadratureRule{ref_shape}(order)
getquadraturerule(qrc::FacetQuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))
