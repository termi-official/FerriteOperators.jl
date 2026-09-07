
"""
    QuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct QuadratureRuleCollection{order}
end

QuadratureRuleCollection(order::Int) = QuadratureRuleCollection{order}()

"""
    getquadraturerule(qrc, cell::AbstractCell[, ::Type{T}])
    getquadraturerule(qrc, sdh::SubDofHandler[, ::Type{T}])

The collection's rule for a reference shape: the cell's own, or — the form
`setup_element_cache` uses — the shape of the subdomain's first cell, shared by
every cell of a `SubDofHandler`. Defined for
[`QuadratureRuleCollection`](@ref) and [`FacetQuadratureRuleCollection`](@ref).

`T` is the rule's scalar type, the device's [`value_type`](@ref) as the
three-argument [`setup_element_cache`](@ref) receives it; without it the rule is
`Float64`.
"""
getquadraturerule(qrc::QuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = QuadratureRule{ref_shape}(order)
getquadraturerule(qrc::QuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))
getquadraturerule(qrc::QuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}, ::Type{T}) where {order,ref_shape,T} = QuadratureRule{ref_shape}(T, order)
getquadraturerule(qrc::QuadratureRuleCollection, sdh::SubDofHandler, ::Type{T}) where {T} = getquadraturerule(qrc, get_first_cell(sdh), T)


"""
    FacetQuadratureRuleCollection(order::Int)

A collection of facet quadrature rules across different cell types.
"""
struct FacetQuadratureRuleCollection{order}
end

FacetQuadratureRuleCollection(order::Int) = FacetQuadratureRuleCollection{order}()

getquadraturerule(qrc::FacetQuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = FacetQuadratureRule{ref_shape}(order)
getquadraturerule(qrc::FacetQuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))
getquadraturerule(qrc::FacetQuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}, ::Type{T}) where {order,ref_shape,T} = FacetQuadratureRule{ref_shape}(T, order)
getquadraturerule(qrc::FacetQuadratureRuleCollection, sdh::SubDofHandler, ::Type{T}) where {T} = getquadraturerule(qrc, get_first_cell(sdh), T)
