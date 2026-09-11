
"""
    QuadratureRuleCollection([::Type{T},] order::Int)

A collection of quadrature rules across different cell types, evaluated in the
scalar type `T` (`Float64` where it is not named).

`T` is the INTEGRATOR's election of the precision its elements evaluate in: the
rules [`getquadraturerule`](@ref) hands out carry it, and an element cache reads
it back through [`element_value_type`](@ref). It is independent of the
[`value_type`](@ref) the device accumulates the global system in.
"""
struct QuadratureRuleCollection{order, T}
end

QuadratureRuleCollection(order::Int) = QuadratureRuleCollection{order, Float64}()
QuadratureRuleCollection(::Type{T}, order::Int) where {T} = QuadratureRuleCollection{order, T}()
QuadratureRuleCollection{order}() where {order} = QuadratureRuleCollection{order, Float64}()

element_value_type(::QuadratureRuleCollection{order, T}) where {order, T} = T

"""
    getquadraturerule(qrc, cell::AbstractCell)
    getquadraturerule(qrc, sdh::SubDofHandler)

The collection's rule for a reference shape: the cell's own, or — the form
`setup_element_cache` uses — the shape of the subdomain's first cell, shared by
every cell of a `SubDofHandler`. Defined for
[`QuadratureRuleCollection`](@ref) and [`FacetQuadratureRuleCollection`](@ref).

The rule carries the collection's scalar type, so the precision an integrator
elected travels with it.
"""
getquadraturerule(qrc::QuadratureRuleCollection{order, T}, cell::AbstractCell{ref_shape}) where {order,T,ref_shape} = QuadratureRule{ref_shape}(T, order)
getquadraturerule(qrc::QuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))


"""
    FacetQuadratureRuleCollection([::Type{T},] order::Int)

A collection of facet quadrature rules across different cell types, evaluated in
the scalar type `T` (`Float64` where it is not named) — the facet counterpart of
[`QuadratureRuleCollection`](@ref).
"""
struct FacetQuadratureRuleCollection{order, T}
end

FacetQuadratureRuleCollection(order::Int) = FacetQuadratureRuleCollection{order, Float64}()
FacetQuadratureRuleCollection(::Type{T}, order::Int) where {T} = FacetQuadratureRuleCollection{order, T}()
FacetQuadratureRuleCollection{order}() where {order} = FacetQuadratureRuleCollection{order, Float64}()

element_value_type(::FacetQuadratureRuleCollection{order, T}) where {order, T} = T

getquadraturerule(qrc::FacetQuadratureRuleCollection{order, T}, cell::AbstractCell{ref_shape}) where {order,T,ref_shape} = FacetQuadratureRule{ref_shape}(T, order)
getquadraturerule(qrc::FacetQuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))
