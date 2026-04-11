@doc raw"""
    SimpleBilinearMassIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int v(x) \cdot D u(x) dx`` for a given Mass value ``D`` and ``u,v`` from the same function space.
"""
struct SimpleBilinearMassIntegrator <: AbstractBilinearIntegrator
    # This is specific to our model
    ρ::Float64
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

"""
The cache associated with [`BilinearMassIntegrator`](@ref) to assemble element Mass matrices.
"""
@device_element struct SimpleBilinearMassElementCache <: AbstractVolumetricElementCache
    ρ::Float64
    cellvalues::CellValues
end

function assemble_element!(Kₑ::AbstractMatrix, cell, element_cache::SimpleBilinearMassElementCache, time)
    (; cellvalues, ρ) = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    reinit!(cellvalues, cell)

    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:n_basefuncs
            Nᵢ = shape_value(cellvalues, qp, i)
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, qp, j)
                Kₑ[i,j] += ρ * Nⱼ ⋅ Nᵢ * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::SimpleBilinearMassIntegrator, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    field_name = element_model.field_name
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    return SimpleBilinearMassElementCache(element_model.ρ, CellValues(qr, ip, ip_geo))
end
