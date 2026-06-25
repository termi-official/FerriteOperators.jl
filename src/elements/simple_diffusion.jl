@doc raw"""
    SimpleBilinearDiffusionIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int \nabla v(x) \cdot D \nabla u(x) dx`` for a given diffusion value ``D`` and ``u,v`` from the same function space.
"""
struct SimpleBilinearDiffusionIntegrator <: AbstractBilinearIntegrator
    # This is specific to our model
    D::Float64
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

"""
The cache associated with [`BilinearDiffusionIntegrator`](@ref) to assemble element diffusion matrices.
"""
@device_element struct SimpleBilinearDiffusionElementCache <: AbstractVolumetricElementCache
    D::Float64
    cellvalues::CellValues
end

function assemble_element!(Kₑ::AbstractMatrix, cell, element_cache::SimpleBilinearDiffusionElementCache, time)
    (; cellvalues, D) = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    reinit!(cellvalues, cell)

    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, qp, j)
                Kₑ[i,j] += D * ∇Nⱼ ⋅ ∇Nᵢ * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::SimpleBilinearDiffusionIntegrator, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    field_name = element_model.field_name
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    return SimpleBilinearDiffusionElementCache(element_model.D, CellValues(qr, ip, ip_geo))
end
