@doc raw"""
    SimpleHyperelasticityIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int \nabla v(x) \cdot D \nabla u(x) dx`` for a given diffusion value ``D`` and ``u,v`` from the same function space.
"""
struct SimpleHyperelasticityIntegrator{EnergyType} <: AbstractBilinearIntegrator
    # This is specific to our model
    ψ::EnergyType
    # Every integrator needs these
    ipc::VectorInterpolationCollection
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

"""
The cache associated with [`BilinearDiffusionIntegrator`](@ref) to assemble element diffusion matrices.
"""
struct SimpleHyperelasticityElementCache{EnergyType, CV <: CellValues} <: AbstractVolumetricElementCache
    ψ::EnergyType
    cellvalues::CV
end

function duplicate_for_device(device, cache::SimpleHyperelasticityElementCache)
    return SimpleHyperelasticityElementCache(
        cache.ψ,
        duplicate_for_device(device, cache.cellvalues),
    )
end

function assemble_element!(Kₑ::AbstractMatrix, cell, element_cache::SimpleHyperelasticityElementCache, time)
    (; cellvalues, D) = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    reinit!(cellvalues, cell)

    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, qp, j)
                Kₑ[i,j] -= D * ∇Nⱼ ⋅ ∇Nᵢ * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::SimpleHyperelasticityIntegrator, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    field_name = element_model.field_name
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    return SimpleHyperelasticityElementCache(element_model.D, CellValues(qr, ip, ip_geo))
end
