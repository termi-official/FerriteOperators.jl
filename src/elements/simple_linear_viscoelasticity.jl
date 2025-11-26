@kwdef struct MaxwellParameters
    E₀::Float64 = 70e3
    E₁::Float64 = 20e3
    μ::Float64  = 1e3
    η₁::Float64 = 1e3
    ν::Float64  = 0.3
end

@doc raw"""
    SimpleCondensedLinearViscoelasticity
"""
struct SimpleCondensedLinearViscoelasticity <: AbstractNonlinearIntegrator
    material_parameters::MaxwellParameters
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    displacement_name::Symbol
    viscosity_name::Symbol
end

struct SimpleCondensedLinearViscoelasticityCache{EnergyType, CV <: CellValues} <: AbstractVolumetricElementCache
    material_parameters::MaxwellParameters
    displacement_range::UnitRange{Int64}
    viscosity_range::UnitRange{Int64}
    cv::CV
end

function duplicate_for_device(device, cache::SimpleCondensedLinearViscoelasticityCache)
    return SimpleCondensedLinearViscoelasticityCache(
        cache.material_parameters,
        cache.displacement_range,
        cache.viscosity_range,
        duplicate_for_device(device, cache.cv),
    )
end

# Element residual
function assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, element_cache::SimpleHyperelasticityElementCache, p)
    (; ψ, cv) = element_cache
    (; dt, t, uₑprev) = p

    ndofs = getnbasefunctions(cv)

    reinit!(cv, cell)

    @inbounds for qp ∈ 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        P = Tensors.gradient(F_ad -> ψ(F_ad), F)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ
        end
    end
end

# # jac
# function assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, element_cache::SimpleHyperelasticityElementCache, p)
#     (; ψ, cv) = element_cache

#     ndofs = getnbasefunctions(cv)

#     reinit!(cv, cell)

#     @inbounds for qp ∈ 1:getnquadpoints(cv)
#         dΩ = getdetJdV(cv, qp)

#         # Compute deformation gradient F
#         ∇u = function_gradient(cv, qp, uₑ)
#         F = one(∇u) + ∇u

#         # Compute stress and tangent
#         ∂P∂F = Tensors.hessian(F_ad -> ψ(F_ad), F)

#         # Loop over test functions
#         for i in 1:ndofs
#             ∇δui = shape_gradient(cv, qp, i)

#             ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
#             for j in 1:ndofs
#                 ∇δuj = shape_gradient(cv, qp, j)
#                 # Add contribution to the tangent
#                 Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
#             end
#         end
#     end
# end

# # Combined residual and jac
# function assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, element_cache::SimpleHyperelasticityElementCache, p)
#     (; ψ, cv) = element_cache

#     ndofs = getnbasefunctions(cv)

#     reinit!(cv, cell)

#     @inbounds for qp ∈ 1:getnquadpoints(cv)
#         dΩ = getdetJdV(cv, qp)

#         # Compute deformation gradient F
#         ∇u = function_gradient(cv, qp, uₑ)
#         F = one(∇u) + ∇u

#         # Compute stress and tangent
#         ∂P∂F, P = Tensors.hessian(F_ad -> ψ(F_ad), F, :all)

#         # Loop over test functions
#         for i in 1:ndofs
#             ∇δui = shape_gradient(cv, qp, i)

#             # Add contribution to the residual from this test function
#             residualₑ[i] += ∇δui ⊡ P * dΩ

#             ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
#             for j in 1:ndofs
#                 ∇δuj = shape_gradient(cv, qp, j)
#                 # Add contribution to the tangent
#                 Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
#             end
#         end
#     end
# end

function setup_element_cache(element_model::SimpleHyperelasticityIntegrator, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    field_name = element_model.field_name
    ip         = Ferrite.getfieldinterpolation(sdh, displacement_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    return SimpleHyperelasticityElementCache(element_model.ψ, CellValues(qr, ip, ip_geo))
end
