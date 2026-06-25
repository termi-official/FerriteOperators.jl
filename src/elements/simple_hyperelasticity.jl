@doc raw"""
    SimpleHyperelasticityIntegrator{EnergyType}
"""
struct SimpleHyperelasticityIntegrator{EnergyType} <: AbstractNonlinearIntegrator
    # This is specific to our model
    ψ::EnergyType
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

@device_element struct SimpleHyperelasticityElementCache{EnergyType} <: AbstractVolumetricElementCache
    ψ::EnergyType # NOTE: since `EnergyType` is readonly & `isbits`, then we don't need to duplicate it for the device (GPU in particular).
    cv::CellValues
end

# Element residual
function assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, element_cache::SimpleHyperelasticityElementCache, p)
    (; ψ, cv) = element_cache

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

# jac
function assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, element_cache::SimpleHyperelasticityElementCache, p)
    (; ψ, cv) = element_cache

    ndofs = getnbasefunctions(cv)

    reinit!(cv, cell)

    @inbounds for qp ∈ 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        ∂P∂F = Tensors.hessian(F_ad -> ψ(F_ad), F)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

# Combined residual and jac
function assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, element_cache::SimpleHyperelasticityElementCache, p)
    (; ψ, cv) = element_cache

    ndofs = getnbasefunctions(cv)

    reinit!(cv, cell)

    @inbounds for qp ∈ 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        ∂P∂F, P = Tensors.hessian(F_ad -> ψ(F_ad), F, :all)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::SimpleHyperelasticityIntegrator, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    field_name = element_model.field_name
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    return SimpleHyperelasticityElementCache(element_model.ψ, CellValues(qr, ip, ip_geo))
end
