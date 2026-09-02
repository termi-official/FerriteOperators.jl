@doc raw"""
    SimpleHyperelasticityIntegrator{EnergyType}

Hyperelasticity with strain energy density ``\psi(F)``, ``F = I + \nabla u``.
"""
struct SimpleHyperelasticityIntegrator{EnergyType} <: AbstractNonlinearIntegrator
    # This is specific to our model
    ψ::EnergyType
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

"""
The cache associated with [`SimpleHyperelasticityIntegrator`](@ref). It serves
the residual, the Jacobian and the fused Jacobian-residual analytically, all
from `Tensors` derivatives of the energy.
"""
struct SimpleHyperelasticityElementCache{EnergyType, CV <: CellValues} <: AbstractVolumetricElementCache
    ψ::EnergyType
    cv::CV
end

function duplicate_for_device(device, cache::SimpleHyperelasticityElementCache)
    return SimpleHyperelasticityElementCache(
        cache.ψ,
        duplicate_for_device(device, cache.cv),
    )
end

Ferrite.getnquadpoints(e::SimpleHyperelasticityElementCache) = getnquadpoints(e.cv)
reinit_values!(e::SimpleHyperelasticityElementCache, cell) = Ferrite.reinit!(e.cv, cell)

function assemble_cell!(req::ResidualRequest, element_cache::SimpleHyperelasticityElementCache, args::CellArgs)
    residualₑ = req.r
    uₑ = args.states.u
    cell = args.cell
    (; ψ, cv) = element_cache

    ndofs = getnbasefunctions(cv)

    @inbounds for qp ∈ 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        P = Tensors.gradient(F_ad -> ψ(F_ad), F)

        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            residualₑ[i] += ∇δui ⊡ P * dΩ
        end
    end
end

function assemble_cell!(req::JacobianRequest{:u}, element_cache::SimpleHyperelasticityElementCache, args::CellArgs)
    Kₑ = req.K
    uₑ = args.states.u
    cell = args.cell
    (; ψ, cv) = element_cache

    ndofs = getnbasefunctions(cv)

    @inbounds for qp ∈ 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        ∂P∂F = Tensors.hessian(F_ad -> ψ(F_ad), F)

        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

function assemble_cell!(req::JacobianResidualRequest, element_cache::SimpleHyperelasticityElementCache, args::CellArgs)
    Kₑ = req.K
    residualₑ = req.r
    uₑ = args.states.u
    cell = args.cell
    (; ψ, cv) = element_cache

    ndofs = getnbasefunctions(cv)

    @inbounds for qp ∈ 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)

        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # `:all` returns the stress alongside the tangent, in one pass
        ∂P∂F, P = Tensors.hessian(F_ad -> ψ(F_ad), F, :all)

        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
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

provides_analytic(::Type{<:SimpleHyperelasticityElementCache}, ::Union{JacobianKind{:u}, JacobianResidualKind}) = true
