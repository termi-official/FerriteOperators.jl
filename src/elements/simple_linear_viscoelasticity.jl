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

struct SimpleCondensedLinearViscoelasticityCache{CV <: CellValues} <: AbstractVolumetricElementCache
    material_parameters::MaxwellParameters
    displacement_range::UnitRange{Int}
    viscosity_range::UnitRange{Int}
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
function assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, element_cache::SimpleCondensedLinearViscoelasticityCache, p)
    (; material_parameters, displacement_range, viscosity_range, cv) = element_cache
    (; dt, uₑprev) = p
    Δt = dt
    (; E₀, E₁, μ, η₁, ν) = material_parameters

    nqp = getnquadpoints(cv)
    ndofs = getnbasefunctions(cv)

    dₑ = @view uₑ[displacement_range]
    qₑ = @view uₑ[viscosity_range]
    qₑmat = reshape(qₑ, (6, nqp))
    qprevₑ = @view uₑprev[viscosity_range]
    qₑprevmat = reshape(qprevₑ, (6, nqp))

    reinit!(cv, cell)

    @inbounds for qp ∈ 1:nqp
        dΩ = getdetJdV(cv, qp)

        # Compute strain tensor
        ∇u = function_gradient(cv, qp, dₑ)
        ε = symmetric(∇u)

        # Extract viscous strain tensor
        εᵛ₀flat = @view qₑprevmat[:, qp]
        εᵛ₀ = SymmetricTensor{2,3}(εᵛ₀flat)

        # This is the used discretization:
        #     dεᵛdt = E₁/η₁ c : (ε - εᵛ)
        # <=> (εᵛ₁ - εᵛ₀) / Δt = E₁/η₁ c : (ε - εᵛ₁) = E₁/η₁ c : ε - E₁/η₁ c : εᵛ₁
        # <=> εᵛ₁ / Δt + E₁/η₁ c : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # <=> (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε

        # Predictor
        I = one(ε)
        c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
        c₂ = 1 / (1+ν) * one(c₁)
        ℂ = c₁ + c₂

        # FIXME non-allocating version by using state_cache nlsolver
        A = tomandel(SMatrix, one(ℂ)/Δt + E₁/η₁ * ℂ)
        b = tomandel(SVector, εᵛ₀/Δt + E₁/η₁ * ℂ ⊡ ε)
        εᵛ₁flat = @view qₑmat[:, qp]
        εᵛ₁ = frommandel(typeof(ε), A \ b)

        # Store solution
        εᵛ₁flat .= εᵛ₁.data

        # Compute stress and tangent
        σ = E₀ * ℂ ⊡ ε + E₁ * ℂ ⊡ (ε - εᵛ₁)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ σ * dΩ
        end
    end
end

# jac
function assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, element_cache::SimpleCondensedLinearViscoelasticityCache, p)
    (; material_parameters, displacement_range, viscosity_range, cv) = element_cache
    (; dt, uₑprev) = p
    Δt = dt
    (; E₀, E₁, μ, η₁, ν) = material_parameters

    nqp = getnquadpoints(cv)
    ndofs = getnbasefunctions(cv)

    dₑ = @view uₑ[displacement_range]
    qₑ = @view uₑ[viscosity_range]
    qₑmat = reshape(qₑ, (6, nqp))
    qprevₑ = @view uₑprev[viscosity_range]
    qₑprevmat = reshape(qprevₑ, (6, nqp))

    reinit!(cv, cell)

    @inbounds for qp ∈ 1:nqp
        dΩ = getdetJdV(cv, qp)

        # Compute strain tensor
        ∇u = function_gradient(cv, qp, dₑ)
        ε = symmetric(∇u)

        # Extract viscous strain tensor
        εᵛ₀flat = @view qₑprevmat[:, qp]
        εᵛ₀ = SymmetricTensor{2,3}(εᵛ₀flat)

        # This is the used discretization:
        #     dεᵛdt = E₁/η₁ c : (ε - εᵛ)
        # <=> (εᵛ₁ - εᵛ₀) / Δt = E₁/η₁ c : (ε - εᵛ₁) = E₁/η₁ c : ε - E₁/η₁ c : εᵛ₁
        # <=> εᵛ₁ / Δt + E₁/η₁ c : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # <=> (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε

        # Predictor
        I = one(ε)
        c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
        c₂ = 1 / (1+ν) * one(c₁)
        ℂ = c₁ + c₂

        # FIXME non-allocating version by using state_cache nlsolver
        A = tomandel(SMatrix, one(ℂ)/Δt + E₁/η₁ * ℂ)
        b = tomandel(SVector, εᵛ₀/Δt + E₁/η₁ * ℂ ⊡ ε)
        εᵛ₁flat = @view qₑmat[:, qp]
        εᵛ₁ = frommandel(typeof(ε), A \ b)

        # Store solution
        εᵛ₁flat .= εᵛ₁.data

        # Corrector
        # Local problem: (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # =>  dLdQ = 𝐈 / Δt + E₁/η₁ c   := A
        # => -dLdF = E₁/η₁ c            := B

        # FIXME non-allocating version by using state_cache nlsolver
        B = tomandel(SMatrix, E₁/η₁ * ℂ)
        dqdε = frommandel(typeof(ℂ), A \ B)
        ∂σ∂q = - E₁ * ℂ

        # Compute tangent
        ∂σ∂ε = (E₀ + E₁) * ℂ + ∂σ∂q ⊡ dqdε

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            ∇δui∂σ∂ε = ∇δui ⊡ ∂σ∂ε # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂σ∂ε ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

# Combined residual and jac
function assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, element_cache::SimpleCondensedLinearViscoelasticityCache, p)
    (; material_parameters, displacement_range, viscosity_range, cv) = element_cache
    (; dt, uₑprev) = p
    Δt = dt
    (; E₀, E₁, μ, η₁, ν) = material_parameters

    nqp = getnquadpoints(cv)
    ndofs = getnbasefunctions(cv)

    dₑ = @view uₑ[displacement_range]
    qₑ = @view uₑ[viscosity_range]
    qₑmat = reshape(qₑ, (6, nqp))
    qprevₑ = @view uₑprev[viscosity_range]
    qₑprevmat = reshape(qprevₑ, (6, nqp))

    reinit!(cv, cell)

    @inbounds for qp ∈ 1:nqp
        dΩ = getdetJdV(cv, qp)

        # Compute strain tensor
        ∇u = function_gradient(cv, qp, dₑ)
        ε = symmetric(∇u)

        # Extract viscous strain tensor
        εᵛ₀flat = @view qₑprevmat[:, qp]
        εᵛ₀ = SymmetricTensor{2,3}(εᵛ₀flat)

        # This is the used discretization:
        #     dεᵛdt = E₁/η₁ c : (ε - εᵛ)
        # <=> (εᵛ₁ - εᵛ₀) / Δt = E₁/η₁ c : (ε - εᵛ₁) = E₁/η₁ c : ε - E₁/η₁ c : εᵛ₁
        # <=> εᵛ₁ / Δt + E₁/η₁ c : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # <=> (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε

        # Predictor
        I = one(ε)
        c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
        c₂ = 1 / (1+ν) * one(c₁)
        ℂ = c₁ + c₂

        # FIXME non-allocating version by using state_cache nlsolver
        A = tomandel(SMatrix, one(ℂ)/Δt + E₁/η₁ * ℂ)
        b = tomandel(SVector, εᵛ₀/Δt + E₁/η₁ * ℂ ⊡ ε)
        εᵛ₁flat = @view qₑmat[:, qp]
        εᵛ₁ = frommandel(typeof(ε), A \ b)
        # Store solution
        εᵛ₁flat .= εᵛ₁.data

        # Corrector
        # Local problem: (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # =>  dLdQ = 𝐈 / Δt + E₁/η₁ c   := A
        # => -dLdF = E₁/η₁ c            := B

        # FIXME non-allocating version by using state_cache nlsolver
        B = tomandel(SMatrix, E₁/η₁ * ℂ)
        dqdε = frommandel(typeof(ℂ), A \ B)
        ∂σ∂q = - E₁ * ℂ

        # Compute stress and tangent
        σ = E₀ * ℂ ⊡ ε + E₁ * ℂ ⊡ (ε - εᵛ₁)
        ∂σ∂ε = (E₀ + E₁) * ℂ + ∂σ∂q ⊡ dqdε

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ σ * dΩ

            ∇δui∂σ∂ε = ∇δui ⊡ ∂σ∂ε # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂σ∂ε ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::SimpleCondensedLinearViscoelasticity, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    nqp        = getnquadpoints(qr)
    ip         = Ferrite.getfieldinterpolation(sdh, element_model.displacement_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)

    displacement_range = dof_range(sdh, element_model.displacement_name)
    viscosity_range    = (displacement_range[end]+1):(displacement_range[end]+6nqp)

    return SimpleCondensedLinearViscoelasticityCache(
        element_model.material_parameters,
        displacement_range,
        viscosity_range,
        CellValues(qr, ip, ip_geo),
    )
end

function load_element_unknowns!(uₑ, u, cell, element::SimpleCondensedLinearViscoelasticityCache)
    uₑ[element.displacement_range] .= u[celldofs(cell)]
    nqp                             = getnquadpoints(element.cv)
    id                              = cellid(cell)
    viscoidx_beg                    = ndofs(cell.dh.dh)+(id-1)*6nqp+1
    viscoidx_end                    = ndofs(cell.dh.dh)+(id-0)*6nqp
    uₑ[element.viscosity_range]    .= u[viscoidx_beg:viscoidx_end]
    return nothing
end

function store_condensed_element_unknowns!(uₑ, u, cell, element::SimpleCondensedLinearViscoelasticityCache)
    nqp                           = getnquadpoints(element.cv)
    id                            = cellid(cell)
    viscoidx_beg                  = ndofs(cell.dh.dh)+(id-1)*6nqp+1
    viscoidx_end                  = ndofs(cell.dh.dh)+(id-0)*6nqp
    u[viscoidx_beg:viscoidx_end] .= uₑ[element.viscosity_range]
    return nothing
end

@concrete struct ImplicitEulerInfo
    uprev
    dt
    t
end

@concrete struct ImplicitEulerElementInfo
    uₑprev
    dt
    t
end

allocate_element_unknown_vector(element::SimpleCondensedLinearViscoelasticityCache, sdh) = zeros(getnbasefunctions(element.cv)+6getnquadpoints(element.cv))

function query_element_parameters(element::SimpleCondensedLinearViscoelasticityCache, cell, p)
    (; cv) = element
    (; uprev, dt, t) = p
    # TODO query pe from taskbuffer
    uₑprev = zeros(getnbasefunctions(cv)+6getnquadpoints(cv))
    load_element_unknowns!(uₑprev, uprev, cell, element)
    return ImplicitEulerElementInfo(uₑprev, dt, t)
end
