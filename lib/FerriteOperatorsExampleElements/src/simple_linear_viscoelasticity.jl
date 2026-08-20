"""
Material parameters of the standard linear solid used by
[`SimpleCondensedLinearViscoelasticity`](@ref): the two spring moduli `E₀`,
`E₁`, the shear modulus `μ`, the dashpot viscosity `η₁` and Poisson's ratio `ν`.
"""
@kwdef struct MaxwellParameters
    E₀::Float64 = 70e3
    E₁::Float64 = 20e3
    μ::Float64  = 1e3
    η₁::Float64 = 1e3
    ν::Float64  = 0.3
end

@doc raw"""
    SimpleCondensedLinearViscoelasticity

Linear viscoelasticity (standard linear solid) with the viscous strain εᵛ as a
condensed per-quadrature-point internal variable. The element owns the LOCAL
stage problem for εᵛ (scaled by `stage_scaling(ctx)`); the previous state arrives through
the `uprev` slot and the trial state is written back into the element-local
`u` buffer per the condensation contract.
"""
struct SimpleCondensedLinearViscoelasticity <: AbstractCondensedNonlinearIntegrator
    material_parameters::MaxwellParameters
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    displacement_name::Symbol
    viscosity_name::Symbol
end

"""
The cache associated with [`SimpleCondensedLinearViscoelasticity`](@ref). It
carries the element-local dof ranges of the displacement and of the condensed
viscous strain, and declares [`has_internal_state`](@ref).
"""
struct SimpleCondensedLinearViscoelasticityCache{CV <: CellValues} <: AbstractVolumetricElementCache
    material_parameters::MaxwellParameters
    displacement_range::UnitRange{Int}
    viscosity_range::UnitRange{Int}
    cv::CV
end

Ferrite.getnquadpoints(e::SimpleCondensedLinearViscoelasticityCache) = getnquadpoints(e.cv)
reinit_values!(e::SimpleCondensedLinearViscoelasticityCache, cell) = Ferrite.reinit!(e.cv, cell)

function duplicate_for_device(device, cache::SimpleCondensedLinearViscoelasticityCache)
    return SimpleCondensedLinearViscoelasticityCache(
        cache.material_parameters,
        cache.displacement_range,
        cache.viscosity_range,
        duplicate_for_device(device, cache.cv),
    )
end

function get_number_of_internal_dofs_per_element(element_model, cache::SimpleCondensedLinearViscoelasticityCache, sdh)
    return [length(cache.viscosity_range) for i in sdh.cellset]
end

provides_analytic(::Type{<:SimpleCondensedLinearViscoelasticityCache}, ::Union{JacobianKind, JacobianResidualKind}) = true
has_internal_state(::Type{<:SimpleCondensedLinearViscoelasticityCache}) = true

# The elastic stiffness (unit modulus) — shared by predictor and corrector.
@inline function _sls_unit_stiffness(ε, ν)
    I = one(ε)
    c₁ = ν / ((ν + 1) * (1 - 2ν)) * I ⊗ I
    c₂ = 1 / (1 + ν) * one(c₁)
    return c₁ + c₂
end

# Local stage problem (backward-Euler form scaled by γ̃):
#     dεᵛdt = E₁/η₁ ℂ : (ε - εᵛ)
# <=> (𝐈/γ̃ + E₁/η₁ ℂ) : εᵛ₁ = εᵛ₀/γ̃ + E₁/η₁ ℂ : ε
# Returns (ℂ, A, εᵛ₁); A is the Mandel matrix of the local operator, reused by
# the corrector.
@inline function _sls_local_solve(cache::SimpleCondensedLinearViscoelasticityCache, ε, εᵛ₀, γ̃)
    (; E₁, η₁, ν) = cache.material_parameters
    ℂ = _sls_unit_stiffness(ε, ν)
    # FIXME non-allocating version by using state_cache nlsolver
    A = tomandel(SMatrix, one(ℂ) / γ̃ + E₁ / η₁ * ℂ)
    b = tomandel(SVector, εᵛ₀ / γ̃ + E₁ / η₁ * ℂ ⊡ ε)
    εᵛ₁ = frommandel(typeof(ε), A \ b)
    return ℂ, A, εᵛ₁
end

# Corrector: consistent (algorithmic) tangent through the local solve.
@inline function _sls_consistent_tangent(cache::SimpleCondensedLinearViscoelasticityCache, ℂ, A)
    (; E₀, E₁, η₁) = cache.material_parameters
    # FIXME non-allocating version by using state_cache nlsolver
    B = tomandel(SMatrix, E₁ / η₁ * ℂ)
    dqdε = frommandel(typeof(ℂ), A \ B)
    ∂σ∂q = -E₁ * ℂ
    return (E₀ + E₁) * ℂ + ∂σ∂q ⊡ dqdε
end

@inline function _sls_stage_scaling(ctx)
    ctx === nothing && throw(ArgumentError(
        "SimpleCondensedLinearViscoelasticity requires a TimeIntegrationContext: " *
        "the local εᵛ stage problem scales by stage_scaling(ctx)."))
    return stage_scaling(ctx)
end

# One concrete entry method per provided kernel (no blanket request method:
# it would satisfy every `hasmethod` probe in the setup-time validation).
assemble_cell!(req::ResidualRequest, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs) = _sls_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u}, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs) = _sls_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs) = _sls_assemble!(req, cache, args)

function _sls_assemble!(req::Union{ResidualRequest, JacobianRequest{:u}, JacobianResidualRequest}, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs)
    (; displacement_range, viscosity_range, cv) = cache
    (; E₀, E₁ ) = cache.material_parameters
    γ̃ = _sls_stage_scaling(args.ctx)
    uₑ     = args.states.u
    uₑprev = args.states.uprev

    nqp   = getnquadpoints(cv)
    ndofs = getnbasefunctions(cv)

    dₑ         = @view uₑ[displacement_range]
    qₑmat      = reshape((@view uₑ[viscosity_range]), (6, nqp))
    qₑprevmat  = reshape((@view uₑprev[viscosity_range]), (6, nqp))

    @inbounds for qp in 1:nqp
        dΩ = getdetJdV(cv, qp)
        ε  = symmetric(function_gradient(cv, qp, dₑ))
        εᵛ₀ = SymmetricTensor{2, 3}(@view qₑprevmat[:, qp])

        ℂ, A, εᵛ₁ = _sls_local_solve(cache, ε, εᵛ₀, γ̃)
        # Trial write-back into the element-local buffer (the condensation
        # contract; the framework's store step propagates it into global u).
        (@view qₑmat[:, qp]) .= εᵛ₁.data

        if req isa Union{ResidualRequest, JacobianResidualRequest}
            σ = E₀ * ℂ ⊡ ε + E₁ * ℂ ⊡ (ε - εᵛ₁)
            for i in 1:ndofs
                req.r[i] += shape_gradient(cv, qp, i) ⊡ σ * dΩ
            end
        end
        if req isa Union{JacobianRequest{:u}, JacobianResidualRequest}
            ∂σ∂ε = _sls_consistent_tangent(cache, ℂ, A)
            for i in 1:ndofs
                ∇δui∂σ∂ε = shape_gradient(cv, qp, i) ⊡ ∂σ∂ε
                for j in 1:ndofs
                    req.K[i, j] += (∇δui∂σ∂ε ⊡ shape_gradient(cv, qp, j)) * dΩ
                end
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

function get_element_internal_index_range(cell, ivh, element::SimpleCondensedLinearViscoelasticityCache)
    nqp = getnquadpoints(element.cv)
    id  = cellid(cell)
    offset = internal_variable_offset(ivh, id)
    internal_beg = offset+1
    internal_end = offset+6nqp
    return internal_beg:internal_end
end

function load_element_unknowns!(uₑ, u, cell, ivh, element::SimpleCondensedLinearViscoelasticityCache)
    internal_range                         = get_element_internal_index_range(cell, ivh, element)
    @views uₑ[element.displacement_range] .= u[celldofs(cell)]
    @views uₑ[element.viscosity_range]    .= u[internal_range]
    return nothing
end

function store_condensed_element_unknowns!(uₑ, u, cell, ivh, element::SimpleCondensedLinearViscoelasticityCache)
    internal_range    = get_element_internal_index_range(cell, ivh, element)
    u[internal_range] .= uₑ[element.viscosity_range]
    return nothing
end

allocate_element_unknown_vector(element::SimpleCondensedLinearViscoelasticityCache, sdh) = zeros(getnbasefunctions(element.cv)+6getnquadpoints(element.cv))
