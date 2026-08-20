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
    SimpleCondensedLinearViscoelasticity(material_parameters, qrc, displacement_name, viscosity_name;
                                          condensation = Separate(), corrector = Stored())

Linear viscoelasticity (standard linear solid) with the viscous strain εᵛ as a
condensed per-quadrature-point internal variable. The element owns the LOCAL
stage problem for εᵛ (scaled by `stage_scaling(ctx)`), solved once per
quadrature point by [`condense_cell!`](@ref) — [`condense_internal!`](@ref)
must run before any evaluation sweep. The Mandel factorization `A` of the
local operator is retained per quadrature point and read by the `Consistent`
kernel.

`condensation`/`corrector` are construction-time seams
([`CondensationElection`](@ref)/[`CorrectorElection`](@ref)); only their
defaults (`Separate()`/`Stored()`) are implemented.
"""
struct SimpleCondensedLinearViscoelasticity{Cond <: CondensationElection, Corr <: CorrectorElection} <: AbstractCondensedNonlinearIntegrator
    material_parameters::MaxwellParameters
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    displacement_name::Symbol
    viscosity_name::Symbol
    condensation::Cond
    corrector::Corr
end
function SimpleCondensedLinearViscoelasticity(material_parameters, qrc, displacement_name, viscosity_name;
        condensation = Separate(), corrector = Stored())
    condensation isa Separate || condensation_election_error(condensation)
    corrector isa Stored || corrector_election_error(corrector)
    return SimpleCondensedLinearViscoelasticity(material_parameters, qrc, displacement_name, viscosity_name,
                                                 condensation, corrector)
end
FerriteOperators.condensation_election(integrator::SimpleCondensedLinearViscoelasticity) = integrator.condensation
FerriteOperators.corrector_election(integrator::SimpleCondensedLinearViscoelasticity) = integrator.corrector

"""
The cache associated with [`SimpleCondensedLinearViscoelasticity`](@ref). It
carries the retained per-quadrature-point Mandel factorization `A`
(`correctors`, populated by [`condense_cell!`](@ref) and read by the
`Consistent` kernel), and declares [`has_internal_state`](@ref).
"""
struct SimpleCondensedLinearViscoelasticityCache{NQP, CV <: CellValues} <: AbstractVolumetricElementCache
    material_parameters::MaxwellParameters
    cv::CV
    correctors::ItemStates{SVector{NQP, SMatrix{6, 6, Float64, 36}}}
end

Ferrite.getnquadpoints(e::SimpleCondensedLinearViscoelasticityCache) = getnquadpoints(e.cv)
reinit_values!(e::SimpleCondensedLinearViscoelasticityCache, cell) = Ferrite.reinit!(e.cv, cell)

function duplicate_for_device(device, cache::SimpleCondensedLinearViscoelasticityCache)
    return SimpleCondensedLinearViscoelasticityCache(
        cache.material_parameters,
        duplicate_for_device(device, cache.cv),
        duplicate_for_device(device, cache.correctors),
    )
end

function get_number_of_internal_dofs_per_element(element_model, cache::SimpleCondensedLinearViscoelasticityCache, sdh)
    nqp = getnquadpoints(cache.cv)
    return [6nqp for i in sdh.cellset]
end

provides_analytic(::Type{<:SimpleCondensedLinearViscoelasticityCache}, ::Union{JacobianKind, JacobianResidualKind}) = true
has_internal_state(::Type{<:SimpleCondensedLinearViscoelasticityCache}) = true

function FerriteOperators.invalidate_correctors!(cache::SimpleCondensedLinearViscoelasticityCache)
    FerriteOperators.invalidate_item_states!(cache.correctors)
    return nothing
end

# The elastic stiffness (unit modulus) — shared by predictor and corrector,
# and by the pure residual kernel (no local-solve dependence).
@inline function _sls_unit_stiffness(ε, ν)
    I = one(ε)
    c₁ = ν / ((ν + 1) * (1 - 2ν)) * I ⊗ I
    c₂ = 1 / (1 + ν) * one(c₁)
    return c₁ + c₂
end

# Local stage problem (backward-Euler form scaled by γ̃):
#     dεᵛdt = E₁/η₁ ℂ : (ε - εᵛ)
# <=> (𝐈/γ̃ + E₁/η₁ ℂ) : εᵛ₁ = εᵛ₀/γ̃ + E₁/η₁ ℂ : ε
# Returns (ℂ, A, εᵛ₁); A is the Mandel matrix of the local operator, retained
# by `condense_cell!` and reused by the consistent tangent.
@inline function _sls_local_solve(cache::SimpleCondensedLinearViscoelasticityCache, ε, εᵛ₀, γ̃)
    (; E₁, η₁, ν) = cache.material_parameters
    ℂ = _sls_unit_stiffness(ε, ν)
    # FIXME non-allocating version by using state_cache nlsolver
    A = tomandel(SMatrix, one(ℂ) / γ̃ + E₁ / η₁ * ℂ)
    b = tomandel(SVector, εᵛ₀ / γ̃ + E₁ / η₁ * ℂ ⊡ ε)
    εᵛ₁ = frommandel(typeof(ε), A \ b)
    return ℂ, A, εᵛ₁
end

# Corrector: consistent (algorithmic) tangent through the local solve, reading
# the retained factorization `A` instead of receiving it from an inline solve.
@inline function _sls_consistent_tangent(cache::SimpleCondensedLinearViscoelasticityCache, ℂ, A)
    (; E₀, E₁, η₁) = cache.material_parameters
    # FIXME non-allocating version by using state_cache nlsolver
    B = tomandel(SMatrix, E₁ / η₁ * ℂ)
    dqdε = frommandel(typeof(ℂ), A \ B)
    ∂σ∂q = -E₁ * ℂ
    return (E₀ + E₁) * ℂ + ∂σ∂q ⊡ dqdε
end
# The FrozenQ partial: dq/dε = 0, so only the elastic + relaxed-branch term
# survives — three lines, the same expression with the correction dropped.
@inline function _sls_frozen_tangent(cache::SimpleCondensedLinearViscoelasticityCache, ℂ)
    (; E₀, E₁) = cache.material_parameters
    return (E₀ + E₁) * ℂ
end

@inline function _sls_stage_scaling(ctx)
    ctx === nothing && throw(ArgumentError(
        "SimpleCondensedLinearViscoelasticity requires a TimeIntegrationContext: " *
        "the local εᵛ stage problem scales by stage_scaling(ctx)."))
    return stage_scaling(ctx)
end

"""
    condense_cell!(cache::SimpleCondensedLinearViscoelasticityCache, args, weights) -> CondensationReport

Solve the local viscous-strain problem at every quadrature point (a direct
linear solve — always converged, zero inner iterations), write the trial εᵛ
and retain the Mandel factorization `A`.
"""
function FerriteOperators.condense_cell!(cache::SimpleCondensedLinearViscoelasticityCache{NQP}, args::CellArgs, weights::NamedTuple) where {NQP}
    cv  = cache.cv
    γ̃   = _sls_stage_scaling(args.ctx)
    id  = cellid(args.cell)

    dₑ        = args.states.u
    qₑmat     = reshape(args.states.q, (6, NQP))
    qₑprevmat = reshape(args.states.qprev, (6, NQP))

    As = MVector{NQP, SMatrix{6, 6, Float64, 36}}(undef)
    @inbounds for qp in 1:NQP
        ε   = symmetric(function_gradient(cv, qp, dₑ))
        εᵛ₀ = SymmetricTensor{2, 3}(@view qₑprevmat[:, qp])
        ℂ, A, εᵛ₁ = _sls_local_solve(cache, ε, εᵛ₀, γ̃)
        (@view qₑmat[:, qp]) .= εᵛ₁.data
        As[qp] = A
    end
    FerriteOperators.set_item_state!(cache.correctors, id, SVector{NQP}(As))
    return CondensationReport(true, NQP, 0, 0, id, 0, 0.0, 1.0)
end

# One concrete entry method per provided kernel (no blanket request method:
# it would satisfy every `hasmethod` probe in the setup-time validation).
assemble_cell!(req::ResidualRequest, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs) = _sls_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u, Consistent}, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs) = _sls_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u, FrozenQ}, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs) = _sls_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest{Consistent}, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs) = _sls_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest{FrozenQ}, cache::SimpleCondensedLinearViscoelasticityCache, args::CellArgs) = _sls_assemble!(req, cache, args)

const _SLSJacobianLike = Union{JacobianRequest{:u, Consistent}, JacobianRequest{:u, FrozenQ}, JacobianResidualRequest{Consistent}, JacobianResidualRequest{FrozenQ}}
const _SLSFrozenLike   = Union{JacobianRequest{:u, FrozenQ}, JacobianResidualRequest{FrozenQ}}

# Pure evaluation at the FROZEN εᵛ the last `condense_internal!` wrote: no
# solve, no write-back, ℂ recomputed freely (a pure function of ε, no local
# solve dependence). `item_state` throws, naming the cell, if
# `condense_internal!` never ran for it.
function _sls_assemble!(req::Union{ResidualRequest, _SLSJacobianLike}, cache::SimpleCondensedLinearViscoelasticityCache{NQP}, args::CellArgs) where {NQP}
    (; E₀, E₁, ν) = cache.material_parameters
    cv = cache.cv
    id = cellid(args.cell)
    dₑ = args.states.u
    qₑmat = reshape(args.states.q, (6, NQP))
    ndofs = getnbasefunctions(cv)

    needs_jac = req isa _SLSJacobianLike
    As = (needs_jac && !(req isa _SLSFrozenLike)) ? FerriteOperators.item_state(cache.correctors, id) : nothing

    @inbounds for qp in 1:NQP
        dΩ  = getdetJdV(cv, qp)
        ε   = symmetric(function_gradient(cv, qp, dₑ))
        εᵛ₁ = SymmetricTensor{2, 3}(@view qₑmat[:, qp])
        ℂ   = _sls_unit_stiffness(ε, ν)

        if req isa Union{ResidualRequest, JacobianResidualRequest}
            σ = E₀ * ℂ ⊡ ε + E₁ * ℂ ⊡ (ε - εᵛ₁)
            for i in 1:ndofs
                req.r[i] += shape_gradient(cv, qp, i) ⊡ σ * dΩ
            end
        end
        if needs_jac
            ∂σ∂ε = req isa _SLSFrozenLike ? _sls_frozen_tangent(cache, ℂ) : _sls_consistent_tangent(cache, ℂ, As[qp])
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
    ncells     = getncells(Ferrite.get_grid(sdh.dh))

    return SimpleCondensedLinearViscoelasticityCache(
        element_model.material_parameters,
        CellValues(qr, ip, ip_geo),
        ItemStates{SVector{nqp, SMatrix{6, 6, Float64, 36}}}(ncells),
    )
end
