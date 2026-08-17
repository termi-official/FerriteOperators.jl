@doc raw"""
    SimpleLinearIntegrator

Represents the integrand of the linear form ``b(v) = f v(x) dx`` for a given constant ``f`` and ``v`` from the test function space.
"""
struct SimpleLinearIntegrator <: AbstractLinearIntegrator
    # This is specific to our model
    f::Float64
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

"""
The cache associated with [`SimpleLinearIntegrator`](@ref) to assemble element "constant" vectors.
"""
struct SimpleLinearElementCache{CV <: CellValues} <: AbstractVolumetricElementCache
    f::Float64
    cellvalues::CV
end

Ferrite.getnquadpoints(e::SimpleLinearElementCache) = getnquadpoints(e.cellvalues)
reinit_values!(e::SimpleLinearElementCache, cell) = Ferrite.reinit!(e.cellvalues, cell)
function setup_element_cache(element_model::SimpleLinearIntegrator, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    field_name = element_model.field_name
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    return SimpleLinearElementCache(element_model.f, CellValues(qr, ip, ip_geo))
end

# The load form is state-independent: the residual kernel reads nothing from
# `args.states`.
function assemble_cell!(req::ResidualRequest, cache::SimpleLinearElementCache, args)
    (; cellvalues, f) = cache
    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:getnbasefunctions(cellvalues)
            req.r[i] += f * shape_value(cellvalues, qp, i) * dΩ
        end
    end
end

duplicate_for_device(device, cache::SimpleLinearElementCache) = SimpleLinearElementCache(cache.f, duplicate_for_device(device, cache.cellvalues))

@doc raw"""
    SimpleBilinearMassIntegrator

Represents the integrand of the bilinear form ``a(u,v) = \int v(x) \cdot D u(x) dx`` for a given Mass value ``D`` and ``u,v`` from the same function space.
"""
struct SimpleBilinearMassIntegrator <: AbstractBilinearIntegrator
    # This is specific to our model
    ρ::Float64
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

"""
The cache associated with [`SimpleBilinearMassIntegrator`](@ref) to assemble element Mass matrices.
"""
struct SimpleBilinearMassElementCache{CV <: CellValues} <: AbstractVolumetricElementCache
    ρ::Float64
    cellvalues::CV
end

Ferrite.getnquadpoints(e::SimpleBilinearMassElementCache) = getnquadpoints(e.cellvalues)
reinit_values!(e::SimpleBilinearMassElementCache, cell) = Ferrite.reinit!(e.cellvalues, cell)
function duplicate_for_device(device, cache::SimpleBilinearMassElementCache)
    return SimpleBilinearMassElementCache(
        cache.ρ,
        duplicate_for_device(device, cache.cellvalues),
    )
end

function setup_element_cache(element_model::SimpleBilinearMassIntegrator, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    field_name = element_model.field_name
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    return SimpleBilinearMassElementCache(element_model.ρ, CellValues(qr, ip, ip_geo))
end

# The bilinear form induces a linear operator, so its residual is the element
# matrix acting on the element vector.
provides_analytic(::Type{<:SimpleBilinearMassElementCache}, ::JacobianKind) = true
function assemble_cell!(req::JacobianRequest{:u}, cache::SimpleBilinearMassElementCache, args)
    (; cellvalues, ρ) = cache
    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:getnbasefunctions(cellvalues)
            Nᵢ = shape_value(cellvalues, qp, i)
            for j in 1:getnbasefunctions(cellvalues)
                req.K[i, j] += ρ * shape_value(cellvalues, qp, j) ⋅ Nᵢ * dΩ
            end
        end
    end
end
function assemble_cell!(req::ResidualRequest, cache::SimpleBilinearMassElementCache, args)
    (; cellvalues, ρ) = cache
    uₑ = args.states.u
    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        uval = function_value(cellvalues, qp, uₑ)
        for i in 1:getnbasefunctions(cellvalues)
            req.r[i] += ρ * uval ⋅ shape_value(cellvalues, qp, i) * dΩ
        end
    end
end
