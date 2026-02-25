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
struct SimpleBilinearMassElementCache{CV <: CellValues} <: AbstractVolumetricElementCache
    ρ::Float64
    cellvalues::CV
end

function duplicate_for_device(device, cache::SimpleBilinearMassElementCache)
    return SimpleBilinearMassElementCache(
        cache.ρ,
        duplicate_for_device(device, cache.cellvalues),
    )
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

@doc raw"""
    MassProlongatorIntegrator
"""
struct MassProlongatorIntegrator <: AbstractTransferIntegrator
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

"""
The cache associated with [`MassProlongatorIntegrator`](@ref) to assemble element Mass matrices.
"""
struct MassProlongatorElementCache{CV1 <: CellValues, CV2 <: CellValues, K} <: AbstractTransferElementCache
    cv1::CV1
    cv2::CV2
    Mₑbuf::K
    Pₑbuf::K
end

function duplicate_for_device(device, cache::MassProlongatorElementCache)
    return MassProlongatorElementCache(
        duplicate_for_device(device, cache.cv1),
        duplicate_for_device(device, cache.cv2),
        similar(cache.Mₑbuf),
        similar(cache.Pₑbuf),
    )
end

function assemble_transfer_element!(Pₑ::AbstractMatrix, cell, element_cache::MassProlongatorElementCache, p)
    (; cv1, cv2, Mₑbuf, Pₑbuf) = element_cache
    n1 = getnbasefunctions(cv1)
    n2 = getnbasefunctions(cv2)

    reinit!(cv1, cell)
    reinit!(cv2, cell)

    fill!(Mₑbuf, zero(eltype(Mₑbuf)))
    fill!(Pₑbuf, zero(eltype(Pₑbuf)))

    # Single quadrature pass: accumulate Mₑ (upper triangle, M is SPD) and Pₑbuf together.
    # Merging the two loops halves the number of shape_value(cv1,...) evaluations.
    @inbounds for qp in 1:getnquadpoints(cv1)
        dΩ = getdetJdV(cv1, qp)
        for i in 1:n1
            Nᵢ = shape_value(cv1, qp, i)
            for j in i:n1  # upper triangle only
                Mₑbuf[i, j] += shape_value(cv1, qp, j) ⋅ Nᵢ * dΩ
            end
            for j in 1:n2
                Pₑbuf[i, j] += shape_value(cv2, qp, j) ⋅ Nᵢ * dΩ
            end
        end
    end

    # In-place Cholesky on the SPD mass matrix.  cholesky! overwrites the upper triangle of
    # Mₑbuf with the Cholesky factor and returns a lightweight wrapper (no large allocation),
    # unlike qr() which copies the matrix and allocates O(n²) for the reflectors.
    C = cholesky!(Symmetric(Mₑbuf))
    ldiv!(Pₑ, C, Pₑbuf)
end

function setup_transfer_element_cache(element_model::MassProlongatorIntegrator, sdh_row::SubDofHandler, sdh_col::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh_row)
    field_name = element_model.field_name
    ip1        = Ferrite.getfieldinterpolation(sdh_row, field_name)
    ip2        = Ferrite.getfieldinterpolation(sdh_col, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh_row)
    Mₑ = zeros(getnbasefunctions(ip1), getnbasefunctions(ip1))
    Pₑ = zeros(getnbasefunctions(ip1), getnbasefunctions(ip2))
    return MassProlongatorElementCache(CellValues(qr, ip1, ip_geo), CellValues(qr, ip2, ip_geo), Mₑ, Pₑ)
end
