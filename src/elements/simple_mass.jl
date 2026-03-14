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

P_ij = M^{-1}_{e} ∫ ϕ_i(x) ⋅ ϕc_j(x) dx (ϕc is the coarse basis function)
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

@doc raw"""
    NestedMassProlongatorIntegrator

Integrator for assembling a prolongation operator between a fine and a coarse grid that
are **hierarchically nested** (geometric multigrid).

Unlike [`MassProlongatorIntegrator`](@ref) (same grid, two DofHandlers), this variant
evaluates the coarse basis functions at fine quadrature points by mapping through the
reference-space affine map stored in [`NestedGridCellCache`](@ref):

```math
\xi_\text{coarse} = \sum_i N_i^{\text{geo,fine}}(\xi_\text{fine})\; \hat{x}_i
```

where ``\hat{x}_i`` are the positions of the fine cell's nodes in the *parent* (coarse)
reference element (`child_ref_coords`).
"""
struct NestedMassProlongatorIntegrator <: AbstractTransferIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

"""
The cache associated with [`NestedMassProlongatorIntegrator`](@ref).
"""
struct NestedMassProlongatorElementCache{CV_fine, IP_coarse, IP_geo, K} <: AbstractTransferElementCache
    cv_fine::CV_fine       # CellValues for fine space (fine quadrature points)
    ip_coarse::IP_coarse   # scalar interpolation for coarse space
    ip_geo_fine::IP_geo    # scalar geometric interpolation of fine element (for ref-space map)
    Mₑbuf::K
    Pₑbuf::K
end

function duplicate_for_device(device, cache::NestedMassProlongatorElementCache)
    return NestedMassProlongatorElementCache(
        duplicate_for_device(device, cache.cv_fine),
        cache.ip_coarse,    # immutable, safe to share across threads
        cache.ip_geo_fine,  # immutable, safe to share across threads
        similar(cache.Mₑbuf),
        similar(cache.Pₑbuf),
    )
end

function setup_transfer_element_cache(
        integrator::NestedMassProlongatorIntegrator,
        sdh_fine::SubDofHandler,
        sdh_coarse::SubDofHandler,
    )
    qr          = getquadraturerule(integrator.qrc, sdh_fine)
    field_name  = integrator.field_name
    ip_fine     = Ferrite.getfieldinterpolation(sdh_fine,   field_name)
    ip_coarse   = Ferrite.getfieldinterpolation(sdh_coarse, field_name)
    ip_geo_vec  = geometric_subdomain_interpolation(sdh_fine)  # vectorized form for CellValues
    ip_geo_fine = Ferrite.geometric_interpolation(typeof(get_first_cell(sdh_fine)))  # scalar for ref-space map
    cv_fine     = CellValues(qr, ip_fine, ip_geo_vec)
    n_fine      = getnbasefunctions(ip_fine)
    n_coarse    = getnbasefunctions(ip_coarse)
    return NestedMassProlongatorElementCache(cv_fine, ip_coarse, ip_geo_fine,
                                             zeros(n_fine, n_fine), zeros(n_fine, n_coarse))
end

function assemble_transfer_element!(
        Pₑ::AbstractMatrix,
        tc::NestedGridCellCache,
        cache::NestedMassProlongatorElementCache,
        p,
    )
    (; cv_fine, ip_coarse, ip_geo_fine, Mₑbuf, Pₑbuf) = cache
    n_fine   = getnbasefunctions(cv_fine)
    n_coarse = getnbasefunctions(ip_coarse)

    reinit!(cv_fine, tc)   # delegates to Ferrite.reinit!(cv, tc::NestedGridCellCache)
    child_nodes = get_child_ref_coords(tc)  # fine cell node positions in coarse ref element

    fill!(Mₑbuf, zero(eltype(Mₑbuf)))
    fill!(Pₑbuf, zero(eltype(Pₑbuf)))

    qr_points = cv_fine.qr.points

    @inbounds for qp in 1:getnquadpoints(cv_fine)
        dΩ     = getdetJdV(cv_fine, qp)
        ξ_fine = qr_points[qp]

        # Map fine reference quadrature point to coarse reference coordinates via the
        # affine map defined by child_nodes (the fine cell's corners in parent ref space).
        ξ_coarse = sum(
            Ferrite.reference_shape_value(ip_geo_fine, ξ_fine, i) * child_nodes[i]
            for i in eachindex(child_nodes)
        )

        for i in 1:n_fine
            Nᵢ = shape_value(cv_fine, qp, i)
            for j in i:n_fine   # upper triangle only (M is SPD)
                Mₑbuf[i, j] += shape_value(cv_fine, qp, j) ⋅ Nᵢ * dΩ
            end
            for j in 1:n_coarse
                Nⱼ_c = Ferrite.reference_shape_value(ip_coarse, ξ_coarse, j)
                Pₑbuf[i, j] += Nⱼ_c ⋅ Nᵢ * dΩ
            end
        end
    end

    C = cholesky!(Symmetric(Mₑbuf))
    ldiv!(Pₑ, C, Pₑbuf)
end
