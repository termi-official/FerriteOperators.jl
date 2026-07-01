"""
    QVector{T, VT, OT, NT} <: AbstractVector{T}

A flat storage vector for quadrature-point data across all cells, with per-cell
random-access via [`get_data_for_index`](@ref).

Fields:
- `data`: flat storage (`AbstractVector{T}`) holding all quadrature values
- `offsets`: `offsets[cellid]` is the 1-based start index in `data` for cell `cellid`
- `npoints`: `npoints[cellid]` is the number of quadrature points for cell `cellid`

Use [`setup_qvector`](@ref) to build a `QVector` from a [`DofHandler`](@ref) or an
assembled operator. Use [`get_data_for_index`](@ref) to obtain a mutable view into
the slice owned by a particular cell.
"""
struct QVector{T, VT <: AbstractVector{T}, OT <: AbstractVector, NT <: AbstractVector} <: AbstractVector{T}
    data::VT
    offsets::OT
    npoints::NT
end

Base.size(v::QVector)             = (length(v.data),)
Base.getindex(v::QVector, i::Int) = getindex(v.data, i)
Base.eltype(::QVector{T}) where T = T

"""
    get_data_for_index(q::QVector, cellid::Integer)

Return a mutable view into the slice of `q` that belongs to cell `cellid`.
The view has length `q.npoints[cellid]`.
"""
@inline function get_data_for_index(r::QVector, i::Integer)
    i1 = r.offsets[i]
    n  = r.npoints[i]
    return @view r.data[i1:i1+n-1]
end

"""
    setup_qvector(::Type{T}, dh::AbstractDofHandler, qrc) -> QVector{T}

Build a [`QVector`](@ref) with element type `T` whose layout matches the quadrature
structure defined by `qrc` over all cells in `dh`.

For each `SubDofHandler` in `dh`, the number of quadrature points per cell is
determined by `getnquadpoints(getquadraturerule(qrc, sdh))`. Cells not belonging
to any subdomain receive zero quadrature points.
"""
function setup_qvector(::Type{T}, dh::AbstractDofHandler, qrc) where {T}
    grid   = get_grid(dh)
    ncells = getncells(grid)
    offsets = zeros(Int, ncells)
    npoints = zeros(Int, ncells)

    for sdh in dh.subdofhandlers
        qr  = getquadraturerule(qrc, sdh)
        nqp = getnquadpoints(qr)
        for cellid in sdh.cellset
            npoints[cellid] = nqp
        end
    end

    # Build 1-based start offsets
    offset = 1
    for cellid in 1:ncells
        offsets[cellid] = offset
        offset += npoints[cellid]
    end
    data = zeros(T, offset - 1)
    return QVector(data, offsets, npoints)
end

"""
    setup_qvector(::Type{T}, operator) -> QVector{T}

Build a [`QVector`](@ref) whose layout matches the quadrature structure of `operator`.

The number of quadrature points per cell is determined from the element caches
stored in the operator's subdomain caches via `getnquadpoints`.
"""
function setup_qvector(::Type{T}, operator) where {T}
    dh     = operator.dh
    grid   = get_grid(dh)
    ncells = getncells(grid)
    offsets = zeros(Int, ncells)
    npoints = zeros(Int, ncells)

    for sc in operator.subdomain_caches
        domain = sc.domain
        nqp    = getnquadpoints(domain.element)
        for cellid in domain.sdh.cellset
            npoints[cellid] = nqp
        end
    end

    offset = 1
    for cellid in 1:ncells
        offsets[cellid] = offset
        offset += npoints[cellid]
    end
    data = zeros(T, offset - 1)
    return QVector(data, offsets, npoints)
end
