"""
    QVector{T, VT, OT, NT} <: AbstractVector{T}

Flat storage for quadrature-point data across all cells, with per-cell random
access through [`get_range_for_cell`](@ref).

- `data`: flat storage (`AbstractVector{T}`) holding all quadrature values
- `offsets`: `offsets[cellid]` is the 1-based start index in `data` for cell `cellid`
- `npoints`: quadrature points of cell `cellid` as `npoints[cellid]`, or one
  count for every cell where that count is uniform

Built by [`setup_qvector`](@ref) from a `Ferrite.DofHandler` or an assembled
operator.
"""
struct QVector{T, VT <: AbstractVector{T}, OT, NT} <: AbstractVector{T}
    data::VT
    offsets::OT
    npoints::NT
end

Base.size(v::QVector)             = (length(v.data),)
Base.getindex(v::QVector, i::Int) = getindex(v.data, i)
Base.eltype(::QVector{T}) where {T} = T

"""
    get_range_for_cell(q::QVector, cellid::Integer)

A mutable view into the slice of `q` that belongs to cell `cellid`, of the
length `q.npoints` gives that cell.
"""
@inline function get_range_for_cell(r::QVector, i::Integer)
    i1 = r.offsets[i]
    n  = _get_npoints_for_cell(r, i)
    return @view r.data[i1:i1+n-1]
end

_get_npoints_for_cell(r::QVector, i) = _get_npoints_for_cell(r, r.npoints, i)
_get_npoints_for_cell(r, npoints::Integer, i) = npoints
_get_npoints_for_cell(r, npoints::AbstractVector, i) = npoints[i]

"""
    setup_qvector(::Type{T}, dh::AbstractDofHandler, qrc) -> QVector{T}

Build a [`QVector`](@ref) with element type `T`, laid out by `qrc` over the
cells of `dh`: every `SubDofHandler` contributes
`getnquadpoints(getquadraturerule(qrc, sdh))` points per cell, and cells
outside every subdomain zero.
"""
function setup_qvector(::Type{T}, dh::AbstractDofHandler, qrc) where {T}
    npoints = zeros(Int, getncells(get_grid(dh)))
    for sdh in dh.subdofhandlers
        qr  = getquadraturerule(qrc, sdh)
        nqp = getnquadpoints(qr)
        for cellid in sdh.cellset
            npoints[cellid] = nqp
        end
    end
    return _qvector_from_npoints(T, npoints)
end

"""
    setup_qvector(::Type{T}, operator) -> QVector{T}

Build a [`QVector`](@ref) laid out by the quadrature structure of `operator`,
the per-cell point count taken via `getnquadpoints` from the element caches in
its subdomain caches.
"""
function setup_qvector(::Type{T}, operator) where {T}
    npoints = zeros(Int, getncells(get_grid(operator.engine.dh)))
    for sc in operator.engine.subdomain_caches
        domain = sc.domain
        # Quadrature storage is per cell; an item family without cells (see
        # `algebraic_items`) contributes no points to the layout.
        domain isa AssemblyDomain || continue
        nqp = getnquadpoints(domain.element)
        for cellid in domain.sdh.cellset
            npoints[cellid] = nqp
        end
    end
    return _qvector_from_npoints(T, npoints)
end

# Shared layout builder: 1-based start offsets from the per-cell point counts,
# compressed to an arithmetic progression when the count is uniform. A zero
# count cannot be a range step, so those layouts stay uncompressed.
function _qvector_from_npoints(::Type{T}, npoints::Vector{Int}) where {T}
    offsets = similar(npoints)
    offset = 1
    for cellid in eachindex(npoints)
        offsets[cellid] = offset
        offset += npoints[cellid]
    end
    data = zeros(T, offset - 1)
    uniform = first(npoints) > 0 && all(==(first(npoints)), npoints)
    final_offsets = uniform ? (offsets[1]:npoints[1]:offsets[end]) : offsets
    final_npoints = uniform ? first(npoints) : npoints
    return QVector(data, final_offsets, final_npoints)
end
