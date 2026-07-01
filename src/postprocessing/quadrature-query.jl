"""
    QuadratureDataQuery{T, QV <: QVector{T}}

A post-processing query buffer that holds quadrature-point results for all (or a
filtered subset of) cells.

Fields:
- `buffer` — [`QVector{T}`](@ref) storing the result at every quadrature point
- `set`    — optional set of cell IDs to evaluate; `nothing` means all cells

Build with [`prepare_quadrature_query`](@ref) and execute with [`process_query!`](@ref).
"""
struct QuadratureDataQuery{T, QV <: QVector{T}}
    buffer::QV
    set::Union{Nothing, AbstractSet{Int}}
end
QuadratureDataQuery(buffer::QVector) = QuadratureDataQuery(buffer, nothing)

"""
    QuadratureDataMultiQuery{Q <: QuadratureDataQuery}

A bundle of [`QuadratureDataQuery`](@ref) objects that are executed together
(one pass per query) via [`process_query!`](@ref).
"""
struct QuadratureDataMultiQuery{Q <: QuadratureDataQuery}
    queries::Vector{Q}
end

"""
    prepare_quadrature_query(::Type{T}, op::QuadratureFerriteOperator; set = nothing)
    prepare_quadrature_query(::Type{T}, prototype::QuadratureDataQuery)

Build a [`QuadratureDataQuery{T}`](@ref).

The first form allocates a fresh [`QVector{T}`](@ref) whose layout matches `op`.
The optional `set` keyword restricts evaluation to the given cell IDs.

The second form reuses the offset/npoints layout of an existing `prototype`
query, useful for building multiple queries over the same mesh without
recomputing the layout.
"""
function prepare_quadrature_query(::Type{T}, op::QuadratureFerriteOperator;
                                  set::Union{Nothing, AbstractSet{Int}} = nothing) where {T}
    buffer = setup_qvector(T, op)
    return QuadratureDataQuery(buffer, set)
end

function prepare_quadrature_query(::Type{T}, proto::QuadratureDataQuery) where {T}
    buffer = QVector(zeros(T, length(proto.buffer)),
                     proto.buffer.offsets,
                     proto.buffer.npoints)
    return QuadratureDataQuery(buffer, proto.set)
end

"""
    process_query!(query::QuadratureDataQuery, op::QuadratureFerriteOperator, u, p, f)
    process_query!(multi::QuadratureDataMultiQuery, op::QuadratureFerriteOperator, u, p, fs)

Evaluate `f(qe, ue, cell, element_cache, pe)` at every quadrature point and store
results in `query.buffer`.  If `query.set` is set, only cells whose ID is in that
set are evaluated; all other cells retain their current (typically zero) values.

The multi-query form calls `process_query!` once per `(query, f)` pair.
"""
function process_query!(query::QuadratureDataQuery, op::QuadratureFerriteOperator, u, p, f)
    evaluate_quadrature!(op, query.buffer, u, p, f, query.set)
    return query
end

function process_query!(multi::QuadratureDataMultiQuery, op::QuadratureFerriteOperator, u, p, fs)
    # TODO fuse fs into a single f, so we do not need to iterate multiple times
    for (query, f) in zip(multi.queries, fs)
        process_query!(query, op, u, p, f)
    end
    return multi
end

"""
    VTKQuadratureFile

A VTK file handler for quadrature-point data, analogous to `Ferrite.VTKGridFile`
but backed by a [`VTKQuadratureGrid`](@ref).

Use the do-block syntax (which calls `close` automatically):

```julia
qgrid = VTKQuadratureGrid(dh, qrc)
VTKQuadratureFile("output", qgrid) do vtk
    write_quadrature_data(vtk, σ, "stress")
end
```
"""
struct VTKQuadratureFile{VTK <: WriteVTK.DatasetFile}
    vtk::VTK
end

function VTKQuadratureFile(filename::String, qgrid::VTKQuadratureGrid; kwargs...)
    coords, cls = Ferrite.create_vtk_griddata(qgrid)
    vtk = WriteVTK.vtk_grid(filename, coords, cls; kwargs...)
    return VTKQuadratureFile(vtk)
end

function VTKQuadratureFile(f::Function, args...; kwargs...)
    vtk = VTKQuadratureFile(args...; kwargs...)
    try
        f(vtk)
    finally
        close(vtk)
    end
    return vtk
end

Base.close(vtk::VTKQuadratureFile) = (WriteVTK.vtk_save(vtk.vtk); vtk)

function Base.show(io::IO, ::MIME"text/plain", vtk::VTKQuadratureFile)
    open_str = isopen(vtk.vtk) ? "open" : "closed"
    print(io, "VTKQuadratureFile for the $(open_str) file \"$(vtk.vtk.path)\".")
end

function WriteVTK.collection_add_timestep(pvd::WriteVTK.CollectionFile, datfile::VTKQuadratureFile, time::Real)
    return WriteVTK.collection_add_timestep(pvd, datfile.vtk, time)
end
Base.setindex!(pvd::WriteVTK.CollectionFile, datfile::VTKQuadratureFile, time::Real) =
    WriteVTK.collection_add_timestep(pvd, datfile, time)

"""
    write_quadrature_data(vtk::VTKQuadratureFile, q::QVector, name)

Write quadrature-point data from `q` to the VTK point-data field `name`.
Supports both scalar (`QVector{<:Real}`) and vector (`QVector{Vec{dim,T}}`) data.
"""
write_quadrature_data(vtk::VTKQuadratureFile, q::QVector{<:Real}, name) =
    (vtk.vtk[name, VTKBase.VTKPointData()] = q.data; vtk)

function write_quadrature_data(vtk::VTKQuadratureFile, q::QVector{Vec{dim, T}}, name) where {dim, T}
    flat = reshape(reinterpret(T, q.data), (dim, length(q)))
    vtk.vtk[name, VTKBase.VTKPointData()] = flat
    return vtk
end

"""
    write_quadrature_data(vtk::VTKQuadratureFile, q::QuadratureDataQuery, name)

Write the buffer inside `q` to the VTK point-data field `name`.
"""
write_quadrature_data(vtk::VTKQuadratureFile, q::QuadratureDataQuery, name) =
    write_quadrature_data(vtk, q.buffer, name)
