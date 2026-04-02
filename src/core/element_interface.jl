"""
Supertype for all caches to integrate over volumes.

General Interface:

    setup_element_cache(model, qr, sdh)

Specialized Interface for Condensed Problems:

    get_number_of_internal_dofs_per_element(model, element_cache, sdh)

"""
abstract type AbstractVolumetricElementCache end
#FIXME: still some element uses old api.
allocate_element_matrix(::AbstractCPUDevice, element_cache::AbstractVolumetricElementCache, sdh)          = zeros(ndofs_per_cell(sdh), ndofs_per_cell(sdh))
allocate_element_unknown_vector(::AbstractCPUDevice, element_cache::AbstractVolumetricElementCache, sdh)  = zeros(ndofs_per_cell(sdh))
allocate_element_residual_vector(::AbstractCPUDevice, element_cache::AbstractVolumetricElementCache, sdh) = zeros(ndofs_per_cell(sdh))

function allocate_element_matrix(device::AbstractGPUDevice, element_cache, sdh)
    N = sdh.ndofs_per_cell
    return KA.zeros(KA.backend(device), value_type(device), N, N, total_nthreads(device))
end
function allocate_element_unknown_vector(device::AbstractGPUDevice, element_cache, sdh)
    N = sdh.ndofs_per_cell
    return KA.zeros(KA.backend(device), value_type(device), N, total_nthreads(device))
end
function allocate_element_residual_vector(device::AbstractGPUDevice, element_cache, sdh)
    N = sdh.ndofs_per_cell
    return KA.zeros(KA.backend(device), value_type(device), N, total_nthreads(device))
end
load_element_unknowns!(uₑ, u, cell, ivh, element_cache::AbstractVolumetricElementCache)   = uₑ .= @view u[celldofs(cell)]
store_condensed_element_unknowns!(uₑ, u, cell, ivh, element_cache::AbstractVolumetricElementCache) = nothing

@doc raw"""
    assemble_element!(Kₑ::AbstractMatrix, cell::CellCache, element_cache::AbstractVolumetricElementCache, time)
Main entry point for bilinear operators

    assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, element_cache::AbstractVolumetricElementCache, time)
Update element matrix in nonlinear operators

    assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::AbstractVolumetricElementCache, time)
Update element matrix and residual in nonlinear operators

    assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::AbstractVolumetricElementCache, time)
Update residual in nonlinear operators

The notation is as follows.
* $K_e$ the element stiffness matrix
* $u_e$ the element unknowns
* $residual_e$ the element residual
"""
assemble_element!


"""
    Utility to execute noop assembly.
"""
struct EmptyVolumetricElementCache <: AbstractVolumetricElementCache end
# Main entry point for bilinear operators
assemble_element!(Kₑ::AbstractMatrix, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update element matrix in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update element matrix and residual in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update residual in nonlinear operators
assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing

"""
Supertype for all caches to integrate over surfaces.

Interface:

    setup_boundary_cache(model, qr, sdh)

"""
abstract type AbstractSurfaceElementCache end

@doc raw"""
    assemble_face!(Kₑ::AbstractMatrix, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Main entry point for bilinear operators

    assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix in nonlinear operators

    assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix and residual in nonlinear operators

    assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update residual in nonlinear operators

The notation is as follows.
* $K_e$ the element stiffness matrix
* $u_e$ the element unknowns
* $residual_e$ the element residual
"""
assemble_face!


"""
    Utility to execute noop assembly.
"""
struct EmptySurfaceElementCache <: AbstractSurfaceElementCache end
# Update element matrix in nonlinear operators
assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
# Update element matrix and residual in nonlinear operators
assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
# Update residual in nonlinear operators
assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
@inline is_facet_in_cache(::FacetIndex, cell, ::EmptySurfaceElementCache) = false

"""
Supertype for all caches to integrate over interfaces.

Interface:

    setup_interface_cache(model, qr, ip, sdh)

"""
abstract type AbstractInterfaceElementCache end

@doc raw"""
    assemble_interface!(Kₑ::AbstractMatrix, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Main entry point for bilinear operators

    assemble_interface!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix in nonlinear operators

    assemble_interface!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix and residual in nonlinear operators

    assemble_interface!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update residual in nonlinear operators

The notation is as follows.
* $K_e$ the element pair stiffness matrix
* $u_e$ the element pair unknowns
* $residual_e$ the element pair residual
"""
assemble_interface!


"""
Utility to execute noop assembly.
"""
struct EmptyInterfaceCache <: AbstractInterfaceElementCache end
# Update element matrix in nonlinear operators
assemble_interface!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptyInterfaceCache, time)            = nothing
# Update element matrix and residual in nonlinear operators
assemble_interface!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptyInterfaceCache, time) = nothing
# Update residual in nonlinear operators
assemble_interface!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptyInterfaceCache, time) = nothing

## GPU element infrastructure ##

# per_thread: extract per-thread instance from a container struct.
# Default: identity (scalars, isbits, non-container fields pass through).
per_thread(x, tid) = x
per_thread(x::Ferrite.CellValuesContainer, tid) = x[tid]

macro per_thread_structure(T)
    quote
        @inline function Base.getindex(x::$(esc(T)), tid)
            fields = ntuple(i -> per_thread(getfield(x, i), tid), fieldcount(typeof(x)))
            typeof(x)(fields...)
        end
    end
end

"""
    @device_element struct MyElementCache <: AbstractVolumetricElementCache
        D::Float64
        cellvalues::CellValues
    end

    # Generates:
    # struct MyElementCache{_T1, _T2} <: AbstractVolumetricElementCache
    #     D::_T1
    #     cellvalues::_T2
    # end
    # Adapt.@adapt_structure MyElementCache
    # @per_thread_structure MyElementCache
"""
macro device_element(expr)
    expr.head == :struct || error("@device_element expects a struct definition")
    expr.args[1] && error("@device_element does not support mutable structs")

    struct_header = expr.args[2]
    struct_body = expr.args[3]

    if struct_header isa Expr && struct_header.head == :(<:)
        struct_name = struct_header.args[1]
        supertype = struct_header.args[2]
    else
        struct_name = struct_header
        supertype = nothing
    end

    # Parametrize all typed fields
    type_params = Symbol[]
    new_fields = Expr[]
    param_counter = 0

    for line in struct_body.args
        if line isa Expr && line.head == :(::)
            field_name = line.args[1]
            param_counter += 1
            param = Symbol("_T", param_counter)
            push!(type_params, param)
            push!(new_fields, :($field_name::$param))
        elseif line isa LineNumberNode
            continue
        else
            push!(new_fields, line)
        end
    end

    if isempty(type_params)
        param_name = struct_name
    else
        param_name = :($struct_name{$(type_params...)})
    end

    full_header = supertype !== nothing ? :($param_name <: $supertype) : param_name

    esc(quote
        Base.@__doc__ struct $full_header
            $(new_fields...)
        end
        @inline function Adapt.adapt_structure(to, x::$struct_name)
            fields = ntuple(i -> Adapt.adapt(to, getfield(x, i)), fieldcount(typeof(x)))
            typeof(x).name.wrapper(fields...)
        end
        @inline function Base.getindex(x::$struct_name, tid)
            fields = ntuple(i -> per_thread(getfield(x, i), tid), fieldcount(typeof(x)))
            typeof(x).name.wrapper(fields...)
        end
    end)
end
