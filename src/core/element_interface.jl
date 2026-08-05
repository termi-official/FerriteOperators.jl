"""
Supertype for all caches to integrate over volumes.

General Interface:

    setup_element_cache(integrator, sdh)

Specialized Interface for Condensed Problems:

    get_number_of_internal_dofs_per_element(model, element_cache, sdh)

"""
abstract type AbstractVolumetricElementCache end

allocate_element_matrix(element_cache, sdh)          = zeros(ndofs_per_cell(sdh), ndofs_per_cell(sdh))
allocate_element_unknown_vector(element_cache, sdh)  = zeros(ndofs_per_cell(sdh))
allocate_element_residual_vector(element_cache, sdh) = zeros(ndofs_per_cell(sdh))
load_element_unknowns!(uₑ, u, cell, ivh, element_cache)   = uₑ .= @view u[celldofs(cell)]
store_condensed_element_unknowns!(uₑ, u, cell, ivh, element_cache) = nothing

Ferrite.getnquadpoints(element_cache::AbstractVolumetricElementCache) = getnquadpoints(element_cache.cv)
Ferrite.reinit!(element_cache::AbstractVolumetricElementCache, cell) = Ferrite.reinit!(element_cache.cv, cell)

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
# Main entry point for linear operators
assemble_element!(rₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update element matrix in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update element matrix and residual in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update residual in nonlinear operators
assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Utils
Ferrite.getnquadpoints(element_cache::EmptyVolumetricElementCache) = 0
Ferrite.reinit!(element_cache::EmptyVolumetricElementCache, cell) = nothing

"""
    setup_element_cache(integrator, sdh)

Setup the element on a given subdofhandler.
"""
setup_element_cache(integrator, sdh) = EmptyVolumetricElementCache()

"""
Supertype for all caches to integrate over surfaces.

Interface:

    setup_boundary_cache(integrator, sdh)

"""
abstract type AbstractSurfaceElementCache end

@doc raw"""
    assemble_facet!(Kₑ::AbstractMatrix, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Main entry point for bilinear operators

    assemble_facet!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix in nonlinear operators

    assemble_facet!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix and residual in nonlinear operators

    assemble_facet!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update residual in nonlinear operators

The notation is as follows.
* $K_e$ the element stiffness matrix
* $u_e$ the element unknowns
* $residual_e$ the element residual
"""
assemble_facet!

# If we compose a face cache into an element cache, then we loop over the faces of the elements and try to assemble
function assemble_element!(Kₑ::AbstractMatrix, cell::CellCache, facet_cache::AbstractSurfaceElementCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        if is_facet_in_cache(FacetIndex(cellid(cell), local_facet_index), cell, facet_cache)
            assemble_facet!(Kₑ, cell, local_facet_index, facet_cache, time)
        end
    end
end
function assemble_element!(rₑ::AbstractVector, cell::CellCache, facet_cache::AbstractSurfaceElementCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        if is_facet_in_cache(FacetIndex(cellid(cell), local_facet_index), cell, facet_cache)
            assemble_facet!(rₑ, cell, local_facet_index, facet_cache, time)
        end
    end
end
function assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, facet_cache::AbstractSurfaceElementCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        if is_facet_in_cache(FacetIndex(cellid(cell), local_facet_index), cell, facet_cache)
            assemble_facet!(Kₑ, uₑ, cell, local_facet_index, facet_cache, time)
        end
    end
end
function assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, facet_cache::AbstractSurfaceElementCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        if is_facet_in_cache(FacetIndex(cellid(cell), local_facet_index), cell, facet_cache)
            assemble_facet!(Kₑ, residualₑ, uₑ, cell, local_facet_index, facet_cache, time)
        end
    end
end
function assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, facet_cache::AbstractSurfaceElementCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        if is_facet_in_cache(FacetIndex(cellid(cell), local_facet_index), cell, facet_cache)
            assemble_facet!(residualₑ, uₑ, cell, local_facet_index, facet_cache, time)
        end
    end
end

Ferrite.getnquadpoints(element_cache::AbstractSurfaceElementCache) = Ferrite.getnquadpoints(element_cache.fv)
Ferrite.reinit!(element_cache::AbstractSurfaceElementCache, cell) = Ferrite.reinit!(element_cache.fv, cell)

"""
    Utility to execute noop assembly.
"""
struct EmptySurfaceElementCache <: AbstractSurfaceElementCache end
# Linear
assemble_facet!(rₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(rₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
# Bilinear
assemble_facet!(Kₑ::AbstractMatrix, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(Kₑ::AbstractMatrix, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
# Update element matrix in nonlinear operators
assemble_facet!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
# Update element matrix and residual in nonlinear operators
assemble_facet!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
# Update residual in nonlinear operators
assemble_facet!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
@inline is_facet_in_cache(::FacetIndex, cell, ::EmptySurfaceElementCache) = false

"""
    setup_boundary_cache(integrator, sdh)

Setup the boundary element on a given subdofhandler.
"""
setup_boundary_cache(integrator, sdh) = EmptySurfaceElementCache()

Ferrite.getnquadpoints(element_cache::EmptySurfaceElementCache) = 0
Ferrite.reinit!(element_cache::EmptySurfaceElementCache, cell) = nothing

"""
Supertype for all caches to integrate over interfaces.

Interface:

    setup_interface_cache(integrator, sdh)

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

"""
    setup_interface_cache(integrator, sdh)

Setup the interface element on a given subdofhandler.
"""
setup_interface_cache(integrator, sdh)

Ferrite.getnquadpoints(element_cache::EmptyInterfaceCache) = 0
Ferrite.reinit!(element_cache::EmptyInterfaceCache, cell) = nothing
