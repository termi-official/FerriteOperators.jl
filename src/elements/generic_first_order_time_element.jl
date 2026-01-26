@concrete struct GenericFirstOrderTimeParameters
    p
    t
    Δt
    uprev
end

@concrete struct GenericFirstOrderTimeElementParameters
    pₑ
    t
    Δt
    uₑprev
end


abstract type AbstractGenericFirstOrderTimeVolumetricElementCache <: AbstractVolumetricElementCache end

function assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, element_cache::AbstractGenericFirstOrderTimeVolumetricElementCache, pfot::GenericFirstOrderTimeElementParameters)
    (; pₑ, t, uₑprev, Δt) = pfot
    assemble_element_gto1!(residualₑ, uₑ, uₑprev, cell, element_cache, pₑ, t, Δt)
end

function assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, element_cache::AbstractGenericFirstOrderTimeVolumetricElementCache, pfot::GenericFirstOrderTimeElementParameters)
    (; pₑ, t, uₑprev, Δt) = pfot
    assemble_element_gto1!(Kₑ, residualₑ, uₑ, uₑprev, cell, element_cache, pₑ, t, Δt)
end

function assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, element_cache::AbstractGenericFirstOrderTimeVolumetricElementCache, pfot::GenericFirstOrderTimeElementParameters)
    (; pₑ, t, uₑprev, Δt) = pfot
    assemble_element_gto1!(Kₑ, uₑ, uₑprev, cell, element_cache, pₑ, t, Δt)
end

function query_element_parameters(element::AbstractGenericFirstOrderTimeVolumetricElementCache, cell, p::GenericFirstOrderTimeParameters)
    (; cv) = element
    (; uprev, Δt, t) = p
    uₑprev = allocate_element_unknown_vector(element, cell)
    load_element_unknowns!(uₑprev, uprev, cell, element)
    return GenericFirstOrderTimeElementParameters(p.p, t, Δt, uₑprev)
end


abstract type AbstractGenericFirstOrderTimeSurfaceElementCache <: AbstractSurfaceElementCache end

function assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_facet_index::Int, surface_cache::AbstractGenericFirstOrderTimeSurfaceElementCache, pfot::GenericFirstOrderTimeElementParameters)
    (; pₑ, t, uₑprev, Δt) = pfot
    assemble_face_gto1!(residualₑ, uₑ, uₑprev, cell, local_facet_index, surface_cache, pₑ, t, Δt)
end

function assemble_face!!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_facet_index::Int, surface_cache::AbstractGenericFirstOrderTimeSurfaceElementCache, pfot::GenericFirstOrderTimeElementParameters)
    (; pₑ, t, uₑprev, Δt) = pfot
    assemble_face_gto1!(Kₑ, residualₑ, uₑ, uₑprev, cell, local_facet_index, surface_cache, pₑ, t, Δt)
end

function assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, local_facet_index::Int, surface_cache::AbstractGenericFirstOrderTimeSurfaceElementCache, pfot::GenericFirstOrderTimeElementParameters)
    (; pₑ, t, uₑprev, Δt) = pfot
    assemble_face_gto1!(Kₑ, uₑ, uₑprev, cell, local_facet_index, surface_cache, pₑ, t, Δt)
end


function query_element_parameters(element::AbstractGenericFirstOrderTimeSurfaceElementCache, cell, p::GenericFirstOrderTimeParameters)
    (; cv) = element
    (; uprev, Δt, t) = p
    uₑprev = allocate_element_unknown_vector(element, cell)
    load_element_unknowns!(uₑprev, uprev, cell, element)
    return GenericFirstOrderTimeElementParameters(p.p, t, Δt, uₑprev)
end
