"""
Supertype for all caches to integrate over volumes.

The contract: elements implement request-typed kernels

    assemble_cell!(req, cache, args)

with a mandatory [`ResidualRequest`](@ref) method (the AD basis) and optional
analytic methods declared via [`provides_analytic`](@ref). Element caches own
`reinit!` of their values objects, selecting per request kind; the loop owns
only the geometry cache. Setup happens through

    setup_element_cache(integrator, sdh)

Condensed elements additionally implement
`get_number_of_internal_dofs_per_element(model, cache, sdh)` and the
`load_element_unknowns!`/`store_condensed_element_unknowns!` pair.
"""
abstract type AbstractVolumetricElementCache end

allocate_element_matrix(element_cache, sdh)          = zeros(ndofs_per_cell(sdh), ndofs_per_cell(sdh))
allocate_element_unknown_vector(element_cache, sdh)  = zeros(ndofs_per_cell(sdh))
allocate_element_residual_vector(element_cache, sdh) = zeros(ndofs_per_cell(sdh))

"""
    reinit_values!(cache, cell)
    reinit_values!(cache, cell, kind)

Reinitialize the values objects an element cache carries for the given cell.
The two-arg method is mandatory (validated at setup) and reinitializes all of
them. The three-arg form is called by the engine once per cell and sweep,
before any kernel of that sweep; specialize it on the request `kind` to
reinitialize only the values that kind needs. Kernels are pure evaluation —
repeated kernel invocations within one sweep (AD chunk passes, split
Jacobian-then-residual fallbacks) do not reinitialize again.
"""
function reinit_values! end
reinit_values!(cache, cell, kind) = reinit_values!(cache, cell)

"""
    evaluate_cell_functional(kind::FunctionalKind, cache, args) -> value

Element kernel for functional (reduction) queries: returns this cell's
contribution to the functional named by `kind` — a `Number` or a Tensors
tensor, summed across cells — or `nothing` for no contribution.
"""
function evaluate_cell_functional end

"""
    get_number_of_internal_dofs_per_element(integrator, cache, sdh) -> AbstractVector{Int}

Number of condensed internal dofs each cell of `sdh` owns, in `sdh.cellset`
order. Queried once at setup to build the [`InternalVariableHandler`](@ref);
there is no fallback, so only condensed elements implement it.
"""
function get_number_of_internal_dofs_per_element end

load_element_unknowns!(uₑ, u, cell, ivh, element_cache)   = uₑ .= @view u[celldofs(cell)]
store_condensed_element_unknowns!(uₑ, u, cell, ivh, element_cache) = nothing

"""
    Utility to execute noop assembly.
"""
struct EmptyVolumetricElementCache <: AbstractVolumetricElementCache end
assemble_cell!(req::AbstractAssemblyRequest, ::EmptyVolumetricElementCache, args) = nothing
provides_analytic(::Type{EmptyVolumetricElementCache}, kind) = true
Ferrite.getnquadpoints(::EmptyVolumetricElementCache) = 0
reinit_values!(::EmptyVolumetricElementCache, cell) = nothing
evaluate_cell_functional(kind, ::EmptyVolumetricElementCache, args) = nothing

"""
    setup_element_cache(integrator, sdh)

Setup the element cache on a given subdofhandler. There is deliberately no
silent no-op fallback: a missing method is a loud setup error, not an
operator that assembles nothing.
"""
function setup_element_cache(integrator, sdh)
    throw(ArgumentError(
        "No `setup_element_cache` method for integrator type $(typeof(integrator)). " *
        "Implement `setup_element_cache(integrator, sdh)` (return `EmptyVolumetricElementCache()` " *
        "explicitly if the integrator has no volumetric term)."))
end

"""
Supertype for all caches to integrate over surfaces.

The contract: facet kernels are request-typed,

    assemble_facet!(req, cache, args, local_facet_index::Int)

gated by `is_facet_in_cache(::FacetIndex, cell, cache)`. The framework's
boundary driver walks the facets of each cell; facet parameters are queried
separately per facet via [`query_facet_parameters`](@ref). Setup happens
through

    setup_boundary_cache(integrator, sdh)
"""
abstract type AbstractSurfaceElementCache end

"""
    assemble_facet!(req, cache, args, local_facet_index::Int)

The volumetric-kernel analogue for facets: accumulate this facet's
contribution to `req`'s buffers. Facet kernels reinitialize their own
`FacetValues` for `local_facet_index`, and have no automatic-differentiation
fallback — a surface cache serves the sweep's request analytically or not at
all.
"""
function assemble_facet! end

"""
    is_facet_in_cache(facet::FacetIndex, cell, cache) -> Bool

Gate of the framework's facet driver: `true` iff `cache` contributes on
`facet`. The driver walks every facet of every cell and calls
[`assemble_facet!`](@ref) only where this returns `true`, so a surface cache
states its facet set here instead of re-deriving it per kernel call.
"""
function is_facet_in_cache end

"""
    Utility to execute noop assembly.
"""
struct EmptySurfaceElementCache <: AbstractSurfaceElementCache end
assemble_facet!(req::AbstractAssemblyRequest, ::EmptySurfaceElementCache, args, local_facet_index::Int) = nothing
@inline is_facet_in_cache(::FacetIndex, cell, ::EmptySurfaceElementCache) = false
Ferrite.getnquadpoints(::EmptySurfaceElementCache) = 0
Ferrite.reinit!(::EmptySurfaceElementCache, cell) = nothing

"""
    setup_boundary_cache(integrator, sdh)

Setup the boundary element cache on a given subdofhandler. Defaults to the
empty cache — "no boundary terms" is the legitimate common case.
"""
setup_boundary_cache(integrator, sdh) = EmptySurfaceElementCache()

"""
Supertype for all caches to integrate over interfaces (facet pairs).

Reserved for the DG work: interface kernels are request-typed over a
two-sided argument bundle (see [`InterfaceKernelArgs`](@ref)); the
one-to-many generalization for non-local coupling reserves its own shape.
Concrete interface caches and their setup hook land with that work.
"""
abstract type AbstractInterfaceElementCache end
