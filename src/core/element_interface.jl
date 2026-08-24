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
`get_number_of_internal_dofs_per_element(model, cache, sdh)` and
[`condense_cell!`](@ref); `q`, their condensed internal state, is an ordinary
slot sourced by [`InternalSource`](@ref) (see [`condense_internal!`](@ref)).
"""
abstract type AbstractVolumetricElementCache end

"""
    allocate_element_matrix(element_cache, sdh)
    allocate_element_unknown_vector(element_cache, sdh)
    allocate_element_residual_vector(element_cache, sdh)

The element-local buffers of one item, sized in the FIELD SPACE — the
`ndofs_per_cell(sdh)` dofs `celldofs` carries. Where the integrator declares
[`global_dofs`](@ref) the engine PADS what these return by the declared count,
so an element that overrides them states the field-space size and never the
augmented one.
"""
allocate_element_matrix(element_cache, sdh)          = zeros(ndofs_per_cell(sdh), ndofs_per_cell(sdh))
@doc (@doc allocate_element_matrix) allocate_element_unknown_vector(element_cache, sdh)  = zeros(ndofs_per_cell(sdh))
@doc (@doc allocate_element_matrix) allocate_element_residual_vector(element_cache, sdh) = zeros(ndofs_per_cell(sdh))

# The padding itself: `similar` keeps whatever array type the element chose.
function pad_element_matrix(Ke, n::Int)
    n == 0 && return Ke
    m = size(Ke, 1) + n
    return fill!(similar(Ke, m, m), zero(eltype(Ke)))
end
function pad_element_vector(v, n::Int)
    n == 0 && return v
    return fill!(similar(v, length(v) + n), zero(eltype(v)))
end

"""
    global_dofs(integrator, sdh) -> AbstractVector{Int}

The GLOBAL dofs an element of `sdh` carries in its local system beyond
`celldofs` — dofs of `sdh.dh` that belong to no cell (Ferrite's
`algebraic_dofs(sdh.dh, :name)`) or that are shared by every item of the
subdomain (a lumped chamber pressure). Ordered; resolved once at setup.
Defaults to `()`, "no global dofs".

The declaration lives on the INTEGRATOR, one per subdomain, and is shared by
the volumetric AND the boundary kernel of that subdomain. The local system is
then, by contract,

    [ celldofs(cell) ; the declared global dofs, in declaration order ]

so the tail occupies [`global_dof_range`](@ref) — an element cache resolves
that range once in `setup_element_cache` and stores it, since the framework
passes no extra channel and `CellArgs`/`FacetArgs` stay at their four fields.
The engine sizes `Ke`/`re`/the slot buffers to the augmented length and
scatters through the augmented dof vector; the AD fallback differentiates the
full augmented system.

Two restrictions, both raised at setup: a declaration excludes
[`ColoredScheduling`](@ref) (a dof shared by every item cannot be isolated by
coloring — the parallel route is atomic scatter under
[`SequentialScheduling`](@ref)) and the [`ElementAssembly`](@ref) form (its dof
maps are celldofs-based).

The sparsity entries for the coupling this creates are NOT inferred from the
declaration: which items couple to the dofs is the user's Ferrite coupling
descriptor, passed through
[`StandardOperatorSpecification`](@ref)/[`BlockedOperatorSpecification`](@ref).
A missing descriptor surfaces as Ferrite's missing-sparsity-entry error on the
first assembly.
"""
global_dofs(integrator, sdh) = ()

"""
    global_dof_range(integrator, sdh) -> UnitRange{Int}

Where the dofs [`global_dofs`](@ref) declares sit in the element-local system:
`ndofs_per_cell(sdh) .+ (1:length(global_dofs(integrator, sdh)))`. An element
cache resolves this at setup and stores it — the local layout is a contract, so
this is the one place that spells it.
"""
global_dof_range(integrator, sdh) = ndofs_per_cell(sdh) .+ (1:length(global_dofs(integrator, sdh)))

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
contribution to `req`'s buffers. `args` is a [`FacetArgs`](@ref); annotating
the parameter is permitted. Facet kernels reinitialize their own `FacetValues`
for `local_facet_index`, and have no automatic-differentiation fallback — a
surface cache serves the sweep's request analytically or not at all.
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
    facet_items(integrator, sdh) -> iterable of FacetIndex

The facets of `sdh` that assemble as their own work items instead of riding
the cell sweep. Defaults to `()`, "no facet items"; a non-empty declaration
additionally needs [`setup_facet_item_cache`](@ref).

Every declared facet's cell must lie in `sdh.cellset` — a facet item's local
system is its owning cell's — and no facet may be declared twice; both are
setup errors. Facets of ONE cell always form one item together, so a cell's
declared facets are assembled and scattered as a single local system.

The DECLARED SET IS THE GATE on this route: [`is_facet_in_cache`](@ref) is not
consulted. That gate exists so the fused route can rediscover membership while
walking every facet of every cell, which is exactly the cost this family
avoids for a term supported on a small fraction of the boundary.

Two routes, one kernel set: the facets declared here are served by the same
`assemble_facet!(req, cache, args::FacetArgs, lfi)` methods the fused route
calls, over the same [`FacetArgs`](@ref). Moving a term between routes is a
change of declaration with no element edits.
"""
facet_items(integrator, sdh) = ()

"""
    setup_facet_item_cache(integrator, sdh) -> AbstractSurfaceElementCache

The surface cache serving the facets [`facet_items`](@ref) declares for `sdh`
— one cache per subdomain, exactly like [`setup_boundary_cache`](@ref), and
returning the same kind of object. A cache built here may equally be handed to
the fused route and vice versa.

There is deliberately no silent fallback: an integrator declaring facet items
without this method is a loud setup error, not an operator whose boundary term
quietly assembles nothing.
"""
function setup_facet_item_cache(integrator, sdh)
    throw(ArgumentError(
        "$(typeof(integrator)) declares `facet_items` but implements no " *
        "`setup_facet_item_cache(integrator, sdh)` method. One surface cache serves every " *
        "declared facet of the subdomain, and it is the same kind of cache " *
        "`setup_boundary_cache` returns."))
end

"""
Supertype for all caches to integrate over interfaces (facet pairs). It carries
no subtypes and no setup hook: nothing in this package traverses interfaces
yet, so the type names the seam and nothing more.
"""
abstract type AbstractInterfaceElementCache end

####################################
## Algebraic items — terms with no mesh support
####################################

"""
    algebraic_items(integrator, dh) -> collection of dof vectors

The items of the algebraic family: one vector of GLOBAL dof indices of `dh` per
term that lives on no cell — a 0D model's own rows, an `AlgebraicCoupling`-only
block, a lumped balance equation. An item's local system is exactly those dofs
in declaration order, so a circulation model contributing one row per chamber
declares one single-dof item per chamber. Defaults to `()`, "no algebraic
terms".

All items of one declaration carry the SAME number of dofs — that is what keeps
a worker's local buffers fixed-size — validated at setup, as are the dofs
themselves (in bounds, and unique within an item). Dofs shared BETWEEN items
are the normal case; [`AlgebraicItems`](@ref) derives the partition from that.

The sparsity entries the items need are not inferred from this declaration:
which dofs couple is the caller's Ferrite coupling descriptor
(`AlgebraicCoupling`), passed through
[`StandardOperatorSpecification`](@ref)/[`BlockedOperatorSpecification`](@ref),
exactly as for [`global_dofs`](@ref).
"""
algebraic_items(integrator, dh) = ()

"""
    setup_algebraic_cache(integrator, dh) -> cache

The cache serving every item [`algebraic_items`](@ref) declares. ONE cache
serves the whole declaration — the analogue of one element cache per
`SubDofHandler` serving all of its cells: items are positioned per sweep, and
the kernel reads which item it stands on from `args.item`
([`AlgebraicItem`](@ref)).

There is deliberately no silent fallback: an integrator declaring items without
this method is a loud setup error, not an operator whose algebraic rows quietly
assemble nothing.
"""
function setup_algebraic_cache(integrator, dh)
    throw(ArgumentError(
        "$(typeof(integrator)) declares `algebraic_items` but implements no " *
        "`setup_algebraic_cache(integrator, dh)` method. One cache serves every declared item; " *
        "the item a kernel stands on arrives as `args.item`."))
end

"""
    get_number_of_internal_dofs_per_algebraic_item(integrator, cache, items) -> AbstractVector{Int}

Number of condensed internal dofs each item of `items` owns, in declaration
order — the [`get_number_of_internal_dofs_per_element`](@ref) counterpart for
the algebraic-item family. Queried once at setup, only when
[`has_internal_state`](@ref) holds for `cache`, to build the item block of the
[`InternalVariableHandler`](@ref) (`[ū | q_cells | q_items]`). Every entry
must be equal: the fixed-size local buffers the uniform-item-size rule keeps
([`resolve_algebraic_items`](@ref)) extends to the internal-dof count too,
validated loudly at setup. There is no fallback, so only a condensed algebraic
cache implements it.
"""
function get_number_of_internal_dofs_per_algebraic_item end

"""
    assemble_algebraic!(req::AbstractAssemblyRequest, cache, args)

The kernel entry point of the algebraic item family — one entry point per
integration domain kind, alongside [`assemble_cell!`](@ref) and
[`assemble_facet!`](@ref). Accumulate the current item's contribution into
`req`'s buffers, which are sized by the item's dof count.

The [`ResidualRequest`](@ref) method is mandatory and validated at setup: it is
the basis of the AD-derived Jacobians and sensitivities, exactly as for a
volumetric cache. Every other request is served by the cache's own kernel where
[`provides_analytic`](@ref) declares one, and by ForwardDiff over the residual
kernel otherwise.

`args` is an [`AlgebraicArgs`](@ref); annotating the parameter is permitted.
"""
function assemble_algebraic! end

"""
    evaluate_algebraic_functional(kind::FunctionalKind, cache, args) -> value

The [`evaluate_cell_functional`](@ref) counterpart of the algebraic family:
this item's contribution to the functional named by `kind`, or `nothing` for no
contribution. The default IS `nothing` — a term with no mesh support carries no
volume, so it enters a domain integral only where its author says otherwise —
which is what keeps [`evaluate_functional`](@ref) working on an operator that
has algebraic items.
"""
evaluate_algebraic_functional(kind, cache, args) = nothing
