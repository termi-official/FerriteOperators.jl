"""
Supertype for all caches to integrate over volumes.

Elements implement request-typed kernels

    assemble_cell!(req, cache, args)

with a mandatory [`ResidualRequest`](@ref) method (the AD basis) and optional
analytic methods declared via [`provides_analytic`](@ref). The cache owns
`reinit!` of its values objects, selecting per request kind; the loop owns
only the geometry cache. Setup happens through
`setup_element_cache(integrator, sdh)`.

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
[`global_dofs`](@ref) the engine PADS what these return, so an override states
the field-space size and never the augmented one.
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

The GLOBAL dofs a CELL of `sdh` carries in its local system beyond `celldofs` —
dofs of `sdh.dh` that belong to no cell (Ferrite's
`algebraic_dofs(sdh.dh, :name)`) and that every cell of the subdomain couples
to (a stress-driven RVE's macroscopic strain). Ordered; resolved once at setup.
Defaults to `()`.

The declaration lives on the INTEGRATOR, one per subdomain, and augments the
CELL family alone: the subdomain's volumetric kernel and the boundary kernel
riding its sweep. A term supported on a facet SET declares its own tail through
[`facet_item_global_dofs`](@ref), which leaves the cell sweep in the pure field
space. The local system is then, by contract,

    [ celldofs(cell) ; the declared global dofs, in declaration order ]

so the tail occupies [`global_dof_range`](@ref), which an element cache
resolves once in `setup_element_cache` and stores — the framework passes no
extra channel, and `CellArgs`/`FacetArgs` stay at their four fields. The
engine sizes `Ke`/`re`/the slot buffers to the augmented length and scatters
through the augmented dof vector; the AD fallback differentiates the full
augmented system.

One restriction, raised at setup and shared with
[`facet_item_global_dofs`](@ref): a declaration excludes
[`ColoredScheduling`](@ref) — a dof shared by every item cannot be isolated by
coloring, so the parallel route is atomic scatter under
[`SequentialScheduling`](@ref).

The sparsity entries for the resulting coupling are NOT inferred: which items
couple to the dofs is the user's Ferrite coupling descriptor, passed through
[`StandardOperatorSpecification`](@ref)/[`BlockedOperatorSpecification`](@ref).
A missing descriptor surfaces as Ferrite's missing-sparsity-entry error on the
first assembly.
"""
global_dofs(integrator, sdh) = ()

"""
    global_dof_range(integrator, sdh) -> UnitRange{Int}

Where the dofs [`global_dofs`](@ref) declares sit in the element-local system:
`ndofs_per_cell(sdh) .+ (1:length(global_dofs(integrator, sdh)))`. The local
layout is a contract, and this is the one place that spells it.
"""
global_dof_range(integrator, sdh) = ndofs_per_cell(sdh) .+ (1:length(global_dofs(integrator, sdh)))

"""
    reinit_values!(cache, cell)
    reinit_values!(cache, cell, kind)

Reinitialize the values objects an element cache carries for the given cell.
The two-arg method is mandatory (validated at setup) and reinitializes all of
them. The engine calls the three-arg form once per cell and sweep, before any
kernel of that sweep; specialize it on the request `kind` to reinitialize only
what that kind needs. Kernels are pure evaluation — repeated invocations
within one sweep (AD chunk passes, split Jacobian-then-residual fallbacks) do
not reinitialize again.
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

"Utility to execute noop assembly."
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

Facet kernels are request-typed,

    assemble_facet!(req, cache, args, local_facet_index::Int)

and run over the facets [`facet_items`](@ref) declares; facet parameters are
queried separately per facet via [`query_facet_parameters`](@ref). Setup happens
through [`setup_facet_item_cache`](@ref).
"""
abstract type AbstractSurfaceElementCache end

"""
    assemble_facet!(req, cache, args, local_facet_index::Int)

The volumetric-kernel analogue for facets: accumulate this facet's
contribution to `req`'s buffers. `args` is a [`FacetArgs`](@ref). Facet
kernels reinitialize their own `FacetValues` for `local_facet_index` and have
no automatic-differentiation fallback — a surface cache serves the sweep's
request analytically or not at all.
"""
function assemble_facet! end

"""
    evaluate_facet_functional(kind::FunctionalKind, cache, args, local_facet_index) -> value

The [`evaluate_cell_functional`](@ref) counterpart of the facet-item family:
returns this facet's contribution to the functional named by `kind` — a
`Number` or a Tensors tensor, summed across facets — or `nothing` for no
contribution. `args` is a [`FacetArgs`](@ref), and the kernel reinitializes its
own `FacetValues` for `local_facet_index`, exactly as [`assemble_facet!`](@ref)
does.
"""
function evaluate_facet_functional end

"Utility to execute noop assembly."
struct EmptySurfaceElementCache <: AbstractSurfaceElementCache end
assemble_facet!(req::AbstractAssemblyRequest, ::EmptySurfaceElementCache, args, local_facet_index::Int) = nothing
evaluate_facet_functional(kind, ::EmptySurfaceElementCache, args, local_facet_index::Int) = nothing

"""
    facet_items(integrator, sdh) -> iterable of FacetIndex

The facets of `sdh` a boundary term is supported on. They assemble as their own
work items — one item per owning cell — which is the ONE route a surface term
takes. Defaults to `()`; a non-empty declaration additionally needs
[`setup_facet_item_cache`](@ref).

Every declared facet's cell must lie in `sdh.cellset` — a facet item's local
system is its owning cell's — and no facet may be declared twice; both are
setup errors. Facets of ONE cell form a single item, assembled and scattered
as one local system.

THE DECLARED SET IS THE TRAVERSAL: a cell whose facets are undeclared is never
visited by this family, which is what makes a term supported on a small
fraction of the boundary cost a small fraction of the boundary.
"""
facet_items(integrator, sdh) = ()

"""
    setup_facet_item_cache(integrator, sdh) -> AbstractSurfaceElementCache

The [`AbstractSurfaceElementCache`](@ref) serving the facets
[`facet_items`](@ref) declares for `sdh` — one per subdomain, the surface
counterpart of [`setup_element_cache`](@ref).

There is deliberately no silent fallback: declaring facet items without this
method is a loud setup error, not a boundary term that quietly assembles
nothing.
"""
function setup_facet_item_cache(integrator, sdh)
    throw(ArgumentError(
        "$(typeof(integrator)) declares `facet_items` but implements no " *
        "`setup_facet_item_cache(integrator, sdh)` method. One surface cache serves every " *
        "declared facet of the subdomain."))
end

"""
    facet_item_global_dofs(integrator, sdh) -> AbstractVector{Int}

The [`global_dofs`](@ref) counterpart of the facet item family: the GLOBAL dofs
a FACET ITEM of `sdh` carries beyond `celldofs` of its owning cell — the lumped
chamber pressure a tying surface couples to. Ordered; resolved once at setup;
defaults to `()`. The local system is
`[celldofs(cell); the declared dofs]`, whose tail is
[`facet_item_global_dof_range`](@ref).

Each family sizes its own local system from its OWN declaration, which is what
this hook buys over declaring [`global_dofs`](@ref): the facet items'
`Ke`/`re`/slot buffers and their scatter are augmented, the subdomain's CELL
sweep stays in the pure field space. The coupling sparsity can then be declared
over the tying facets alone (Ferrite's `FacetCoupling`) instead of every cell of
the subdomain, and a condensed volumetric element may sit beside the tying term
— the rejection in [`ADElementCache`](@ref) applies to the cell declaration.

The restrictions of [`global_dofs`](@ref) hold here too: the declaration
excludes [`ColoredScheduling`](@ref), and the sparsity entries are the caller's
coupling descriptor, never inferred.
"""
facet_item_global_dofs(integrator, sdh) = ()

"""
    facet_item_global_dof_range(integrator, sdh) -> UnitRange{Int}

Where [`facet_item_global_dofs`](@ref) sits in a facet item's local system, the
[`global_dof_range`](@ref) counterpart: a facet item's local system is its
owning cell's, so the tail starts after `ndofs_per_cell(sdh)` here as well.
"""
facet_item_global_dof_range(integrator, sdh) =
    ndofs_per_cell(sdh) .+ (1:length(facet_item_global_dofs(integrator, sdh)))

####################################
## Algebraic items — terms with no mesh support
####################################

"""
    algebraic_items(integrator, dh) -> collection of dof vectors

The items of the algebraic family: one vector of GLOBAL dof indices of `dh`
per term that lives on no cell — a 0D model's own rows, an
`AlgebraicCoupling`-only block, a lumped balance equation. An item's local
system is exactly those dofs in declaration order, so a circulation model
contributing one row per chamber declares one single-dof item per chamber.
Defaults to `()`.

All items of one declaration carry the SAME number of dofs — that is what
keeps a worker's local buffers fixed-size — validated at setup, as are the
dofs themselves (in bounds, and unique within an item). Dofs shared BETWEEN
items are the normal case; [`AlgebraicItems`](@ref) derives the partition from
that.

The sparsity entries are not inferred from this declaration: which dofs couple
is the caller's Ferrite coupling descriptor (`AlgebraicCoupling`), passed
through
[`StandardOperatorSpecification`](@ref)/[`BlockedOperatorSpecification`](@ref),
exactly as for [`global_dofs`](@ref).
"""
algebraic_items(integrator, dh) = ()

"""
    setup_algebraic_cache(integrator, dh) -> cache

The cache serving every item [`algebraic_items`](@ref) declares — the analogue
of one element cache per `SubDofHandler` serving all of its cells. Items are
positioned per sweep, and the kernel reads which item it stands on from
`args.item` ([`AlgebraicItem`](@ref)).

There is deliberately no silent fallback: declaring items without this method
is a loud setup error, not algebraic rows that quietly assemble nothing.
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
must be equal: the uniform-item-size rule that keeps local buffers fixed-size
([`resolve_algebraic_items`](@ref)) extends to the internal-dof count,
validated loudly at setup. There is no fallback, so only a condensed algebraic
cache implements it.
"""
function get_number_of_internal_dofs_per_algebraic_item end

"""
    assemble_algebraic!(req::AbstractAssemblyRequest, cache, args)

The kernel entry point of the algebraic item family — one per integration
domain kind, alongside [`assemble_cell!`](@ref) and
[`assemble_facet!`](@ref). Accumulate the current item's contribution into
`req`'s buffers, which are sized by the item's dof count.

The [`ResidualRequest`](@ref) method is mandatory and validated at setup; every
other request is served analytically where [`provides_analytic`](@ref) declares
it and by ForwardDiff over the residual kernel otherwise, exactly as for a
volumetric cache. `args` is an [`AlgebraicArgs`](@ref).
"""
function assemble_algebraic! end

"""
    evaluate_algebraic_functional(kind::FunctionalKind, cache, args) -> value

The [`evaluate_cell_functional`](@ref) counterpart of the algebraic family:
this item's contribution to the functional named by `kind`, or `nothing` for no
contribution. The default IS `nothing` — a term with no mesh support carries no
volume, so it enters a domain integral only where its author says otherwise —
which keeps [`evaluate_functional`](@ref) working on an operator that has
algebraic items.
"""
evaluate_algebraic_functional(kind, cache, args) = nothing
