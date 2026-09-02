"""
Per-inner parameter views produced by a composite cache's
[`query_cell_parameters`](@ref) / [`query_facet_parameters`](@ref). The fan-out
re-seeds `args.p` per inner from this bundle; any other `p` reaches every inner
unchanged, so a hand-built [`CellArgs`](@ref)/[`FacetArgs`](@ref) still works.
"""
struct CompositeParameters{Views <: Tuple}
    views::Views
end

"""
Combines multiple element caches over the same volume.

Scope bound: the inners share ONE domain, ONE evaluation context and ONE
scatter target. Terms evaluated at different contexts (generalized-α's two
times) or scattered into different targets (term-split operators, IMEX row
blocks) are separate sweeps over separate integrators, not inners of one
composite.

No values objects are shared by construction — each inner sets up and
reinitializes its own; deliberate sharing stays the sharing element's
responsibility. Each inner is handed its OWN parameter view, produced by that
inner's [`query_cell_parameters`](@ref).

The composite is analytic for a kind iff every inner is; otherwise the whole
composite falls back to AD over the fused residual (one Dual sweep, one
geometry pass). That routing is all-or-nothing.

Inners carrying condensed internal state are rejected at setup — see
[`validate_element_cache`](@ref).
"""
struct CompositeVolumetricElementCache{CacheTupleType <: Tuple} <: AbstractVolumetricElementCache
    inner_caches::CacheTupleType
end

query_cell_parameters(composite::CompositeVolumetricElementCache, cell, p) =
    CompositeParameters(map(inner -> query_cell_parameters(inner, cell, p), composite.inner_caches))

assemble_cell!(req::AbstractAssemblyRequest, composite::CompositeVolumetricElementCache, args) =
    _composite_assemble_cell!(req, composite.inner_caches, args, args.p)

_composite_assemble_cell!(req, inner_caches, args, p::CompositeParameters) =
    _fan_out_cell!(req, inner_caches, args, p.views)
@unroll function _composite_assemble_cell!(req, inner_caches, args, p)
    @unroll for inner in inner_caches
        assemble_cell!(req, inner, args)
    end
end

_fan_out_cell!(req, ::Tuple{}, args, ::Tuple{}) = nothing
function _fan_out_cell!(req, inner_caches::Tuple, args, views::Tuple)
    assemble_cell!(req, first(inner_caches), with_parameters(args, first(views)))
    return _fan_out_cell!(req, Base.tail(inner_caches), args, Base.tail(views))
end

provides_analytic(::Type{CompositeVolumetricElementCache{CT}}, kind) where {CT <: Tuple} = _all_provide(CT, kind)
_all_provide(::Type{Tuple{}}, kind) = true
_all_provide(::Type{T}, kind) where {T <: Tuple} =
    provides_analytic(Base.tuple_type_head(T), kind) && _all_provide(Base.tuple_type_tail(T), kind)
# The fan-out is all-or-nothing, so an inner that is itself decorated carries
# its decoration's coverage into the composite's answer.
serves_kind(::Type{CompositeVolumetricElementCache{CT}}, kind) where {CT <: Tuple} = _all_serve(CT, kind)
_all_serve(::Type{Tuple{}}, kind) = true
_all_serve(::Type{T}, kind) where {T <: Tuple} =
    serves_kind(Base.tuple_type_head(T), kind) && _all_serve(Base.tuple_type_tail(T), kind)
has_internal_state(::Type{CompositeVolumetricElementCache{CT}}) where {CT <: Tuple} = _any_internal(CT)
_any_internal(::Type{Tuple{}}) = false
_any_internal(::Type{T}) where {T <: Tuple} =
    has_internal_state(Base.tuple_type_head(T)) || _any_internal(Base.tuple_type_tail(T))
# A composite is insensitive for a kind iff every STATEFUL inner declares so —
# stateless inners differentiate exactly under plain AD regardless.
internal_state_insensitive(::Type{CompositeVolumetricElementCache{CT}}, kind) where {CT <: Tuple} =
    _all_stateful_insensitive(CT, kind)
_all_stateful_insensitive(::Type{Tuple{}}, kind) = true
function _all_stateful_insensitive(::Type{T}, kind) where {T <: Tuple}
    H = Base.tuple_type_head(T)
    return (!has_internal_state(H) || internal_state_insensitive(H, kind)) &&
        _all_stateful_insensitive(Base.tuple_type_tail(T), kind)
end

# The blanket fan-out method satisfies any `hasmethod` check, so validation must
# recurse: each inner is its own validation subject, kernels and admissibility
# alike.
function _validate_element_kernels(composite::CompositeVolumetricElementCache, declared_requests::Tuple)
    stateful = filter(inner -> has_internal_state(typeof(inner)), composite.inner_caches)
    isempty(stateful) || throw(ArgumentError(
        "Composing condensed elements is not supported yet, but $(_cache_names(stateful)) " *
        "carries condensed internal state. `get_number_of_internal_dofs_per_element` dispatches " *
        "on the composite, which implements it for none of its inners, and the internal " *
        "variable handler keys on the outer integrator — so the internal dofs would never be " *
        "allocated and `condense_internal!`'s write-back would have nowhere to land. Assemble " *
        "the condensed element as its own operator term."))
    foreach(cache -> validate_element_cache(cache, declared_requests), composite.inner_caches)
    return nothing
end

# The failure is a property of specific inners, not of the whole, so the
# rejection names them.
function assert_sensitivity_admissible(::Type{CompositeVolumetricElementCache{CT}}, kind) where {CT <: Tuple}
    has_internal_state(CompositeVolumetricElementCache{CT}) || return nothing
    serves_kind(CompositeVolumetricElementCache{CT}, kind) && return nothing
    internal_state_insensitive(CompositeVolumetricElementCache{CT}, kind) && return nothing
    missing_kind = _inner_types_without(CT, kind)
    throw(ArgumentError(
        "A composite carrying condensed internal state is inadmissible for $(typeof(kind)) " *
        "unless every inner serves that kind, but $(missing_kind) do(es) not. " *
        "AD-from-residual on the pure residual kernel would compute only the frozen-q " *
        "partial, silently missing the ∂F/∂q·dq/d· correction this kind's total needs. " *
        "Implement the analytic kernel on the listed inner(s), declare " *
        "`internal_state_insensitive` where the local equations do not depend on the seeded " *
        "quantity, or assemble the condensed element as its own operator term."))
end
_inner_types_without(::Type{Tuple{}}, kind) = ()
function _inner_types_without(::Type{T}, kind) where {T <: Tuple}
    H = Base.tuple_type_head(T)
    rest = _inner_types_without(Base.tuple_type_tail(T), kind)
    return serves_kind(H, kind) ? rest : (nameof(H), rest...)
end
_cache_names(caches::Tuple) = map(c -> nameof(typeof(c)), caches)

# Values deliberately shared between inners are reinitialized once per inner
# holding them.
reinit_values!(composite::CompositeVolumetricElementCache, cell) =
    _composite_reinit!(composite.inner_caches, cell)
reinit_values!(composite::CompositeVolumetricElementCache, cell, kind) =
    _composite_reinit!(composite.inner_caches, cell, kind)
@unroll function _composite_reinit!(inner_caches, cell)
    @unroll for inner in inner_caches
        reinit_values!(inner, cell)
    end
end
@unroll function _composite_reinit!(inner_caches, cell, kind)
    @unroll for inner in inner_caches
        reinit_values!(inner, cell, kind)
    end
end

# Empty inners carry no rule and are ignored; the rest must agree.
function Ferrite.getnquadpoints(composite::CompositeVolumetricElementCache)
    contributing = drop_empty_caches(composite.inner_caches)
    isempty(contributing) && return 0
    counts = map(getnquadpoints, contributing)
    allequal(counts) || throw(ArgumentError(
        "The inners of a composite disagree on the number of quadrature points " *
        "($(join(map((c, n) -> "$(nameof(typeof(c))) => $n", contributing, counts), ", "))). " *
        "Quadrature-point queries address one rule per cell, so composed elements " *
        "sharing a domain must share their quadrature rule."))
    return first(counts)
end

function duplicate_for_device(device, cache::CompositeVolumetricElementCache)
    return CompositeVolumetricElementCache(
        map(inner_cache -> duplicate_for_device(device, inner_cache), cache.inner_caches),
    )
end

# Interface composition (facet pairs) is not implemented.

"""
Combines multiple facet-item caches over one subdomain. The subdomain declares
ONE facet set — the union of the inners' [`facet_items`](@ref) — and the
fan-out gates on the facets each inner declared, so an inner integrates exactly
its own set. The per-inner sets are resolved once, at setup, and shared
read-only during a sweep.

Same scope bound as [`CompositeVolumetricElementCache`](@ref): one domain, one
context, one sink, no values objects shared by construction. Each inner is
handed its own per-facet parameter view, produced by that inner's
[`query_facet_parameters`](@ref).

A weighted sweep routes PER INNER — every covering inner is handed the same
[`WeightedJacobianRequest`](@ref) and accumulates its own fused contribution,
which is what puts a boundary spring (∂F/∂u only) next to a dashpot (∂F/∂v
only) in one `W`. This route composes no per-slot Jacobians, so an inner
without the fused weighted facet kernel is rejected by
[`assert_facet_item_route`](@ref) rather than folded around.
"""
struct CompositeFacetItemCache{CacheTupleType <: Tuple, SetTupleType <: Tuple} <: AbstractSurfaceElementCache
    inner_caches::CacheTupleType
    facet_sets::SetTupleType
end

query_facet_parameters(composite::CompositeFacetItemCache, cell, local_facet_index, p) =
    CompositeParameters(map(inner -> query_facet_parameters(inner, cell, local_facet_index, p), composite.inner_caches))

duplicate_for_device(device, cache::CompositeFacetItemCache) = CompositeFacetItemCache(
    map(inner_cache -> duplicate_for_device(device, inner_cache), cache.inner_caches),
    cache.facet_sets,                                # shared: read-only during a sweep
)

function assemble_facet!(req::AbstractAssemblyRequest, composite::CompositeFacetItemCache, args, local_facet_index::Int)
    idx = FacetIndex(cellid(args.cell), local_facet_index)
    return _composite_facet_item!(req, composite.inner_caches, composite.facet_sets, idx, args, local_facet_index, args.p)
end

_composite_facet_item!(req, caches, sets, idx, args, lfi, p::CompositeParameters) =
    _fan_out_facet_item!(req, caches, sets, idx, args, lfi, p.views)
_composite_facet_item!(req, caches, sets, idx, args, lfi, p) =
    _fan_out_facet_item!(req, caches, sets, idx, args, lfi)

_fan_out_facet_item!(req, ::Tuple{}, ::Tuple{}, idx, args, lfi, ::Tuple{}) = nothing
function _fan_out_facet_item!(req, caches::Tuple, sets::Tuple, idx, args, lfi, views::Tuple)
    idx in first(sets) && assemble_facet!(req, first(caches), with_parameters(args, first(views)), lfi)
    return _fan_out_facet_item!(req, Base.tail(caches), Base.tail(sets), idx, args, lfi, Base.tail(views))
end

_fan_out_facet_item!(req, ::Tuple{}, ::Tuple{}, idx, args, lfi) = nothing
function _fan_out_facet_item!(req, caches::Tuple, sets::Tuple, idx, args, lfi)
    idx in first(sets) && assemble_facet!(req, first(caches), args, lfi)
    return _fan_out_facet_item!(req, Base.tail(caches), Base.tail(sets), idx, args, lfi)
end

function evaluate_facet_functional(kind, composite::CompositeFacetItemCache, args, local_facet_index::Int)
    idx = FacetIndex(cellid(args.cell), local_facet_index)
    return _composite_facet_functional(kind, composite.inner_caches, composite.facet_sets, idx, args, local_facet_index, args.p)
end

_composite_facet_functional(kind, caches, sets, idx, args, lfi, p::CompositeParameters) =
    _fan_out_facet_functional(kind, caches, sets, idx, args, lfi, p.views)
_composite_facet_functional(kind, caches, sets, idx, args, lfi, p) =
    _fan_out_facet_functional(kind, caches, sets, idx, args, lfi)

_fan_out_facet_functional(kind, ::Tuple{}, ::Tuple{}, idx, args, lfi, ::Tuple{}) = nothing
function _fan_out_facet_functional(kind, caches::Tuple, sets::Tuple, idx, args, lfi, views::Tuple)
    head = idx in first(sets) ?
        evaluate_facet_functional(kind, first(caches), with_parameters(args, first(views)), lfi) : nothing
    return _add_facet_partials(head,
        _fan_out_facet_functional(kind, Base.tail(caches), Base.tail(sets), idx, args, lfi, Base.tail(views)))
end

_fan_out_facet_functional(kind, ::Tuple{}, ::Tuple{}, idx, args, lfi) = nothing
function _fan_out_facet_functional(kind, caches::Tuple, sets::Tuple, idx, args, lfi)
    head = idx in first(sets) ? evaluate_facet_functional(kind, first(caches), args, lfi) : nothing
    return _add_facet_partials(head,
        _fan_out_facet_functional(kind, Base.tail(caches), Base.tail(sets), idx, args, lfi))
end

# A facet's contribution is the sum over the inners covering it, and `nothing`
# — "no contribution" — is the fold's identity, exactly as it is for the
# family's own sweep.
_add_facet_partials(::Nothing, ::Nothing) = nothing
_add_facet_partials(a, ::Nothing) = a
_add_facet_partials(::Nothing, b) = b
_add_facet_partials(a, b) = a + b

# Facet kernels have no AD fallback, so the fan-out is all-or-nothing per kind
# just as the volumetric composite's is: the composite serves a kind iff every
# inner does.
provides_analytic(::Type{CompositeFacetItemCache{CT, ST}}, kind) where {CT <: Tuple, ST} = _all_provide(CT, kind)
serves_kind(::Type{CompositeFacetItemCache{CT, ST}}, kind) where {CT <: Tuple, ST} = _all_serve(CT, kind)
has_internal_state(::Type{CompositeFacetItemCache{CT, ST}}) where {CT <: Tuple, ST} = _any_internal(CT)
internal_state_insensitive(::Type{CompositeFacetItemCache{CT, ST}}, kind) where {CT <: Tuple, ST} =
    _all_stateful_insensitive(CT, kind)

# The blanket fan-out satisfies any `hasmethod` probe, so validation recurses:
# each inner is its own validation subject, kernels and route election alike.
_validate_facet_item_kernels(composite::CompositeFacetItemCache, declared_requests::Tuple) =
    foreach(cache -> validate_facet_item_cache(cache, declared_requests), composite.inner_caches)

# ... and so does the per-item route election, which is a statement about the
# cache that actually runs the kernel.
assert_facet_item_route(kind::WeightedJacobianKind, composite::CompositeFacetItemCache) =
    _assert_inner_facet_item_routes(kind, composite.inner_caches)
@unroll function _assert_inner_facet_item_routes(kind, inner_caches)
    @unroll for inner in inner_caches
        assert_facet_item_route(kind, inner)
    end
end

####################################
## Composition of element caches
####################################

"""
    compose_element_caches(caches::Tuple)

Combine per-inner volumetric caches into the cache the engine sees. Empty
caches carry no term and are dropped, so an all-empty composition collapses to
[`EmptyVolumetricElementCache`](@ref) and a single survivor is returned
unwrapped — the engine's empty-cache and single-cache fast paths survive
composition.
"""
compose_element_caches(caches::Tuple) = _collapse(drop_empty_caches(caches), EmptyVolumetricElementCache(), CompositeVolumetricElementCache)

function _collapse(kept::Tuple, empty, Composite)
    length(kept) == 0 && return empty
    length(kept) == 1 && return only(kept)
    return Composite(kept)
end

# Type-stable recursive filter: the dropped types are decided per element type,
# so the result type is inferred rather than reduced to an abstract eltype.
drop_empty_caches(::Tuple{}) = ()
drop_empty_caches(caches::Tuple) = _drop_empty_head(first(caches), Base.tail(caches))
_drop_empty_head(::EmptyVolumetricElementCache, rest::Tuple) = drop_empty_caches(rest)
_drop_empty_head(cache, rest::Tuple) = (cache, drop_empty_caches(rest)...)

####################################
## Composite integrators
####################################

"""
    NonlinearCompositeIntegrator(subintegrators::Tuple)
    BilinearCompositeIntegrator(subintegrators::Tuple)
    LinearCompositeIntegrator(subintegrators::Tuple)

Integrator stacking several sub-integrators over the SAME domain into one
element: cache setup maps every sub-integrator over the subdomain and combines
the results through [`compose_element_caches`](@ref).

Scope bound: one domain, one evaluation context, one scatter target — as for
[`CompositeVolumetricElementCache`](@ref). Hence the single field: terms needing
their own context, weight or domain are separate operator terms, not inners.

Nested composites are flattened at construction, which also rejects an empty
tuple, a sub-integrator carrying condensed internal state, and cross-sink
mixes. A **bilinear** sub-integrator inside a nonlinear composite is legitimate
— the operator its form induces scatters into the same matrix and residual, its
residual being the element matrix acting on the element vector — whereas an
[`AbstractLinearIntegrator`](@ref) describes a load form, whose sink is a
vector alone, and never composes into a nonlinear or bilinear operator.

Routing and composition compose in one order: a `*MultiDomainIntegrator` whose
values are composite integrators. A composite of routers is not supported.
"""
struct NonlinearCompositeIntegrator{T <: Tuple} <: AbstractNonlinearIntegrator
    subintegrators::T
    function NonlinearCompositeIntegrator(subintegrators::Tuple)
        flat = flatten_subintegrators(subintegrators)
        validate_composite_members(flat, Union{AbstractNonlinearIntegrator, AbstractBilinearIntegrator},
                                   "NonlinearCompositeIntegrator",
                                   "only nonlinear and bilinear terms share its sink")
        return new{typeof(flat)}(flat)
    end
end
NonlinearCompositeIntegrator(subintegrators...) = NonlinearCompositeIntegrator(subintegrators)

@doc (@doc NonlinearCompositeIntegrator)
struct BilinearCompositeIntegrator{T <: Tuple} <: AbstractBilinearIntegrator
    subintegrators::T
    function BilinearCompositeIntegrator(subintegrators::Tuple)
        flat = flatten_subintegrators(subintegrators)
        validate_composite_members(flat, AbstractBilinearIntegrator, "BilinearCompositeIntegrator",
                                   "only bilinear terms share its sink")
        return new{typeof(flat)}(flat)
    end
end
BilinearCompositeIntegrator(subintegrators...) = BilinearCompositeIntegrator(subintegrators)

@doc (@doc NonlinearCompositeIntegrator)
struct LinearCompositeIntegrator{T <: Tuple} <: AbstractLinearIntegrator
    subintegrators::T
    function LinearCompositeIntegrator(subintegrators::Tuple)
        flat = flatten_subintegrators(subintegrators)
        validate_composite_members(flat, AbstractLinearIntegrator, "LinearCompositeIntegrator",
                                   "only linear terms share its sink")
        return new{typeof(flat)}(flat)
    end
end
LinearCompositeIntegrator(subintegrators...) = LinearCompositeIntegrator(subintegrators)

const AnyCompositeIntegrator = Union{NonlinearCompositeIntegrator, BilinearCompositeIntegrator, LinearCompositeIntegrator}

setup_element_cache(element_model::AnyCompositeIntegrator, sdh::SubDofHandler) =
    compose_element_caches(map(sub -> setup_element_cache(sub, sdh), element_model.subintegrators))

# The inners share one local system, so they share its tail: a composite
# declares what its inners declare, and silent inners (the default `()`) read
# the tail a declaring inner puts there. Each family answers from its OWN hook,
# since each sizes its own local system.
global_dofs(integrator::AnyCompositeIntegrator, sdh::SubDofHandler) = _composite_declared_dofs(
    integrator, map(sub -> global_dofs(sub, sdh), integrator.subintegrators), "global_dofs")

facet_item_global_dofs(integrator::AnyCompositeIntegrator, sdh::SubDofHandler) = _composite_declared_dofs(
    integrator, map(sub -> facet_item_global_dofs(sub, sdh), integrator.subintegrators), "facet_item_global_dofs")

function _composite_declared_dofs(integrator, declarations::Tuple, hook)
    declared = filter(!isempty, declarations)
    isempty(declared) && return ()
    reference = first(declared)
    all(d -> length(d) == length(reference) && all(d .== reference), declared) || throw(ArgumentError(
        "The sub-integrators of $(nameof(typeof(integrator))) declare different `$hook` " *
        "for this subdomain ($(map(collect, declared))). Composed terms fill ONE local system, " *
        "whose tail is `[celldofs(cell); global dofs]` — with two different declarations there " *
        "is no unambiguous tail. Declare the same dofs in the same order, or assemble the " *
        "terms as separate operators."))
    return reference
end

# A composite answers every declaration hook out of its sub-integrators', so
# each of them is a declaration subject in its own right.
function _declaration_subjects!(subjects, integrator::AnyCompositeIntegrator)
    push!(subjects, integrator)
    foreach(sub -> _declaration_subjects!(subjects, sub), integrator.subintegrators)
    return subjects
end

"""
    facet_items(integrator::AnyCompositeIntegrator, sdh)
    setup_facet_item_cache(integrator::AnyCompositeIntegrator, sdh)

The facet-item declaration of a composite: the sorted union of its
sub-integrators' [`facet_items`](@ref), served by ONE
[`CompositeFacetItemCache`](@ref) that gates the fan-out on each inner's own
set. Inners declaring nothing contribute nothing — to the declaration and to
the cache — so an all-silent composite keeps the additive `()` default and a
single declaring inner's cache is returned unwrapped, mirroring the
[`compose_element_caches`](@ref) collapse rules.

The union is what makes overlap legal: two terms supported on the SAME facet
are one item declared once and assembled by both inners, which the family's
declared-twice rejection would otherwise refuse.
"""
function facet_items(integrator::AnyCompositeIntegrator, sdh::SubDofHandler)
    declared = map(sub -> facet_items(sub, sdh), integrator.subintegrators)
    all(isempty, declared) && return ()
    merged = Set{FacetIndex}()
    for set in declared, facet in set
        push!(merged, facet)
    end
    # Sorted for the same reason `resolve_facet_items` sorts: neither a `Set`'s
    # iteration order nor the order the inners happen to sit in may decide the
    # item order.
    return sort!(collect(merged); by = facet -> (facet[1], facet[2]))
end

function setup_facet_item_cache(integrator::AnyCompositeIntegrator, sdh::SubDofHandler)
    declaring = filter(sub -> !isempty(facet_items(sub, sdh)), integrator.subintegrators)
    isempty(declaring) && throw(ArgumentError(
        "No sub-integrator of $(nameof(typeof(integrator))) declares `facet_items` for this " *
        "subdomain, so there is no facet-item cache to build."))
    caches = map(sub -> setup_facet_item_cache(sub, sdh), declaring)
    length(caches) == 1 && return only(caches)
    return CompositeFacetItemCache(caches, map(sub -> Set{FacetIndex}(facet_items(sub, sdh)), declaring))
end

flatten_subintegrators(::Tuple{}) = ()
flatten_subintegrators(subintegrators::Tuple) =
    (_flatten_head(first(subintegrators))..., flatten_subintegrators(Base.tail(subintegrators))...)
_flatten_head(sub) = (sub,)
_flatten_head(sub::AnyCompositeIntegrator) = sub.subintegrators

function validate_composite_members(subintegrators::Tuple, allowed::Type, name, sink_rule)
    isempty(subintegrators) && throw(ArgumentError(
        "$name needs at least one sub-integrator. An empty composite assembles nothing, " *
        "which is a setup error rather than a no-op."))
    for sub in subintegrators
        sub isa AbstractCondensedNonlinearIntegrator && throw(ArgumentError(
            "$(typeof(sub)) carries condensed internal state and cannot be composed. " *
            "`get_number_of_internal_dofs_per_element` dispatches on the composite cache, " *
            "which implements it for none of its inners, so the internal dofs would never be " *
            "allocated and `condense_internal!`'s write-back would have nowhere to land. " *
            "Assemble the condensed element as its own operator term."))
        sub isa allowed || throw(ArgumentError(
            "$(typeof(sub)) cannot be composed into a $name — $sink_rule. Terms scattering " *
            "into different targets are separate operators, not inners of one composite."))
    end
    return nothing
end
