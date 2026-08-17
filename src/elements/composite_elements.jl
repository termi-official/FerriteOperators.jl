"""
Per-inner parameter views produced by a composite cache's
[`query_cell_parameters`](@ref) / [`query_facet_parameters`](@ref). The fan-out
re-seeds `args.p` per inner from this bundle; any other `p` object reaches
every inner unchanged, so a hand-built `KernelArgs` still works.
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

No values objects are shared by construction through this path — each inner
sets up and reinitializes its own. Deliberate sharing between elements is an
element-side concern and stays the sharing element's responsibility.

Composition is pure iteration: the request carries the buffers, so one generic
fan-out serves every request type. Each inner is handed its OWN parameter
view, produced by that inner's [`query_cell_parameters`](@ref).

The composite is analytic for a kind iff every inner is; otherwise the whole
composite falls back to AD over the fused residual (one Dual sweep, one
geometry pass for the fused Jacobian). That routing is all-or-nothing.

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

# The composite's blanket fan-out method would satisfy any hasmethod check,
# so validation must recurse into the inner caches.
function validate_element_cache(composite::CompositeVolumetricElementCache, declared_requests::Tuple = (), ::Type{A} = KernelArgs) where {A}
    stateful = filter(inner -> has_internal_state(typeof(inner)), composite.inner_caches)
    isempty(stateful) || throw(ArgumentError(
        "Composing condensed elements is not supported yet, but $(_cache_names(stateful)) " *
        "carries condensed internal state. The condensation hooks " *
        "(`get_number_of_internal_dofs_per_element`, `load_element_unknowns!`, " *
        "`store_condensed_element_unknowns!`) dispatch on the composite, which implements " *
        "none of them, and the internal variable handler keys on the outer integrator — so " *
        "the internal dofs would never be allocated and the trial write-back would be " *
        "dropped. Assemble the condensed element as its own operator term."))
    foreach(cache -> validate_element_cache(cache, declared_requests, A), composite.inner_caches)
    return nothing
end

# Naming the inners is what turns a composed-admissibility rejection into an
# actionable one: the failure is a property of specific inners, not the whole.
function assert_sensitivity_admissible(::Type{CompositeVolumetricElementCache{CT}}, kind) where {CT <: Tuple}
    has_internal_state(CompositeVolumetricElementCache{CT}) || return nothing
    provides_analytic(CompositeVolumetricElementCache{CT}, kind) && return nothing
    internal_state_insensitive(CompositeVolumetricElementCache{CT}, kind) && return nothing
    missing_kind = _inner_types_without(CT, kind)
    throw(ArgumentError(
        "A composite carrying condensed internal state is inadmissible for $(typeof(kind)) " *
        "unless every inner serves that kind analytically, but $(missing_kind) do(es) not. " *
        "AD-from-residual through the condensed inner's local solve would be silently wrong. " *
        "Implement the analytic kernel on the listed inner(s), declare " *
        "`internal_state_insensitive` where the local equations do not depend on the seeded " *
        "quantity, or assemble the condensed element as its own operator term."))
end
_inner_types_without(::Type{Tuple{}}, kind) = ()
function _inner_types_without(::Type{T}, kind) where {T <: Tuple}
    H = Base.tuple_type_head(T)
    rest = _inner_types_without(Base.tuple_type_tail(T), kind)
    return provides_analytic(H, kind) ? rest : (nameof(H), rest...)
end
_cache_names(caches::Tuple) = map(c -> nameof(typeof(c)), caches)

# Inner scratch declarations survive composition (later inners win on name
# collisions, like the solver-vs-element merge at setup).
declare_scratch(composite::CompositeVolumetricElementCache) =
    reduce(merge, map(declare_scratch, composite.inner_caches); init = (;))

# Values instances deliberately shared between inners are reinitialized once
# per inner that holds them.
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

# Quadrature-point queries address ONE quadrature rule, so the inners must
# agree on it. Empty inners carry no rule and are ignored.
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

"""
Combines multiple element caches over the same surface. The boundary driver
gates on the composite (`is_facet_in_cache` = any inner covers the facet); the
fan-out re-gates per inner cache, since inner caches may cover different facet
sets. Each inner is handed its own per-facet parameter view.

Same scope bound as [`CompositeVolumetricElementCache`](@ref): one domain, one
context, one sink, no values objects shared by construction.
"""
struct CompositeSurfaceElementCache{CacheTupleType <: Tuple} <: AbstractSurfaceElementCache
    inner_caches::CacheTupleType
end

query_facet_parameters(composite::CompositeSurfaceElementCache, cell, local_facet_index, p) =
    CompositeParameters(map(inner -> query_facet_parameters(inner, cell, local_facet_index, p), composite.inner_caches))

function duplicate_for_device(device, cache::CompositeSurfaceElementCache)
    return CompositeSurfaceElementCache(
        map(inner_cache -> duplicate_for_device(device, inner_cache), cache.inner_caches),
    )
end

assemble_facet!(req::AbstractAssemblyRequest, composite::CompositeSurfaceElementCache, args, local_facet_index::Int) =
    _composite_assemble_facet!(req, composite.inner_caches, args, local_facet_index, args.p)

_composite_assemble_facet!(req, inner_caches, args, local_facet_index, p::CompositeParameters) =
    _fan_out_facet!(req, inner_caches, args, local_facet_index, p.views)
@unroll function _composite_assemble_facet!(req, inner_caches, args, local_facet_index, p)
    @unroll for inner in inner_caches
        if is_facet_in_cache(FacetIndex(cellid(args.cell), local_facet_index), args.cell, inner)
            assemble_facet!(req, inner, args, local_facet_index)
        end
    end
end

_fan_out_facet!(req, ::Tuple{}, args, local_facet_index, ::Tuple{}) = nothing
function _fan_out_facet!(req, inner_caches::Tuple, args, local_facet_index, views::Tuple)
    inner = first(inner_caches)
    if is_facet_in_cache(FacetIndex(cellid(args.cell), local_facet_index), args.cell, inner)
        assemble_facet!(req, inner, with_parameters(args, first(views)), local_facet_index)
    end
    return _fan_out_facet!(req, Base.tail(inner_caches), args, local_facet_index, Base.tail(views))
end

is_facet_in_cache(idx::FacetIndex, cell, composite::CompositeSurfaceElementCache) =
    _any_facet_in_cache(idx, cell, composite.inner_caches)
@unroll function _any_facet_in_cache(idx, cell, inner_caches)
    @unroll for inner in inner_caches
        is_facet_in_cache(idx, cell, inner) && return true
    end
    return false
end

# Interface composition is not implemented; DG support will follow the same
# request fan-out pattern as cells and facets.

####################################
## Composition of element caches
####################################

"""
    compose_element_caches(caches::Tuple)
    compose_boundary_caches(caches::Tuple)

Combine per-inner caches into the cache the engine sees. Empty caches carry no
term and are dropped, so an all-empty composition collapses to the empty cache
and a single surviving cache is returned unwrapped — the engine's empty-cache
and single-cache fast paths therefore survive composition.
"""
compose_element_caches(caches::Tuple) = _collapse(drop_empty_caches(caches), EmptyVolumetricElementCache(), CompositeVolumetricElementCache)

@doc (@doc compose_element_caches)
compose_boundary_caches(caches::Tuple) = _collapse(drop_empty_caches(caches), EmptySurfaceElementCache(), CompositeSurfaceElementCache)

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
_drop_empty_head(::EmptySurfaceElementCache, rest::Tuple) = drop_empty_caches(rest)
_drop_empty_head(cache, rest::Tuple) = (cache, drop_empty_caches(rest)...)

####################################
## Composite integrators
####################################

"""
    NonlinearCompositeIntegrator(subintegrators::Tuple)
    BilinearCompositeIntegrator(subintegrators::Tuple)
    LinearCompositeIntegrator(subintegrators::Tuple)

Integrator stacking several sub-integrators over the SAME domain into one
element: element and boundary cache setup map every sub-integrator over the
subdomain and combine the results through [`compose_element_caches`](@ref) /
[`compose_boundary_caches`](@ref).

Scope bound: one domain, one evaluation context, one scatter target — the same
bound as [`CompositeVolumetricElementCache`](@ref). The type carries exactly
one field: terms needing their own context, weight or domain are separate
operator terms, not inners.

Nested composite integrators are flattened at construction. Composition is
rejected loudly at construction for an empty tuple, for any sub-integrator
carrying condensed internal state, and for cross-sink mixes. A **bilinear**
sub-integrator inside a nonlinear composite is legitimate — a bilinear form
induces a linear operator whose residual is the element matrix acting on the
element vector — but a linear (load) form has a different sink and never
composes into a nonlinear or bilinear operator.

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

setup_element_cache(element_model::NonlinearCompositeIntegrator, sdh::SubDofHandler) =
    compose_element_caches(map(sub -> setup_element_cache(sub, sdh), element_model.subintegrators))
setup_boundary_cache(element_model::NonlinearCompositeIntegrator, sdh::SubDofHandler) =
    compose_boundary_caches(map(sub -> setup_boundary_cache(sub, sdh), element_model.subintegrators))

setup_element_cache(element_model::BilinearCompositeIntegrator, sdh::SubDofHandler) =
    compose_element_caches(map(sub -> setup_element_cache(sub, sdh), element_model.subintegrators))
setup_boundary_cache(element_model::BilinearCompositeIntegrator, sdh::SubDofHandler) =
    compose_boundary_caches(map(sub -> setup_boundary_cache(sub, sdh), element_model.subintegrators))

setup_element_cache(element_model::LinearCompositeIntegrator, sdh::SubDofHandler) =
    compose_element_caches(map(sub -> setup_element_cache(sub, sdh), element_model.subintegrators))
setup_boundary_cache(element_model::LinearCompositeIntegrator, sdh::SubDofHandler) =
    compose_boundary_caches(map(sub -> setup_boundary_cache(sub, sdh), element_model.subintegrators))

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
            "The condensation hooks dispatch on the composite cache, which implements none " *
            "of them, so the internal dofs would never be allocated and the trial write-back " *
            "would be dropped. Assemble the condensed element as its own operator term."))
        sub isa allowed || throw(ArgumentError(
            "$(typeof(sub)) cannot be composed into a $name — $sink_rule. Terms scattering " *
            "into different targets are separate operators, not inners of one composite."))
    end
    return nothing
end
