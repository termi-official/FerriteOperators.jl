"""
    CellFamily()

The cell item family's registration marker: the cells of every subdomain,
positioned by [`assembly_iterator`](@ref) and enumerated by
[`item_provider`](@ref). Registered by every operator.

A family marker is a singleton and subtypes nothing — it only selects a
[`setup_family_caches`](@ref) method, so a downstream family declares its own
`struct` and needs no supertype from this package.

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
struct CellFamily end

"""
    FacetItemFamily()

Registration marker for the boundary items [`facet_items`](@ref) declares, one
`SubdomainCache` per subdomain that declares any.

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
struct FacetItemFamily end

"""
    AlgebraicItemFamily()

Registration marker for the items [`algebraic_items`](@ref) declares — a dof set
and nothing else — served by a single `SubdomainCache` over the whole handler.

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
struct AlgebraicItemFamily end

"""
    item_families(integrator, dh) -> Tuple

The item families `integrator`'s operator carries over `dh`, in TRAVERSAL
ORDER. The tuple's order IS the order of `engine.subdomain_caches`, which a
reduction's determinism rests on.

The default is [`CellFamily`](@ref) always, [`FacetItemFamily`](@ref) where some
subdomain declares [`facet_items`](@ref), [`AlgebraicItemFamily`](@ref) where the
handler declares [`algebraic_items`](@ref). An overload REPLACES that tuple:

    FerriteOperators.item_families(::MyIntegrator, dh) = (CellFamily(), MyFamily())

The composite and multi-domain wrappers do NOT forward this hook, though they do
forward [`facet_items`](@ref) and [`algebraic_items`](@ref): a wrapper's own
answer is the operator's, and a sub-integrator's registration is not seen.

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
function item_families(integrator, dh)
    facets    = any(sdh -> !isempty(facet_items(integrator, sdh)), dh.subdofhandlers)
    algebraic = !isempty(algebraic_items(integrator, dh))
    return (CellFamily(),
            (facets    ? (FacetItemFamily(),)    : ())...,
            (algebraic ? (AlgebraicItemFamily(),) : ())...)
end

"""
    setup_family_caches(family, strategy, integrator, dh, shared) -> SubdomainCaches

One registered family's `SubdomainCache`s, built once at
[`setup_engine`](@ref). The ONLY route a family's caches enter the engine by:
the three shipped families answer it like any downstream one.

A method builds, per subdomain it serves, the three things a `SubdomainCache`
holds: the family's domain descriptor, the device instances of the workspace a
sweep positions, and the partition its item set implies
([`compute_partition`](@ref)). An empty vector declines the operator.

`shared` is what [`setup_engine`](@ref) resolved before any family ran:

| field | what it carries |
|---|---|
| `slots` | the declared state slot names the per-worker buffers are sized for |
| `needs_sensitivity` | whether the workspaces carry [`SensitivityBuffers`](@ref) |
| `ivh` | the resolved [`InternalVariableHandler`](@ref) |
| `ad_backend` | the [`ADElementCache`](@ref) backend, `nothing` where wrapping is opted out |
| `declared_kinds` | the request kinds whose setup-time validation runs at setup |
| `element_caches` | the resolved, decorated CELL element caches, one per subdomain |
| `global_dof_sets` | each subdomain's validated [`global_dofs`](@ref) declaration |
| `facet_item_global_dof_sets` | each subdomain's validated [`facet_item_global_dofs`](@ref) declaration |
| `algebraic_domain` | [`resolve_algebraic_domain`](@ref)'s `(cache, items)`, or `nothing` |

The shipped methods live in `operators/setup.jl` ([`CellFamily`](@ref)),
`core/facet-task.jl` and `core/algebraic-task.jl`.
[`foreach_patch`](@ref) and [`setup_transfer_operator`](@ref) deliberately do
NOT register; `devdocs/design.md` states why.

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
function setup_family_caches end

# A single family hands its own vector straight back, so the common cells-only
# operator keeps a CONCRETELY typed `subdomain_caches`.
function _family_subdomain_caches(families::Tuple, strategy, integrator, dh, shared)
    length(families) == 1 &&
        return setup_family_caches(only(families), strategy, integrator, dh, shared)
    caches = SubdomainCache[]
    for family in families
        append!(caches, setup_family_caches(family, strategy, integrator, dh, shared))
    end
    return caches
end
