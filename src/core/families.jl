## The item-family registration seam: WHICH families an operator's engine
## carries, and how each family's `SubdomainCache`s are built. The three shipped
## families reach the engine through this seam and no other — there is no
## privileged setup path beside it.

"""
    CellFamily()

The cell item family's registration marker: the cells of every subdomain,
positioned by [`assembly_iterator`](@ref) and enumerated by
[`item_provider`](@ref). Registered by every operator, since
[`item_families`](@ref)'s default always names it.

A family marker is a singleton and subtypes nothing: it exists to select a
[`setup_family_caches`](@ref) method, so a downstream family declares its own
`struct` and needs no supertype from this package.

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
struct CellFamily end

"""
    FacetItemFamily()

The facet item family's registration marker: the boundary items
[`facet_items`](@ref) declares, one `SubdomainCache` per subdomain that declares
any. See [`CellFamily`](@ref) for what a marker is.

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
struct FacetItemFamily end

"""
    AlgebraicItemFamily()

The algebraic item family's registration marker: the items
[`algebraic_items`](@ref) declares — a dof set and nothing else — served by a
single `SubdomainCache` over the whole handler. See [`CellFamily`](@ref) for
what a marker is.

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
struct AlgebraicItemFamily end

"""
    item_families(integrator, dh) -> Tuple

The item families `integrator`'s operator carries over `dh`, in TRAVERSAL
ORDER. [`setup_engine`](@ref) calls [`setup_family_caches`](@ref) once per
element and concatenates the results, so the tuple's order IS the order of
`engine.subdomain_caches` — which a reduction's determinism rests on.

The default derives the tuple from the declarations that already decide what an
operator carries: [`CellFamily`](@ref) always, [`FacetItemFamily`](@ref) where
some subdomain declares [`facet_items`](@ref), [`AlgebraicItemFamily`](@ref)
where the handler declares [`algebraic_items`](@ref). Declaring nothing
therefore registers cells alone, exactly as before this seam existed.

A downstream family is registered by overloading this — the tuple it returns
REPLACES the default, so an integrator adding a family beside the cells returns
both markers, and one whose items are not cells at all returns its own marker
alone.

    FerriteOperators.item_families(::MyIntegrator, dh) = (CellFamily(), MyFamily())

!!! warning "Experimental surface"
    The registration seam may change in a minor release.

**Scope limits, stated rather than discovered.** The composite and multi-domain
wrappers forward [`facet_items`](@ref) and [`algebraic_items`](@ref) — so the
built-in families of a wrapped sub-integrator are carried through the default
above — but they do not forward this hook: a wrapper's own `item_families`
answer is the operator's. And a family whose driver is not the engine's does not
register at all: patch items run through [`foreach_patch`](@ref) and transfer
operators through [`setup_transfer_operator`](@ref), for the reasons
[`setup_family_caches`](@ref) states.
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
[`setup_engine`](@ref). This is the ONLY route a family's caches enter the
engine by — the three shipped families answer it like any downstream one, so
there is no privileged setup path to reach for.

A method builds, per subdomain it serves, the three things a `SubdomainCache`
holds: the family's domain descriptor, the device instances of the workspace a
sweep positions, and the partition its item set implies
([`compute_partition`](@ref)). Returning an empty vector is how a registered
family declines a given operator.

`shared` is what [`setup_engine`](@ref) resolved before any family ran, and
every field is available to every family:

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

The shipped methods are the worked examples: the [`CellFamily`](@ref) one
(`operators/setup.jl`) resolves an iterator and a provider per subdomain and is
the only one that also builds a device-resident handler; the
[`FacetItemFamily`](@ref) one (`core/facet-task.jl`) skips a subdomain that
declares no items; the [`AlgebraicItemFamily`](@ref) one
(`core/algebraic-task.jl`) serves the whole handler with a single cache.

**Two families deliberately do NOT register**, and the reasons are properties of
those families rather than of this seam. [`foreach_patch`](@ref) is
sequential-CPU-only because the callback's collectors are the CALLER's, so this
package cannot duplicate them per worker, and `PatchAssemblyWorkspace` positions
through a `Ref` resolved against its provider — a CPU-scoped positioning
mechanism. [`setup_transfer_operator`](@ref) restricts to sequential full
assembly by design and assembles a RECTANGULAR matrix through its own driver.
Both adopt the iteration seams ([`assembly_iterator`](@ref),
[`iterator_dofs`](@ref), [`item_provider`](@ref)) and keep their own entry
points. The guarantee this seam makes is therefore the true one: **no
engine-registered family has a privileged setup path.**

!!! warning "Experimental surface"
    The registration seam may change in a minor release.
"""
function setup_family_caches end

# The registered families' caches, concatenated in declaration order. One
# family hands its own vector straight back, so the very common cells-only
# operator keeps a CONCRETELY typed `subdomain_caches` and widens to the
# abstract element type only where a second family is registered.
function _family_subdomain_caches(families::Tuple, strategy, integrator, dh, shared)
    length(families) == 1 &&
        return setup_family_caches(only(families), strategy, integrator, dh, shared)
    caches = SubdomainCache[]
    for family in families
        append!(caches, setup_family_caches(family, strategy, integrator, dh, shared))
    end
    return caches
end
