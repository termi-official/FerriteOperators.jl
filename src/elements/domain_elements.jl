"""
    NonlinearMultiDomainIntegrator(subintegrators::Dict{String})
    LinearMultiDomainIntegrator(subintegrators::Dict{String})
    BilinearMultiDomainIntegrator(subintegrators::Dict{String})

Integrator carrying one sub-integrator per named subdomain, so a single
operator hosts different physics per subdomain. The keys are names of
**volumetric cellsets** in the grid's cellset registry.

A name resolves *both* the element cache and the boundary cache of the
subdomain it claims; facetset names play no part. Which facets actually carry a
term is decided per facet by [`is_facet_in_cache`](@ref), as for any other
surface cache.

Each subdomain must lie entirely within one declared cellset. **Production mode
assumes this holds and does not verify it** — a subdomain's first cell alone
decides its claim. [`FerriteOperators.debug_mode`](@ref) replaces the sample
with an exhaustive check of every cell.

Resolution runs once per operator setup. `ArgumentError`s in **both** modes:

- a declared name that is not a cellset of the grid,
- a subdomain whose first cell is owned by no declared name,
- a subdomain whose first cell is owned by more than one declared name,
- a declared name that claims no subdomain.

Debug mode additionally rejects, naming the offending cell:

- two declared cellsets sharing a cell, whether or not a subdomain touches it,
- a subdomain with any cell owned by no declared name,
- a subdomain whose cells span several declared names.

There is deliberately no lenient fallback: an unclaimed subdomain, or a
mistyped name silently contributing nothing, is what setup-time validation
exists to catch.

Sub-integrators must share the operator's sink: a nonlinear router accepts
nonlinear *and* bilinear sub-integrators — the operator a bilinear form induces
has the same sink, its residual being the element matrix acting on the element
vector — while the bilinear and linear routers accept only their own kind.
"""
struct NonlinearMultiDomainIntegrator{DictType <: Dict{String, <:Union{AbstractNonlinearIntegrator, AbstractBilinearIntegrator}}} <: AbstractNonlinearIntegrator
    subintegrators::DictType
end

@doc (@doc NonlinearMultiDomainIntegrator)
struct LinearMultiDomainIntegrator{DictType <: Dict{String, <:AbstractLinearIntegrator}} <: AbstractLinearIntegrator
    subintegrators::DictType
end

@doc (@doc NonlinearMultiDomainIntegrator)
struct BilinearMultiDomainIntegrator{DictType <: Dict{String, <:AbstractBilinearIntegrator}} <: AbstractBilinearIntegrator
    subintegrators::DictType
end

const AnyMultiDomainIntegrator = Union{NonlinearMultiDomainIntegrator, BilinearMultiDomainIntegrator, LinearMultiDomainIntegrator}

# Resolution is hoisted to the plural hooks: twice per operator setup (elements,
# boundaries) instead of once per subdomain and hook. Decoration stays
# per-subdomain, since neighbouring sub-integrators may differ in needing it.
function setup_elements(integrator::AnyMultiDomainIntegrator, dh::AbstractDofHandler, ad_backend, n_global_dofs)
    resolved = zip(subintegrators_per_subdomain(integrator, dh), dh.subdofhandlers, n_global_dofs)
    needs_ad_decoration(integrator) || return [setup_element_cache(sub, sdh) for (sub, sdh, _) in resolved]
    return [decorate_element_cache(setup_element_cache(sub, sdh), sdh, ad_backend, n) for (sub, sdh, n) in resolved]
end

# A subdomain's global dofs are its sub-integrator's, like its caches.
global_dofs(integrator::AnyMultiDomainIntegrator, sdh::SubDofHandler) =
    global_dofs(subintegrator_for_subdomain(integrator.subintegrators, sdh), sdh)

setup_boundaries(integrator::AnyMultiDomainIntegrator, dh::AbstractDofHandler) =
    [setup_boundary_cache(sub, sdh) for (sub, sdh) in zip(subintegrators_per_subdomain(integrator, dh), dh.subdofhandlers)]

# Each subdomain routes its own facet set through its own surface cache.
facet_items(integrator::AnyMultiDomainIntegrator, sdh::SubDofHandler) =
    facet_items(subintegrator_for_subdomain(integrator.subintegrators, sdh), sdh)
setup_facet_item_cache(integrator::AnyMultiDomainIntegrator, sdh::SubDofHandler) =
    setup_facet_item_cache(subintegrator_for_subdomain(integrator.subintegrators, sdh), sdh)

# Available for direct use, but each pays a full resolution — the engine calls
# the plural hooks above.
setup_element_cache(element_model::AnyMultiDomainIntegrator, sdh::SubDofHandler) =
    setup_element_cache(subintegrator_for_subdomain(element_model.subintegrators, sdh), sdh)
setup_boundary_cache(element_model::AnyMultiDomainIntegrator, sdh::SubDofHandler) =
    setup_boundary_cache(subintegrator_for_subdomain(element_model.subintegrators, sdh), sdh)

"""
    subintegrators_per_subdomain(integrator, dh) -> Vector

The sub-integrator claiming each subdomain of `dh`, in `dh.subdofhandlers`
order.
"""
subintegrators_per_subdomain(integrator::AnyMultiDomainIntegrator, dh::AbstractDofHandler) =
    [integrator.subintegrators[name] for name in resolve_subdomain_claims(integrator.subintegrators, dh)]

"""
    subintegrator_for_subdomain(subintegrators::Dict{String}, sdh::SubDofHandler)

The sub-integrator claiming `sdh`. Validates the whole `DofHandler`'s subdomain
mapping, so every setup hook reaches the same verdict.
"""
function subintegrator_for_subdomain(subintegrators::Dict{String}, sdh::SubDofHandler)
    dh = sdh.dh
    claims = resolve_subdomain_claims(subintegrators, dh)
    index = findfirst(candidate -> candidate === sdh, dh.subdofhandlers)
    index === nothing && throw(ArgumentError(
        "The given SubDofHandler does not belong to the DofHandler it names as its own."))
    return subintegrators[claims[index]]
end

"""
    resolve_subdomain_claims(subintegrators::Dict{String}, dh) -> Vector{String}

The declared name claiming each subdomain of `dh`, in `dh.subdofhandlers`
order. Exhaustive under [`FerriteOperators.debug_mode`](@ref), sampling
otherwise; the flag is a compile-time constant, so the unused form is not
reachable code.
"""
resolve_subdomain_claims(subintegrators::Dict{String}, dh::AbstractDofHandler) =
    resolve_subdomain_claims(subintegrators, dh, DEBUG ? Val(:full) : Val(:sample))

# `Val(:sample)` — production: one membership query per subdomain, so nothing
# scales with the cell count. `Val(:full)` — debug: two linear passes over a
# transient cell → owner array, which is not retained.
function resolve_subdomain_claims(subintegrators::Dict{String}, dh::AbstractDofHandler, mode::Union{Val{:full}, Val{:sample}})
    grid = get_grid(dh)
    cellsets = Ferrite.getcellsets(grid)
    declared = sort!(collect(keys(subintegrators)))

    for name in declared
        haskey(cellsets, name) || throw(ArgumentError(
            "$(repr(name)) is not a cellset of the grid. Declared subdomains must be " *
            "volumetric cellsets; the grid has $(sort!(collect(keys(cellsets))))."))
    end

    claims = Vector{String}(undef, length(dh.subdofhandlers))
    claimed = falses(length(declared))
    for (index, id) in enumerate(claiming_name_ids(dh, cellsets, declared, mode))
        claims[index] = declared[id]
        claimed[id] = true
    end

    unused = declared[.!claimed]
    isempty(unused) || throw(ArgumentError(
        "The declared names $unused claim no subdomain of this DofHandler. " *
        "A sub-integrator that never assembles is a setup error, not a no-op."))

    return claims
end

function claiming_name_ids(dh, cellsets, declared::Vector{String}, ::Val{:sample})
    return [sampled_name_id(sdh, cellsets, declared, index) for (index, sdh) in enumerate(dh.subdofhandlers)]
end

function claiming_name_ids(dh, cellsets, declared::Vector{String}, ::Val{:full})
    owner = fill_subdomain_owners(get_grid(dh), cellsets, declared)
    return [scanned_name_id(sdh, owner, declared, index) for (index, sdh) in enumerate(dh.subdofhandlers)]
end

# The production claim: the subdomain's first cell decides. Uniform ownership
# of the remaining cells is the caller's assumption, checked only in debug mode.
function sampled_name_id(sdh::SubDofHandler, cellsets, declared::Vector{String}, index::Int)
    isempty(sdh.cellset) && throw(ArgumentError(
        "Subdomain $index contains no cells, so no declared name can claim it."))
    probe = first(sdh.cellset)
    matches = findall(name -> probe in cellsets[name], declared)
    if isempty(matches)
        throw(ArgumentError(
            "Subdomain $index is owned by no declared name (its first cell, $probe, lies in " *
            "none of $declared). Every subdomain must be claimed by exactly one sub-integrator."))
    elseif length(matches) > 1
        throw(ArgumentError(
            "Subdomain $index is claimed by more than one declared name: its first cell, " *
            "$probe, lies in $(declared[matches]). Declared subdomains must be pairwise " *
            "disjoint, so a cell owned by two sub-integrators has no unambiguous term."))
    end
    return only(matches)
end

# Pass one of the debug form: cell -> index into `declared`, zero for unowned.
# The narrow eltype keeps the transient array small on the meshes this survives.
function fill_subdomain_owners(grid, cellsets, declared::Vector{String})
    T = length(declared) ≤ typemax(Int8) ? Int8 : Int32
    owner = zeros(T, getncells(grid))
    for (id, name) in enumerate(declared)
        for cell in cellsets[name]
            previous = owner[cell]
            previous == 0 || throw(ArgumentError(
                "Cell $cell lies in both declared cellsets $(repr(declared[previous])) and " *
                "$(repr(name)). Declared subdomains must be pairwise disjoint: a cell owned " *
                "by two sub-integrators has no unambiguous term."))
            owner[cell] = id
        end
    end
    return owner
end

# Pass two of the debug form: one scan, collecting distinct owners with the
# first cell exhibiting each, so every rejection can name a cell.
function scanned_name_id(sdh::SubDofHandler, owner::Vector, declared::Vector{String}, index::Int)
    seen = Int[]
    witness = Int[]
    unowned_count = 0
    first_unowned = 0
    for cell in sdh.cellset
        id = Int(owner[cell])
        if id == 0
            unowned_count += 1
            first_unowned == 0 && (first_unowned = cell)
        elseif !(id in seen)
            push!(seen, id)
            push!(witness, cell)
        end
    end

    if unowned_count > 0
        detail = isempty(seen) ? "no cell of it is owned by a declared name" :
            "its remaining cells lie in $(declared[seen]), so the declaration covers it only partially"
        throw(ArgumentError(
            "Subdomain $index has $unowned_count cell(s) owned by no declared name " *
            "(first: cell $first_unowned) — $detail. Every subdomain must be claimed by " *
            "exactly one sub-integrator."))
    elseif length(seen) > 1
        spans = join(("cell $(witness[i]) in $(repr(declared[seen[i]]))" for i in eachindex(seen)), ", ")
        throw(ArgumentError(
            "Subdomain $index spans several declared names ($spans). A subdomain assembles " *
            "one term, so its cells must all be owned by the same sub-integrator."))
    elseif isempty(seen)
        throw(ArgumentError(
            "Subdomain $index contains no cells, so no declared name can claim it."))
    end

    return only(seen)
end
