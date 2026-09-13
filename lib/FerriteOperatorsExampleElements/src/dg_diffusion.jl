@doc raw"""
    SIPGDiffusionIntegrator(D, η, qrc, fqrc, field_name)

The symmetric interior penalty (SIPG) discretization of ``-\nabla \cdot (D \nabla u)``
over a DISCONTINUOUS space (`DiscontinuousLagrange`), as one two-sided element: the
item is an INTERIOR FACET and the local system is the `2Nb × 2Nb` block over
`[celldofs(here); celldofs(there)]`.

```math
a(u,v) = \sum_K \int_K D \nabla u \cdot \nabla v \,\mathrm{d}x
  - \sum_F \int_F \{D \nabla u\} \cdot [\![v]\!]
  - \sum_F \int_F \{D \nabla v\} \cdot [\![u]\!]
  + \sum_F \int_F \sigma_F [\![u]\!] \cdot [\![v]\!] \,\mathrm{d}s
```

with Ferrite's conventions for the average ``\{w\} = (w^- + w^+)/2`` and the
vector-valued jump ``[\![v]\!] = v^- \boldsymbol{n}^- + v^+ \boldsymbol{n}^+``
(`shape_value_jump` is ``v^+ - v^-``, hence the `-getnormal(iv, qp)` factor
below). The second and third face terms are the consistency and the
adjoint-consistency term; dropping the third gives IIPG and flipping its sign
NIPG, neither of which this element implements.

**The penalty.** ``\sigma_F = \eta \frac{(p+1)(p+d)}{d} \frac{|F|}{\min(|K^-|,|K^+|)}``,
the standard SIPG scaling ``\propto p^2/h`` with ``h = \min(|K^-|,|K^+|)/|F|``
the smaller adjacent cell's measure per unit facet area. Coercivity needs the
penalty to grow at least that fast, and `η ≥ 1` is the user's safety factor
over the constant. Both measures are read off the values objects the item
already reinitialized, so the element needs no mesh metric of its own.

**The volume term rides the facet items.** An interior-penalty operator's
natural traversals are two — cells for the volume term, interior facets for the
face terms — and one operator resolves ONE traversal per sweep kind. The volume
term is therefore APPORTIONED: cell `K`'s volume block is added to the own-side
diagonal block of each item touching it, weighted `1 / n_interior_facets(K)`.
The weights of a cell sum to one, so this is exact, not an approximation, and it
is what lets the same element serve `FullAssembly` and
[`BlockRowAssembly`](@ref) unchanged. A cell with NO interior facet has no item
to carry its volume term, and `setup_element_cache` rejects such a subdomain by
name.

`D` is a constant scalar diffusivity, `qrc` the volume rule and `fqrc` the facet
rule (`FerriteOperators.FacetQuadratureRuleCollection`, which that package does
not export).

# Use

Assembling (the reference arm) — the interior-facet coupling is not in the
`DofHandler`'s cell pattern, so the specification declares it:

```julia
spec = StandardOperatorSpecification(; sparsity_entries = interior_facet_entries!)
op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = FullAssembly(spec)),
                    SIPGDiffusionIntegrator(1.0, 10.0, QuadratureRuleCollection(2),
                                            FerriteOperators.FacetQuadratureRuleCollection(2), :u),
                    dh)
```

Matrix-free, over the condensed block-row store:

```julia
op = setup_operator(AssemblyStrategy(MatrixFreeAction(; storage = BlockRowAssembly()),
                                     ColoredScheduling(), SequentialCPUDevice()),
                    integrator, dh)
```
"""
struct SIPGDiffusionIntegrator <: AbstractBilinearIntegrator
    D::Float64
    η::Float64
    qrc::QuadratureRuleCollection
    fqrc::FacetQuadratureRuleCollection
    field_name::Symbol
end

"""
The interior-facet item tables and the values objects the FILL reads, held as ONE
field of [`SIPGDiffusionElementCache`](@ref) so that one `nothing` states the whole
of what a device without host `InterfaceValues` does not get.

`facets[item, :]` is `(here cell, here facet, there cell, there facet)`,
`windows[item, :]` is `[celldofs(here); celldofs(there)]` — the item's dof window,
laid out once at setup so positioning an item is a row lookup and a refill builds
no topology — and `shares[cellid]` is the cell's interior-facet count, the
apportionment denominator.

`grid`, `facets`, `windows` and `shares` are read-only and shared between workers;
the three values objects and the two coordinate buffers are per worker.
"""
struct SIPGInteriorFacetFill{G, CV, IV, X}
    grid::G
    cv_here::CV
    cv_there::CV
    iv::IV
    coords_here::Vector{X}
    coords_there::Vector{X}
    facets::Matrix{Int32}
    windows::Matrix{Int}
    shares::Vector{Int32}
end

"""
The cache of [`SIPGDiffusionIntegrator`](@ref).

`fill` is the [`SIPGInteriorFacetFill`](@ref) bundle on the host and on a
host-resident device, a `Vector` of one per worker where the device batches, and
`nothing` on a real GPU — which has no device `InterfaceValues` to fill with, and
whose block-row store therefore arrives already filled from the host mirror.
That `nothing` is also what keeps this cache `isbits` for the block-row ACTION
kernel it rides into as `BlockRowAssemblyCache`'s inner cache.
"""
struct SIPGDiffusionElementCache{T, F} <: AbstractVolumetricElementCache
    D::T
    σ::T
    fill::F
end

element_value_type(::SIPGDiffusionElementCache{T}) where {T} = T

# The local system spans TWO cells.
allocate_element_matrix(::SIPGDiffusionElementCache{T}, sdh) where {T} =
    zeros(T, 2ndofs_per_cell(sdh), 2ndofs_per_cell(sdh))
allocate_element_unknown_vector(::SIPGDiffusionElementCache{T}, sdh) where {T} =
    zeros(T, 2ndofs_per_cell(sdh))
allocate_element_residual_vector(::SIPGDiffusionElementCache{T}, sdh) where {T} =
    zeros(T, 2ndofs_per_cell(sdh))

####################################
## The item: what a sweep positions on
####################################

"""
One item's dof window as a VIEW into [`SIPGInteriorFacetFill`](@ref)'s window
table — `[celldofs(here); celldofs(there)]`, which is what the assembler scatters
through and what the block-row condensation reads.
"""
struct InterfaceDofWindow{I <: Integer, M <: AbstractMatrix{I}} <: AbstractVector{I}
    windows::M
    item::Int
    n::Int
end
Base.size(w::InterfaceDofWindow) = (w.n,)
Base.IndexStyle(::Type{<:InterfaceDofWindow}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(w::InterfaceDofWindow, i::Int) = w.windows[w.item, i]

"""
    InteriorFacetCache

What a [`SIPGDiffusionIntegrator`](@ref) sweep positions on: ONE interior facet,
naming the two cells it separates and their local facet numbers. Both halves read
the tables [`SIPGInteriorFacetFill`](@ref) laid out once, so positioning is a row
index and nothing else.

This is the HOST shape, positioned IN PLACE — the sequential CPU device discards
what [`position_iterator`](@ref) returns and reads the object it handed in.
[`DeviceInteriorFacetCursor`](@ref) is the device shape, positioned by
construction; `Ferrite.cellid` answers the HERE cell on both, the representative
an item spanning two cells must name and the one [`BlockRowAssembly`](@ref)'s
condensation keys its slot on.
"""
mutable struct InteriorFacetCache{SDH, FT <: AbstractMatrix{Int32}, WT <: AbstractMatrix{Int}}
    const sdh::SDH
    const facets::FT
    const windows::WT
    item::Int
end

"""
    DeviceInteriorFacetCursor

[`InteriorFacetCache`](@ref)'s device-resident shape: the same tables, moved by
[`adapt_shared`](@ref), positioned by CONSTRUCTION.

It carries no `Adapt` rule, and that is a statement about this traversal: the fill
reads host `InterfaceValues`, so it launches only on a HOST-RESIDENT backend
(`KernelAbstractions.CPU()`), where `adapt` is the identity. A device cursor a
real GPU kernel runs over needs `Adapt.@adapt_structure`, so that its array
fields are converted at the launch boundary.
"""
struct DeviceInteriorFacetCursor{SDH, FT <: AbstractMatrix{Int32}, WT <: AbstractMatrix{Int}}
    sdh::SDH
    facets::FT
    windows::WT
    item::Int
end

const InteriorFacetItem = Union{InteriorFacetCache, DeviceInteriorFacetCursor}

"""
    interface_cells(it) -> (here, facet_here, there, facet_there)

The two cells the positioned interior-facet item separates and the local facet
number each of them sees it as.
"""
@inline interface_cells(it::InteriorFacetItem) =
    (Int(@inbounds it.facets[it.item, 1]), Int(@inbounds it.facets[it.item, 2]),
     Int(@inbounds it.facets[it.item, 3]), Int(@inbounds it.facets[it.item, 4]))

Ferrite.cellid(it::InteriorFacetItem) = Int(@inbounds it.facets[it.item, 1])
iterator_dofs(it::InteriorFacetItem) = InterfaceDofWindow(it.windows, it.item, size(it.windows, 2))
iterator_handler(it::InteriorFacetItem) = it.sdh

@inline position_iterator(it::InteriorFacetCache, item, flags::Ferrite.UpdateFlags) =
    (it.item = Int(item); it)
@inline position_iterator(it::DeviceInteriorFacetCursor, item, flags::Ferrite.UpdateFlags) =
    DeviceInteriorFacetCursor(it.sdh, it.facets, it.windows, Int(item))

# The tables are read-only and shared; only the item index is the worker's own.
duplicate_for_device(::AbstractCPUDevice, it::InteriorFacetCache) =
    InteriorFacetCache(it.sdh, it.facets, it.windows, it.item)
setup_device_instances(device::AbstractGPUDevice, it::DeviceInteriorFacetCursor, n::Int) =
    DeviceInteriorFacetCursor(it.sdh, adapt_shared(device, it.facets),
                              adapt_shared(device, it.windows), it.item)
device_worker_view(it::DeviceInteriorFacetCursor, worker) = it

"""
    InteriorFacetItems(facets)

The work-item provider of [`SIPGDiffusionIntegrator`](@ref): the interior facets
of one `SubDofHandler`, each counted once.

The [`ColoredScheduling`](@ref) partition is greedy over CELLS, not over dofs:
over a discontinuous space every dof belongs to exactly one cell, so two items
with no cell in common share no dof either, which is the promise
[`compute_partition`](@ref) demands.
"""
struct InteriorFacetItems{FT}
    facets::FT
end

compute_partition(::SequentialScheduling, p::InteriorFacetItems) = (collect(axes(p.facets, 1)),)

function compute_partition(::ColoredScheduling, p::InteriorFacetItems)
    colors  = Vector{Int}[]
    claimed = Set{Int}[]
    for item in axes(p.facets, 1)
        here, there = Int(p.facets[item, 1]), Int(p.facets[item, 3])
        c = findfirst(taken -> here ∉ taken && there ∉ taken, claimed)
        if c === nothing
            push!(colors, [item]); push!(claimed, Set{Int}((here, there)))
        else
            push!(colors[c], item); push!(claimed[c], here); push!(claimed[c], there)
        end
    end
    return colors
end

# Narrowed on the CACHE, per the spelling rule: the kind stays open, so the
# `QuadratureDataKind` fill of a `BlockRowAssembly()` operator and the plain
# assembling sweep resolve the same two-sided traversal.
assembly_iterator(kind, c::SIPGDiffusionElementCache, sdh) =
    InteriorFacetCache(sdh, c.fill.facets, c.fill.windows, 1)
# The tables come from the HOST cache, the handler the cursor reports from the
# device one; `setup_device_instances` above is what moves the tables.
device_assembly_iterator(kind, c::SIPGDiffusionElementCache, sdh, device_sdh) =
    DeviceInteriorFacetCursor(device_sdh, c.fill.facets, c.fill.windows, 1)
item_provider(kind, c::SIPGDiffusionElementCache, sdh) = InteriorFacetItems(c.fill.facets)

####################################
## Values
####################################

function reinit_values!(c::SIPGDiffusionElementCache, it::InteriorFacetItem)
    f = c.fill
    here, facet_here, there, facet_there = interface_cells(it)
    getcoordinates!(f.coords_here, f.grid, here)
    getcoordinates!(f.coords_there, f.grid, there)
    cell_here, cell_there = getcells(f.grid, here), getcells(f.grid, there)
    Ferrite.reinit!(f.cv_here, cell_here, f.coords_here)
    Ferrite.reinit!(f.cv_there, cell_there, f.coords_there)
    Ferrite.reinit!(f.iv, cell_here, f.coords_here, facet_here,
                    cell_there, f.coords_there, facet_there)
    return nothing
end

# Under `storage = BlockRowAssembly()` the ACTION positions a cell-with-neighbours
# cursor and visits no quadrature point at all; the two-sided kernels run only at
# FILL time, on the cursor above.
reinit_values!(::SIPGDiffusionElementCache, ::CellNeighbourCursor) = nothing

####################################
## The kernels
####################################

provides_analytic(::Type{<:SIPGDiffusionElementCache}, ::JacobianKind{:u}) = true

# No `element_matrix_symmetry` election, though the local system IS symmetric:
# the packed layout it selects belongs to `ElementAssemblyCache`, which a
# two-sided element cannot elect at all, and `BlockRowAssembly()`'s
# `(slot, 1+Nf, Nb, Nb)` store keeps full blocks.

function assemble_cell!(req::JacobianRequest{:u}, c::SIPGDiffusionElementCache, args::CellArgs)
    Kₑ = req.K
    f  = c.fill
    (; iv, cv_here, cv_there) = f
    D  = c.D
    nb = getnbasefunctions(cv_here)
    here, _, there, _ = interface_cells(args.cell)
    σ = c.σ * _measure(iv) / min(_measure(cv_here), _measure(cv_there))

    for qp in 1:getnquadpoints(iv)
        dΓ = getdetJdV(iv, qp)
        n  = getnormal(iv, qp)
        for i in 1:2nb
            jumpᵢ = shape_value_jump(iv, qp, i) * (-n)
            avgᵢ  = shape_gradient_average(iv, qp, i)
            for j in 1:2nb
                jumpⱼ = shape_value_jump(iv, qp, j) * (-n)
                avgⱼ  = shape_gradient_average(iv, qp, j)
                Kₑ[i, j] += (σ * (jumpᵢ ⋅ jumpⱼ) - D * (jumpᵢ ⋅ avgⱼ) - D * (avgᵢ ⋅ jumpⱼ)) * dΓ
            end
        end
    end
    _add_volume_block!(Kₑ, cv_here, D / f.shares[here], 0)
    _add_volume_block!(Kₑ, cv_there, D / f.shares[there], nb)
    return nothing
end

# The bilinear form induces a linear operator, so its residual is the element
# matrix acting on the element vector — mandatory, so the element composes into
# nonlinear operators and AD-based sensitivities.
function assemble_cell!(req::ResidualRequest, c::SIPGDiffusionElementCache, args::CellArgs)
    f  = c.fill
    (; iv, cv_here, cv_there) = f
    D  = c.D
    uₑ = args.states.u
    nb = getnbasefunctions(cv_here)
    here, _, there, _ = interface_cells(args.cell)
    σ = c.σ * _measure(iv) / min(_measure(cv_here), _measure(cv_there))

    for qp in 1:getnquadpoints(iv)
        dΓ = getdetJdV(iv, qp)
        n  = getnormal(iv, qp)
        jumpᵤ = function_value_jump(iv, qp, uₑ) * (-n)
        avgᵤ  = function_gradient_average(iv, qp, uₑ)
        for i in 1:2nb
            jumpᵢ = shape_value_jump(iv, qp, i) * (-n)
            avgᵢ  = shape_gradient_average(iv, qp, i)
            req.r[i] += (σ * (jumpᵢ ⋅ jumpᵤ) - D * (jumpᵢ ⋅ avgᵤ) - D * (avgᵢ ⋅ jumpᵤ)) * dΓ
        end
    end
    _add_volume_residual!(req.r, cv_here, uₑ, D / f.shares[here], 0, nb)
    _add_volume_residual!(req.r, cv_there, uₑ, D / f.shares[there], nb, nb)
    return nothing
end

function _add_volume_block!(Kₑ, cv, w, offset)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        for i in 1:getnbasefunctions(cv)
            ∇Nᵢ = shape_gradient(cv, qp, i)
            for j in 1:getnbasefunctions(cv)
                Kₑ[offset + i, offset + j] += w * (shape_gradient(cv, qp, j) ⋅ ∇Nᵢ) * dΩ
            end
        end
    end
    return nothing
end

function _add_volume_residual!(r, cv, uₑ, w, offset, nb)
    u_side = @view uₑ[(offset + 1):(offset + nb)]
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, u_side)
        for i in 1:nb
            r[offset + i] += w * (∇u ⋅ shape_gradient(cv, qp, i)) * dΩ
        end
    end
    return nothing
end

function _measure(values)
    m = zero(getdetJdV(values, 1))
    for qp in 1:getnquadpoints(values)
        m += getdetJdV(values, qp)
    end
    return m
end

####################################
## Setup
####################################

function setup_element_cache(m::SIPGDiffusionIntegrator, sdh::SubDofHandler)
    T      = element_value_type(m.qrc)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    qr     = getquadraturerule(m.qrc, sdh)
    fqr    = getquadraturerule(m.fqrc, sdh)
    iv = InterfaceValues(FacetValues(T, fqr, ip, ip_geo),
                         FacetValues(T, deepcopy(fqr), ip, ip_geo))
    fill = _interior_facet_fill(sdh, CellValues(T, qr, ip, ip_geo), CellValues(T, qr, ip, ip_geo), iv)
    p, d = Ferrite.getorder(ip), Ferrite.getrefdim(ip)
    return SIPGDiffusionElementCache(T(m.D), T(m.η * (p + 1) * (p + d) / d), fill)
end

function _interior_facet_fill(sdh::SubDofHandler, cv_here, cv_there, iv)
    grid   = Ferrite.get_grid(sdh.dh)
    top    = Ferrite.ExclusiveTopology(grid)
    nb     = ndofs_per_cell(sdh)
    inside = falses(getncells(grid))
    for cellid in sdh.cellset
        inside[cellid] = true
    end
    shares = zeros(Int32, getncells(grid))
    items  = NTuple{4, Int32}[]
    for cellid in sdh.cellset, facet in 1:Ferrite.nfacets(getcells(grid, cellid))
        found = Ferrite.getneighborhood(top, grid, FacetIndex(cellid, facet))
        length(found) <= 1 || throw(ArgumentError(
            "Facet $facet of cell $cellid has $(length(found)) neighbours. " *
            "`SIPGDiffusionIntegrator` keeps one item per interior facet, so it covers " *
            "conforming meshes only."))
        (isempty(found) || !inside[first(found)[1]]) && continue
        other, facet_other = first(found)
        shares[cellid] += one(Int32)
        cellid < other && push!(items, (cellid, facet, other, facet_other))
    end
    for cellid in sdh.cellset
        shares[cellid] > 0 || throw(ArgumentError(
            "Cell $cellid has no interior facet inside this subdomain. " *
            "`SIPGDiffusionIntegrator` apportions each cell's VOLUME term across the interior " *
            "facets touching it, so an isolated cell has no item to carry it."))
    end

    facets  = zeros(Int32, length(items), 4)
    windows = zeros(Int, length(items), 2nb)
    for (item, (here, facet_here, there, facet_there)) in enumerate(items)
        facets[item, :] .= (here, facet_here, there, facet_there)
        celldofs!(view(windows, item, 1:nb), sdh.dh, Int(here))
        celldofs!(view(windows, item, (nb + 1):(2nb)), sdh.dh, Int(there))
    end
    X = Vec{Ferrite.getspatialdim(grid), Ferrite.get_coordinate_eltype(grid)}
    nnodes = Ferrite.nnodes_per_cell(grid, first(sdh.cellset))
    return SIPGInteriorFacetFill(grid, cv_here, cv_there, iv,
                                 zeros(X, nnodes), zeros(X, nnodes), facets, windows, shares)
end

"""
    interior_facet_entries!(sp, dh)

The sparsity entries an interior-facet traversal couples and the `DofHandler`'s
cell pattern does not carry: both off-diagonal blocks of every interior facet's
`2Nb × 2Nb` system. Pass it as `StandardOperatorSpecification`'s
`sparsity_entries` to assemble a [`SIPGDiffusionIntegrator`](@ref).

It derives the interior facets from the grid's topology, the same set
`setup_element_cache` derives the items from, and rebuilds that topology per
call — the specification re-runs this for every matrix it allocates.
"""
function interior_facet_entries!(sp, dh)
    grid = Ferrite.get_grid(dh)
    top  = Ferrite.ExclusiveTopology(grid)
    for cellid in 1:getncells(grid), facet in 1:Ferrite.nfacets(getcells(grid, cellid))
        found = Ferrite.getneighborhood(top, grid, FacetIndex(cellid, facet))
        isempty(found) && continue
        other = first(found)[1]
        cellid < other || continue
        for i in celldofs(dh, cellid), j in celldofs(dh, other)
            Ferrite.add_entry!(sp, i, j)
            Ferrite.add_entry!(sp, j, i)
        end
    end
    return nothing
end

####################################
## Device layout
####################################

duplicate_for_device(::AbstractCPUDevice, c::SIPGDiffusionElementCache) =
    SIPGDiffusionElementCache(c.D, c.σ, _duplicate_fill(c.fill))

# The fill reads HOST `InterfaceValues`, and `update_operator!` drives the
# block-row fill through the engine on exactly the devices that are host-resident
# (`_engine_driven_fill`, `KernelAbstractionsDevice(KA.CPU())` among them). The
# values therefore travel to those and to no others: a real GPU fills through the
# host mirror instead, and a host container would in any case not survive the
# ACTION launch this cache rides into.
setup_device_instances(device::AbstractGPUDevice, c::SIPGDiffusionElementCache, n) =
    SIPGDiffusionElementCache(c.D, c.σ,
        _engine_driven_fill(device) ? [_duplicate_fill(c.fill) for _ in 1:n] : nothing)
device_worker_view(c::SIPGDiffusionElementCache, worker) =
    SIPGDiffusionElementCache(c.D, c.σ, _worker_fill(c.fill, worker))

_worker_fill(fill::AbstractVector, worker) = fill[worker]
_worker_fill(::Nothing, worker) = nothing

# Read-only between workers: `grid` and the three tables are shared, the values
# objects and coordinate buffers are the worker's own.
_duplicate_fill(f::SIPGInteriorFacetFill) = SIPGInteriorFacetFill(
    f.grid, copy(f.cv_here), copy(f.cv_there), copy(f.iv),
    copy(f.coords_here), copy(f.coords_there), f.facets, f.windows, f.shares)
