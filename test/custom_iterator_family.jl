# The DEFINITIONS half of `test/test_custom_iterator.jl`'s demo — types,
# protocol methods and the reference assembler — `include`d both by that file's
# testsets and by `test/gpu/`'s CUDA arm, so the same family is exercised on
# both backends from one source.

using FerriteOperators
using LinearAlgebra

import Adapt
import FerriteOperators: assembly_iterator, item_provider, iterator_dofs, iterator_handler,
    compute_partition, duplicate_for_device, setup_element_cache, reinit_values!,
    allocate_element_matrix, allocate_element_unknown_vector, allocate_element_residual_vector,
    provides_analytic, assemble_cell!, apply_element_action!, item_families, setup_family_caches,
    device_assembly_iterator, position_iterator, setup_device_instances, device_worker_view

####################################
## The physics, chosen so the reference is closed form
####################################
# A weighted jump penalty across pairs of neighbouring cells,
#
#     a(u, v) = η Σ_pairs (ū_L − α ū_R)(v̄_L − α v̄_R),   ū_K = (1/N) Σ_{i ∈ K} u_i
#
# so one item's element matrix is the rank-one block `Kₑ = η g gᵀ` over the
# concatenated dof window `[celldofs(L); celldofs(R)]`, with
# `g = [+1/N … | −α/N …]`. There is no quadrature and no `InterfaceValues`: the
# demo tests the ITERATION protocol, not Ferrite's DG machinery.
#
# `α ≠ 1` is load bearing: with α = 1 the block is invariant under swapping the
# two sides, so a traversal that mixed up left and right would still assemble
# the right answer.

const JUMP_α = 2.0
const JUMP_η = 3.5

struct JumpPenaltyIntegrator <: AbstractBilinearIntegrator
    η::Float64
    α::Float64
    pairs::Vector{Tuple{Int, Int}}
end

# The item set rides the ELEMENT CACHE, which both protocol seams are keyed on,
# so the iterator and the provider read one list without a second channel.
# `g` and `pairs` are TYPE PARAMETERS, not hard-typed `Vector`s: the device
# instance below moves `g` to the backend's array type and drops `pairs`
# entirely, and a concretely-typed field would admit neither.
struct JumpPenaltyCache{G, P} <: AbstractVolumetricElementCache
    η::Float64
    n::Int                       # dofs per cell; the local system is 2n
    g::G
    pairs::P
end

function setup_element_cache(m::JumpPenaltyIntegrator, sdh::SubDofHandler)
    n = ndofs_per_cell(sdh)
    g = [fill(1 / n, n); fill(-m.α / n, n)]
    return JumpPenaltyCache(m.η, n, g, m.pairs)
end

# Read-only between workers: sharing it is what a per-worker copy would produce.
duplicate_for_device(device, c::JumpPenaltyCache) = c

# `g` is read by every worker, so it moves to the device once. `pairs` only
# built the iterator's `lefts`/`rights` lists (`device_assembly_iterator`
# below) and the kernel never reads it — but isbits is a requirement of
# crossing the launch boundary regardless of whether a field is read, so the
# device instance drops it rather than shipping it unconverted.
setup_device_instances(device::KernelAbstractionsDevice, c::JumpPenaltyCache, n) =
    JumpPenaltyCache(c.η, c.n, Adapt.adapt(device.backend, c.g), nothing)
device_worker_view(c::JumpPenaltyCache, worker) = c

# The local system spans TWO cells, so every element-local buffer is 2n.
allocate_element_matrix(c::JumpPenaltyCache, sdh)          = zeros(2c.n, 2c.n)
allocate_element_unknown_vector(c::JumpPenaltyCache, sdh)  = zeros(2c.n)
allocate_element_residual_vector(c::JumpPenaltyCache, sdh) = zeros(2c.n)

# How many items the sweep actually visited — what catches a traversal silently
# falling back to the cell family.
const VISITED = Threads.Atomic{Int}(0)

provides_analytic(::Type{<:JumpPenaltyCache}, ::JacobianKind{:u}) = true

# `pairs` is what tells the two apart: `duplicate_for_device` (SequentialCPUDevice,
# PolyesterDevice) keeps it, `setup_device_instances` (KernelAbstractionsDevice,
# KA.CPU() as much as CUDA — one compiled kernel body for both backends) drops
# it to `nothing`. Only the HOST-only overload may touch a host global.
function assemble_cell!(req::JacobianRequest{:u}, c::JumpPenaltyCache{<:Any, <:Vector}, args::CellArgs)
    Threads.atomic_add!(VISITED, 1)
    _fill_jump_block!(req.K, c)
    return nothing
end
function assemble_cell!(req::JacobianRequest{:u}, c::JumpPenaltyCache{<:Any, Nothing}, args::CellArgs)
    _fill_jump_block!(req.K, c)
    return nothing
end
function _fill_jump_block!(K, c::JumpPenaltyCache)
    g = c.g
    for j in eachindex(g), i in eachindex(g)
        K[i, j] += c.η * g[i] * g[j]
    end
    return nothing
end

function assemble_cell!(req::ResidualRequest, c::JumpPenaltyCache, args::CellArgs)
    g = c.g
    s = c.η * dot(g, args.states.u)
    for i in eachindex(g)
        req.r[i] += s * g[i]
    end
    return nothing
end

# The matrix-free action of the same rank-one block, `yₑ += η g (g·uₑ)`: the
# ELEMENT axis, and the only method the `MatrixFreeAction` arm below adds.
function apply_element_action!(yₑ, c::JumpPenaltyCache, uₑ, args::CellArgs)
    g = c.g
    # An element kernel that runs on a device path must stay generic: a BLAS
    # fast-path like `dot` is not GPU-compilable, so the reduction is a loop.
    acc = zero(eltype(g))
    for i in eachindex(g)
        acc += g[i] * uₑ[i]
    end
    s = c.η * acc
    for i in eachindex(g)
        yₑ[i] += s * g[i]
    end
    return nothing
end

####################################
## The iterator: what a sweep POSITIONS on
####################################

# Positioned in place on a pair index, staging both cells' geometry and the
# concatenated dof window. `cellid` answers with the LEFT cell — the
# representative an iterator whose item is a set of cells has to name.
mutable struct PairCache{G, SDH, X}
    const grid::G
    const sdh::SDH
    const pairs::Vector{Tuple{Int, Int}}
    left::Int
    right::Int
    const dofs::Vector{Int}
    const coords_left::Vector{X}
    const coords_right::Vector{X}
end

function PairCache(sdh::SubDofHandler, prs::Vector{Tuple{Int, Int}})
    grid = Ferrite.get_grid(sdh.dh)
    n = ndofs_per_cell(sdh)
    c₀ = first(prs)[1]
    return PairCache(grid, sdh, prs, -1, -1, zeros(Int, 2n),
                     getcoordinates(grid, c₀), getcoordinates(grid, c₀))
end

function Ferrite.reinit!(pc::PairCache, item::Int)
    pc.left, pc.right = pc.pairs[item]
    n = ndofs_per_cell(pc.sdh)
    celldofs!(view(pc.dofs, 1:n), pc.sdh, pc.left)
    celldofs!(view(pc.dofs, (n + 1):(2n)), pc.sdh, pc.right)
    getcoordinates!(pc.coords_left, pc.grid, pc.left)
    getcoordinates!(pc.coords_right, pc.grid, pc.right)
    return pc
end

# The three the FRAMEWORK requires, and no more.
Ferrite.cellid(pc::PairCache)   = pc.left
iterator_dofs(pc::PairCache)    = pc.dofs
iterator_handler(pc::PairCache) = pc.sdh

# An independent copy per threaded worker; reached through `iterator_handler`,
# never through a `.dh` field this iterator does not have.
duplicate_for_device(::AbstractCPUDevice, pc::PairCache) = PairCache(pc.sdh, pc.pairs)

####################################
## The DEVICE iterator
####################################

# The dof window over the device handler's flat `cell_dofs`, positioned by
# construction — `DeviceCellCursor`'s `DeviceCellDofs` for a single cell,
# concatenated over two.
struct PairDofs{I <: Integer, V <: AbstractVector{I}} <: AbstractVector{I}
    cell_dofs::V
    lbase::Int
    rbase::Int
    n::Int
end
Base.size(d::PairDofs) = (2d.n,)
Base.IndexStyle(::Type{<:PairDofs}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(d::PairDofs, i::Int) =
    i <= d.n ? d.cell_dofs[d.lbase + i] : d.cell_dofs[d.rbase + (i - d.n)]

# Positioned by CONSTRUCTION, like `DeviceCellCursor`: `lefts`/`rights` are the
# pair list moved to the device ONCE and shared read-only, and `left`/`right`
# are this item's own cell ids, looked up by pair index at `position_iterator`
# time.
struct DevicePairCursor{SDH, V <: AbstractVector{Int}}
    sdh::SDH
    lefts::V
    rights::V
    left::Int
    right::Int
    n::Int
end

Adapt.@adapt_structure DevicePairCursor

Ferrite.cellid(c::DevicePairCursor)   = c.left
iterator_dofs(c::DevicePairCursor)    = PairDofs(c.sdh.cell_dofs,
    Int(@inbounds c.sdh.cell_dofs_offset[c.left]) - 1,
    Int(@inbounds c.sdh.cell_dofs_offset[c.right]) - 1, c.n)
iterator_handler(c::DevicePairCursor) = c.sdh

position_iterator(c::DevicePairCursor, item, flags) =
    DevicePairCursor(c.sdh, c.lefts, c.rights,
        @inbounds(c.lefts[Int(item)]), @inbounds(c.rights[Int(item)]), c.n)

# `lefts`/`rights` are shared, not per-worker, so there is nothing to slice.
device_worker_view(c::DevicePairCursor, worker) = c

# The HOST `sdh` builds the pair list once. This constructs the SHAPE only;
# `setup_device_instances` below moves `lefts`/`rights` onto the device.
device_assembly_iterator(kind, c::JumpPenaltyCache, sdh, device_sdh) =
    DevicePairCursor(device_sdh, first.(c.pairs), last.(c.pairs), -1, -1, ndofs_per_cell(sdh))

setup_device_instances(device::KernelAbstractionsDevice, c::DevicePairCursor, n) =
    DevicePairCursor(c.sdh, Adapt.adapt(device.backend, c.lefts),
        Adapt.adapt(device.backend, c.rights), c.left, c.right, c.n)

# Annotated on the DEVICE iterator type too — the same no-op, but a SEPARATE
# method: `reinit_values!` dispatches on whichever iterator positioned
# `args.cell`, and the device sweep positions a `DevicePairCursor`.
reinit_values!(::JumpPenaltyCache, ::DevicePairCursor) = nothing

####################################
## The provider: what the ITEMS are
####################################

struct PairItems{SDH}
    sdh::SDH
    pairs::Vector{Tuple{Int, Int}}
end

# One chunk; the atomic scatter resolves the cells consecutive pairs share.
compute_partition(::SequentialScheduling, p::PairItems) = (collect(eachindex(p.pairs)),)

# The provider's own promise, which the framework cannot check: no two items of
# one chunk share a SCATTER DOF. The colouring is derived from the dof windows
# rather than from cell-disjointness, because the promise is about dofs — over a
# continuous space two cell-disjoint items still share a common facet's dofs.
function compute_partition(::ColoredScheduling, p::PairItems)
    colors = Vector{Int}[]
    claimed = Set{Int}[]
    for (i, (l, r)) in pairs(p.pairs)
        d = Set{Int}(vcat(celldofs(p.sdh.dh, l), celldofs(p.sdh.dh, r)))
        c = findfirst(taken -> isdisjoint(taken, d), claimed)
        if c === nothing
            push!(colors, [i]); push!(claimed, d)
        else
            push!(colors[c], i); union!(claimed[c], d)
        end
    end
    return colors
end

####################################
## THE WHOLE EXTENSION: two methods
####################################

assembly_iterator(kind, c::JumpPenaltyCache, sdh) = PairCache(sdh, c.pairs)
item_provider(kind, c::JumpPenaltyCache, sdh)     = PairItems(sdh, c.pairs)

####################################
## THE REGISTRATION: the family those two belong to
####################################
# A family marker plus one `setup_family_caches` method puts this traversal on
# the engine's family list. The declaration REPLACES the derived default
# `(CellFamily(),)`, so the operator below carries no cell family at all — an
# engine that still hand-appended a cell setup would assemble the cells too, and
# every count below would be 12 rather than 9.
#
# The method delegates to the CELL family's own body, this family's subdomain
# caches being cell-shaped. Reaching the shipped body from test code IS the
# dogfood claim.

struct JumpPairFamily end

# How many times the engine reached THIS family's setup. One per `setup_operator`.
const FAMILY_SETUPS = Ref(0)

item_families(::JumpPenaltyIntegrator, dh) = (JumpPairFamily(),)

function setup_family_caches(::JumpPairFamily, strategy, integrator, dh, shared)
    FAMILY_SETUPS[] += 1
    return setup_family_caches(CellFamily(), strategy, integrator, dh, shared)
end

# Annotated on the ITERATOR type — the spelling setup validation must accept,
# since it probes the RESOLVED iterator type. A no-op: this family stages
# nothing per sweep.
reinit_values!(::JumpPenaltyCache, ::PairCache) = nothing

# The third iterator this cache is validated against: under
# `storage = BlockRowAssembly()` the ACTION positions a cell-with-neighbours
# cursor, and the two-sided kernels run only at FILL time, on the pair iterator
# above. Also a no-op, and for the same reason.
reinit_values!(::JumpPenaltyCache, ::CellNeighbourCursor) = nothing

####################################
## The testbed
####################################

# Horizontal neighbours of a structured (nx, ny) quadrilateral grid: cell `c`
# and `c+1` unless `c` sits on the right edge. For (4, 3): 9 items against 12
# cells — the counts differ, which is what the item-count assertion rests on.
horizontal_pairs(nx, ny) = [(c, c + 1) for c in 1:(nx * ny) if mod(c, nx) != 0]

# The entries this item family couples and the DofHandler's cell pattern does
# not carry: the two off-diagonal blocks of every pair's 2n × 2n system.
pair_sparsity(prs) = function (sp, dh)
    for (l, r) in prs
        dl, dr = celldofs(dh, l), celldofs(dh, r)
        for i in dl, j in dr
            Ferrite.add_entry!(sp, i, j)
            Ferrite.add_entry!(sp, j, i)
        end
    end
    return nothing
end

# The space is DISCONTINUOUS, and structurally so: a two-sided item's local
# system is indexed by `[celldofs(L); celldofs(R)]`, which Ferrite's assembler
# requires to be DUPLICATE FREE. Over a continuous space two face-neighbours
# share the dofs on their common facet, the window repeats them, and the scatter
# fails. A jump penalty across an interface is a discontinuous-space term anyway.
function jump_testbed(dims = (4, 3))
    grid = generate_grid(Quadrilateral, dims)
    dh   = DofHandler(grid)
    add!(dh, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
    close!(dh)
    prs  = horizontal_pairs(dims...)
    spec = StandardOperatorSpecification(; sparsity_entries = pair_sparsity(prs))
    return (; grid, dh, prs,
            integrator = JumpPenaltyIntegrator(JUMP_η, JUMP_α, prs),
            spec)
end

strategy_for(spec, device, scheduling = SequentialScheduling()) =
    AssemblyStrategy(device; form = FullAssembly(spec), scheduling)

# The reference: a plain loop over the pair list and `celldofs(dh, c)`, sharing
# no code with the engine.
function reference_matrix(dh, prs, n)
    g = [fill(1 / n, n); fill(-JUMP_α / n, n)]
    A = zeros(ndofs(dh), ndofs(dh))
    for (l, r) in prs
        d = [celldofs(dh, l); celldofs(dh, r)]
        for j in eachindex(d), i in eachindex(d)
            A[d[i], d[j]] += JUMP_η * g[i] * g[j]
        end
    end
    return A
end

sweep!(op) = (Threads.atomic_xchg!(VISITED, 0); update_operator!(op, nothing); op.A)
