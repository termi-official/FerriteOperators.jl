# A work-item iterator written entirely OUTSIDE the package: a two-sided
# interface traversal whose item is a PAIR of neighbouring cells, positioned by
# an iterator of its own and enumerated by a provider of its own. Nothing in
# `src/` knows this family exists; it reaches the engine through the two
# protocol seams and the three required accessors and nothing else.
#
# PUBLIC-SURFACE COUNT. Every FerriteOperators name this file reaches for is
# EXPORTED — `using FerriteOperators` brings all of them into scope, and the
# `import` below exists only because Julia requires it to ADD METHODS to a
# function owned by another module. FerriteOperators names reachable only
# through the module (i.e. unexported): 0. Two of Ferrite's own unexported
# names are used (`Ferrite.get_grid`, `Ferrite.add_entry!`), neither of them
# this package's surface.
#
# What the family actually costs its author, measured on this file:
#   2 protocol methods   — `assembly_iterator`, `item_provider`
#   3 required accessors — `Ferrite.cellid`, `iterator_dofs`, `iterator_handler`
#   2 partition methods  — `compute_partition` for the two scheduling policies
#   1 duplication method — `duplicate_for_device`, for the threaded CPU route
#   1 declaration        — `sparsity_entries`, the entries a two-sided item
#                          couples and the cell pattern does not carry
#   1 family marker      — `JumpPairFamily`, the type the registration dispatches on
#   2 registration methods — `item_families`, `setup_family_caches`
# everything else in the file is the ELEMENT axis (already open before this
# round) and the assertions.
#
# The last three are what REGISTRATION costs, and they are optional: a family
# whose items ride the cell route needs only the nine above, because the default
# `item_families` already registers `CellFamily()` and the two protocol seams
# narrow what that family positions on and enumerates. They are written here
# because registering is what proves the seam — see the registration testset.
#
# THE DEVICE HALF (optional; `KernelAbstractionsDevice(KA.CPU())`). A separate,
# device-resident iterator (`DevicePairCursor`) plus 4 more methods —
# `device_assembly_iterator`, `setup_device_instances` × 2 (the iterator and
# the element cache), `device_worker_view` × 2 — and one `reinit_values!`
# overload for the device iterator type. Not counted against the CPU-only tally
# above: the family opens on the host with the two protocol methods alone, and
# a downstream author who never targets `KernelAbstractionsDevice` writes none
# of this.

using FerriteOperators
using Test
using SparseArrays
using LinearAlgebra
using Polyester
# FerriteKAExt — the device handler, `distribute_to_workers` and every `Adapt`
# rule the device kernel builds on — is triggered by these four together, not
# by KernelAbstractions alone (matches `test/test_ka_device.jl`,
# `test/test_matrix_free.jl`). No CUDA: the device arm below targets
# `KernelAbstractionsDevice(KA.CPU())` only.
import Adapt, GPUArrays, GPUArraysCore
import KernelAbstractions as KA

import FerriteOperators: assembly_iterator, item_provider, iterator_dofs, iterator_handler,
    compute_partition, duplicate_for_device, setup_element_cache, reinit_values!,
    allocate_element_matrix, allocate_element_unknown_vector, allocate_element_residual_vector,
    provides_analytic, assemble_cell!, item_families, setup_family_caches,
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
# `α ≠ 1` is load bearing. With α = 1 the block is invariant under swapping the
# two sides, so a traversal that mixed up left and right would still assemble
# the right answer; α = 2 makes the block side-asymmetric and the assertion
# below real.

const JUMP_α = 2.0
const JUMP_η = 3.5

struct JumpPenaltyIntegrator <: AbstractBilinearIntegrator
    η::Float64
    α::Float64
    pairs::Vector{Tuple{Int, Int}}
end

# The item set rides the ELEMENT CACHE, which is what both protocol seams are
# keyed on — so the iterator and the provider read the same list without a
# second channel through `setup_operator`.
struct JumpPenaltyCache <: AbstractVolumetricElementCache
    η::Float64
    n::Int                       # dofs per cell; the local system is 2n
    g::Vector{Float64}
    pairs::Vector{Tuple{Int, Int}}
end

function setup_element_cache(m::JumpPenaltyIntegrator, sdh::SubDofHandler)
    n = ndofs_per_cell(sdh)
    g = [fill(1 / n, n); fill(-m.α / n, n)]
    return JumpPenaltyCache(m.η, n, g, m.pairs)
end

# Read-only between workers: sharing it is what a per-worker copy would produce.
duplicate_for_device(device, c::JumpPenaltyCache) = c

# The KA device's own pair — `g` is read by every worker, so it moves once;
# `pairs` is a HOST list the device kernel never touches (it exists only to
# build the iterators), so it rides along unadapted.
setup_device_instances(device::KernelAbstractionsDevice, c::JumpPenaltyCache, n) =
    JumpPenaltyCache(c.η, c.n, Adapt.adapt(device.backend, c.g), c.pairs)
device_worker_view(c::JumpPenaltyCache, worker) = c

# The local system spans TWO cells, so every element-local buffer is 2n.
allocate_element_matrix(c::JumpPenaltyCache, sdh)          = zeros(2c.n, 2c.n)
allocate_element_unknown_vector(c::JumpPenaltyCache, sdh)  = zeros(2c.n)
allocate_element_residual_vector(c::JumpPenaltyCache, sdh) = zeros(2c.n)

# How many items the sweep actually visited — the assertion that catches a
# traversal silently falling back to the cell family (gate failure mode 1).
const VISITED = Threads.Atomic{Int}(0)

provides_analytic(::Type{<:JumpPenaltyCache}, ::JacobianKind{:u}) = true

function assemble_cell!(req::JacobianRequest{:u}, c::JumpPenaltyCache, args::CellArgs)
    Threads.atomic_add!(VISITED, 1)
    g = c.g
    for j in eachindex(g), i in eachindex(g)
        req.K[i, j] += c.η * g[i] * g[j]
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

####################################
## The iterator: what a sweep POSITIONS on
####################################

# Positioned in place on a pair index, staging both cells' geometry and the
# concatenated dof window. `cellid` answers with the LEFT cell — the
# representative this family names, as an iterator whose item is a set of cells
# must.
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
## The DEVICE iterator: A7, landed (was blocked on the missing `_batch_iterator`
## fallback — S2's gate report §3; S3 adds it and this is the reproduction).
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
# pair list moved to the device ONCE (shared, read-only — every worker reads
# the same array), and `left`/`right` are this item's own cell ids, looked up
# by pair index at `position_iterator` time.
struct DevicePairCursor{SDH, V <: AbstractVector{Int}}
    sdh::SDH
    lefts::V
    rights::V
    left::Int
    right::Int
    n::Int
end

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

# The HOST `sdh` builds the pair list once; `setup_device_instances` below is
# what actually moves `lefts`/`rights` onto the device — this constructs the
# SHAPE only, mirroring `DeviceCellCursor`'s `coords = nothing` placeholder.
device_assembly_iterator(kind, c::JumpPenaltyCache, sdh, device_sdh) =
    DevicePairCursor(device_sdh, first.(c.pairs), last.(c.pairs), -1, -1, ndofs_per_cell(sdh))

setup_device_instances(device::KernelAbstractionsDevice, c::DevicePairCursor, n) =
    DevicePairCursor(c.sdh, Adapt.adapt(device.backend, c.lefts),
        Adapt.adapt(device.backend, c.rights), c.left, c.right, c.n)

# Annotated on the DEVICE iterator type too — the same no-op as the host one,
# but a SEPARATE method: `reinit_values!(cache, cell, kind)`'s generic fallback
# dispatches on whichever iterator positioned `args.cell`, and the device sweep
# positions a `DevicePairCursor`, never a `PairCache`.
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

# The provider's own promise, and the framework cannot check it: no two items of
# one chunk share a SCATTER DOF. The colouring is derived from the dof windows
# themselves rather than from cell-disjointness, because the promise is about
# dofs — over a continuous space two cell-disjoint items still share the dofs of
# a common facet, and only the dof windows would say so.
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
# A family marker plus one `setup_family_caches` method is what puts this
# traversal on the engine's family list. The declaration REPLACES the derived
# default `(CellFamily(),)`, so the operator below carries the pair family and
# no cell family at all — an engine that still hand-appended a cell setup would
# assemble the cells too, and every count below would be 12 rather than 9.
#
# The method delegates to the CELL family's own body: the subdomain caches this
# family needs are cell-shaped (one per subdomain, over the iterator and the
# provider the two seams above resolve to), and reaching the shipped body from
# test code IS the dogfood claim — no engine-registered family has a privileged
# setup path.

struct JumpPairFamily end

# How many times the engine reached THIS family's setup. One per `setup_operator`.
const FAMILY_SETUPS = Ref(0)

item_families(::JumpPenaltyIntegrator, dh) = (JumpPairFamily(),)

function setup_family_caches(::JumpPairFamily, strategy, integrator, dh, shared)
    FAMILY_SETUPS[] += 1
    return setup_family_caches(CellFamily(), strategy, integrator, dh, shared)
end

# Annotated on the ITERATOR type, which is the natural spelling and the one
# setup validation must accept (it probes the RESOLVED iterator type). This
# family stages nothing per sweep, so it is a no-op.
reinit_values!(::JumpPenaltyCache, ::PairCache) = nothing

####################################
## The testbed
####################################

# Horizontal neighbours of a structured (nx, ny) quadrilateral grid, derived
# arithmetically: cell `c` and `c+1` are neighbours unless `c` sits on the right
# edge. For (4, 3): 9 items against 12 cells — the counts differ, which is what
# the item-count assertion rests on.
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

# The space is DISCONTINUOUS, and that is structural rather than decorative:
# the local system of a two-sided item is indexed by `[celldofs(L); celldofs(R)]`,
# and Ferrite's assembler requires that window to be DUPLICATE FREE. Over a
# continuous space two face-neighbours share the dofs on their common facet, the
# window repeats them, and the scatter fails. A jump penalty across an interface
# is a discontinuous-space term anyway, so the demo is the shape the family
# really has.
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

@testset "custom item iterator (two-sided interface family)" begin
    tb = jump_testbed()
    n  = ndofs_per_cell(first(tb.dh.subdofhandlers))

    # A8 / C4b — setup validation probes the RESOLVED iterator type. The cache
    # annotates `reinit_values!` against `PairCache` and against nothing else,
    # so a probe hard-wired to `CellCache` would reject it by name here.
    @test !hasmethod(reinit_values!, Tuple{JumpPenaltyCache, Ferrite.CellCache})
    op = setup_operator(strategy_for(tb.spec, SequentialCPUDevice()), tb.integrator, tb.dh)
    element = first(get_subdomain_caches(op)).domain.element
    sdh     = first(tb.dh.subdofhandlers)
    @test assembly_iterator(nothing, element, sdh) isa PairCache
    @test hasmethod(reinit_values!, Tuple{JumpPenaltyCache, PairCache})

    @testset "A2 — the sweep visits the PROVIDER's items, not the cells" begin
        @test length(tb.prs) == 9
        @test getncells(tb.grid) == 12
        partition = first(get_subdomain_caches(op)).partition
        @test length(partition[1]) == 9
        sweep!(op)
        @test VISITED[] == 9
    end

    @testset "A3 — the assembled matrix equals the hand-built reference" begin
        A = Matrix(sweep!(op))
        @test A ≈ reference_matrix(tb.dh, tb.prs, n) rtol = 1.0e-12
        # Side-asymmetry: with α ≠ 1 the block is not invariant under swapping
        # the two sides, so a left/right mix-up would show here.
        @test !(A ≈ reference_matrix(tb.dh, [(r, l) for (l, r) in tb.prs], n))
    end

    @testset "A4 — the traversal really is two-sided" begin
        A = sweep!(op)
        cells = allocate_matrix(tb.dh)          # the CELL pattern alone
        d1, d2 = celldofs(tb.dh, 1), celldofs(tb.dh, 2)
        @test isempty(intersect(d1, d2))        # nothing derivable from one cell
        for i in d1, j in d2
            @test A[i, j] != 0                  # coupled by the pair item
            @test !(i in view(rowvals(cells), nzrange(cells, j)))  # in no cell's block
        end
    end

    @testset "A5 — the same family under a threaded, coloured device" begin
        seq = copy(sweep!(op))
        colored = setup_operator(
            strategy_for(tb.spec, PolyesterDevice(; min_items_per_worker = 1), ColoredScheduling()),
            tb.integrator, tb.dh)
        @test length(first(get_subdomain_caches(colored)).device_cache) > 1  # really per-worker
        par = copy(sweep!(colored))
        @test VISITED[] == 9
        @test Matrix(par) ≈ Matrix(seq) rtol = 1.0e-12
        # A repeat of the same configuration reproduces itself exactly; the
        # cross-device comparison above is `≈` because the summation order
        # differs, not because anything is approximate.
        @test sweep!(colored) == par
    end

    @testset "A7 — the device half, landed: KernelAbstractionsDevice(KA.CPU())" begin
        seq = copy(sweep!(op))
        device_op = setup_operator(
            strategy_for(tb.spec, KernelAbstractionsDevice(KA.CPU(); items_per_worker = 1),
                ColoredScheduling()),
            tb.integrator, tb.dh)
        # A low `items_per_worker` for the same reason A5's Polyester arm lowers
        # `min_items_per_worker`: a 9-item set at the default (2) would still
        # engage several workers, but this makes it explicit rather than
        # incidental.
        @test size(first(get_subdomain_caches(device_op)).device_cache.Ke, 1) > 1
        dev = copy(sweep!(device_op))
        @test VISITED[] == 9
        # Host-vs-device exactness, not a tolerance: the device sweep's dof
        # windows and element math are the same `Int`/`Float64` arithmetic as
        # the host's, only the traversal moved.
        @test maximum(abs, Matrix(dev) .- Matrix(seq)) == 0.0
    end

    @testset "A6 — the colouring promise the provider makes, asserted directly" begin
        provider = item_provider(nothing, element, sdh)
        @test provider isa PairItems
        partition = compute_partition(ColoredScheduling(), provider)
        @test length(partition) > 1
        window(i) = Set{Int}(vcat(celldofs(tb.dh, tb.prs[i][1]), celldofs(tb.dh, tb.prs[i][2])))
        for color in partition, i in color, j in color
            i == j && continue
            @test isdisjoint(window(i), window(j))     # no shared SCATTER DOF
        end
        @test sort!(reduce(vcat, partition)) == collect(1:9)
    end

    @testset "the conventional accessor half is the iterator's own" begin
        pc = assembly_iterator(nothing, element, sdh)
        Ferrite.reinit!(pc, 5)
        l, r = tb.prs[5]
        @test cellid(pc) == l
        @test iterator_handler(pc) === sdh
        @test iterator_dofs(pc) == [celldofs(tb.dh, l); celldofs(tb.dh, r)]
        # Ferrite's assembler addresses a scatter through this window and needs
        # it duplicate free — the reason the demo's space is discontinuous.
        @test allunique(iterator_dofs(pc))
        @test pc.coords_left == getcoordinates(tb.grid, l)
        @test pc.coords_right == getcoordinates(tb.grid, r)
    end

    @testset "the registration seam: a family declared entirely from test code" begin
        # Every operator above was built through THIS file's
        # `setup_family_caches` method — the engine reached it by dispatch on
        # the declared marker, and by nothing else.
        before = FAMILY_SETUPS[]
        fresh  = setup_operator(strategy_for(tb.spec, SequentialCPUDevice()), tb.integrator, tb.dh)
        @test FAMILY_SETUPS[] == before + 1
        @test item_families(tb.integrator, tb.dh) === (JumpPairFamily(),)
        # The cell family is NOT registered, so nothing swept the 12 cells: the
        # 9 pair items are the whole traversal.
        @test length(get_subdomain_caches(fresh)) == length(tb.dh.subdofhandlers)
        @test length(first(get_subdomain_caches(fresh)).partition[1]) == 9
        @test Matrix(sweep!(fresh)) ≈ reference_matrix(tb.dh, tb.prs, n) rtol = 1.0e-12
        # The three shipped families answer the SAME generic function this one
        # does — that is the whole of the dogfood claim, from outside `src/`.
        for family in (CellFamily(), FacetItemFamily(), AlgebraicItemFamily(), JumpPairFamily())
            @test hasmethod(setup_family_caches, Tuple{typeof(family), Any, Any, Any, Any})
        end
    end

    @testset "the defaults are unchanged for every other family" begin
        @test item_provider(nothing, nothing, sdh) isa CellItems
        @test item_families(nothing, tb.dh) === (CellFamily(),)
    end
end
