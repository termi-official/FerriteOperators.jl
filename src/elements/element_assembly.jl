####################################
## The ELEMENT assembly level
####################################
#
# The third member of a matrix-free operator's storage ladder
# ([`MatrixFreeAction`](@ref)): the dense element matrices, kept between actions.
# It is element-AGNOSTIC — a decorator over whatever cache the integrator built —
# which is what lets it serve a cache with no matrix-free kernel at all.

"""
    UnitElementVector{T}(n, j)

The `j`-th unit vector of length `n`, as an `AbstractVector{T}` that stores
nothing. It exists so an element's action kernel can be applied to a basis of
the local space without a per-worker scratch vector to hold one — which is what
fills a column of an element matrix from an action
([`ElementAssemblyCache`](@ref)).
"""
struct UnitElementVector{T} <: AbstractVector{T}
    n::Int
    j::Int
end
Base.size(v::UnitElementVector) = (v.n,)
@inline Base.getindex(v::UnitElementVector{T}, i::Int) where {T} = ifelse(i == v.j, one(T), zero(T))

"""
    MatrixKernelFill()
    ActionKernelFill()

How an [`ElementAssemblyCache`](@ref) fills one cell's matrix: from the
element's own element-matrix kernel (`assemble_cell!` on a
`JacobianRequest{:u}`, where [`provides_analytic`](@ref) declares one), or from
`ndofs_per_cell` applications of [`apply_element_action!`](@ref) to the unit
vectors of the local space. Resolved once at setup and carried as a field, so
the fill kernel has no runtime branch and the cache's type parameters stay
determined by its fields.

The action route is exact and costs `ndofs_per_cell` element actions per cell
ONCE per fill — a setup-time price a sum-factorized element pays instead of
growing a second, conventional implementation of the same form.
"""
struct MatrixKernelFill end
@doc (@doc MatrixKernelFill) struct ActionKernelFill end

"""
    ElementAssemblyCache <: AbstractElementCacheDecorator

The cache a `storage = ElementAssembly()` operator runs
([`MatrixFreeAction`](@ref)): the wrapped element cache, the subdomain's
element matrices, and the cell → slot map that addresses them.

`K`'s layout is the [`element_matrix_symmetry`](@ref) election, read once off
the wrapped cache at construction and carried as `symmetry`:

- [`GeneralElementMatrix`](@ref) (the default) — `K` is `(slot, i, j)` dense.
  The CELL index is stride-1, so the lanes of one `(i, j)` step read adjacent
  addresses.
- [`SymmetricElementMatrix`](@ref) — `K` is `(slot, t)`, `t` the packed
  upper-triangle index (row-major, diagonal included); `slot` stays leading and
  stride-1. `Kₑ = Kₑᵀ` is TRUSTED, not checked: the action never reads the
  lower triangle.

`slots[cellid]` is the cell's row, `0` for a cell outside this subdomain, which
keeps the store the size of the SUBDOMAIN rather than of the grid.

`local_size` is `Val(ndofs_per_cell)`, so the dense product's extents are a
compile-time constant rather than the store's runtime dimensions
([`element_local_length`](@ref)): a device kernel whose trip counts are known
keeps the element vectors in registers, which on an RTX 2080 is the difference
between 0.69 ms and 0.27 ms for one action over 132k trilinear hexahedra.

The matrices are shared read-only by the action and written per cell by the
fill, so `K`/`slots` are not per worker and the device layout batches neither:
[`setup_device_instances`](@ref) moves them across once ([`adapt_shared`](@ref))
and recurses into the inner cache, whose own per-worker fields are the only
OTHER batched ones. `scratch` is the exception — a per-worker `ndofs_per_cell²`
buffer the SYMMETRIC fill packs from (0×0 and unbatched under
[`GeneralElementMatrix`](@ref), where the fill writes `K`'s own view
directly).

!!! warning "Experimental surface"
    This decorator and the elections that build it may change in a minor
    release.
"""
struct ElementAssemblyCache{Inner, KT, ST, R, Sym, SCT, ND} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
    K::KT
    slots::ST
    route::R
    symmetry::Sym
    scratch::SCT
    local_size::Val{ND}
end

element_local_length(c::ElementAssemblyCache) = c.local_size

# The stored matrices ARE the element matrix, so the engine's per-worker `Ke`
# would be a second copy of one: the fill writes into the store's own slot (or,
# under SymmetricElementMatrix(), the cache's own `scratch`) and the action
# reads `K`, neither touching the workspace buffer.
allocate_element_matrix(cache::ElementAssemblyCache, sdh) = zeros(element_value_type(cache), 0, 0)

# The ELEMENT level's action reads the cell's stored matrix and nothing else: the
# wrapped cache's values objects are consumed by the FILL, so positioning them
# per action would re-derive geometry no kernel then reads.
reinit_values!(::ElementAssemblyCache, cell, ::MatrixFreeActionKind) = nothing
item_update_flags(::MatrixFreeActionKind, ::ElementAssemblyCache) =
    Ferrite.UpdateFlags(nodes = false, coords = false, dofs = true)

duplicate_for_device(device, c::ElementAssemblyCache) =
    ElementAssemblyCache(duplicate_for_device(device, c.inner), c.K, c.slots, c.route, c.symmetry,
                         similar(c.scratch), c.local_size)
setup_device_instances(device::AbstractGPUDevice, c::ElementAssemblyCache, n) =
    ElementAssemblyCache(setup_device_instances(device, c.inner, n),
                         adapt_shared(device, c.K), adapt_shared(device, c.slots), c.route, c.symmetry,
                         setup_device_instances(device, c.scratch, n), c.local_size)
device_worker_view(c::ElementAssemblyCache, worker) =
    ElementAssemblyCache(device_worker_view(c.inner, worker), c.K, c.slots, c.route, c.symmetry,
                         device_worker_view(c.scratch, worker), c.local_size)

####################################
## Setup
####################################

# The framework's half of the storage election: any bilinear cache becomes an
# element-assembling one, whatever it implements. The route is probed on the
# cache being WRAPPED, which is the one the fill calls; the symmetry election
# (design.md C11) is read off the same cache.
with_action_storage(cache, ::ElementAssembly, sdh::SubDofHandler) =
    ElementAssemblyCache(cache, sdh, element_matrix_fill_route(typeof(cache)), element_matrix_symmetry(cache))

function ElementAssemblyCache(cache, sdh::SubDofHandler, route, symmetry)
    T  = element_value_type(cache)
    nd = ndofs_per_cell(sdh)
    slots = zeros(Int32, getncells(get_grid(sdh.dh)))
    for (slot, cellid) in enumerate(sdh.cellset)
        slots[cellid] = slot
    end
    K, scratch = _allocate_element_matrix_store(symmetry, T, length(sdh.cellset), nd)
    return ElementAssemblyCache(cache, K, slots, route, symmetry, scratch, Val(nd))
end

# GeneralElementMatrix(): `K` dense, no scratch needed — the fill writes `K`'s
# own view directly, as it always has.
_allocate_element_matrix_store(::GeneralElementMatrix, T, ncells, nd) =
    (zeros(T, ncells, nd, nd), zeros(T, 0, 0))
# SymmetricElementMatrix(): `K` packed (upper triangle, diagonal included), and
# a per-worker `nd × nd` scratch the fill packs from (design.md C11).
_allocate_element_matrix_store(::SymmetricElementMatrix, T, ncells, nd) =
    (zeros(T, ncells, (nd * (nd + 1)) ÷ 2), zeros(T, nd, nd))

"""
    element_matrix_fill_route(::Type{C}) -> MatrixKernelFill() or ActionKernelFill()

Which route an [`ElementAssemblyCache`](@ref) over `C` fills its matrices
through, and the setup-time capability check behind the ELEMENT level: the
element's own element-matrix kernel where [`provides_analytic`](@ref) declares
one, the action applied to the unit vectors where
[`apply_element_action!`](@ref) exists, and a loud rejection naming both where
neither does.
"""
function element_matrix_fill_route(::Type{C}) where {C}
    provides_analytic(C, JacobianKind{:u}()) && return MatrixKernelFill()
    hasmethod(apply_element_action!, Tuple{AbstractVector, C, AbstractVector, CellArgs}) &&
        return ActionKernelFill()
    throw(ArgumentError(
        "$(C) can serve neither route of the `ElementAssembly` storage level: it declares no " *
        "analytic `JacobianKind{:u}` kernel (`provides_analytic`), so its element matrix cannot " *
        "be assembled, and implements no `apply_element_action!`, so the matrix cannot be " *
        "filled column by column from the action either. Implement one of them, or elect " *
        "`storage = Stored()`/`Recompute()`."))
end

####################################
## The action and the fill
####################################

"""
    apply_element_action!(yₑ, cache::ElementAssemblyCache, uₑ, args)

The ELEMENT-level action: the dense product `yₑ += Kₑ·uₑ` over the matrix this
cell's slot holds. No quadrature point is visited and no contraction runs — the
element's own kernels are consumed by the FILL, not by the action.

The extents come from the cache's `local_size`, not from the store's
dimensions: a compile-time trip count is what lets a device kernel hold `uₑ` and
the row accumulator in registers instead of walking two arrays in memory.
"""
@inline apply_element_action!(yₑ, cache::ElementAssemblyCache, uₑ, args::CellArgs) =
    _element_matrix_action!(cache.symmetry, yₑ, cache.K, (@inbounds cache.slots[cellid(args.cell)]), uₑ, cache.local_size)

@inline function _element_matrix_action!(::GeneralElementMatrix, yₑ, K, slot, uₑ, ::Val{ND}) where {ND}
    for i in 1:ND
        acc = zero(eltype(K))
        for j in 1:ND
            @inbounds acc += K[slot, i, j] * uₑ[j]
        end
        @inbounds yₑ[i] += acc
    end
    return nothing
end

# The packed read: `K[slot, i, j]` becomes `K[slot, t(i, j)]` with `t` the
# canonical (row ≤ col) packed index — the symmetric expansion happens in
# registers, never by reading the (absent) lower triangle.
@inline function _element_matrix_action!(::SymmetricElementMatrix, yₑ, K, slot, uₑ, ::Val{ND}) where {ND}
    for i in 1:ND
        acc = zero(eltype(K))
        for j in 1:ND
            @inbounds acc += K[slot, _packed_index(i, j, Val(ND))] * uₑ[j]
        end
        @inbounds yₑ[i] += acc
    end
    return nothing
end

"""
    element_action_row(cache::ElementAssemblyCache, uₑ, args, i)

The ELEMENT level's row: the dot product of row `i` of this cell's stored matrix
with `uₑ`, accumulated in ONE register and returned by value — what a
[`LanesPerElement`](@ref) lane owns.

The two layouts differ only in where the row's entries live. Dense reads
`K[slot, i, j]` along `j`; packed reads `K[slot, t(i, j)]`, whose `j`-walk
crosses the packed triangle's rows below the diagonal and runs along it above.
Neither reads the lower triangle.
"""
@inline element_action_row(cache::ElementAssemblyCache, uₑ, args::CellArgs, i::Int) =
    _element_matrix_action_row(cache.symmetry, cache.K, (@inbounds cache.slots[cellid(args.cell)]),
                               uₑ, i, cache.local_size)

@inline function _element_matrix_action_row(::GeneralElementMatrix, K, slot, uₑ, i::Int, ::Val{ND}) where {ND}
    acc = zero(eltype(K))
    for j in 1:ND
        @inbounds acc += K[slot, i, j] * uₑ[j]
    end
    return acc
end

@inline function _element_matrix_action_row(::SymmetricElementMatrix, K, slot, uₑ, i::Int, ::Val{ND}) where {ND}
    acc = zero(eltype(K))
    for j in 1:ND
        @inbounds acc += K[slot, _packed_index(i, j, Val(ND))] * uₑ[j]
    end
    return acc
end

"""
    _packed_index(i, j, ::Val{ND}) -> t

The 1-based linear index of `(i, j)` in a row-major packed upper triangle
(diagonal included) of an `ND × ND` matrix — `t(i, j) == t(j, i)`, so a caller
addressing either triangle reads the SAME entry. `ND*(ND+1)÷2` entries total.
"""
@inline function _packed_index(i::Integer, j::Integer, ::Val{ND}) where {ND}
    a, b = i <= j ? (i, j) : (j, i)
    return ((a - 1) * (2ND - a + 2)) ÷ 2 + (b - a + 1)
end

"""
    fill_quadrature_data!(cache::ElementAssemblyCache, args)

Fill this cell's element matrix — the ELEMENT level's answer to the same fill
sweep ([`QuadratureDataKind`](@ref)) the PARTIAL level runs, so `setup_operator`
and [`update_operator!`](@ref) need no second entry point and the freshness
contract is one contract.

The inner cache is filled FIRST where it keeps a store of its own, so an
element assembled through its action route reads factors as fresh as the sweep
that is filling it.
"""
function fill_quadrature_data!(cache::ElementAssemblyCache, args::CellArgs)
    fill_quadrature_data!(cache.inner, args)
    slot = @inbounds cache.slots[cellid(args.cell)]
    _fill_element_matrix!(cache.route, cache.symmetry, cache, slot, args)
    return nothing
end

function _fill_element_matrix!(::MatrixKernelFill, ::GeneralElementMatrix, cache::ElementAssemblyCache,
        slot::Integer, args::CellArgs)
    Kₑ = @inbounds @view cache.K[slot, :, :]
    fill!(Kₑ, zero(eltype(Kₑ)))
    assemble_cell!(JacobianRequest{:u}(Kₑ), cache.inner, args)
    return nothing
end

function _fill_element_matrix!(::ActionKernelFill, ::GeneralElementMatrix, cache::ElementAssemblyCache,
        slot::Integer, args::CellArgs)
    T, nd = eltype(cache.K), size(cache.K, 3)
    for j in 1:nd
        column = @inbounds @view cache.K[slot, :, j]
        fill!(column, zero(T))
        apply_element_action!(column, cache.inner, UnitElementVector{T}(nd, j), args)
    end
    return nothing
end

# SymmetricElementMatrix(): `K`'s packed row has no square view for the
# element's kernel to write, so the fill writes the cache's own per-worker
# `ndofs_per_cell²` `scratch` — exactly as the general route writes `K`'s view
# — and then packs its upper triangle. Paid at FILL time only
# (`setup_operator`/`update_operator!`), never on the action.
function _fill_element_matrix!(::MatrixKernelFill, ::SymmetricElementMatrix, cache::ElementAssemblyCache,
        slot::Integer, args::CellArgs)
    Kₑ = cache.scratch
    fill!(Kₑ, zero(eltype(Kₑ)))
    assemble_cell!(JacobianRequest{:u}(Kₑ), cache.inner, args)
    _pack_symmetric!(cache.K, slot, Kₑ, cache.local_size)
    return nothing
end

function _fill_element_matrix!(::ActionKernelFill, ::SymmetricElementMatrix, cache::ElementAssemblyCache,
        slot::Integer, args::CellArgs)
    T, nd = eltype(cache.scratch), size(cache.scratch, 1)
    for j in 1:nd
        column = @inbounds @view cache.scratch[:, j]
        fill!(column, zero(T))
        apply_element_action!(column, cache.inner, UnitElementVector{T}(nd, j), args)
    end
    _pack_symmetric!(cache.K, slot, cache.scratch, cache.local_size)
    return nothing
end

# The packed commit: only the upper triangle (row ≤ col) of the square `Kₑ` is
# read — the SAME entries `_element_matrix_action!`'s `SymmetricElementMatrix`
# route later reads back through `_packed_index`. A wrongly-declared election
# silently drops the lower triangle here rather than erroring — see
# `element_matrix_symmetry`'s trust contract.
@inline function _pack_symmetric!(K, slot, Kₑ, ::Val{ND}) where {ND}
    for i in 1:ND, j in i:ND
        @inbounds K[slot, _packed_index(i, j, Val(ND))] = Kₑ[i, j]
    end
    return nothing
end
