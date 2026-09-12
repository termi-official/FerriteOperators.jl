####################################
## The ELEMENT assembly level
####################################

# The `j`-th unit vector of length `n`, storing nothing, so the action route can
# fill a column of an element matrix without a scratch vector.
struct UnitElementVector{T} <: AbstractVector{T}
    n::Int
    j::Int
end
Base.size(v::UnitElementVector) = (v.n,)
@inline Base.getindex(v::UnitElementVector{T}, i::Int) where {T} = ifelse(i == v.j, one(T), zero(T))

# How an `ElementAssemblyCache` fills one cell's matrix — from the element's own
# element-matrix kernel, or from `ndofs_per_cell` applications of the action to
# the unit vectors. Resolved once at setup and carried as a field, so the fill
# has no runtime branch.
struct MatrixKernelFill end
struct ActionKernelFill end

"""
    ElementAssemblyCache <: AbstractElementCacheDecorator

The cache a `storage = ElementAssembly()` operator runs
([`MatrixFreeAction`](@ref)): the wrapped element cache, the subdomain's
element matrices, and the cell → slot map that addresses them. It is
element-AGNOSTIC, so it serves a cache with no matrix-free kernel at all.

`K`'s layout is the [`element_matrix_symmetry`](@ref) election, read once off
the wrapped cache at construction and carried as `symmetry`:

- [`GeneralElementMatrix`](@ref) (the default) — `K` is `(slot, i, j)` dense.
- [`SymmetricElementMatrix`](@ref) — `K` is `(slot, t)`, `t` the packed
  upper-triangle index (row-major, diagonal included). `Kₑ = Kₑᵀ` is TRUSTED,
  not checked: the action never reads the lower triangle.

`slot` is leading and stride-1 either way, so the lanes of one `(i, j)` step
read adjacent addresses. `slots[cellid]` is the cell's row, `0` outside this
subdomain, keeping the store the size of the SUBDOMAIN rather than of the grid.

`local_size` is `Val(ndofs_per_cell)` ([`element_local_length`](@ref)), so a
device kernel's trip counts are compile-time constants and the element vectors
stay in registers.

`K`/`slots` are shared read-only by the action, so the device layout batches
neither ([`adapt_shared`](@ref)) and recurses into the inner cache. `scratch`
is the exception — a per-worker `ndofs_per_cell²` buffer the SYMMETRIC fill
packs from, 0×0 and unbatched under [`GeneralElementMatrix`](@ref).

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

# The stored matrices ARE the element matrix; nothing touches the workspace `Ke`.
allocate_element_matrix(cache::ElementAssemblyCache, sdh) = zeros(element_value_type(cache), 0, 0)

# The wrapped cache's values objects are consumed by the FILL, so positioning
# them per action would re-derive geometry no kernel reads.
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

# The route and the symmetry election are read off the cache being WRAPPED,
# which is the one the fill calls.
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

_allocate_element_matrix_store(::GeneralElementMatrix, T, ncells, nd) =
    (zeros(T, ncells, nd, nd), zeros(T, 0, 0))
_allocate_element_matrix_store(::SymmetricElementMatrix, T, ncells, nd) =
    (zeros(T, ncells, (nd * (nd + 1)) ÷ 2), zeros(T, nd, nd))

"""
    element_matrix_fill_route(::Type{C}) -> MatrixKernelFill() or ActionKernelFill()

Which route an [`ElementAssemblyCache`](@ref) over `C` fills its matrices
through: the element's own element-matrix kernel where
[`provides_analytic`](@ref) declares one, the action applied to the unit vectors
where [`apply_element_action!`](@ref) exists, a rejection where neither does.
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
cell's slot holds. No quadrature point is visited — the element's own kernels
are consumed by the FILL. The extents come from `local_size`, not from the
store's dimensions.
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
[`LanesPerElement`](@ref) lane owns. Neither layout reads the lower triangle.
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

# 1-based index of `(i, j)` in a row-major packed upper triangle (diagonal
# included) of an `ND × ND` matrix, `ND*(ND+1)÷2` entries. `t(i, j) == t(j, i)`,
# so either triangle addresses the SAME entry.
@inline function _packed_index(i::Integer, j::Integer, ::Val{ND}) where {ND}
    a, b = i <= j ? (i, j) : (j, i)
    return ((a - 1) * (2ND - a + 2)) ÷ 2 + (b - a + 1)
end

"""
    fill_quadrature_data!(cache::ElementAssemblyCache, args)

Fill this cell's element matrix — the ELEMENT level's answer to the same
[`QuadratureDataKind`](@ref) sweep the PARTIAL level runs.

The inner cache is filled FIRST where it keeps a store of its own, so an element
assembled through its action route reads factors as fresh as the sweep filling
it.
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

# `K`'s packed row has no square view for the element's kernel to write, so the
# fill writes `scratch` and packs its upper triangle afterwards.
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

# Only the upper triangle is read. A wrongly-declared election silently drops the
# lower one here rather than erroring; see `element_matrix_symmetry`.
@inline function _pack_symmetric!(K, slot, Kₑ, ::Val{ND}) where {ND}
    for i in 1:ND, j in i:ND
        @inbounds K[slot, _packed_index(i, j, Val(ND))] = Kₑ[i, j]
    end
    return nothing
end
