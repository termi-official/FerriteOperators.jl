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
([`MatrixFreeAction`](@ref)): the wrapped element cache, the subdomain's dense
element matrices, and the cell → slot map that addresses them.

`K` is laid out `(slot, i, j)` — the CELL index is stride-1, so the lanes of one
`(i, j)` step read adjacent addresses. `slots[cellid]` is the cell's row, `0`
for a cell outside this subdomain, which keeps the store the size of the
SUBDOMAIN rather than of the grid.

The matrices are shared read-only by the action and written per cell by the
fill, so nothing here is per worker and the device layout batches nothing:
[`setup_device_instances`](@ref) moves the store across once
([`adapt_shared`](@ref)) and recurses into the inner cache, whose own
per-worker fields are the only batched ones.

!!! warning "Experimental surface"
    This decorator and the election that builds it may change in a minor
    release.
"""
struct ElementAssemblyCache{Inner, KT, ST, R} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
    K::KT
    slots::ST
    route::R
end

# The stored matrices ARE the element matrix, so the engine's per-worker `Ke`
# would be a second copy of one: the fill writes into the store's own slot and
# the action reads it, neither touching the workspace buffer.
allocate_element_matrix(cache::ElementAssemblyCache, sdh) = zeros(element_value_type(cache), 0, 0)

duplicate_for_device(device, c::ElementAssemblyCache) =
    ElementAssemblyCache(duplicate_for_device(device, c.inner), c.K, c.slots, c.route)
setup_device_instances(device::AbstractGPUDevice, c::ElementAssemblyCache, n) =
    ElementAssemblyCache(setup_device_instances(device, c.inner, n),
                         adapt_shared(device, c.K), adapt_shared(device, c.slots), c.route)
device_worker_view(c::ElementAssemblyCache, worker) =
    ElementAssemblyCache(device_worker_view(c.inner, worker), c.K, c.slots, c.route)

####################################
## Setup
####################################

# The framework's half of the storage election: any bilinear cache becomes an
# element-assembling one, whatever it implements. The route is probed on the
# cache being WRAPPED, which is the one the fill calls.
with_action_storage(cache, ::ElementAssembly, sdh::SubDofHandler) =
    ElementAssemblyCache(cache, sdh, element_matrix_fill_route(typeof(cache)))

function ElementAssemblyCache(cache, sdh::SubDofHandler, route)
    T  = element_value_type(cache)
    nd = ndofs_per_cell(sdh)
    slots = zeros(Int32, getncells(get_grid(sdh.dh)))
    for (slot, cellid) in enumerate(sdh.cellset)
        slots[cellid] = slot
    end
    return ElementAssemblyCache(cache, zeros(T, length(sdh.cellset), nd, nd), slots, route)
end

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
"""
function apply_element_action!(yₑ, cache::ElementAssemblyCache, uₑ, args::CellArgs)
    K = cache.K
    slot = @inbounds cache.slots[cellid(args.cell)]
    for i in 1:size(K, 2)
        acc = zero(eltype(K))
        for j in 1:size(K, 3)
            @inbounds acc += K[slot, i, j] * uₑ[j]
        end
        @inbounds yₑ[i] += acc
    end
    return nothing
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
    _fill_element_matrix!(cache.route, cache, slot, args)
    return nothing
end

function _fill_element_matrix!(::MatrixKernelFill, cache::ElementAssemblyCache, slot::Integer, args::CellArgs)
    Kₑ = @inbounds @view cache.K[slot, :, :]
    fill!(Kₑ, zero(eltype(Kₑ)))
    assemble_cell!(JacobianRequest{:u}(Kₑ), cache.inner, args)
    return nothing
end

function _fill_element_matrix!(::ActionKernelFill, cache::ElementAssemblyCache, slot::Integer, args::CellArgs)
    T, nd = eltype(cache.K), size(cache.K, 3)
    for j in 1:nd
        column = @inbounds @view cache.K[slot, :, j]
        fill!(column, zero(T))
        apply_element_action!(column, cache.inner, UnitElementVector{T}(nd, j), args)
    end
    return nothing
end
