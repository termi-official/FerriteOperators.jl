####################################
## Row-sum lumping
####################################

"""
    RowSumLumped(inner::AbstractBilinearIntegrator)

The row-sum lumping of `inner`: the bilinear integrator whose element matrix is
the DIAGONAL `diag(mᵢ)`, `mᵢ = Σⱼ Mᵢⱼ` over `inner`'s own element matrix. It
declares [`DiagonalElementMatrix`](@ref), so its element buffer is an
`ndofs_per_cell` vector and a [`FullAssembly`](@ref) operator over it holds a
`Diagonal`.

**Element-level row-sum IS global row-sum.** The global matrix is
`M = Σₑ Pₑᵀ Mₑ Pₑ` and the global row sum is `M·1`, so lumping cell by cell and
assembling gives `Σₑ Pₑᵀ (Mₑ·1ₑ)`, which is exactly `M·1` — the scatter is
linear and `Pₑ·1 = 1ₑ`. The two orders agree entry for entry, on a continuous
space as on a discontinuous one, which is why this decorator needs no global
pass.

It is DISCRETIZATION's building block, not a treatment the solver elects: which
mass a model carries is the model owner's, and [`RateFormIntegrator`](@ref)
reads the structure of whatever mass it is handed. A lumped mass is
unconditionally invertible only where the row sums are non-zero — positive for
a mass over a non-negative basis, and checked where the reciprocal is formed.

`inner` must serve an analytic `JacobianKind{:u}` kernel
([`provides_analytic`](@ref)): the lumping reads its element matrix.
"""
struct RowSumLumped{I <: AbstractBilinearIntegrator} <: AbstractBilinearIntegrator
    inner::I
end

"""
The cache of [`RowSumLumped`](@ref): the wrapped cache and the `Nb × Nb` scratch
its element matrix is read into before the rows are summed. The scratch is
per-worker, as [`ElementAssemblyCache`](@ref)'s is.
"""
struct RowSumLumpedElementCache{Inner, SCT} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
    scratch::SCT
end

element_matrix_structure(::RowSumLumpedElementCache) = DiagonalElementMatrix()
rewrap(d::RowSumLumpedElementCache, inner) = RowSumLumpedElementCache(inner, d.scratch)

# The decorator layer forwards `allocate_element_matrix` to the inner, whose
# buffer is the SQUARE this cache sums the rows of — the one hook where a
# structure-changing decorator must answer for itself, or the engine would hand
# this cache's `req.K[i]` kernel a matrix.
allocate_element_matrix(c::RowSumLumpedElementCache, sdh) =
    zeros(element_value_type(c), ndofs_per_cell(sdh))

function setup_element_cache(integrator::RowSumLumped, sdh::SubDofHandler)
    inner = setup_element_cache(integrator.inner, sdh)
    provides_analytic(typeof(inner), JacobianKind{:u}()) || throw(ArgumentError(
        "`RowSumLumped` sums the ROWS of its inner's element matrix, and " *
        "$(nameof(typeof(inner))) declares no analytic `JacobianKind{:u}` kernel " *
        "(`provides_analytic`), so there is no element matrix to sum. Lump a mass whose element " *
        "kernel fills `JacobianRequest{:u}`."))
    # The scratch is the inner's AUGMENTED local system, `global_dofs` tail
    # included: the engine pads what `allocate_element_matrix` returns, and the
    # inner writes the padded square this reads its row sums from.
    n = ndofs_per_cell(sdh) + length(global_dofs(integrator, sdh))
    return RowSumLumpedElementCache(inner, zeros(element_value_type(inner), n, n))
end

# The inner sits under one local system, so its declarations are this
# integrator's; the drift check sees it in its own right.
global_dofs(integrator::RowSumLumped, sdh::SubDofHandler) = global_dofs(integrator.inner, sdh)
facet_items(integrator::RowSumLumped, sdh::SubDofHandler) = facet_items(integrator.inner, sdh)
facet_item_global_dofs(integrator::RowSumLumped, sdh::SubDofHandler) =
    facet_item_global_dofs(integrator.inner, sdh)
setup_facet_item_cache(integrator::RowSumLumped, sdh::SubDofHandler) =
    setup_facet_item_cache(integrator.inner, sdh)
algebraic_items(integrator::RowSumLumped, dh::AbstractDofHandler) = algebraic_items(integrator.inner, dh)
setup_algebraic_cache(integrator::RowSumLumped, dh::AbstractDofHandler) =
    setup_algebraic_cache(integrator.inner, dh)
function _declaration_subjects!(subjects, integrator::RowSumLumped)
    push!(subjects, integrator)
    _declaration_subjects!(subjects, integrator.inner)
    return subjects
end

duplicate_for_device(device, c::RowSumLumpedElementCache) =
    RowSumLumpedElementCache(duplicate_for_device(device, c.inner), similar(c.scratch))
setup_device_instances(device::AbstractGPUDevice, c::RowSumLumpedElementCache, n) =
    RowSumLumpedElementCache(setup_device_instances(device, c.inner, n),
                             setup_device_instances(device, c.scratch, n))
device_worker_view(c::RowSumLumpedElementCache, worker) =
    RowSumLumpedElementCache(device_worker_view(c.inner, worker), device_worker_view(c.scratch, worker))

provides_analytic(::Type{<:RowSumLumpedElementCache}, ::JacobianKind{:u}) = true

# `req.K[i] += Σⱼ Mᵢⱼ` over the inner's element matrix; `req.K` is the diagonal
# VECTOR `DiagonalElementMatrix` allocates.
function assemble_cell!(req::JacobianRequest{:u}, cache::RowSumLumpedElementCache, args::CellArgs)
    Mₑ = cache.scratch
    fill!(Mₑ, zero(eltype(Mₑ)))
    assemble_cell!(JacobianRequest{:u}(Mₑ), cache.inner, args)
    for i in axes(Mₑ, 1)
        acc = zero(eltype(Mₑ))
        for j in axes(Mₑ, 2)
            @inbounds acc += Mₑ[i, j]
        end
        @inbounds req.K[i] += acc
    end
    return nothing
end

# The action of the lumped form: `rᵢ = mᵢ uᵢ`, the diagonal against the cell's
# own state — NOT the inner's residual, which is the unlumped form's.
function assemble_cell!(req::ResidualRequest, cache::RowSumLumpedElementCache, args::CellArgs)
    Mₑ = cache.scratch
    fill!(Mₑ, zero(eltype(Mₑ)))
    assemble_cell!(JacobianRequest{:u}(Mₑ), cache.inner, args)
    uₑ = args.states.u
    for i in axes(Mₑ, 1)
        acc = zero(eltype(Mₑ))
        for j in axes(Mₑ, 2)
            @inbounds acc += Mₑ[i, j]
        end
        @inbounds req.r[i] += acc * uₑ[i]
    end
    return nothing
end
