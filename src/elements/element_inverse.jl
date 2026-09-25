####################################
## The element inverse
####################################

"""
    ElementInverse(inner::AbstractBilinearIntegrator)

The bilinear integrator whose element matrix is `Mₑ⁻¹`, `Mₑ` the element
matrix of `inner`. Assembled, it is `M⁻¹` exactly where every dof belongs to
ONE cell (a discontinuous space), since `M` is then block diagonal by cell;
over a continuous space the sum of inverse blocks is not the inverse, and
`setup_element_cache` refuses the handler. The structure follows `inner`: a
[`DiagonalElementMatrix`](@ref) inverts entry-wise, a dense one through its
Cholesky factor, so `inner` must be symmetric positive definite cell by cell.

`inner` must serve an analytic `JacobianKind{:u}` kernel
([`provides_analytic`](@ref)) and declare no global dofs, facet items or
algebraic items. [`RateFormIntegrator`](@ref) builds its per-cell `M⁻¹` from
this integrator under [`ElementAssembly`](@ref).
"""
struct ElementInverse{I <: AbstractBilinearIntegrator} <: AbstractBilinearIntegrator
    inner::I
end

# `scratch` holds the inner's element matrix in the inner's own shape; `work`
# is the action route's solve vector. Both per-worker.
struct ElementInverseElementCache{Inner, SCT, WT} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
    scratch::SCT
    work::WT
end

rewrap(d::ElementInverseElementCache, inner) = ElementInverseElementCache(inner, d.scratch, d.work)

function setup_element_cache(integrator::ElementInverse, sdh::SubDofHandler)
    _assert_cell_only_term(integrator.inner, sdh, sdh.dh, "ElementInverse")
    inner = setup_element_cache(integrator.inner, sdh)
    provides_analytic(typeof(inner), JacobianKind{:u}()) || throw(ArgumentError(
        "`ElementInverse` inverts its inner's element matrix, and $(nameof(typeof(inner))) " *
        "declares no analytic `JacobianKind{:u}` kernel (`provides_analytic`), so there is no " *
        "element matrix to invert."))
    d = _first_shared_cell_dof(sdh.dh)
    d == 0 || throw(ArgumentError(
        "`ElementInverse` requires a DISCONTINUOUS space: dof $(d) is shared by two cells of " *
        "this handler, so the assembled matrix is not block diagonal by cell and the sum of the " *
        "per-cell inverses is not its inverse. Lump the term (`RowSumLumped`) and invert the " *
        "assembled diagonal instead."))
    return ElementInverseElementCache(inner, allocate_element_matrix(inner, sdh),
                                      allocate_element_unknown_vector(inner, sdh))
end

# A cell term declaring a local-system tail or a second item family would be
# inverted as a strict subset of itself; refused by name instead.
function _assert_cell_only_term(term, sdh, dh, who)
    for (hook, declaration) in ((global_dofs(term, sdh), "global_dofs"),
                                (facet_items(term, sdh), "facet_items"),
                                (algebraic_items(term, dh), "algebraic_items"))
        isempty(hook) || throw(ArgumentError(
            "`$(who)` inverts a CELL term's element matrices alone, and this one declares " *
            "`$(declaration)` — the declared items and dofs would be dropped without a word."))
    end
    return nothing
end

# The first dof two cells of the WHOLE handler share, `0` where none do; one
# bitvector across the subdomains catches a dof shared over a subdomain seam.
function _first_shared_cell_dof(dh::AbstractDofHandler)
    seen = falses(ndofs(dh))
    for sdh in dh.subdofhandlers
        d = _shared_cell_dof(sdh, seen)
        d == 0 || return d
    end
    return 0
end

duplicate_for_device(device, c::ElementInverseElementCache) =
    ElementInverseElementCache(duplicate_for_device(device, c.inner), similar(c.scratch), similar(c.work))
setup_device_instances(device::AbstractGPUDevice, c::ElementInverseElementCache, n) =
    ElementInverseElementCache(setup_device_instances(device, c.inner, n),
                               setup_device_instances(device, c.scratch, n),
                               setup_device_instances(device, c.work, n))
device_worker_view(c::ElementInverseElementCache, worker) =
    ElementInverseElementCache(device_worker_view(c.inner, worker), device_worker_view(c.scratch, worker),
                               device_worker_view(c.work, worker))

provides_analytic(::Type{<:ElementInverseElementCache}, ::JacobianKind{:u}) = true

function assemble_cell!(req::JacobianRequest{:u}, cache::ElementInverseElementCache, args::CellArgs)
    Mₑ = cache.scratch
    fill!(Mₑ, zero(eltype(Mₑ)))
    assemble_cell!(JacobianRequest{:u}(Mₑ), cache.inner, args)
    _accumulate_inverse!(req.K, Mₑ)
    return nothing
end

function assemble_cell!(req::ResidualRequest, cache::ElementInverseElementCache, args::CellArgs)
    Mₑ = cache.scratch
    fill!(Mₑ, zero(eltype(Mₑ)))
    assemble_cell!(JacobianRequest{:u}(Mₑ), cache.inner, args)
    _accumulate_inverse_action!(req.r, Mₑ, args.states.u, cache.work)
    return nothing
end

# Stateless, so the throw compiles inside a device kernel.
struct SingularElementMatrixError <: Exception end
Base.showerror(io::IO, ::SingularElementMatrixError) = print(io,
    "SingularElementMatrixError: `ElementInverse` met an element matrix that is not symmetric ",
    "positive definite (a non-positive Cholesky pivot or a zero diagonal), so its inverse does ",
    "not exist. Check the density and the quadrature rule the inner integrator carries.")

@noinline _throw_singular_element() = throw(SingularElementMatrixError())

function _accumulate_inverse!(K::AbstractVector, m::AbstractVector)
    @inbounds for i in eachindex(m)
        m[i] > 0 || _throw_singular_element()
        K[i] += inv(m[i])
    end
    return nothing
end

# `Mₑ ← L` (lower Cholesky factor, in place), `L ← L⁻¹`, then
# `Kᵢⱼ += Σₖ L⁻¹ₖᵢ L⁻¹ₖⱼ`. Plain loops over the per-worker scratch: no
# allocation, so the fill runs inside a device kernel.
function _accumulate_inverse!(K::AbstractMatrix, S::AbstractMatrix)
    _cholesky_lower!(S)
    _invert_lower!(S)
    n = size(S, 1)
    @inbounds for j in 1:n, i in 1:n
        acc = zero(eltype(S))
        for k in max(i, j):n
            acc += S[k, i] * S[k, j]
        end
        K[i, j] += acc
    end
    return nothing
end

function _accumulate_inverse_action!(r::AbstractVector, m::AbstractVector, u, work)
    @inbounds for i in eachindex(m)
        m[i] > 0 || _throw_singular_element()
        r[i] += u[i] / m[i]
    end
    return nothing
end

# Two triangular solves against the Cholesky factor: `L w = u`, `Lᵀ x = w`.
function _accumulate_inverse_action!(r::AbstractVector, S::AbstractMatrix, u, w)
    _cholesky_lower!(S)
    n = size(S, 1)
    @inbounds for i in 1:n
        acc = u[i]
        for k in 1:(i - 1)
            acc -= S[i, k] * w[k]
        end
        w[i] = acc / S[i, i]
    end
    @inbounds for i in n:-1:1
        acc = w[i]
        for k in (i + 1):n
            acc -= S[k, i] * w[k]
        end
        w[i] = acc / S[i, i]
        r[i] += w[i]
    end
    return nothing
end

# Lower Cholesky factor of a symmetric positive definite `S`, written into its
# lower triangle. The upper triangle is never read.
function _cholesky_lower!(S::AbstractMatrix)
    n = size(S, 1)
    @inbounds for j in 1:n
        s = S[j, j]
        for k in 1:(j - 1)
            s -= S[j, k]^2
        end
        s > 0 || _throw_singular_element()
        d = sqrt(s)
        S[j, j] = d
        for i in (j + 1):n
            t = S[i, j]
            for k in 1:(j - 1)
                t -= S[i, k] * S[j, k]
            end
            S[i, j] = t / d
        end
    end
    return nothing
end

# In-place inverse of the lower triangle, column by column: row `i` of column
# `j` reads the not-yet-overwritten `L[i, j:i]` and the already-inverted
# `X[j:i-1, j]`.
function _invert_lower!(L::AbstractMatrix)
    n = size(L, 1)
    @inbounds for j in 1:n
        L[j, j] = inv(L[j, j])
        for i in (j + 1):n
            s = zero(eltype(L))
            for k in j:(i - 1)
                s += L[i, k] * L[k, j]
            end
            L[i, j] = -s / L[i, i]
        end
    end
    return nothing
end
