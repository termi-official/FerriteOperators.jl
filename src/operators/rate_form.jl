####################################
## The rate form: `M⁻¹ · rhs`
####################################

"""
    BilinearRateFormIntegrator(rhs, mass)

The rate form of a BILINEAR right-hand side — `RateFormIntegrator(rhs, mass)`
elects it. See [`RateFormIntegrator`](@ref).
"""
struct BilinearRateFormIntegrator{R <: AbstractBilinearIntegrator, M} <: AbstractBilinearIntegrator
    rhs::R
    mass::M
end

"""
    LinearRateFormIntegrator(rhs, mass)

The rate form of a LINEAR right-hand side — `RateFormIntegrator(rhs, mass)`
elects it. See [`RateFormIntegrator`](@ref).
"""
struct LinearRateFormIntegrator{R <: AbstractLinearIntegrator, M} <: AbstractLinearIntegrator
    rhs::R
    mass::M
end

const AnyRateFormIntegrator = Union{BilinearRateFormIntegrator, LinearRateFormIntegrator}

"""
    RateFormIntegrator(rhs, mass)

The INVERSE-MASS-WEIGHTED (rate) form of `rhs`: the term whose operator is

    action(u) = M⁻¹ · (A · u)      # `rhs` a bilinear form, `A` the operator it induces
    payload   = M⁻¹ · b            # `rhs` a linear form, `b` its load vector

One spelling; which type it builds follows `rhs`'s own family
([`BilinearRateFormIntegrator`](@ref), [`LinearRateFormIntegrator`](@ref)),
since a bilinear right-hand side induces an operator and a linear one a vector.
`mass` is an ordinary bilinear MASS integrator — the model's own, lumped
([`RowSumLumped`](@ref)) or not. The PAIR is the whole declaration: how `M⁻¹` is
realized is the storage's business and never the caller's.

**The mass is checked STRUCTURALLY at setup, never approximated silently.**
[`element_matrix_structure`](@ref) of the mass's element cache and the space's
dof sharing decide the treatment:

| mass structure | space | `M⁻¹` is |
|---|---|---|
| [`DiagonalElementMatrix`](@ref) | any | the reciprocal of the ASSEMBLED global diagonal |
| [`DenseElementMatrix`](@ref) | cell-disjoint dofs (discontinuous) | the per-cell block inverse |
| [`DenseElementMatrix`](@ref) | continuous | REFUSED at setup |

A dense mass over a continuous space has no cell-local inverse, and this
package will not substitute one: lump the mass (`RowSumLumped(mass)`), or give
the model a mass that is diagonal by construction — a collocated (spectral)
element declaring [`DiagonalElementMatrix`](@ref). Both are DISCRETIZATION
decisions and both are spelled by the caller.

**What each realization does with the pair.** The rhs is assembled exactly as
it would be alone; only where `M⁻¹` is applied differs:

| form / storage | realization |
|---|---|
| [`FullAssembly`](@ref) | the rhs matrix, its rows then scaled (diagonal mass) or block-solved per cell (dense mass) — same sparsity, no second matrix |
| [`MatrixFreeAction`](@ref) + [`BlockRowAssembly`](@ref) | `M⁻¹` fused into the block-row store at fill ([`finalize_action_storage!`](@ref)) |
| [`MatrixFreeAction`](@ref) + [`Stored`](@ref)/[`Recompute`](@ref)/[`ElementAssembly`](@ref) | the rhs action, then the reciprocal scaling of `y` — a DIAGONAL mass only, a dense one refused by name |
| linear `rhs` | `b ← M⁻¹b` at every fill — under an ASSEMBLING form, a load vector having no action to evaluate |

The scaling of the ACTION is the whole of `M⁻¹` and not a per-cell
approximation of it: a diagonal `M⁻¹` commutes with the scatter, so scaling
after it gives what scaling each cell's rows before it would — which scaling
per cell could NOT, a continuous space's row collecting several cells.

`M⁻¹` is as fresh as the operator's last fill: [`update_operator!`](@ref)
re-derives it with that call's `(p, ctx)`, and [`setup_operator`](@ref) with the
`initial_parameters`/`initial_context` pair, so a mass reading
[`evaluation_time`](@ref) is evaluated at the same point the rhs is.

!!! warning "Experimental surface"
    The rate form, its operator types and the realizations above may change in a
    minor release.
"""
RateFormIntegrator(rhs::AbstractBilinearIntegrator, mass) = BilinearRateFormIntegrator(rhs, mass)
RateFormIntegrator(rhs::AbstractLinearIntegrator, mass) = LinearRateFormIntegrator(rhs, mass)
RateFormIntegrator(rhs, mass) = throw(ArgumentError(
    "A rate form weights a BILINEAR or a LINEAR right-hand side by `M⁻¹` (got " *
    "$(nameof(typeof(rhs)))). A nonlinear residual is not the action of an operator, so `M⁻¹F(u)` " *
    "is not a term this package assembles; scale the residual solver-side."))

# A rate form never reaches an element cache: `setup_operator` takes the pair
# apart and builds the engine over the rhs alone. Reaching here means it was
# composed into someone else's local system — which `M⁻¹` does not distribute
# over.
setup_element_cache(i::AnyRateFormIntegrator, sdh::SubDofHandler) = throw(ArgumentError(
    "A rate form is an OPERATOR-level term: `M⁻¹` weights the assembled right-hand side, not one " *
    "element's local system, and `M⁻¹(A₁ + A₂) ≠ M⁻¹A₁ + A₂` — so it composes with no other term " *
    "into a shared element cache. Build it on its own — `setup_operator(strategy, " *
    "RateFormIntegrator(rhs, mass), dh)` — and add the terms at the operator level."))

####################################
## The term the ENGINE assembles
####################################

# The rhs as the engine sees it, carrying the mass ONLY where the storage fuses
# `M⁻¹` into its own store (`BlockRowAssembly`). It exists for two reasons: the
# inner operator is then built through the ordinary `setup_operator` methods —
# a rate-form integrator would dispatch back into this file — and the mass
# reaches `with_action_storage`, which is keyed on the element cache and sees no
# integrator at all.
struct _RateFormRHS{R, M} <: AbstractBilinearIntegrator
    rhs::R
    mass::M
end

setup_element_cache(i::_RateFormRHS, sdh::SubDofHandler) =
    _rate_form_cache(setup_element_cache(i.rhs, sdh), i.mass)
_rate_form_cache(cache, ::Nothing) = cache
_rate_form_cache(cache, mass) = RateFormElementCache(cache, mass)

global_dofs(i::_RateFormRHS, sdh::SubDofHandler) = global_dofs(i.rhs, sdh)
facet_items(i::_RateFormRHS, sdh::SubDofHandler) = facet_items(i.rhs, sdh)
facet_item_global_dofs(i::_RateFormRHS, sdh::SubDofHandler) = facet_item_global_dofs(i.rhs, sdh)
setup_facet_item_cache(i::_RateFormRHS, sdh::SubDofHandler) = setup_facet_item_cache(i.rhs, sdh)
algebraic_items(i::_RateFormRHS, dh::AbstractDofHandler) = algebraic_items(i.rhs, dh)
setup_algebraic_cache(i::_RateFormRHS, dh::AbstractDofHandler) = setup_algebraic_cache(i.rhs, dh)
function _declaration_subjects!(subjects, integrator::_RateFormRHS)
    push!(subjects, integrator)
    _declaration_subjects!(subjects, integrator.rhs)
    return subjects
end

# Carries the mass to the one storage that fuses `M⁻¹` into its own store; the
# store's cache takes the mass and wraps the rhs cache directly.
struct RateFormElementCache{Inner, M} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
    mass::M
end
rewrap(d::RateFormElementCache, inner) = RateFormElementCache(inner, d.mass)

with_action_storage(d::RateFormElementCache, storage::BlockRowAssembly, sdh::SubDofHandler) =
    _block_row_storage(d.inner, sdh, d.mass)

####################################
## The mass sweep
####################################

# Hands every cell's element mass to `f(slot, cellid, cell, Mₑ)`, `Mₑ` in the
# shape the cache's `element_matrix_structure` declares. A host cell sweep of
# its own: a mass is a cell term and the rhs need not be one.
function foreach_element_mass(f, integrator, sdh::SubDofHandler, p, ctx)
    cache = setup_element_cache(integrator, sdh)
    _assert_mass_matrix_kernel(cache)
    Mₑ    = allocate_element_matrix(cache, sdh)
    it    = assembly_iterator(nothing, cache, sdh)
    flags = item_update_flags(nothing, cache)
    for (slot, cellid) in enumerate(sdh.cellset)
        cell = position_iterator(it, cellid, flags)
        reinit_values!(cache, cell)
        fill!(Mₑ, zero(eltype(Mₑ)))
        assemble_cell!(JacobianRequest{:u}(Mₑ), cache,
                       CellArgs((;), cell, query_cell_parameters(cache, cell, p), ctx))
        f(slot, cellid, cell, Mₑ)
    end
    return nothing
end

# The mass rides a CELL sweep of its own ([`foreach_element_mass`](@ref)), which
# has no local-system tail, no facet items and no algebraic items to visit. A
# mass declaring any of them would be assembled as a strict subset of itself,
# silently, so the rate form refuses it here instead.
function _assert_cell_only_mass(mass, sdh, dh)
    for (hook, declaration) in ((global_dofs(mass, sdh), "global_dofs"),
                                (facet_items(mass, sdh), "facet_items"),
                                (algebraic_items(mass, dh), "algebraic_items"))
        isempty(hook) || throw(ArgumentError(
            "The mass of a rate form declares `$(declaration)`, and `M⁻¹` is derived from a CELL " *
            "sweep over the mass's element matrices alone — the declared items and dofs would be " *
            "dropped from it without a word. Invert a mass supported on the cells."))
    end
    return nothing
end

function _assert_mass_matrix_kernel(cache)
    provides_analytic(typeof(cache), JacobianKind{:u}()) || throw(ArgumentError(
        "A rate form reads its mass's ELEMENT MATRIX, and $(nameof(typeof(cache))) declares no " *
        "analytic `JacobianKind{:u}` kernel (`provides_analytic`). A mass whose element matrix is " *
        "only reachable by differentiating its residual kernel cannot be the `M` of `M⁻¹A`."))
    return nothing
end

####################################
## The retained inverse mass
####################################

# `M⁻¹` of a diagonal mass: the reciprocal of the ASSEMBLED global diagonal, so
# one datum serves a continuous space (several cells per entry) as well as a
# discontinuous one. The sweep accumulates in `host`; `minv` is the same array
# on a host device, its device mirror otherwise.
struct DiagonalInverseMass{HV, DV}
    host::HV
    minv::DV
end

# `M⁻¹` of a dense mass over a cell-disjoint space: one inverse block per cell
# with that cell's dofs beside it, one entry per subdomain in
# `dh.subdofhandlers` order.
struct BlockInverseMass{T}
    blocks::Vector{Array{T, 3}}    # (nslots, Nb, Nb)
    dofs::Vector{Matrix{Int}}      # (nslots, Nb)
end

# The structural election itself: ONE treatment for the whole handler. It runs
# whether or not the storage goes on to keep `M⁻¹` as data, so a mass this
# package cannot invert cell-locally is refused with the rate form's own
# message and not with some store's.
function _resolve_mass_structure(mass, dh::AbstractDofHandler)
    isempty(dh.subdofhandlers) && throw(ArgumentError(
        "A rate form needs a mass to invert and this `DofHandler` carries no subdomain."))
    structures = map(dh.subdofhandlers) do sdh
        _assert_cell_only_mass(mass, sdh, dh)
        cache = setup_element_cache(mass, sdh)
        _assert_mass_matrix_kernel(cache)
        element_matrix_structure(cache)
    end
    allequal(map(typeof, structures)) || throw(ArgumentError(
        "The mass of a rate form declares " *
        "$(join(map(s -> string(nameof(typeof(s)), "()"), structures), ", ")) across this " *
        "handler's subdomains. `M⁻¹` is ONE treatment for the whole operator, so the mass must " *
        "have the same `element_matrix_structure` on every subdomain."))
    structure = first(structures)
    structure isa DenseElementMatrix && _assert_cell_local_mass(dh)
    return structure
end

# A DENSE mass is invertible cell by cell exactly where every dof belongs to ONE
# cell — of the whole HANDLER, not of one subdomain: a dof shared across a
# subdomain seam would otherwise pass each subdomain's own check and have `M⁻¹`
# applied to its row once per subdomain.
function _assert_cell_local_mass(dh::AbstractDofHandler)
    seen = falses(ndofs(dh))
    for sdh in dh.subdofhandlers
        d = _shared_cell_dof(sdh, seen)
        d == 0 || throw(ArgumentError(
            "The mass of a rate form has a DENSE element matrix over a CONTINUOUS space: dof $(d) " *
            "is shared by two cells of this handler, so `M` is not block diagonal by cell and " *
            "`M⁻¹` is a global solve this package will not hide inside an operator. Two escapes, both a " *
            "DISCRETIZATION decision and neither of them silent: wrap the mass in " *
            "`RowSumLumped(mass)`, whose element matrix IS its diagonal, or give the model a " *
            "collocated (spectral) mass element declaring " *
            "`element_matrix_structure(cache) = DiagonalElementMatrix()`."))
    end
    return nothing
end

function _build_inverse_mass(::DiagonalElementMatrix, strategy, dh)
    host = zeros(value_type(strategy.device), ndofs(dh))
    return DiagonalInverseMass(host, adapt_shared(strategy.device, host))
end

function _build_inverse_mass(::DenseElementMatrix, strategy, dh)
    T = value_type(strategy.device)
    blocks = Array{T, 3}[]
    dofs   = Matrix{Int}[]
    for sdh in dh.subdofhandlers
        nb = ndofs_per_cell(sdh)
        push!(blocks, zeros(T, length(sdh.cellset), nb, nb))
        D   = Matrix{Int}(undef, length(sdh.cellset), nb)
        buf = Vector{Int}(undef, nb)
        for (slot, cellid) in enumerate(sdh.cellset)
            celldofs!(buf, sdh.dh, cellid)
            D[slot, :] .= buf
        end
        push!(dofs, D)
    end
    return BlockInverseMass(blocks, dofs)
end

function _refresh_inverse_mass!(data::DiagonalInverseMass, mass, dh, p, ctx)
    m = data.host
    fill!(m, zero(eltype(m)))
    for sdh in dh.subdofhandlers
        buf = Vector{Int}(undef, ndofs_per_cell(sdh))
        foreach_element_mass(mass, sdh, p, ctx) do slot, cellid, cell, mₑ
            celldofs!(buf, sdh.dh, cellid)
            for i in eachindex(buf)
                @inbounds m[buf[i]] += mₑ[i]
            end
        end
    end
    for d in eachindex(m)
        v = @inbounds m[d]
        iszero(v) && throw(ArgumentError(
            "The assembled lumped mass has a zero diagonal entry at dof $(d), so `M⁻¹` does not " *
            "exist. A rate form checks its mass STRUCTURALLY at setup; a vanishing entry is a " *
            "property of its VALUES — a dof no cell of this handler carries mass for, or a " *
            "density that integrates to zero."))
        @inbounds m[d] = inv(v)
    end
    data.minv === m || copyto!(data.minv, m)
    return data
end

function _refresh_inverse_mass!(data::BlockInverseMass, mass, dh, p, ctx)
    for (index, sdh) in enumerate(dh.subdofhandlers)
        blocks = data.blocks[index]
        foreach_element_mass(mass, sdh, p, ctx) do slot, cellid, cell, Mₑ
            @inbounds blocks[slot, :, :] .= inv(Mₑ)
        end
    end
    return data
end

# `y ← M⁻¹y`. The block route's cells are dof-disjoint, so their order is free.
_apply_inverse_mass!(y::AbstractVector, data::DiagonalInverseMass) = (y .*= data.minv; y)

function _apply_inverse_mass!(y::AbstractVector, data::BlockInverseMass)
    for (blocks, D) in zip(data.blocks, data.dofs)
        nb = size(D, 2)
        v  = zeros(eltype(y), nb)
        w  = zeros(eltype(y), nb)
        for slot in axes(D, 1)
            for i in 1:nb
                @inbounds v[i] = y[D[slot, i]]
            end
            mul!(w, (@inbounds @view blocks[slot, :, :]), v)
            for i in 1:nb
                @inbounds y[D[slot, i]] = w[i]
            end
        end
    end
    return y
end

####################################
## Fusing `M⁻¹` into an assembled matrix
####################################

# Per cell, the `nonzeros(A)` positions of its `Nb` rows laid out `(Nb, ncols)`,
# built once against the allocated pattern. A cell's rows must share ONE column
# set; a ragged block row is refused rather than partly fused.
struct BlockRowPlan
    idx::Vector{Int}        # flat `(Nb, ncols)` index block per slot
    offsets::Vector{Int}    # slot → its first entry, `nslots + 1` terminated
end

_build_fusion_plan(::DiagonalInverseMass, A, dh) = nothing
_build_fusion_plan(data::BlockInverseMass, A, dh) = [_block_row_plan(A, D) for D in data.dofs]

function _block_row_plan(A::AbstractSparseMatrixCSC, D::Matrix{Int})
    rowptr, rowidx, rowcol = _csc_row_index(A)
    nb      = size(D, 2)
    idx     = Int[]
    offsets = Vector{Int}(undef, size(D, 1) + 1)
    for slot in axes(D, 1)
        offsets[slot] = length(idx) + 1
        r1    = D[slot, 1]
        ncols = rowptr[r1 + 1] - rowptr[r1]
        base  = length(idx)
        resize!(idx, base + nb * ncols)
        for i in 1:nb
            r = D[slot, i]
            (rowptr[r + 1] - rowptr[r]) == ncols || _throw_ragged_block_row(r1, r)
            for c in 1:ncols
                rowcol[rowptr[r] + c - 1] == rowcol[rowptr[r1] + c - 1] || _throw_ragged_block_row(r1, r)
                idx[base + (c - 1) * nb + i] = rowidx[rowptr[r] + c - 1]
            end
        end
    end
    offsets[end] = length(idx) + 1
    return BlockRowPlan(idx, offsets)
end

_block_row_plan(A, D) = throw(ArgumentError(
    "A rate form with a DENSE mass fuses the per-cell block solve into the assembled matrix, " *
    "which this package does for a host `SparseMatrixCSC` and not for $(typeof(A)). Assemble the " *
    "operator on a host device, or elect `MatrixFreeAction(; storage = BlockRowAssembly())`, " *
    "whose store fuses `M⁻¹` per cell without a global matrix."))

@noinline _throw_ragged_block_row(r1, r) = throw(ArgumentError(
    "The sparsity pattern couples rows $(r1) and $(r) of ONE cell to different columns, so that " *
    "cell's block row is ragged and the per-cell solve `M⁻¹A` cannot be applied in place. A rate " *
    "form with a dense mass needs the whole cell block present in the pattern — declare the " *
    "coupling through `StandardOperatorSpecification(; sparsity_entries = …)`."))

# `nonzeros` positions and their columns, grouped by ROW: one counting pass over
# the CSC, so the plan above reads a row's entries without searching the columns.
function _csc_row_index(A::AbstractSparseMatrixCSC)
    nrows  = size(A, 1)
    rows   = rowvals(A)
    rowptr = zeros(Int, nrows + 1)
    for k in eachindex(rows)
        rowptr[rows[k] + 1] += 1
    end
    rowptr[1] = 1
    for r in 2:(nrows + 1)
        rowptr[r] += rowptr[r - 1]
    end
    cursor = copy(rowptr)
    rowidx = Vector{Int}(undef, length(rows))
    rowcol = Vector{Int}(undef, length(rows))
    for j in axes(A, 2)
        for k in nzrange(A, j)
            r = rows[k]
            p = cursor[r]
            rowidx[p] = k
            rowcol[p] = j
            cursor[r] = p + 1
        end
    end
    return rowptr, rowidx, rowcol
end

# `A ← M⁻¹A` in place on the pattern `A` already has — `M⁻¹` is diagonal, or
# block diagonal over dofs one cell owns, so the sparsity is preserved exactly.
function _fuse_inverse_mass!(A::AbstractSparseMatrixCSC, data::DiagonalInverseMass, ::Nothing)
    minv = data.host
    rows = rowvals(A)
    vals = nonzeros(A)
    for k in eachindex(vals)
        @inbounds vals[k] *= minv[rows[k]]
    end
    return A
end

function _fuse_inverse_mass!(A::AbstractSparseMatrixCSC, data::BlockInverseMass, plans::Vector{BlockRowPlan})
    vals = nonzeros(A)
    for (blocks, plan) in zip(data.blocks, plans)
        nb = size(blocks, 2)
        v  = zeros(eltype(vals), nb)
        w  = zeros(eltype(vals), nb)
        for slot in axes(blocks, 1)
            base  = plan.offsets[slot] - 1
            ncols = (plan.offsets[slot + 1] - plan.offsets[slot]) ÷ nb
            M     = @inbounds @view blocks[slot, :, :]
            for c in 1:ncols
                for i in 1:nb
                    @inbounds v[i] = vals[plan.idx[base + (c - 1) * nb + i]]
                end
                mul!(w, M, v)
                for i in 1:nb
                    @inbounds vals[plan.idx[base + (c - 1) * nb + i]] = w[i]
                end
            end
        end
    end
    return A
end

_fuse_inverse_mass!(A, data, plan) = throw(ArgumentError(
    "A rate form under `FullAssembly` fuses `M⁻¹` into the assembled matrix in place, which this " *
    "package does for a host `SparseMatrixCSC` and not for $(typeof(A))."))

####################################
## Where `M⁻¹` is applied
####################################

# One singleton/struct per realization, so every entry point below dispatches
# instead of branching. `StoreFusedMass` carries no data at all: the inner
# store's own fill re-derives and applies the mass.
struct StoreFusedMass end

# `M⁻¹` fused into the inner operator's assembled matrix after every fill.
struct MatrixFusedMass{D, P}
    data::D
    plan::P
end

# `M⁻¹` applied to the inner operator's action, or to a load vector.
struct AppliedMass{D}
    data::D
end

# The MATRIX route holds `M⁻¹A`, so the matrix action needs nothing further —
# but `evaluate!` runs the element RESIDUAL kernels, which know only `A`, so the
# weighting is applied to the result there. The two hooks are exactly that
# distinction and not a duplicate.
_weight_action!(::StoreFusedMass, y) = y
_weight_action!(::MatrixFusedMass, y) = y
_weight_action!(t::AppliedMass, y) = _apply_inverse_mass!(y, t.data)

_weight_evaluation!(::StoreFusedMass, y) = y
_weight_evaluation!(t::MatrixFusedMass, y) = _apply_inverse_mass!(y, t.data)
_weight_evaluation!(t::AppliedMass, y) = _apply_inverse_mass!(y, t.data)

_refresh_treatment!(::StoreFusedMass, mass, dh, p, ctx) = nothing
_refresh_treatment!(t, mass, dh, p, ctx) = _refresh_inverse_mass!(t.data, mass, dh, p, ctx)

_fuse_treatment!(::StoreFusedMass, inner) = nothing
_fuse_treatment!(::AppliedMass, inner) = nothing
_fuse_treatment!(t::MatrixFusedMass, inner) = (_fuse_inverse_mass!(get_matrix(inner), t.data, t.plan); nothing)

####################################
## The operators
####################################

"""
    RateFormFerriteOperator <: AbstractBilinearOperator

The operator [`setup_operator`](@ref) returns for a
[`BilinearRateFormIntegrator`](@ref): the rhs's own operator, plus wherever
`M⁻¹` is applied ([`RateFormIntegrator`](@ref) tabulates the realizations).

Its surface is the inner operator's — `mul!` in both forms,
[`evaluate!`](@ref), [`update_operator!`](@ref), `size`, `eltype`, and
`get_matrix` where the inner has a matrix, which is then the FUSED `M⁻¹A` and
not `A`. `y` and `u` must not alias, the matrix-free inner operator's rule.

!!! warning "Experimental surface"
    This operator may change in a minor release.
"""
@concrete struct RateFormFerriteOperator <: AbstractBilinearOperator
    inner
    integrator
    treatment
    scratch        # the unweighted action of the 5-argument `mul!`; `nothing` where unused
end

Base.size(op::RateFormFerriteOperator) = size(op.inner)
Base.size(op::RateFormFerriteOperator, axis) = size(op.inner, axis)
Base.eltype(op::RateFormFerriteOperator) = eltype(op.inner)
get_matrix(op::RateFormFerriteOperator) = get_matrix(op.inner)
operator_payload(op::RateFormFerriteOperator) = operator_payload(op.inner)
get_dof_handler(op::RateFormFerriteOperator) = get_dof_handler(op.inner)
get_strategy(op::RateFormFerriteOperator) = get_strategy(op.inner)
get_subdomain_caches(op::RateFormFerriteOperator) = get_subdomain_caches(op.inner)

"""
    rate_form_rhs(op) -> operator

The operator of the rate form's right-hand side — the one `M⁻¹` weights. Where
it holds a matrix, that matrix is ALREADY fused: a rate form is `M⁻¹A` and
keeps no unweighted copy.
"""
rate_form_rhs(op::RateFormFerriteOperator) = op.inner

mul!(y::AbstractVector, op::RateFormFerriteOperator, u::AbstractVector) =
    (mul!(y, op.inner, u); _weight_action!(op.treatment, y); y)

function mul!(y::AbstractVector, op::RateFormFerriteOperator, u::AbstractVector, α, β)
    op.scratch === nothing && return mul!(y, op.inner, u, α, β)
    mul!(op.scratch, op.inner, u)
    _weight_action!(op.treatment, op.scratch)
    # `β = 0` ASSIGNS rather than scales, the LinearAlgebra convention a `NaN`
    # already in `y` would otherwise propagate through.
    if iszero(β)
        y .= α .* op.scratch
    else
        y .= α .* op.scratch .+ β .* y
    end
    return y
end

evaluate!(op::RateFormFerriteOperator, y::AbstractVector, states::NamedTuple, p, ctx) =
    (evaluate!(op.inner, y, states, p, ctx); _weight_evaluation!(op.treatment, y); y)
evaluate!(op::RateFormFerriteOperator, y::AbstractVector, u::AbstractVector, p) =
    evaluate!(op, y, (u = u,), p, nothing)

"""
    update_operator!(op::RateFormFerriteOperator, p, ctx = nothing)

Refill the rhs and re-derive `M⁻¹` at `(p, ctx)`, then re-apply the weighting
the storage elected. The mass is swept here exactly as the rhs is, so the two
halves of `M⁻¹A` are never evaluated at different points.
"""
function update_operator!(op::RateFormFerriteOperator, p, ctx = nothing)
    update_operator!(op.inner, p, ctx)
    _refresh_treatment!(op.treatment, op.integrator.mass, get_dof_handler(op), p, ctx)
    _fuse_treatment!(op.treatment, op.inner)
    return nothing
end

"""
    LinearRateFormFerriteOperator <: AbstractLinearOperator

The operator [`setup_operator`](@ref) returns for a
[`LinearRateFormIntegrator`](@ref): the rhs's load vector, weighted — `op.b` IS
`M⁻¹b`, which is what a rate-form source contributes to `du/dt`.

!!! warning "Experimental surface"
    This operator may change in a minor release.
"""
@concrete struct LinearRateFormFerriteOperator <: AbstractLinearOperator
    b
    engine
    integrator
    data
end

"""
    update_operator!(op::LinearRateFormFerriteOperator, p, ctx = nothing)

Assemble the rhs into `op.b`, re-derive `M⁻¹` at the same `(p, ctx)` and weight
the vector — the entry point a time-varying source is refilled through.
"""
function update_operator!(op::LinearRateFormFerriteOperator, p, ctx = nothing)
    assemble_into!(LinearKind(), (op.b,), op, (;), p, ctx)
    _refresh_inverse_mass!(op.data, op.integrator.mass, op.engine.dh, p, ctx)
    _apply_inverse_mass!(op.b, op.data)
    return nothing
end

####################################
## Setup
####################################

"""
    setup_operator(strategy, integrator::BilinearRateFormIntegrator, dh; …)
    setup_operator(strategy, integrator::LinearRateFormIntegrator, dh; …)

Build the rate form's operator: the rhs's own operator under `strategy`, plus
the `M⁻¹` treatment the mass's structure and the storage elect
([`RateFormIntegrator`](@ref) tabulates both, and states which pairs are
refused). `initial_parameters`/`initial_context` are the pair the mass — and,
under [`MatrixFreeAction`](@ref), the rhs store — are first evaluated with.

An ASSEMBLED rate form holds an unfilled matrix until the first
[`update_operator!`](@ref), exactly as any other [`FullAssembly`](@ref) operator
does; what setup does evaluate is the mass, so a singular one is an error here
and not at the first refill.
"""
setup_operator(strategy::AbstractAssemblyStrategy, integrator::BilinearRateFormIntegrator,
        dh::AbstractDofHandler; kwargs...) = _setup_rate_form(strategy, integrator, dh; kwargs...)
setup_operator(strategy::AbstractAssemblyStrategy, integrator::LinearRateFormIntegrator,
        dh::AbstractDofHandler; kwargs...) = _setup_rate_form(strategy, integrator, dh; kwargs...)

# The matrix-free form has `setup_operator` methods of its OWN for the bilinear
# and linear families, each more specific in the STRATEGY than the two above and
# less specific in the integrator: without these two the pair would be
# ambiguous, and a rate form under `MatrixFreeAction` — the shipped GPU route —
# would not resolve at all.
setup_operator(strategy::AssemblyStrategy{<:MatrixFreeAction}, integrator::BilinearRateFormIntegrator,
        dh::AbstractDofHandler; kwargs...) = _setup_rate_form(strategy, integrator, dh; kwargs...)
setup_operator(strategy::AssemblyStrategy{<:MatrixFreeAction}, integrator::LinearRateFormIntegrator,
        dh::AbstractDofHandler; kwargs...) = _setup_rate_form(strategy, integrator, dh; kwargs...)

function _setup_rate_form(strategy, integrator::BilinearRateFormIntegrator, dh::AbstractDofHandler;
        initial_parameters = nothing, initial_context = nothing, kwargs...)
    structure = _resolve_mass_structure(integrator.mass, dh)
    # The store-fused route keeps `M⁻¹` nowhere but in the store it fuses into,
    # so no inverse-mass data is built for it at all. What the storage cannot
    # weight is refused HERE, before the rhs operator is built: the matrix-free
    # setup fills its store, and a refusal after that fill would have paid for a
    # whole sweep to say no.
    fused = _fuses_in_store(strategy.form)
    fused || _assert_weightable(strategy.form, structure)
    inner = _setup_rate_form_rhs(strategy, _RateFormRHS(integrator.rhs, fused ? integrator.mass : nothing),
                                 dh, initial_parameters, initial_context; kwargs...)
    treatment = fused ? StoreFusedMass() :
        _weighting_treatment(strategy.form, _build_inverse_mass(structure, strategy, dh), inner, dh)
    _refresh_treatment!(treatment, integrator.mass, dh, initial_parameters, initial_context)
    return RateFormFerriteOperator(inner, integrator, treatment, _action_scratch(treatment))
end

function _setup_rate_form(strategy, integrator::LinearRateFormIntegrator, dh::AbstractDofHandler;
        initial_parameters = nothing, initial_context = nothing, kwargs...)
    _assert_assembled_linear_rate_form(strategy.form)
    data  = _build_inverse_mass(_resolve_mass_structure(integrator.mass, dh), strategy, dh)
    inner = setup_operator(strategy, integrator.rhs, dh; kwargs...)
    _refresh_inverse_mass!(data, integrator.mass, dh, initial_parameters, initial_context)
    return LinearRateFormFerriteOperator(inner.b, inner.engine, integrator, data)
end

# A linear form has no action for a matrix-free operator to evaluate, so its
# rate form has none either. The refusal is spelled here rather than left to the
# inner `setup_operator`'s, which names the wrapped integrator and says nothing
# about the pair the caller actually wrote.
_assert_assembled_linear_rate_form(form) = nothing
_assert_assembled_linear_rate_form(form::MatrixFreeAction) = throw(ArgumentError(
    "A rate form over a LINEAR right-hand side carries the load vector `M⁻¹b` and has no action " *
    "to evaluate, so it has no `MatrixFreeAction` realization. Assemble the source under " *
    "`FullAssembly` — the vector it holds is what a matrix-free rate operator's own consumer adds " *
    "to `du/dt` — and keep the matrix-free strategy for the bilinear term beside it."))

# The initial evaluation pair is the MATRIX-FREE setup's, that form being the
# one that fills a store at setup; an assembling form fills at
# `update_operator!` and takes no such pair.
_setup_rate_form_rhs(strategy, term, dh, p, ctx; kwargs...) = setup_operator(strategy, term, dh; kwargs...)
_setup_rate_form_rhs(strategy::AssemblyStrategy{<:MatrixFreeAction}, term, dh, p, ctx; kwargs...) =
    setup_operator(strategy, term, dh; initial_parameters = p, initial_context = ctx, kwargs...)

# Only the block-row store fuses `M⁻¹` into what it keeps; `FullAssembly` fuses
# into the matrix afterwards, and the other matrix-free levels keep no row to
# fuse into at all.
_fuses_in_store(form) = false
_fuses_in_store(form::MatrixFreeAction) = form.storage isa BlockRowAssembly

# Which mass a storage that keeps `M⁻¹` as DATA can weight with, decided off the
# structure alone so the refusal precedes the rhs operator's construction.
_assert_weightable(form, structure) = nothing
_assert_weightable(form::MatrixFreeAction, ::DenseElementMatrix) = throw(ArgumentError(
    "A rate form with a DENSE (per-cell block) mass keeps no row to fuse `M⁻¹` into under " *
    "`storage = $(nameof(typeof(form.storage)))()`, so every action would end in a per-cell block " *
    "solve over the whole result — work this package does not hide inside a `mul!`. Elect " *
    "`storage = BlockRowAssembly()`, whose store carries the cell's whole row and fuses `M⁻¹` " *
    "into it once per fill; assemble the operator under `FullAssembly`; or lump the mass " *
    "(`RowSumLumped`), whose inverse is a scaling of the result."))

_weighting_treatment(form::FullAssembly, data, inner, dh) =
    MatrixFusedMass(data, _build_fusion_plan(data, get_matrix(inner), dh))

# Every mass reaching here is diagonal — `_assert_weightable` refused the rest.
_weighting_treatment(form::MatrixFreeAction, data, inner, dh) = AppliedMass(data)

_weighting_treatment(form, data, inner, dh) = throw(ArgumentError(
    "A rate form has no realization under $(nameof(typeof(form))). The shipped forms are " *
    "`FullAssembly` — the rhs matrix, row-scaled or block-solved — and `MatrixFreeAction`."))

# The 5-argument `mul!` of an action-weighted operator needs the unweighted
# action somewhere before it can scale it; every other treatment scales nothing
# there and forwards to the inner operator's own five-argument form.
_action_scratch(treatment) = nothing
_action_scratch(t::AppliedMass) = similar(t.data.minv)
