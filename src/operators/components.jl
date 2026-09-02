####################################
## Component bags over a shared sparsity pattern
####################################

"""
    allocate_components(op, names::NTuple{N, Symbol}) -> NamedTuple of matrices

Allocate `N` square system matrices for `op` that share ONE sparsity pattern
(see [`share_pattern`](@ref)). Each is a plain system matrix — assemble into it
through the ordinary entry points, fold the bag with [`combine!`](@ref).

The shared pattern makes `combine!` a pure `nzval` operation, and is a contract:
structural mutation of a component (`dropzeros!`, inserting an entry) breaks the
whole bag. `apply_zero!`-style value mutation is fine.

```julia
comps = allocate_components(op, (:Ju, :Jdu))
assemble_slot_jacobian!(comps.Ju,  op, JacobianKind{:u}(),  states, p, ctx)
assemble_slot_jacobian!(comps.Jdu, op, JacobianKind{:du}(), states, p, ctx)
combine!(W, comps, (Jdu = 1 / Δt, Ju = 1.0))   # backward Euler Newton matrix
```
"""
function allocate_components(op, names::NTuple{N, Symbol}) where {N}
    N == 0 && throw(ArgumentError("A component bag needs at least one name."))
    allunique(names) || throw(ArgumentError("Component names must be unique, got $names."))
    A = create_system_matrix(op.engine.strategy, op.engine.dh)
    return NamedTuple{names}((A, ntuple(_ -> share_pattern(A), N - 1)...))
end

"""
    share_pattern(A::SparseMatrixCSC, T = eltype(A)) -> SparseMatrixCSC

A `T`-valued matrix aliasing `A`'s `colptr`/`rowval` with a fresh zeroed
`nzval` — "same pattern group as `A`". Pass `T = ComplexF64` for the
combination target of a transformed Radau stage.
"""
share_pattern(A::SparseMatrixCSC, ::Type{T} = eltype(A)) where {T} =
    SparseMatrixCSC(size(A, 1), size(A, 2), getcolptr(A), rowvals(A), zeros(T, nnz(A)))
share_pattern(A::AbstractMatrix, ::Type = Float64) = throw(ArgumentError(
    "Component bags are currently implemented for `SparseMatrixCSC` only, got $(typeof(A))."))

"""
    combine!(W, comps::NamedTuple, weights::NamedTuple) -> W

Fold `W = Σ weights[k] · comps[k]` over the names in `weights`, a subset of
`comps`. Values only: `W` and the components must share the sparsity pattern
(checked), and `W.nzval` is overwritten. Real components with complex weights
combine into a `Complex` `W` — what transformed (diagonalized) Radau needs per
eigenvalue of the tableau inverse.
"""
function combine!(W::SparseMatrixCSC, comps::NamedTuple, weights::NamedTuple)
    isempty(weights) && throw(ArgumentError("`weights` names no component to combine."))
    for k in keys(weights)
        haskey(comps, k) || throw(ArgumentError(
            "`weights` names `:$k`, which is not a component of the bag $(keys(comps))."))
        assert_shared_pattern(W, comps[k], k)
    end
    fill!(W.nzval, zero(eltype(W)))
    for (k, w) in pairs(weights)
        nz = comps[k].nzval
        @. W.nzval += w * nz
    end
    return W
end

# Aliased index arrays are the constructed case and settle the check outright;
# a `W` allocated elsewhere is compared entrywise instead of trusted.
function assert_shared_pattern(W::SparseMatrixCSC, C::SparseMatrixCSC, name::Symbol)
    size(W) == size(C) || throw(DimensionMismatch(
        "component `:$name` has size $(size(C)), target has $(size(W))."))
    nnz(W) == nnz(C) || throw(ArgumentError(
        "component `:$name` has $(nnz(C)) stored entries, target has $(nnz(W)) — " *
        "combined matrices must share one sparsity pattern."))
    getcolptr(W) === getcolptr(C) && rowvals(W) === rowvals(C) && return nothing
    (getcolptr(W) == getcolptr(C) && rowvals(W) == rowvals(C)) || throw(ArgumentError(
        "component `:$name` does not share the target's sparsity pattern. Allocate " *
        "components with `allocate_components` and targets with `share_pattern`."))
    return nothing
end
