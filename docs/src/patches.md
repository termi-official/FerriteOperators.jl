```@meta
CurrentModule = FerriteOperators
```

# Patch items

!!! warning "Experimental"
    The patch surface may still change. Everything else in this documentation
    is contract; this page is a preview of the local-BVP layer.

A patch item is a *set* of cells with its own dof numbering — the unit of work
for error estimators, localized-orthogonal-decomposition correctors, and other
methods whose systems live on a neighbourhood rather than on one cell. The
cell kernels are reused unchanged; only the scatter target is patch-local.

```julia
provider = PatchItems(sdh, cellsets)                 # one patch per cell set
dest = [zeros(patch_ndofs(provider, i), patch_ndofs(provider, i)) for i in eachindex(cellsets)]
assemble_patch_matrices!(dest, op, provider, states, p, ctx)
```

Rows and columns of `dest[i]` follow [`patch_dofs`](@ref), the injection from
patch-local into global numbering.

Solving the delivered local system is deliberately **not** part of the
contract: the caller owns the solve. [`patch_free_dofs`](@ref) /
[`patch_prescribed_dofs`](@ref) describe the interior/boundary split, and
[`PatchItemStates`](@ref) is the item-lifetime storage a retained
factorization belongs in — neither per-worker scratch nor operator-global
frozen data.

## Requests, terms and sinks

A patch request is a kind ([`PatchMatrixKind`](@ref) or
[`PatchVectorKind`](@ref)) carrying a tuple of [`PatchTerm`](@ref)s and a
sink. Every term of one request is evaluated in a single pass over the patch's
cells and accumulates into the same element buffer in tuple order — the
accumulation order is part of the contract. A term's restriction is
[`WholePatch`](@ref) or [`CellGroup`](@ref), and its `data` payload selects
either the element's ordinary cell kernel (`nothing`) or an
[`assemble_patch_cell!`](@ref) term kernel.

The sink is the scatter mode: [`PatchLocalSink`](@ref) writes into a
patch-local matrix or vector per item, [`PatchAssemblerSink`](@ref) goes
through a Ferrite assembler per item, [`PatchGlobalVectorSink`](@ref)
accumulates back into a global vector, and [`PatchTripletSink`](@ref) collects
triplets for a rectangular assembly.

Patch sweeps are pure evaluation: condensed element unknowns are gathered but
never written back, unlike the global sweeps. Sequential CPU only.

## Patch API reference

```@autodocs
Modules = [FerriteOperators]
Pages = ["core/patch-task.jl"]
```
