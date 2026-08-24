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
[`ItemStates`](@ref) is the item-lifetime storage a retained
factorization belongs in — neither per-worker scratch nor operator-global
frozen data. [Driving the patches yourself](@ref) is where that loop is
written.

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
triplets for a rectangular assembly. A sink is three methods —
[`patch_target`](@ref), [`patch_scatter`](@ref), [`patch_emit!`](@ref) — so a
downstream scatter mode is a new sink type, not a fork of the driver.

Patch sweeps are pure evaluation: condensed element unknowns are gathered but
never written back.

## Driving the patches yourself

[`assemble_patches!`](@ref) is one fixed pipeline: one request, one sink, all
patches. A local BVP needs more than that per patch — assemble a matrix, solve
against several right-hand sides, emit each result — so it takes the callback
route instead. [`foreach_patch`](@ref) calls `f(pws, patchid)` once per patch,
in item order, with the patch workspace already positioned on the patch, and
[`assemble_patch_target!`](@ref) is the per-patch assembly primitive callable
as often as the caller likes on that workspace.

```julia
provider = PatchItems(sdh, cellsets; prescribed_facets)
sink     = PatchTripletSink()                    # column-less: emissions name their column
facts    = ItemStates{typeof(lu(zeros(1, 1)))}(npatches(provider))

foreach_patch(op, provider, (u = u,), p) do pws, pid
    n, free = patch_ndofs(provider, pid), patch_free_dofs(provider, pid)
    if !has_item_state(facts, pid)                # retained across sweeps
        K = zeros(n, n)
        assemble_patch_target!(K, (PatchTerm(WholePatch()),), pws, (u = u,), p)
        set_item_state!(facts, pid, lu(K[free, free]))
    end
    F, rows = item_state(facts, pid), patch_dofs(provider, pid)[free]
    rhs = zeros(n)
    for j in 1:ncols
        fill!(rhs, 0.0)                          # the primitive accumulates
        assemble_patch_target!(rhs, (PatchTerm(WholePatch(), MyColumnData(j)),), pws, (u = u,), p)
        emit_patch_column!(sink, rows, (pid - 1) * ncols + j, F \ rhs[free])
    end
end

W = sparse(sink, ndofs(dh), npatches(provider) * ncols)
```

The pieces and their contracts:

- `assemble_patch_target!` **accumulates** into a target it never zeroes, so
  the caller decides what a call adds to. Whether the element's Jacobian or
  residual kernels run follows the target: a matrix (or a Ferrite assembler
  over one) gets the Jacobian, a vector the residual.
- The target of one call is one column of the result. `N` right-hand sides are
  `N` calls with per-call term data and `N` [`emit_patch_column!`](@ref)
  emissions — the sink's `columns` map exists for the pipeline tail only, which
  is why the many-columns-per-patch case uses the column-less
  [`PatchTripletSink`](@ref) and names its columns itself. Duplicate
  `(row, col)` entries are summed by `sparse`, so overlapping patches
  contributing to a shared column need no coordination.
- [`ItemStates`](@ref) is the retained-state slot: a factorization lives
  as long as the item, and FO never writes or invalidates it. Nothing about a
  patch is stored anywhere else.
- The solve, the boundary treatment and the column layout are the caller's.
  [`patch_free_dofs`](@ref) / [`patch_prescribed_dofs`](@ref) give the
  interior/boundary split, and [`augment_prescribed_dofs!`](@ref) pins whatever
  else the local problem needs to be well posed.

## Parallel patch sweeps

Patches are independent work items, so a caller can process them in parallel —
and it *is* the caller who does it. FO ships no threaded patch loop: what a
patch emits is the caller's collector, and FO cannot duplicate what it does not
own. What FO provides are the seams that make a parallel sweep produce exactly
the triplet stream of a sequential one, bit for bit:

```julia
chunks = patch_chunks(provider, nchunks)                     # contiguous, ascending
sinks  = [PatchTripletSink() for _ in chunks]                # one collector per worker
wss    = [patch_workspace(op, provider) for _ in chunks]     # one workspace per worker

@sync for c in eachindex(chunks)
    Threads.@spawn for pid in chunks[c]
        reinit!(wss[c], pid)
        # assemble / solve / emit into sinks[c], exactly as in the callback above
    end
end

merged = PatchTripletSink()
foreach(s -> append!(merged, s), sinks)                      # chunk order ⇒ item order
```

The contract:

- **Chunks are contiguous and ascending.** [`patch_chunks`](@ref) splits the
  items that way; the workers may finish in any order, but merging the
  collectors in chunk order restores the item order a sequential sweep emits
  in.
- **One workspace per worker.** [`patch_workspace`](@ref) builds an independent
  [`PatchAssemblyWorkspace`](@ref) per call, and `duplicate_for_device` copies
  an existing one. Two workspaces share only the provider, which patch sweeps
  read and never write — but a provider mutated at setup time
  ([`augment_prescribed_dofs!`](@ref)) must be finished before any worker
  starts.
- **Collectors are the caller's.** One [`PatchTripletSink`](@ref) per chunk.
  [`PatchGlobalVectorSink`](@ref) is *not* thread safe: overlapping patches hit
  the same entries of the destination with a non-atomic read-modify-write.
- **Item state is disjoint by construction.** [`ItemStates`](@ref) slots
  are indexed by item, so workers processing different patches touch different
  slots. Item lifetime is not worker lifetime — a slot must never be handed to
  a worker-lifetime cache.
- **Bit-identity requires workspace-independent per-patch numerics.** It holds
  because a patch's result depends only on the patch. Anything that makes it
  depend on what the same workspace saw earlier — an accumulating scratch, a
  cache whose fill order changes the values — breaks the guarantee, and no
  amount of chunk ordering restores it.
- **Preparation happens before spawning.** Data that is not safe to fill lazily
  from several workers (a memoized geometric map, a table built on first touch)
  must be complete before the workers start; a sequential warm-up pass over the
  patches is the place for it.

The kind × sink pipelines stay sequential and reject a parallel device loudly:
the sink rides inside the shared `kind`, so every worker would scatter through
the same sink object. [`foreach_patch`](@ref) is sequential for the collector
reason above. Parallelism lives on the seams in this section, not behind a flag.

## Two-grid data and foreign-space evaluation

A local BVP frequently needs data from a function space that is not the
operator's: a coarse space whose basis the patch problem is driven by, an
enrichment space, a field discretized on another grid. FO carries no values
type for that, and none is needed — the pattern is a **downstream-owned values
object**:

1. Evaluate the foreign space with the public Ferrite reference API
   (`reference_shape_value` / `reference_shape_gradient` on the interpolation,
   at the reference points of interest) and store the result in whatever layout
   the term kernel wants.
2. Hand the object to the kernel as a [`PatchTerm`](@ref) `data` payload, where
   it reaches [`assemble_patch_cell!`](@ref) unchanged, or keep it inside a
   downstream element cache.
3. Give the object a `duplicate_for_device` method. It then participates in
   per-worker duplication like every other cache, which is what makes it usable
   from [`patch_workspace`](@ref)-based parallel sweeps.

The reference points themselves are the two-grid question, and that is where
FO's transfer machinery comes in: a [`NestedGridCellCache`](@ref) carries the
fine → coarse cell map and [`get_child_ref_coords`](@ref), the current fine
cell's nodes expressed in the parent coarse reference element. Composing the
fine cell's quadrature points through those coordinates gives the coarse
reference points at which the coarse interpolation is evaluated — one
`reference_shape_value` call per point and basis function, no geometric search.

## Patch API reference

```@autodocs
Modules = [FerriteOperators]
Pages = ["core/patch-task.jl", "core/item_states.jl"]
```
