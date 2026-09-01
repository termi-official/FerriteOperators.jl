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

FO drives the sweep and assembles; everything a local BVP does per patch —
factorize, solve, emit — happens in a callback the caller writes:

```julia
provider = PatchItems(sdh, cellsets; prescribed_facets)
sink     = PatchTripletSink()
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

Rows and columns of a patch-local target follow [`patch_dofs`](@ref), the
injection from patch-local into global numbering. Solving the local system is
deliberately **not** part of the contract: the caller owns the solve.
[`patch_free_dofs`](@ref) / [`patch_prescribed_dofs`](@ref) describe the
interior/boundary split, [`augment_prescribed_dofs!`](@ref) pins whatever else
the local problem needs to be well posed, and [`ItemStates`](@ref) is the
item-lifetime storage a retained factorization belongs in — neither per-worker
scratch nor operator-global frozen data.

The pieces and their contracts:

- [`foreach_patch`](@ref) calls `f(pws, patchid)` once per patch, in ascending
  item order, with the patch workspace already positioned on the patch. Its
  `items` argument names a subset (default: every patch); only those patches
  are visited, which is what a partial re-solve needs — FO writes nothing
  outside `f`, so what an untouched patch contributed to an earlier sweep stays
  as it was.
- [`assemble_patch_target!`](@ref) **accumulates** into a target it never
  zeroes, so the caller decides what a call adds to. Whether the element's
  Jacobian or residual kernels run follows the target: a matrix (or a Ferrite
  assembler over one) gets the Jacobian, a vector the residual.
- The target of one call is one column of the result. `N` right-hand sides are
  `N` calls with per-call term data and `N` [`emit_patch_column!`](@ref)
  emissions. Duplicate `(row, col)` entries are summed by `sparse`, so
  overlapping patches contributing to a shared column need no coordination.
- [`ItemStates`](@ref) is the retained-state slot: a factorization lives as
  long as the item, and FO never writes or invalidates it. Nothing about a
  patch is stored anywhere else.

Patch sweeps are pure evaluation: condensed element unknowns are gathered but
never written back.

## Terms, the patch context and sinks

`assemble_patch_target!` takes a *tuple* of [`PatchTerm`](@ref)s. Every term of
one call is evaluated in a single pass over the patch's cells and accumulates
into the same element buffer in tuple order — the accumulation order is part of
the contract. A term's restriction is [`WholePatch`](@ref) or
[`CellGroup`](@ref), and its `data` payload selects either the element's
ordinary cell kernel (`nothing`, over an ordinary [`CellArgs`](@ref)) or an
[`assemble_patch_cell!`](@ref) term kernel. Terms that must fuse per quadrature
point rather than per cell belong in ONE term whose `data` carries both
sources.

A term kernel receives a [`PatchArgs`](@ref): the four fields of `CellArgs`
plus where in the patch the cell sits — the provider, the patch id, the cell's
`group` tag, and `ldofs`, the cell's dofs in patch-local numbering. That last
one is the window a payload indexes patch-local data through, so a term kernel
needs no per-patch handle of its own:

```julia
function FerriteOperators.assemble_patch_cell!(req::ResidualRequest, cache::MyCache,
                                               args::PatchArgs, data::MySource)
    d = @view data.patch_field[args.ldofs]       # patch-local data on this cell
    parent = args.group                          # e.g. the parent coarse cell
    ...
end
```

`args.ldofs` is the driver's scratch, refilled per cell and scattered through
after the kernels run — read it, never retain it past the call.

A sink is where a *finished* patch quantity goes, once the caller has assembled
and solved it. [`PatchTripletSink`](@ref) collects triplets for a rectangular
assembly through [`emit_patch_column!`](@ref), which names the column per call;
[`PatchGlobalVectorSink`](@ref) accumulates a patch-local vector back into a
global one through the injection ([`patch_emit!`](@ref)). A downstream scatter
mode is a new sink type with its own `patch_emit!` method, not a fork of the
per-patch body.

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
  in. It takes the same `items` argument as `foreach_patch`, so a subset sweep
  chunks and merges exactly like a full one.
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

[`foreach_patch`](@ref) itself is sequential for the collector reason above,
and rejects a parallel device loudly. Parallelism lives on the seams in this
section, not behind a flag.

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

Which coarse cell a fine cell belongs to is also what a [`CellGroup`](@ref)
restriction selects on and what `PatchArgs.group` reports, so a term restricted
to one parent cell needs no map of its own.

## Patch API reference

```@autodocs
Modules = [FerriteOperators]
Pages = ["core/patch-task.jl", "core/item_states.jl"]
```
