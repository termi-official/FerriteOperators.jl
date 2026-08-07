####################################
## Patch items (experimental)
####################################
#
# First non-cell item family: a work item is a SET of cells with a
# patch-local dof numbering. Cell kernels are reused unchanged; only the
# scatter target is patch-local. The corrector/local-BVP layer (view-derived
# free/prescribed dof partition, local solves, item-lifetime state) is NOT
# part of this surface yet — the contract below covers patch-local matrix
# assembly only and is experimental.

"""
    PatchItems(sdh, cellsets)

Work-item provider over patches: item `i` is the cell set `cellsets[i]`
(cells of `sdh`) with its own contiguous patch-local dof numbering. The
injection map back to global dofs is [`patch_dofs`](@ref).

Experimental: the patch item surface may change with the local-BVP work.
"""
struct PatchItems{SDH <: SubDofHandler}
    sdh::SDH
    patches::Vector{Vector{Int}}
    dofs::Vector{Vector{Int}}        # per patch: local → global dof (the injection)
    dofmaps::Vector{Dict{Int, Int}}  # per patch: global → local dof
end

function PatchItems(sdh::SubDofHandler, cellsets)
    dh = sdh.dh
    patches = [sort!(collect(Int, cs)) for cs in cellsets]
    for cells in patches
        issubset(cells, sdh.cellset) || throw(ArgumentError(
            "patch cells $(setdiff(cells, sdh.cellset)) are not part of the SubDofHandler's cellset"))
    end
    dofs = map(patches) do cells
        gd = Int[]
        for c in cells
            append!(gd, celldofs(dh, c))
        end
        sort!(unique!(gd))
    end
    dofmaps = [Dict{Int, Int}(g => l for (l, g) in pairs(d)) for d in dofs]
    return PatchItems(sdh, patches, dofs, dofmaps)
end

npatches(provider::PatchItems) = length(provider.patches)

"Global dofs of patch `i` in patch-local order (the local→global injection)."
patch_dofs(provider::PatchItems, i::Int) = provider.dofs[i]

"Number of dofs of patch `i`."
patch_ndofs(provider::PatchItems, i::Int) = length(provider.dofs[i])

compute_partition(::SequentialScheduling, provider::PatchItems) = (collect(1:npatches(provider)),)

# Wraps a cell workspace; the item is a patch index resolved through the
# provider.
@concrete struct PatchAssemblyWorkspace <: AbstractWorkspace
    provider
    current   # Ref{Int}: the patch index of the item being processed
    inner     # the per-worker cell AssemblyWorkspace
end

Ferrite.reinit!(ws::PatchAssemblyWorkspace, patchid::Int) = (ws.current[] = patchid; nothing)

"Patch-local Jacobian assembly: fills `dest[i]` (a matrix over patch `i`'s dofs)."
struct PatchJacobianKind{D}
    dest::D
end

execute_single_task!(task::AssemblyTask, ws::PatchAssemblyWorkspace) = execute_kind!(task.kind, task, ws)

function execute_kind!(kind::PatchJacobianKind, task, ws::PatchAssemblyWorkspace)
    pid    = ws.current[]
    dofmap = ws.provider.dofmaps[pid]
    KP     = kind.dest[pid]
    iws    = ws.inner
    for cellid in ws.provider.patches[pid]
        reinit!(iws.cell, cellid)
        fill!(iws.Ke, 0.0)
        reinit_values!(iws.element, iws.cell, JacobianKind())
        pₑ = query_cell_parameters(iws.element, iws.cell, task.p)
        statesₑ = load_slots!(iws, task.states)
        v2_cell_kernel!(JacobianKind(), iws.element, iws, statesₑ, pₑ, task.ctx)
        dofs = celldofs(iws.cell)
        for (j, gj) in pairs(dofs), (i, gi) in pairs(dofs)
            KP[dofmap[gi], dofmap[gj]] += iws.Ke[i, j]
        end
    end
    return nothing
end

"""
    assemble_patch_matrices!(dest, op, provider::PatchItems, states, p, ctx = nothing)

Assemble the patch-local Jacobian of every patch into `dest[i]` (a matrix of
size `patch_ndofs(provider, i)` square, zeroed here), reusing the operator's
cell kernels; rows/columns follow [`patch_dofs`](@ref). Sequential CPU only.

Experimental: part of the patch item family; the corrector/local-BVP layer
follows separately.
"""
function assemble_patch_matrices!(dest::AbstractVector, op, provider::PatchItems, states::NamedTuple, p, ctx = nothing)
    _check_declared_slots(op.engine, states)
    op.engine.strategy.device isa SequentialCPUDevice || throw(ArgumentError(
        "patch assembly currently supports SequentialCPUDevice only (got $(typeof(op.engine.strategy.device)))"))
    length(dest) == npatches(provider) || throw(DimensionMismatch(
        "expected $(npatches(provider)) patch targets, got $(length(dest))"))
    sc = findfirst(sc -> sc.domain.sdh === provider.sdh, op.engine.subdomain_caches)
    sc === nothing && throw(ArgumentError("the provider's SubDofHandler is not part of the operator"))
    for KP in dest
        fill!(KP, zero(eltype(KP)))
    end
    ws = PatchAssemblyWorkspace(provider, Ref(0), first(op.engine.subdomain_caches[sc].device_cache))
    task = AssemblyTask(PatchJacobianKind(dest), nothing, states, p, ctx)
    execute_on_device!(task, op.engine.strategy.device, (ws,), compute_partition(SequentialScheduling(), provider))
    return dest
end

assemble_patch_matrices!(dest::AbstractVector, op, provider::PatchItems, u::AbstractVector, p) =
    assemble_patch_matrices!(dest, op, provider, (u = u,), p, nothing)
