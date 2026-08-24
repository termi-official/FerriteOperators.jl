## Cell traversal uses the two-DofHandler iterators of src/core/iterators.jl. The
## rectangular assembler entry point `Ferrite.assemble!(assembler, rdofs, cdofs, Ke)`
## needs Ferrite ≥ 1.4 (https://github.com/Ferrite-FEM/Ferrite.jl/pull/1279).

####################################
## Element interface              ##
####################################

"""
    AbstractTransferIntegrator

Supertype for integrators producing element-local **rectangular** matrices —
contributions to a transfer (prolongation / restriction) operator between two DofHandlers.

Required: `setup_transfer_element_cache(integrator, sdh_row::SubDofHandler,
sdh_col::SubDofHandler)`, returning an [`AbstractTransferElementCache`](@ref).
"""
abstract type AbstractTransferIntegrator end

"""
    AbstractTransferElementCache

Supertype for element caches used in transfer-operator assembly.

Required:

    assemble_transfer_element!(Pe, tc, element_cache, p)

with `tc` a [`SameGridCellCache`](@ref) or [`NestedGridCellCache`](@ref) and `Pe`
the pre-allocated `(nrdofs_per_cell × ncdofs_per_cell)` element matrix.
"""
abstract type AbstractTransferElementCache end

## Allocation helper (may be specialised by concrete caches)
allocate_transfer_element_matrix(::AbstractTransferElementCache, sdh_row, sdh_col) =
    zeros(ndofs_per_cell(sdh_row), ndofs_per_cell(sdh_col))

function setup_transfer_element_cache end


####################################
## Transfer workspace              ##
####################################

"""
    TransferWorkspace

Per-worker workspace for transfer assembly: element cache, pre-allocated
rectangular element matrix, and transfer cell cache.
"""
@concrete struct TransferWorkspace <: AbstractWorkspace
    element
    Pe
    tc
end

Ferrite.reinit!(ws::TransferWorkspace, cellid) = reinit!(ws.tc, cellid)

function duplicate_for_device(device::AbstractCPUDevice, ws::TransferWorkspace)
    return TransferWorkspace(
        duplicate_for_device(device, ws.element),
        copy(ws.Pe),
        duplicate_for_device(device, ws.tc),
    )
end


####################################
## Transfer task                   ##
####################################

@concrete struct AssembleTransferTerm
    inner_assembler
    p
end
duplicate_for_device(device, task::AssembleTransferTerm) = AssembleTransferTerm(duplicate_for_device(device, task.inner_assembler), task.p)

function execute_single_task!(task::AssembleTransferTerm, ws::TransferWorkspace)
    pₑ = query_cell_parameters(ws.element, ws.tc, task.p)

    fill!(ws.Pe, 0.0)
    @timeit_debug "assemble transfer element" assemble_transfer_element!(ws.Pe, ws.tc, ws.element, pₑ)
    assemble!(task.inner_assembler, getrowdofs(ws.tc), getcolumndofs(ws.tc), ws.Pe)
end


####################################
## Operator struct                 ##
####################################

"""
    TransferFerriteOperator

A transfer (prolongation / restriction) operator assembled as a rectangular sparse matrix
`P` of size `(nrdofs × ncdofs)`. Construct via [`setup_transfer_operator`](@ref), update
via [`update_operator!`](@ref), apply with `mul!(out, op, x[, α, β])`.

!!! warning "Experimental surface"
    The transfer constructors and operator types may change in a minor release;
    the assembled matrix and its sparsity are not affected.
"""
@concrete struct TransferFerriteOperator
    P
    strategy
    subdomain_caches
    dh_row
    dh_col
    integrator
end

function _reassemble_transfer!(op, p)
    (; P, strategy, subdomain_caches) = op

    n_row = maximum(sc -> ndofs_per_cell(sc.domain.sdh_row), subdomain_caches; init = 0)
    n_col = maximum(sc -> ndofs_per_cell(sc.domain.sdh_col), subdomain_caches; init = 0)
    assembler = start_assemble(P; fillzero = true, maxcelldofs_hint = max(n_row, n_col))

    task = AssembleTransferTerm(assembler, p)

    execute_on_subdomains!(task, strategy, subdomain_caches)

    return op
end

"""
    update_operator!(op::TransferFerriteOperator, p)
    update_operator!(op::NestedTransferFerriteOperator, p)

Reassemble the rectangular transfer matrix `op.P` from scratch.
"""
update_operator!(op::TransferFerriteOperator, p) = _reassemble_transfer!(op, p)

mul!(out::AbstractVector, op::TransferFerriteOperator, x::AbstractVector) =
    mul!(out, op.P, x)
mul!(out::AbstractVector, op::TransferFerriteOperator, x::AbstractVector, α, β) =
    mul!(out, op.P, x, α, β)

Base.eltype(op::TransferFerriteOperator) = eltype(op.P)
Base.size(op::TransferFerriteOperator, axis) = size(op.P, axis)
Base.size(op::TransferFerriteOperator) = size(op.P)


####################################
## Nested-grid transfer operator   ##
####################################

"""
    NestedTransferFerriteOperator

Transfer operator for hierarchically nested grids (geometric multigrid): the fine and
coarse DofHandlers live on different grids, connected via `fine2coarse` mappings.
Construct via [`setup_nested_transfer_operator`](@ref), update via [`update_operator!`](@ref).

!!! warning "Experimental surface"
    The transfer constructors and operator types may change in a minor release;
    the assembled matrix and its sparsity are not affected.
"""
@concrete struct NestedTransferFerriteOperator
    P
    strategy
    subdomain_caches
    dh_fine
    dh_coarse
    integrator
end

update_operator!(op::NestedTransferFerriteOperator, p) = _reassemble_transfer!(op, p)

mul!(out::AbstractVector, op::NestedTransferFerriteOperator, x::AbstractVector) =
    mul!(out, op.P, x)
mul!(out::AbstractVector, op::NestedTransferFerriteOperator, x::AbstractVector, α, β) =
    mul!(out, op.P, x, α, β)

Base.eltype(op::NestedTransferFerriteOperator) = eltype(op.P)
Base.size(op::NestedTransferFerriteOperator, axis) = size(op.P, axis)
Base.size(op::NestedTransferFerriteOperator) = size(op.P)
