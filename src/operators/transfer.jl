## TransferFerriteOperator
##
## Assembles a rectangular sparse matrix P (nrdofs × ncdofs) from element-local
## rectangular contributions, using the two-DofHandler iterator infrastructure defined in
## src/core/iterators.jl.
##
## The entry point for the rectangular assembler relies on:
##   Ferrite.assemble!(assembler, rdofs, cdofs, Ke)
## which is available in Ferrite ≥ 1.2 (see Ferrite.jl PR #TODO upstream link).
##
## Usage sketch (p-multigrid prolongator):
##
##   integrator = ProlongatorIntegrator(QuadratureRuleCollection(3))
##   strategy   = SequentialAssemblyStrategy(SequentialCPUDevice())
##   op         = setup_transfer_operator(strategy, integrator, fine_dh, coarse_dh)
##   update_operator!(op, nothing)
##   P = op.P

####################################
## Element interface              ##
####################################

"""
    AbstractTransferIntegrator

Supertype for integrators that produce element-local **rectangular** matrices, i.e.
contributions to a transfer (prolongation / restriction) operator between two DofHandlers.

Required methods:
- `setup_transfer_element_cache(integrator, sdh_row::SubDofHandler, sdh_col::SubDofHandler)`

The returned cache must be a subtype of [`AbstractTransferElementCache`](@ref).
"""
abstract type AbstractTransferIntegrator end

"""
    AbstractTransferElementCache

Supertype for element caches used in transfer-operator assembly.

Required method:

    assemble_transfer_element!(Pe, tc, element_cache, p)

where `tc` is a [`SameGridCellCache`](@ref) or a
[`NestedGridCellCache`](@ref) and `Pe` is the pre-allocated rectangular element
matrix of size `(nrdofs_per_cell × ncdofs_per_cell)`.
"""
abstract type AbstractTransferElementCache end

## Allocation helper (may be specialised by concrete caches)
allocate_transfer_element_matrix(::AbstractTransferElementCache, sdh_row, sdh_col) =
    zeros(ndofs_per_cell(sdh_row), ndofs_per_cell(sdh_col))

## Default setup (can be overridden)
function setup_transfer_element_cache end


####################################
## Transfer workspace              ##
####################################

"""
    TransferWorkspace

Per-worker workspace for transfer operator assembly. Holds the element cache,
pre-allocated rectangular element matrix, and the transfer cell cache.
"""
@concrete struct TransferWorkspace
    element
    Pe
    tc
end

Ferrite.reinit!(ws::TransferWorkspace, cellid) = reinit!(ws.tc, cellid)


####################################
## Transfer task                   ##
####################################

struct AssembleTransferTerm{A}
    inner_assembler::A
    p
end
duplicate_for_device(device, task::AssembleTransferTerm) = AssembleTransferTerm(duplicate_for_device(device, task.inner_assembler), task.p)

function execute_single_task!(task::AssembleTransferTerm, ws::TransferWorkspace)
    pₑ = query_element_parameters(ws.element, ws.tc, nothing, task.p)

    fill!(ws.Pe, 0.0)
    @timeit_debug "assemble transfer element" assemble_transfer_element!(ws.Pe, ws.tc, ws.element, pₑ)
    assemble!(task.inner_assembler, getrowdofs(ws.tc), getcolumndofs(ws.tc), ws.Pe)
end


####################################
## Subdomain-level cache           ##
####################################

"""
    TransferSubdomainCache

Holds all pre-allocated data needed to assemble one subdomain's contribution to a transfer
operator.  Analogous to `SubdomainCache` in the square-operator case.
"""
struct TransferSubdomainCache{SDH_row, SDH_col, WS}
    sdh_row::SDH_row            # SubDofHandler for row space (fine / test)
    sdh_col::SDH_col            # SubDofHandler for col space (coarse / trial)
    workspace::WS               # TransferWorkspace
end

function TransferSubdomainCache(sdh_row::SubDofHandler, sdh_col::SubDofHandler, element::AbstractTransferElementCache)
    Pe = allocate_transfer_element_matrix(element, sdh_row, sdh_col)
    tc = SameGridCellCache(sdh_row, sdh_col)
    workspace = TransferWorkspace(element, Pe, tc)
    return TransferSubdomainCache(sdh_row, sdh_col, workspace)
end

####################################
## Operator struct                 ##
####################################

"""
    TransferFerriteOperator

A transfer (prolongation / restriction) operator assembled as a rectangular sparse matrix
`P` of size `(nrdofs × ncdofs)`.

Construct via [`setup_transfer_operator`](@ref) and update via [`update_operator!`](@ref).

    mul!(out, op, x)
    mul!(out, op, x, α, β)

apply the operator (matrix-vector product).
"""
struct TransferFerriteOperator{MatrixType}
    P::MatrixType
    strategy
    subdomain_caches::Vector{TransferSubdomainCache}
end

"""
    update_operator!(op::TransferFerriteOperator, p)

Reassemble the rectangular transfer matrix `op.P` from scratch.
"""
function update_operator!(op::TransferFerriteOperator, p)
    (; P, strategy, subdomain_caches) = op

    n_row = maximum(sc -> ndofs_per_cell(sc.sdh_row), subdomain_caches; init = 0)
    n_col = maximum(sc -> ndofs_per_cell(sc.sdh_col), subdomain_caches; init = 0)
    assembler = start_assemble2(P; fillzero = true, maxcelldofs_hint = max(n_row, n_col))

    task = AssembleTransferTerm(assembler, p)

    for (subdomain_id, sc) in enumerate(subdomain_caches)
        @timeit_debug "assemble transfer subdomain $subdomain_id" begin
            execute_on_device!(task, strategy.device, sc.workspace, (sc.sdh_row.cellset,))
        end
    end

    return op
end

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
    NestedTransferSubdomainCache

Holds pre-allocated data for assembling one subdomain's contribution to a
[`NestedTransferFerriteOperator`](@ref).  The fine and coarse DofHandlers live on
**different** grids connected by `fine2coarse` and `child_ref_coords`.
"""
struct NestedTransferSubdomainCache{SDH_fine, SDH_coarse, WS}
    sdh_fine::SDH_fine
    sdh_coarse::SDH_coarse
    workspace::WS               # TransferWorkspace with NestedGridCellCache
end

"""
    NestedTransferFerriteOperator

Transfer operator for hierarchically nested grids (geometric multigrid).  The fine and
coarse DofHandlers live on different grids connected via `fine2coarse` mappings.

Construct via [`setup_nested_transfer_operator`](@ref); update via [`update_operator!`](@ref).
"""
struct NestedTransferFerriteOperator{MatrixType}
    P::MatrixType
    strategy
    subdomain_caches::Vector{NestedTransferSubdomainCache}
end

function update_operator!(op::NestedTransferFerriteOperator, p)
    (; P, strategy, subdomain_caches) = op

    n_row = maximum(sc -> ndofs_per_cell(sc.sdh_fine),   subdomain_caches; init = 0)
    n_col = maximum(sc -> ndofs_per_cell(sc.sdh_coarse), subdomain_caches; init = 0)
    assembler = start_assemble2(P; fillzero = true, maxcelldofs_hint = max(n_row, n_col))

    task = AssembleTransferTerm(assembler, p)

    for (subdomain_id, sc) in enumerate(subdomain_caches)
        @timeit_debug "assemble nested transfer subdomain $subdomain_id" begin
            execute_on_device!(task, strategy.device, sc.workspace, (sc.sdh_fine.cellset,))
        end
    end

    return op
end

mul!(out::AbstractVector, op::NestedTransferFerriteOperator, x::AbstractVector) =
    mul!(out, op.P, x)
mul!(out::AbstractVector, op::NestedTransferFerriteOperator, x::AbstractVector, α, β) =
    mul!(out, op.P, x, α, β)

Base.eltype(op::NestedTransferFerriteOperator) = eltype(op.P)
Base.size(op::NestedTransferFerriteOperator, axis) = size(op.P, axis)
Base.size(op::NestedTransferFerriteOperator) = size(op.P)
