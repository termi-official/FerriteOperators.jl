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
## Element interface               ##
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

where `tc` is a [`SameGridTransferCellCache`](@ref) or a
[`NestedGridTransferCellCache`](@ref) and `Pe` is the pre-allocated rectangular element
matrix of size `(nrdofs_per_cell × ncdofs_per_cell)`.
"""
abstract type AbstractTransferElementCache end

## Allocation helper (may be specialised by concrete caches)
allocate_transfer_element_matrix(::AbstractTransferElementCache, sdh_row, sdh_col) =
    zeros(ndofs_per_cell(sdh_row), ndofs_per_cell(sdh_col))

## Default setup (can be overridden)
function setup_transfer_element_cache end


####################################
## Subdomain-level cache           ##
####################################

"""
    TransferSubdomainCache

Holds all pre-allocated data needed to assemble one subdomain's contribution to a transfer
operator.  Analogous to `SubdomainCache` in the square-operator case.
"""
struct TransferSubdomainCache{SDH_row, SDH_col, EL, TC}
    sdh_row::SDH_row            # SubDofHandler for row space (fine / test)
    sdh_col::SDH_col            # SubDofHandler for col space (coarse / trial)
    element::EL                 # AbstractTransferElementCache
    Pe::Matrix{Float64}         # pre-allocated element-local rectangular matrix
    tc::TC                      # SameGridTransferCellCache (reused across iterations)
end

function TransferSubdomainCache(sdh_row::SubDofHandler, sdh_col::SubDofHandler, element::AbstractTransferElementCache)
    Pe = allocate_transfer_element_matrix(element, sdh_row, sdh_col)
    tc = SameGridTransferCellCache(sdh_row, sdh_col)
    return TransferSubdomainCache(sdh_row, sdh_col, element, Pe, tc)
end


####################################
## Sparse accumulation helper      ##
####################################

## Accumulate a local rectangular matrix Pe into a global sparse matrix P.
## Uses direct sparse indexing which is safe even before the sparsity pattern is fixed.
@inline function _accumulate_transfer_element!(P::SparseMatrixCSC, Pe::AbstractMatrix,
        rdofs::AbstractVector{Int}, cdofs::AbstractVector{Int})
    @inbounds for (j, Kcol) in enumerate(cdofs)
        for (i, Krow) in enumerate(rdofs)
            P[Krow, Kcol] += Pe[i, j]
        end
    end
    return nothing
end


####################################
## Device dispatch                 ##
####################################

function execute_transfer_on_device!(
        P::SparseMatrixCSC,
        device::SequentialCPUDevice,
        subdomain_cache::TransferSubdomainCache,
        p,
    )
    (; sdh_row, element, Pe, tc) = subdomain_cache
    for cellid in sdh_row.cellset
        reinit!(tc, cellid)
        fill!(Pe, 0.0)
        @timeit_debug "assemble transfer element" assemble_transfer_element!(Pe, tc, element, p)
        _accumulate_transfer_element!(P, Pe, getrowdofs(tc), getcolumndofs(tc))
    end
    return nothing
end

function execute_transfer_on_device!(
        P::SparseMatrixCSC,
        device::PolyesterDevice,
        subdomain_cache::TransferSubdomainCache,
        p,
    )
    # For now fall back to sequential; parallel coloring for transfer operators
    # can be added later once the coloring algorithm is extended to rectangular operators.
    execute_transfer_on_device!(P, SequentialCPUDevice(), subdomain_cache, p)
    return nothing
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

    # Zero out existing entries (preserves sparsity pattern after first assembly)
    fill!(nonzeros(P), zero(eltype(P)))

    for (subdomain_id, subdomain_cache) in enumerate(subdomain_caches)
        @timeit_debug "assemble transfer subdomain $subdomain_id" begin
            execute_transfer_on_device!(P, strategy.device, subdomain_cache, p)
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
