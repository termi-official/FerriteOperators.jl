Ferrite.start_assemble(::AbstractAssemblyStrategy, J::AbstractMatrix; fillzero::Bool=true) = start_assemble(J; fillzero)
Ferrite.start_assemble(::AbstractAssemblyStrategy, J::AbstractMatrix, residual::AbstractVector; fillzero::Bool=true) = start_assemble(J, residual; fillzero)
function Ferrite.start_assemble(::AbstractAssemblyStrategy, residual::AbstractVector{T}; fillzero::Bool=true) where T
    fillzero && fill!(residual, zero(T))
    return residual
end

# FIXME we might want to upstream this
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, Ke::AbstractMatrix, fe::AbstractVector) = assemble!(assembler, celldofs(cell), Ke, fe)
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, Ke::AbstractMatrix) = assemble!(assembler, celldofs(cell), Ke)
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, fe::AbstractVector) = assemble!(assembler, celldofs(cell), fe)
function Ferrite.assemble!(f::AbstractVector, cell::CellCache, fe::AbstractVector)
    assemble!(f, celldofs(cell), fe)
end
finalize_assembly!(assembler::Ferrite.AbstractAssembler) = nothing
finalize_assembly!(assembler::AbstractVector) = nothing

allocate_vector(::Vector{T}, dh) where T = zeros(T, ndofs(dh))

struct CSCAssembler2{Tv, Ti, MT <: AbstractSparseMatrixCSC{Tv, Ti}} <: AbstractCSCAssembler
    K::MT
    f::Vector{Tv}
    rowpermutation::Vector{Int}
    colpermutation::Vector{Int}
    sortedrowdofs::Vector{Int}
    sortedcoldofs::Vector{Int}
end

function start_assemble2(K::AbstractSparseMatrixCSC{T}, f::Vector = T[]; fillzero::Bool = true, maxcelldofs_hint::Int = 0) where {T}
    fillzero && (fillzero!(K); fillzero!(f))
    return CSCAssembler2(K, f, zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint), zeros(Int, maxcelldofs_hint))
end

@propagate_inbounds function Ferrite.assemble!(A::CSCAssembler2, rowdofs::AbstractVector{<:Integer}, coldofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::Union{AbstractVector, Nothing} = nothing)
    return _assemble!(A, rowdofs, coldofs, Ke, fe, false)
end

@propagate_inbounds function _assemble!(A::Union{AbstractCSCAssembler, AbstractCSRAssembler}, rowdofs::AbstractVector{<:Integer}, coldofs::AbstractVector{<:Integer}, Ke::AbstractMatrix, fe::Union{AbstractVector, Nothing}, sym::Bool)
    @boundscheck checkbounds(Ke, keys(rowdofs), keys(coldofs))
    if fe !== nothing
        @boundscheck checkbounds(fe, keys(rowdofs))
        @boundscheck checkbounds(A.f, rowdofs)
        @inbounds assemble!(A.f, rowdofs, fe)
    end

    K = matrix_handle(A)
    @boundscheck checkbounds(K, rowdofs, coldofs)

    # We assume that the input dofs are not sorted, because the cells need the dofs in
    # a specific order, which might not be the sorted order. Hence we sort them.
    # Note that we are not allowed to mutate `dofs` in the process.
    sortedcoldofs, colpermutation = Ferrite._sortdofs_for_assembly!(A.colpermutation, A.sortedcoldofs, coldofs)
    sortedrowdofs, rowpermutation = Ferrite._sortdofs_for_assembly!(A.rowpermutation, A.sortedrowdofs, rowdofs)

    return _assemble_inner!(K, Ke, rowdofs, sortedrowdofs, rowpermutation, coldofs, sortedcoldofs, colpermutation, sym)
end

@propagate_inbounds function _assemble_inner!(
        K::SparseMatrixCSC, Ke::AbstractMatrix,
        rowdofs::AbstractVector, sortedrowdofs::AbstractVector, rowpermutation::AbstractVector,
        coldofs::AbstractVector, sortedcoldofs::AbstractVector, colpermutation::AbstractVector,
        sym::Bool
    )
    current_col = 1
    Krows = rowvals(K)
    Kvals = nonzeros(K)
    ld = length(rowdofs)
    @inbounds for Kcol in sortedcoldofs
        maxlookups = sym ? current_col : ld
        Kecol = colpermutation[current_col]
        ri = 1 # row index pointer for the local matrix
        Ri = 1 # row index pointer for the global matrix
        nzr = nzrange(K, Kcol)
        while Ri <= length(nzr) && ri <= maxlookups
            R = nzr[Ri]
            Krow = Krows[R]
            Kerow = rowpermutation[ri]
            val = Ke[Kerow, Kecol]
            if Krow == rowdofs[Kerow]
                # Match: add the value (if non-zero) and advance the pointers
                if !iszero(val)
                    Kvals[R] += val
                end
                ri += 1
                Ri += 1
            elseif Krow < rowdofs[Kerow]
                # No match yet: advance the global matrix row pointer
                Ri += 1
            else # Krow > rowdofs[Kerow]
                # No match: no entry exist in the global matrix for this row. This is
                # allowed as long as the value which would have been inserted is zero.
                iszero(val) || Ferrite._missing_sparsity_pattern_error(Krow, Kcol)
                # Advance the local matrix row pointer
                ri += 1
            end
        end
        # Make sure that remaining entries in this column of the local matrix are all zero
        for i in ri:maxlookups
            if !iszero(Ke[rowpermutation[i], Kecol])
                Ferrite._missing_sparsity_pattern_error(sortedrowdofs[i], Kcol)
            end
        end
        current_col += 1
    end
    return
end
