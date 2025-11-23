struct GenericE2EVectorIndex{IndexType}
    offset::Index
    length::Index
end

struct GenericE2EMatrixIndex{IndexType}
    offset::IndexType
    nrows::IndexType
    ncols::IndexType
end
Base.zero(GenericE2EMatrixIndex{T}) where T = GenericE2EMatrixIndex(zero(T), zero(T), zero(T))

@concrete struct GenericIndexdData
    data
    index_structure
end

# Generic Element Assembly Data Type
@concrete struct E2EOperator
    device
    # Local matrices
    element_matrices
    # input vector index -> local matrix index
    vector_element_map
    # input vector index <- local matrix index
    element_vector_map
end

function mul!(out::AbstractVector{T}, operator::E2EOperator, in::AbstractVector) where T
    fill!(out, zero(T))
    matrix_free_product!(out, operator, in, operator.device)
end

function matrix_free_product!(out::AbstractVector, A::E2EOperator, in::AbstractVector, device::SequentialCPUDevice)
    # Element loop
    # TODO abstraction layer via iterator
    for (ei, (offset, nrows, ncols)) in enumerate(A.element_matrices.index_structure)
        # TODO abstraction layer for this operation here. A might be computed on-the-fly.
        Aₑ_flattened = @view A.data[offset:(offset+nrows*ncols-1)]
        Aₑ = reshape(matrix_flattened, (nrows, ncols))

        # TODO add function to dispatch on
        in_index_struct = A.vector_element_map.index_structure[ei]
        inₑ_indices = @view in_index_struct.data[in_index_struct.offset:(in_index_struct.offset + in_index_struct.length - 1)]
        # TODO device buffer to store local copy into
        inₑ = @view in[inₑ_indices]

        # TODO add function to dispatch on
        out_index_struct = A.element_vector_map.index_structure[ei]
        outₑ_indices = @view out_index_struct.data[out_index_struct.offset:(out_index_struct.offset + out_index_struct.length - 1)]
        # TODO device buffer to store local copy into
        outₑ = @view out[outₑ_indices]

        # Local product kernel
        mul!(outₑ, Aₑ, inₑ)
    end
end

@concrete struct E2EOperatorAssembler
    system_matrix::E2EOperator
    residual
end

function Ferrite.start_assemble(element_matrix::E2EOperator)
    return E2EOperatorAssembler(element_matrix, nothing)
end

function Ferrite.assemble!(assembler::E2EOperatorAssembler, cell, Kₑ::AbstractMatrix)
    i = cellid(cell)
end

# function Ferrite.assemble!(assembler::E2EOperatorAssembler, cell, Kₑ::AbstractMatrix, rₑ::AbstractVector)
#     #
# end

function create_system_matrix(strategy::ElementAssemblyStrategy, dh) where {ValueType, IndexType}
    (; device) = strategy

    ValueType = value_type(device)
    IndexType = index_type(device)

    grid = get_grid(dh)
    index_structure = index_structure = zeros(IndexType, getncells(grid))

    next_index = 1
    for sdh in enumerate(dh.subdofhandlers)
        ndofs = ndofs_per_cell(sdh)
        for cell in CellIterator(sdh)
            
        end
    end

    data = zeros(ValueType, total_data_size)

    return E2EOperator(device, data, index_structure)
end
