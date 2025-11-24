struct GenericEAVectorIndex{IndexType}
    offset::IndexType
    length::IndexType
end
Base.zero(::GenericEAVectorIndex{T}) where T = GenericEAVectorIndex(zero(T), zero(T))


struct GenericEAMatrixIndex{IndexType}
    offset::IndexType
    nrows::IndexType
    ncols::IndexType
end
Base.zero(::GenericEAMatrixIndex{T}) where T = GenericEAMatrixIndex(zero(T), zero(T), zero(T))


struct GenericIndexdData{IndexType, DataType, IndexStructureType <: AbstractVector{IndexType}}
    data::DataType
    index_structure::IndexStructureType
end

# Generic Element Assembly Data Type
@concrete struct EAOperator
    device
    device_cache
    # Local matrices
    element_matrices
    # input vector index -> local matrix index
    vector_element_map
    # input vector index <- local matrix index
    element_vector_map
end

getnelements(indexed_data::GenericIndexdData) = length(indexed_data.index_structure)
getnelements(ea::EAOperator) = getnelements(ea.element_matrices)



function mul!(out::AbstractVector{T}, operator::EAOperator, in::AbstractVector) where T
    fill!(out, zero(T))
    matrix_free_product!(out, operator, in, operator.device)
end

product_kernel!(outₑ, Aₑ, inₑ, device_cache) = mul!(outₑ, Aₑ, inₑ)

function matrix_free_product!(out::AbstractVector, A::EAOperator, in::AbstractVector, device::SequentialCPUDevice)
    (; device_cache) = A
    # Element loop
    # TODO abstraction layer via iterator
    for ei in 1:getnelements(A)
        # Query Data
        Aₑ   = read_data(A.element_matrices, ei, device, device_cache)
        inₑ  = read_data(A.vector_element_map, in, ei, device, device_cache)
        outₑ = read_data(A.element_vector_map, out, ei, device, device_cache)

        # Local product kernel
        product_kernel!(outₑ, Aₑ, inₑ, device_cache)

        # Maybe write data back
        store_data!(out, outₑ, A.element_vector_map, ei, device, device_cache)
    end
end


struct EAViewCache
end

# TODO atomic flag
product_kernel!(outₑ, Aₑ, inₑ, device_cache::EAViewCache) = mul!(outₑ, Aₑ, inₑ, 1, 1)

function read_data(indexed_data::GenericIndexdData{<:GenericEAMatrixIndex}, i::Integer, device, device_cache::EAViewCache)
    (; offset, nrows, ncols) = indexed_data.index_structure[i]
    Aₑ_flattened = @view indexed_data.data[offset:(offset+nrows*ncols-1)]
    Aₑ = reshape(Aₑ_flattened, (nrows, ncols))
    return Aₑ
end

function read_data(indices::AbstractVector{<:GenericEAVectorIndex}, data::AbstractVector, i::Integer, device, device_cache::EAViewCache)
    data_with_indices = GenericIndexdData(
        data,
        indices,
    )
    return read_data(data_with_indices, i, device, device_cache)
end

function read_data(indexed_data::GenericIndexdData{<:GenericEAVectorIndex}, i::Integer, device, device_cache::EAViewCache)
    (; offset, length) = indexed_data.index_structure[i]
    vₑ = @view indexed_data.data[offset:(offset+length-1)]
    return vₑ
end

function store_data!(A::AbstractMatrix, Aₑ::AbstractMatrix, indexed_data::AbstractVector{<:GenericEAMatrixIndex}, i::Integer, device, device_cache::EAViewCache)
    return nothing
end

function store_data!(out::AbstractVector, outₑ::AbstractVector, indexed_data::AbstractVector{<:GenericEAVectorIndex}, i::Integer, device, device_cache::EAViewCache)
    return nothing
end



@concrete struct PerInstanceEACache
    inₑs
    outₑs
end

function read_data(indexed_data::GenericIndexdData{<:GenericEAMatrixIndex}, i::Integer, device, device_cache::PerInstanceEACache)
    error("Not implemented")
end

function read_data(indexed_data::GenericIndexdData{<:GenericEAVectorIndex}, i::Integer, device, device_cache::PerInstanceEACache)
    error("Not implemented")
end



@concrete struct EARecomputeCache
    # TODO more info about assembly
end

function read_data(indexed_data::GenericIndexdData{<:GenericEAMatrixIndex}, i::Integer, device, device_cache::EARecomputeCache)
    error("Not implemented")
end

function read_data(indexed_data::GenericIndexdData{<:GenericEAVectorIndex}, i::Integer, device, device_cache::EARecomputeCache)
    error("Not implemented")
end



@concrete struct EAOperatorAssembler
    system_matrix::EAOperator
    residual
end

function Ferrite.start_assemble(element_matrix::EAOperator)
    return EAOperatorAssembler(element_matrix, nothing)
end

function Ferrite.assemble!(assembler::EAOperatorAssembler, cell, Kₑ::AbstractMatrix)
    i = cellid(cell)
end

# function Ferrite.assemble!(assembler::EAOperatorAssembler, cell, Kₑ::AbstractMatrix, rₑ::AbstractVector)
#     #
# end

function create_system_matrix(strategy::ElementAssemblyStrategy, dh)
    (; device) = strategy

    ValueType = value_type(device)
    IndexType = index_type(device)

    grid = get_grid(dh)
    index_structure = zeros(IndexType, getncells(grid))

    next_index = 1
    for sdh in enumerate(dh.subdofhandlers)
        ndofs = ndofs_per_cell(sdh)
        for cell in CellIterator(sdh)
            # TODO
        end
    end

    data = zeros(ValueType, total_data_size)

    return EAOperator(device, data, index_structure)
end
