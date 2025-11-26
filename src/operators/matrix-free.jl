struct GenericEAVectorIndex{IndexType}
    offset::IndexType
    length::IndexType
end
Base.zero(::Type{GenericEAVectorIndex{T}}) where T = GenericEAVectorIndex(zero(T), zero(T))


struct GenericEAMatrixIndex{IndexType}
    offset::IndexType
    nrows::IndexType
    ncols::IndexType
end
Base.zero(::Type{GenericEAMatrixIndex{T}}) where T = GenericEAMatrixIndex(zero(T), zero(T), zero(T))


struct GenericIndexedData{IndexType, DataType, IndexStructureType <: AbstractVector{IndexType}}
    data::DataType
    index_structure::IndexStructureType
end
Base.fill!(v::GenericIndexedData, val) = fill!(v.data, val)

@concrete struct EAVector
    # Buffer for the per element data
    data
    # Map from global dof index to element index and local dof index
    dof_to_element_map
end
EAVector(dh::DofHandler) = EAVector(Float64, Int, dh)
function EAVector(::Type{ValueType}, ::Type{IndexType}, dh::DofHandler) where {ValueType, IndexType}
    @assert length(dh.field_names) == 1
    map  = create_dof_to_element_map(dh)
    grid = get_grid(dh)

    eadata      = zeros(ValueType, length(dh.cell_dofs))
    eaoffsets   = zeros(GenericEAVectorIndex{IndexType}, getncells(grid))
    next_offset = 1
    for i in 1:getncells(grid)
        ndofs = ndofs_per_cell(dh, i)
        eadatav = @view eadata[next_offset:(next_offset+ndofs-1)]
        # celldofs!(eadatav, dh, i) # FIXME (https://github.com/Ferrite-FEM/Ferrite.jl/pull/1252)
        eadatav .= celldofs(dh, i)
        eaoffsets[i] = GenericEAVectorIndex(next_offset, ndofs)
        next_offset += ndofs
    end

    return EAVector(
        GenericIndexedData(eadata, eaoffsets),
        map,
    )
end
Base.fill!(v::EAVector, val) = fill!(v.data, val)

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

getnelements(indexed_data::GenericIndexedData) = length(indexed_data.index_structure)
getnelements(ea::EAOperator) = getnelements(ea.element_matrices)



function mul!(out::AbstractVector{T}, operator::EAOperator, in::AbstractVector) where T
    fill!(out, zero(T))
    matrix_free_product!(out, operator, in, operator.device)
end

function matrix_free_product!(out::AbstractVector, A::EAOperator, in::AbstractVector, device::SequentialCPUDevice)
    (; device_cache) = A
    # Element loop
    for ei in 1:getnelements(A)
        # Query Data
        Aₑ   = read_data(A.element_matrices, ei, EAViewCache())
        inₑ  = read_data(A.vector_element_map, in, ei, device_cache)
        outₑ = read_data(A.element_vector_map, out, ei, device_cache)

        # Local product kernel
        mul!(outₑ, Aₑ, inₑ, 1, 1)

        # Write data back
        store_data!(out, outₑ, A.element_vector_map, ei, device_cache)
    end
end

function matrix_free_product!(out::AbstractVector, A::EAOperator, in::AbstractVector, device::PolyesterDevice)
    (; device_cache) = A
    (; chunksize) = device
    # Element loop
    @batch for ei_base in 1:chunksize:getnelements(A)
        for ei in ei_base:min(ei_base+chunksize-1, getnelements(A))
            product_kernel!(out, A, in, ei, A.vector_element_map, A.element_vector_map, device, device_cache)
        end
    end
end

struct EAViewCache
end

# TODO make this per subdomain
struct EAThreadedKernelCache{T}
    inₑ::Vector{Vector{T}}
    Aₑ::Vector{Vector{T}}
    outₑ::Vector{Vector{T}}
end

function product_kernel!(out::AbstractVector{T}, A::EAOperator, in::AbstractVector, ei, vector_to_element_map::GenericIndexedData{<:GenericEAVectorIndex}, element_to_vector_map::GenericIndexedData{<:GenericEAVectorIndex}, device, device_cache) where T
    Aₑ          = read_data(A.element_matrices, ei, EAViewCache())
    in_indices  = get_indices(element_to_vector_map, ei)
    out_indices = get_indices(vector_to_element_map, ei)
    for i in 1:size(Aₑ, 1)
        tmp = zero(T)
        for j in 1:size(Aₑ, 2)
            tmp += Aₑ[i,j] * in[in_indices[j]]
        end
        Atomix.@atomic out[out_indices[i]] += tmp
    end
end

function get_indices(indexed_data::GenericIndexedData{<:GenericEAVectorIndex}, i::Integer)
    (; offset, length) = indexed_data.index_structure[i]
    return @view indexed_data.data[offset:(offset+length-1)]
end
function get_indices(indexed_data::GenericIndexedData{<:Int}, i::Integer)
    i1 = indexed_data.index_structure[i]
    i2 = indexed_data.index_structure[i+1]
    return @view indexed_data.data[i1:(i2-1)]
end

function read_data(indexed_data::GenericIndexedData{<:GenericEAMatrixIndex}, i::Integer, device_cache::EAViewCache)
    (; offset, nrows, ncols) = indexed_data.index_structure[i]
    Aₑ_flattened = @view indexed_data.data[offset:(offset+nrows*ncols-1)]
    Aₑ = reshape(Aₑ_flattened, (nrows, ncols))
    return Aₑ
end
function read_data(indexed_data::GenericIndexedData{<:GenericEAVectorIndex}, data::AbstractVector, i::Integer, device_cache::EAViewCache)
    (; offset, length) = indexed_data.index_structure[i]
    indices = @view indexed_data.data[offset:(offset+length-1)]
    vₑ = @view data[indices]
    return vₑ
end

function store_data!(A::AbstractMatrix, Aₑ::AbstractMatrix, indexed_data::GenericIndexedData{<:GenericEAMatrixIndex}, i::Integer, device_cache::EAViewCache)
    return nothing
end

function store_data!(out::AbstractVector, outₑ::AbstractVector, indexed_data::GenericIndexedData{<:GenericEAVectorIndex}, i::Integer, device_cache::EAViewCache)
    return nothing
end


@concrete struct PerInstanceEACache
    inₑs
    outₑs
end

function read_data(indexed_data::GenericIndexedData{<:GenericEAMatrixIndex}, i::Integer, device_cache::PerInstanceEACache)
    error("Not implemented")
end

function read_data(indexed_data::GenericIndexedData{<:GenericEAVectorIndex}, i::Integer, device_cache::PerInstanceEACache)
    error("Not implemented")
end



@concrete struct EARecomputeCache
    # TODO more info about assembly
end

function read_data(indexed_data::GenericIndexedData{<:GenericEAMatrixIndex}, i::Integer, device_cache::EARecomputeCache)
    error("Not implemented")
end

function read_data(indexed_data::GenericIndexedData{<:GenericEAVectorIndex}, i::Integer, device_cache::EARecomputeCache)
    error("Not implemented")
end



@concrete struct EAOperatorAssembler <: Ferrite.AbstractAssembler
    device
    K_element
    f_element
    f
end
duplicate_for_device(device, assembler::EAOperatorAssembler) = assembler

function Ferrite.start_assemble(strategy::ElementAssemblyOperatorStrategy, f::Vector{T}; fillzero::Bool=true) where T
    fillzero && fill!(f, 0.0)
    fillzero && fill!(strategy.eadata, 0.0)
    return EAOperatorAssembler(strategy.device, nothing, strategy.eadata, f)
end
function Ferrite.start_assemble(strategy::ElementAssemblyOperatorStrategy, element_matrix::EAOperator; fillzero::Bool=true)
    fillzero && fill!(element_matrix.element_matrices.data, 0.0)
    return EAOperatorAssembler(strategy.device, element_matrix, nothing, nothing)
end
function Ferrite.start_assemble(strategy::ElementAssemblyOperatorStrategy, element_matrix::EAOperator, f::AbstractVector; fillzero::Bool=true)
    fillzero && fill!(element_matrix.element_matrices.data, 0.0)
    fillzero && fill!(f, 0.0)
    fillzero && fill!(strategy.eadata, 0.0)
    return EAOperatorAssembler(strategy.device, element_matrix, strategy.eadata, f)
end

function Ferrite.assemble!(assembler::EAOperatorAssembler, cell::CellCache, Kₑ::AbstractMatrix)
    (; element_matrices) = assembler.K_element
    i = cellid(cell)
    (; offset, nrows, ncols) = element_matrices.index_structure[i]
    Aₑ_flattened = @view element_matrices.data[offset:(offset+nrows*ncols-1)]
    Aₑ = reshape(Aₑ_flattened, (nrows, ncols))
    Aₑ .+= Kₑ
    return nothing
end
function Ferrite.assemble!(assembler::EAOperatorAssembler, cell::CellCache, rₑ::AbstractVector)
    i = cellid(cell)
    (; data) = assembler.f_element # f_element is an EAVector
    (; offset, length) = data.index_structure[i]
    fₑ = @view data.data[offset:(offset+length-1)]
    fₑ .+= rₑ
    return nothing
end
function Ferrite.assemble!(assembler::EAOperatorAssembler, cell::CellCache, Kₑ::AbstractMatrix, rₑ::AbstractVector)
    assemble!(assembler, cell, Kₑ)
    assemble!(assembler, cell, rₑ)
end

function ea_collapse!(b::Vector, bes::EAVector, device::AbstractCPUDevice)
    ndofs = size(b, 1)
    for dof ∈ 1:ndofs
        _ea_collapse_kernel!(b, dof, bes)
    end
end
function ea_collapse!(b::Vector, bes::EAVector, device::PolyesterDevice)
    ndofs = size(b, 1)
    @batch minbatch=max(1, device.chunksize) for dof ∈ 1:ndofs
        _ea_collapse_kernel!(b, dof, bes)
    end
end
@inline function _ea_collapse_kernel!(b::AbstractVector, dof::Integer, bes::EAVector)
    for edp ∈ get_indices(bes.dof_to_element_map, dof)
        local_data = get_indices(bes.data, edp.element_index)
        b[dof]  += local_data[edp.local_dof_index]
    end
end

function finalize_assembly!(assembler::EAOperatorAssembler)
    assembler.f === nothing && return

    ea_collapse!(assembler.f, assembler.f_element, assembler.device)
end

# TODO support for DG
# TODO switch input from strategy to a cache holding EA info
function create_system_matrix(strategy::ElementAssemblyOperatorStrategy, dh)
    (; device) = strategy

    ValueType = value_type(device)
    IndexType = index_type(device)

    grid = get_grid(dh)
    matrix_index_structure = zeros(GenericEAMatrixIndex{IndexType}, getncells(grid))
    vector_index_structure = zeros(GenericEAVectorIndex{IndexType}, getncells(grid))

    element_offset = 1
    vector_offset  = 1
    for sdh in dh.subdofhandlers
        ndofs = ndofs_per_cell(sdh)
        for cc in CellIterator(sdh)
            dofs = celldofs(cc)
            nrows = ncols = length(dofs)
            matrix_index_structure[cellid(cc)] = GenericEAMatrixIndex(element_offset, nrows, ncols)
            vector_index_structure[cellid(cc)] = GenericEAVectorIndex(vector_offset, nrows)
            element_offset += nrows*ncols
            vector_offset  += nrows
        end
    end

    element_vector_info = GenericIndexedData(dh.cell_dofs, vector_index_structure)
    element_matrices = GenericIndexedData(zeros(ValueType, element_offset-1), matrix_index_structure)
    # FIXME pass EA info down via device cache instead of hardcoded EAViewCache
    return EAOperator(device, EAViewCache(), element_matrices, element_vector_info, element_vector_info)
end
