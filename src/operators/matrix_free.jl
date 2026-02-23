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

function Adapt.adapt_structure(to, gid::GenericIndexedData)
    GenericIndexedData(Adapt.adapt(to, gid.data), Adapt.adapt(to, gid.index_structure))
end

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
    # Scratch for the device to store its data
    device_cache
    # Local matrices (adapted to device)
    element_matrices
    # input vector index -> local matrix index (adapted to device)
    vector_element_map
    # input vector index <- local matrix index (adapted to device)
    element_vector_map
end

struct EAViewCache
end

getnelements(indexed_data::GenericIndexedData) = length(indexed_data.index_structure)
getnelements(ea::EAOperator) = getnelements(ea.element_matrices)


function mul!(out::AbstractVector{T}, operator::EAOperator, in::AbstractVector) where T
    fill!(out, zero(T))
    matrix_free_product!(out, operator, in, operator.device)
end

function matrix_free_product!(out::AbstractVector, A::EAOperator, in::AbstractVector, device::SequentialCPUDevice)
    for ei in 1:getnelements(A)
        Aₑ   = read_data(A.element_matrices, ei)
        inₑ  = read_data(A.vector_element_map, in, ei)
        outₑ = read_data(A.element_vector_map, out, ei)
        mul!(outₑ, Aₑ, inₑ, 1, 1)
    end
end

function matrix_free_product!(out::AbstractVector, A::EAOperator, in::AbstractVector, device::PolyesterDevice)
    (; chunksize) = device
    @batch for ei_base in 1:chunksize:getnelements(A)
        for ei in ei_base:min(ei_base+chunksize-1, getnelements(A))
            product_kernel!(out, A, in, ei, A.vector_element_map, A.element_vector_map, device)
        end
    end
end

function product_kernel!(out::AbstractVector{T}, A::EAOperator, in::AbstractVector, ei, vector_to_element_map::GenericIndexedData{<:GenericEAVectorIndex}, element_to_vector_map::GenericIndexedData{<:GenericEAVectorIndex}, device) where T
    Aₑ          = read_data(A.element_matrices, ei)
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

# GPU product kernel via KernelAbstractions.jl
# Each work item handles one element: gather xₑ, compute Kₑ*xₑ, atomic scatter into y.
@kernel function _gpu_product_kernel!(out, @Const(Ae_data), @Const(Ae_idx), @Const(map_data), @Const(map_idx), @Const(in_vec))
    ei = @index(Global)

    # Element matrix info
    ae_info = Ae_idx[ei]
    ae_off = ae_info.offset
    nrows = ae_info.nrows
    ncols = ae_info.ncols

    # DOF mapping info
    m_info = map_idx[ei]
    m_off = m_info.offset

    # Local matvec: gather input, multiply, atomic scatter output
    for i in 1:nrows
        tmp = zero(eltype(out))
        for j in 1:ncols
            # Column-major flat index: Aₑ[i,j] = Ae_data[ae_off + (j-1)*nrows + (i-1)]
            tmp += Ae_data[ae_off + (j-1)*nrows + (i-1)] * in_vec[map_data[m_off + j - 1]]
        end
        Atomix.@atomic out[map_data[m_off + i - 1]] += tmp
    end
end

function matrix_free_product!(out::AbstractVector, A::EAOperator, in::AbstractVector, device::AbstractGPUDevice)
    backend = default_backend(device)
    nelem = getnelements(A)

    kernel = _gpu_product_kernel!(backend)
    kernel(
        out,
        A.element_matrices.data,
        A.element_matrices.index_structure,
        A.vector_element_map.data,
        A.vector_element_map.index_structure,
        in;
        ndrange=nelem
    )
    KA.synchronize(backend)
end

# GPU element assembly kernel: each work item computes Kₑ for one element.
# Implements diffusion physics: Ke[i,j] += D * (∇Nⱼ ⋅ ∇Nᵢ) * dΩ
@kernel function _gpu_ea_assembly_kernel!(
    Ke_data,
    @Const(Ke_idx),
    @Const(dNdξ), @Const(dMdξ), @Const(weights),
    @Const(cells), @Const(nodes),
    @Const(cellset),
    D,
    nbf::Int32, nqp::Int32, ngeo::Int32
)
    idx = @index(Global)
    ei = cellset[idx]

    # Load cell connectivity (Ferrite cell types are isbitstype)
    cell = cells[ei]
    node_ids = Ferrite.get_node_ids(cell)

    ke_info = Ke_idx[ei]
    ke_off = ke_info.offset
    nrows = ke_info.nrows

    for qp in Int32(1):nqp
        # Compute Jacobian: J = Σⱼ x[j] ⊗ dMdξ[j, qp]
        J = nodes[node_ids[1]].x ⊗ dMdξ[Int32(1), qp]
        for j in Int32(2):ngeo
            J += nodes[node_ids[j]].x ⊗ dMdξ[j, qp]
        end

        detJ = det(J)
        Jinv = inv(J)
        dΩ = detJ * weights[qp]

        for i in Int32(1):nbf
            ∇Nᵢ = dNdξ[i, qp] ⋅ Jinv
            for j in Int32(1):nbf
                ∇Nⱼ = dNdξ[j, qp] ⋅ Jinv
                Ke_data[ke_off + (j - Int32(1)) * nrows + (i - Int32(1))] += D * (∇Nⱼ ⋅ ∇Nᵢ) * dΩ
            end
        end
    end
end

# Launch GPU assembly kernel for bilinear diffusion term
function _launch_gpu_bilinear_assembly!(device, ea_operator, strategy_cache::GPUElementAssemblyStrategyCache, element_cache)
    cv_data     = strategy_cache.cv_data
    device_grid = strategy_cache.device_grid
    D = element_cache.D

    backend = default_backend(device)
    ncells = length(strategy_cache.cellset_gpu)

    kernel = _gpu_ea_assembly_kernel!(backend)
    kernel(
        ea_operator.element_matrices.data,
        ea_operator.element_matrices.index_structure,
        cv_data.dNdξ, cv_data.dMdξ, cv_data.weights,
        device_grid.cells, device_grid.nodes,
        strategy_cache.cellset_gpu,
        D,
        cv_data.nbf, cv_data.nqp, cv_data.ngeo;
        ndrange=ncells
    )
    KA.synchronize(backend)
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

function read_data(indexed_data::GenericIndexedData{<:GenericEAMatrixIndex}, i::Integer)
    (; offset, nrows, ncols) = indexed_data.index_structure[i]
    Aₑ_flattened = @view indexed_data.data[offset:(offset+nrows*ncols-1)]
    Aₑ = reshape(Aₑ_flattened, (nrows, ncols))
    return Aₑ
end
function read_data(indexed_data::GenericIndexedData{<:GenericEAVectorIndex}, data::AbstractVector, i::Integer)
    (; offset, length) = indexed_data.index_structure[i]
    indices = @view indexed_data.data[offset:(offset+length-1)]
    vₑ = @view data[indices]
    return vₑ
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
    fillzero && fill!(element_matrix.element_matrices.data, 0)
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


ea_collapse!(b::Vector, bes::EAVector, ::AbstractGPUDevice) = ea_collapse!(b, bes, SequentialCPUDevice())

function finalize_assembly!(assembler::EAOperatorAssembler)
    if assembler.f !== nothing
        ea_collapse!(assembler.f, assembler.f_element, assembler.device)
    end
end

# TODO support for DG
# FIXME pass EA info down via device cache instead of hardcoded EAViewCache
function create_system_matrix(strategy::ElementAssemblyOperatorStrategy, dh)
    (; device) = strategy

    ValueType = value_type(device)
    IndexType = index_type(device)

    grid = get_grid(dh)
    matrix_index_structure = zeros(GenericEAMatrixIndex{IndexType}, getncells(grid))
    vector_index_structure = zeros(GenericEAVectorIndex{IndexType}, getncells(grid))

    element_offset = IndexType(1)
    vector_offset  = IndexType(1)
    for sdh in dh.subdofhandlers
        ndofs = ndofs_per_cell(sdh)
        for cc in CellIterator(sdh)
            dofs = celldofs(cc)
            nrows = ncols = IndexType(length(dofs))
            matrix_index_structure[cellid(cc)] = GenericEAMatrixIndex(element_offset, nrows, ncols)
            vector_index_structure[cellid(cc)] = GenericEAVectorIndex(vector_offset, nrows)
            element_offset += nrows*ncols
            vector_offset  += nrows
        end
    end

    element_vector_info = Adapt.adapt(device, GenericIndexedData(dh.cell_dofs, vector_index_structure))
    element_matrices = Adapt.adapt(device, GenericIndexedData(zeros(ValueType, element_offset-1), matrix_index_structure))

    return EAOperator(device, EAViewCache(), element_matrices, element_vector_info, element_vector_info)
end
