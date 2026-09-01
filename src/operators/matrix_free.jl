struct GenericEAVectorIndex{IndexType}
    offset::IndexType
    length::IndexType
end
Base.zero(::Type{GenericEAVectorIndex{T}}) where T = GenericEAVectorIndex(zero(T), zero(T))


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
    length(dh.field_names) == 1 || throw(ArgumentError(
        "ElementAssembly supports single-field DofHandlers only, got $(length(dh.field_names)) fields."))
    map  = create_dof_to_element_map(dh)
    grid = get_grid(dh)

    eadata      = zeros(ValueType, length(dh.cell_dofs))
    eaoffsets   = zeros(GenericEAVectorIndex{IndexType}, getncells(grid))
    next_offset = 1
    for i in 1:getncells(grid)
        ndofs = ndofs_per_cell(dh, i)
        eaoffsets[i] = GenericEAVectorIndex(next_offset, ndofs)
        next_offset += ndofs
    end

    return EAVector(
        GenericIndexedData(eadata, eaoffsets),
        map,
    )
end
Base.fill!(v::EAVector, val) = fill!(v.data, val)

####################################
## EA task and workspace          ##
####################################

mutable struct EAIndexWorkspace <: AbstractWorkspace
    ei::Int
end
Ferrite.reinit!(ws::EAIndexWorkspace, ei::Int) = (ws.ei = ei)
duplicate_for_device(device::AbstractCPUDevice, ws::EAIndexWorkspace) = EAIndexWorkspace(0)

struct EACollapseTask{B, Bes}
    b::B
    bes::Bes
end
duplicate_for_device(device, task::EACollapseTask) = task

function execute_single_task!(task::EACollapseTask, ws::EAIndexWorkspace)
    _ea_collapse_kernel!(task.b, ws.ei, task.bes)
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

@concrete struct EAOperatorAssembler{T, DeviceType <: AbstractDevice{T}} <: Ferrite.AbstractAssembler{T}
    device::DeviceType
    f_element
    f
end
duplicate_for_device(device, assembler::EAOperatorAssembler) = assembler

function Ferrite.start_assemble(strategy::AssemblyStrategy{<:ElementAssemblyData}, f::AbstractVector; fillzero::Bool=true)
    fillzero && fill!(f, 0.0)
    fillzero && fill!(strategy.form.eadata, 0.0)
    return EAOperatorAssembler(strategy.device, strategy.form.eadata, f)
end

function Ferrite.assemble!(assembler::EAOperatorAssembler, cell::CellCache, rₑ::AbstractVector)
    i = cellid(cell)
    (; data) = assembler.f_element # f_element is an EAVector
    (; offset, length) = data.index_structure[i]
    fₑ = @view data.data[offset:(offset+length-1)]
    fₑ .+= rₑ
    return nothing
end

function ea_collapse!(b::AbstractVector, bes::EAVector, device::AbstractCPUDevice)
    items = (1:length(b),)
    nw = n_workers(device, items)
    dc = setup_device_instances(device, EAIndexWorkspace(0), nw)
    execute_on_device!(EACollapseTask(b, bes), device, dc, items)
end
@inline function _ea_collapse_kernel!(b::AbstractVector, dof::Integer, bes::EAVector)
    for edp ∈ get_indices(bes.dof_to_element_map, dof)
        local_data = get_indices(bes.data, edp.element_index)
        b[dof]  += local_data[edp.local_dof_index]
    end
end

finalize_assembly!(assembler::EAOperatorAssembler) =
    ea_collapse!(assembler.f, assembler.f_element, assembler.device)

# The `ElementAssembly` form stores per-element RESIDUAL contributions, so an
# operator whose target is a matrix has nothing to be allocated here. The
# generic `create_system_matrix` would go looking for an operator specification
# the form does not carry, so the rejection is stated where the setup asks.
function create_system_matrix(strategy::AssemblyStrategy{<:Union{ElementAssembly, ElementAssemblyData}}, dh)
    throw(ArgumentError(
        "ElementAssembly supports vector-target (linear) operators only: it stores per-element " *
        "residual contributions and holds no matrix. Build a bilinear or nonlinear operator " *
        "under `FullAssembly`."))
end
