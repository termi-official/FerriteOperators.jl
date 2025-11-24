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
Ferrite.assemble!(f::AbstractVector, cell::CellCache, fe::AbstractVector) = assemble!(f, celldofs(cell), fe)

finalize_assembly!(assembler) = nothing

allocate_vector(::Vector{T}, dh) where T = zeros(T, ndofs(dh))

# @concrete struct PerElementMatrixAssembler
#     element_matrix::PerElementMatrix
#     residual
# end

# function Ferrite.start_assemble(element_matrix::PerElementMatrix)
#     PerElementMatrixAssembler(element_matrix)
# end

# function Ferrite.assemble!(assembler::PerElementMatrixAssembler, cell, Kₑ::AbstractMatrix)
# end

# function Ferrite.assemble!(assembler::PerElementMatrixAssembler, cell, Kₑ::AbstractMatrix, rₑ::AbstractVector)
# end

# function create_system_matrix(strategy::ElementAssemblyStrategy, dh) where {ValueType, IndexType}
#     (; device) = strategy

#     ValueType = value_type(device)
#     IndexType = index_type(device)

#     grid = get_grid(dh)
#     offsets = zeros(IndexType, getncells(grid))

#     next_index = 1
#     for sdh in enumerate(dh.subdofhandlers)
#         ndofs = ndofs_per_cell(sdh)
#         for cell in CellIterator(sdh)
            
#         end
#     end

#     data = zeros(ValueType, total_data_size)

#     PerElementMatrix{ValueType, IndexType, Vector{ValueType}, Vector{IndexType}}
# end
