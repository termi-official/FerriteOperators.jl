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

finalize_assembly!(assembler::Ferrite.AbstractAssembler) = nothing
finalize_assembly!(assembler::AbstractVector) = nothing

allocate_vector(::Vector{T}, dh) where T = zeros(T, ndofs(dh))
