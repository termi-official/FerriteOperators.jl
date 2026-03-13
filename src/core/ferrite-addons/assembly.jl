Ferrite.start_assemble(::AbstractAssemblyStrategy, J::AbstractMatrix; fillzero::Bool=true) = start_assemble(J; fillzero)
Ferrite.start_assemble(::AbstractAssemblyStrategy, J::AbstractMatrix, residual::AbstractVector; fillzero::Bool=true) = start_assemble(J, residual; fillzero)
function Ferrite.start_assemble(::AbstractAssemblyStrategy, residual::AbstractVector{T}; fillzero::Bool=true) where T
    fillzero && fill!(residual, zero(T))
    return residual
end

# FIXME we might want to upstream this
#  NOTE: why we need this? because in specific assembler implementations, e.g. matrix free assembler,
# we cannot do `Ferrite.assemble!(a::EAOperatorAssembler, c::CellCacheType, Kₑ::AbstractMatrix)`, because this will run into ambiguities with the original `Ferrite.assemble!`.
const CellCacheType = Union{CellCache, ImmutableCellCache} #TODO: better naming, maybe? 
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCacheType, Ke::AbstractMatrix, fe::AbstractVector) = assemble!(assembler, celldofs(cell), Ke, fe)
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCacheType, Ke::AbstractMatrix) = assemble!(assembler, celldofs(cell), Ke)
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCacheType, fe::AbstractVector) = assemble!(assembler, celldofs(cell), fe)
function Ferrite.assemble!(f::AbstractVector, cell::CellCacheType, fe::AbstractVector)
    assemble!(f, celldofs(cell), fe)
end
finalize_assembly!(assembler::Ferrite.AbstractAssembler) = nothing
finalize_assembly!(assembler::AbstractVector) = nothing

allocate_vector(::Vector{T}, dh) where T = zeros(T, ndofs(dh))
