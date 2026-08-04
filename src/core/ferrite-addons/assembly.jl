strategy_needs_atomic(::AbstractAssemblyStrategy) = true
strategy_needs_atomic(::PerColorAssemblyStrategy) = false

struct VectorAssembler{T, VT <: AbstractVector{T}, atomic} <: Ferrite.AbstractAssembler{T}
    f::VT
end

Ferrite.start_assemble(strategy::AbstractAssemblyStrategy, J::AbstractMatrix; fillzero::Bool=true) = start_assemble(J, atomic = strategy_needs_atomic(strategy); fillzero)
Ferrite.start_assemble(strategy::AbstractAssemblyStrategy, J::AbstractMatrix, residual::AbstractVector; fillzero::Bool=true) = start_assemble(J, residual, atomic = strategy_needs_atomic(strategy); fillzero)
function Ferrite.start_assemble(strategy::AbstractAssemblyStrategy, residual::AbstractVector{T}; fillzero::Bool=true) where T
    fillzero && fill!(residual, zero(T))
    return VectorAssembler{T, typeof(residual), strategy_needs_atomic(strategy)}(residual)
end
duplicate_for_device(device, a::VectorAssembler) = a

# FIXME we might want to upstream this
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, Ke::AbstractMatrix, fe::AbstractVector) = assemble!(assembler, celldofs(cell), Ke, fe)
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, Ke::AbstractMatrix) = assemble!(assembler, celldofs(cell), Ke)
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, fe::AbstractVector) = assemble!(assembler, celldofs(cell), fe)
function Ferrite.assemble!(assembler::VectorAssembler{<:Any, <:Any, atomic}, cell::CellCache, fe::AbstractVector) where {atomic}
    for (i, dof) in enumerate(celldofs(cell))
        Ferrite._addindex!(assembler.f, dof, fe[i], Val{atomic}())
    end
    return
end
finalize_assembly!(assembler::Ferrite.AbstractAssembler) = nothing
finalize_assembly!(assembler::AbstractVector) = nothing

allocate_vector(::Vector{T}, dh) where T = zeros(T, ndofs(dh))
allocate_vector(::Type{Vector{T}}, dh) where T = zeros(T, ndofs(dh))
