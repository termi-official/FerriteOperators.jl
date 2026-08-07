# Atomic scatter is needed when a parallel device writes into shared global
# storage without color isolation. Atomicity is a property of the SCATTER
# TARGET, not of the strategy alone:
# - dof-scattered shared targets (global sparse matrices, global vectors,
#   dense parameter-Jacobian rows) follow `dof_scatter_needs_atomic` —
#   coloring isolates them, otherwise a parallel device needs atomics,
#   REGARDLESS of the operator form (an EA operator's residual sweep into a
#   plain global vector still races);
# - element-private targets (EA per-element storage) never need atomics;
# - parameter-space accumulators (VJP) are never color-isolated and handle
#   their own atomicity at the entry point.
dof_scatter_needs_atomic(strategy::AssemblyStrategy) =
    !(strategy.device isa SequentialCPUDevice) && !(strategy.scheduling isa ColoredScheduling)
strategy_needs_atomic(strategy::AssemblyStrategy) = dof_scatter_needs_atomic(strategy)

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

# Sensitivity scatter targets. Deliberately not Ferrite.AbstractAssembler:
# their column/entry layout is the parameter space, not the dof space, so the
# celldofs-scatter convenience methods above must never match them.
struct ParameterJacobianAssembler{T, MT <: AbstractMatrix{T}, atomic}
    B::MT   # residual_size × nθ
end
function Ferrite.assemble!(assembler::ParameterJacobianAssembler{<:Any, <:Any, atomic}, cell::CellCache, Bₑ::AbstractMatrix) where {atomic}
    for j in axes(Bₑ, 2)
        for (i, dof) in enumerate(celldofs(cell))
            if atomic
                Atomix.@atomic assembler.B[dof, j] += Bₑ[i, j]
            else
                assembler.B[dof, j] += Bₑ[i, j]
            end
        end
    end
    return
end

struct ParameterVJPAssembler{T, VT <: AbstractVector{T}, atomic}
    g::VT   # length nθ
end
function Ferrite.assemble!(assembler::ParameterVJPAssembler{<:Any, <:Any, atomic}, cell::CellCache, gₑ::AbstractVector) where {atomic}
    for i in eachindex(gₑ)
        if atomic
            Atomix.@atomic assembler.g[i] += gₑ[i]
        else
            assembler.g[i] += gₑ[i]
        end
    end
    return
end

duplicate_for_device(device, a::ParameterJacobianAssembler) = a
duplicate_for_device(device, a::ParameterVJPAssembler) = a
finalize_assembly!(::ParameterJacobianAssembler) = nothing
finalize_assembly!(::ParameterVJPAssembler) = nothing

allocate_vector(::Vector{T}, dh) where T = zeros(T, ndofs(dh))
allocate_vector(::Type{Vector{T}}, dh) where T = zeros(T, ndofs(dh))
