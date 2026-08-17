# Atomic scatter is needed when a parallel device writes into shared global
# storage without color isolation. Atomicity is a property of the SCATTER
# TARGET, not of the strategy alone:
# - dof-scattered shared targets (global sparse matrices, global vectors,
#   dense parameter-Jacobian rows) follow `dof_scatter_needs_atomic` —
#   coloring isolates them, otherwise a parallel device needs atomics,
#   REGARDLESS of the operator form (an EA operator's residual sweep into a
#   plain global vector still races);
# - element-private targets (EA per-element storage) never need atomics;
# - parameter-space accumulators (VJP columns) are never color-isolated —
#   every parallel device needs atomics there.
# Value-returning sweeps have no scatter target at all: they fold per worker
# and reduce host-side, so they never enter this decision.
dof_scatter_needs_atomic(strategy::AssemblyStrategy) =
    !(strategy.device isa SequentialCPUDevice) && !(strategy.scheduling isa ColoredScheduling)
parameter_scatter_needs_atomic(strategy::AssemblyStrategy) =
    !(strategy.device isa SequentialCPUDevice)

# The single point where atomic and plain accumulation differ.
@inline _accum!(::Val{true},  A, v, i)    = Atomix.@atomic A[i] += v
@inline _accum!(::Val{true},  A, v, i, j) = Atomix.@atomic A[i, j] += v
@inline _accum!(::Val{false}, A, v, i)    = A[i] += v
@inline _accum!(::Val{false}, A, v, i, j) = A[i, j] += v

struct VectorAssembler{T, VT <: AbstractVector{T}, atomic} <: Ferrite.AbstractAssembler{T}
    f::VT
end

Ferrite.start_assemble(strategy::AbstractAssemblyStrategy, J::AbstractMatrix; fillzero::Bool=true) = start_assemble(J, atomic = dof_scatter_needs_atomic(strategy); fillzero)
Ferrite.start_assemble(strategy::AbstractAssemblyStrategy, J::AbstractMatrix, residual::AbstractVector; fillzero::Bool=true) = start_assemble(J, residual, atomic = dof_scatter_needs_atomic(strategy); fillzero)
function Ferrite.start_assemble(strategy::AbstractAssemblyStrategy, residual::AbstractVector{T}; fillzero::Bool=true) where T
    fillzero && fill!(residual, zero(T))
    return VectorAssembler{T, typeof(residual), dof_scatter_needs_atomic(strategy)}(residual)
end
duplicate_for_device(device, a::VectorAssembler) = a

# FIXME we might want to upstream this
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, Ke::AbstractMatrix, fe::AbstractVector) = assemble!(assembler, celldofs(cell), Ke, fe)
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, Ke::AbstractMatrix) = assemble!(assembler, celldofs(cell), Ke)
Ferrite.assemble!(assembler::Ferrite.AbstractAssembler, cell::CellCache, fe::AbstractVector) = assemble!(assembler, celldofs(cell), fe)
function Ferrite.assemble!(assembler::VectorAssembler{<:Any, <:Any, atomic}, cell::CellCache, fe::AbstractVector) where {atomic}
    for (i, dof) in enumerate(celldofs(cell))
        _accum!(Val(atomic), assembler.f, fe[i], dof)
    end
    return
end
finalize_assembly!(assembler::Ferrite.AbstractAssembler) = nothing
finalize_assembly!(assembler::AbstractVector) = nothing
finalize_assembly!(::Nothing) = nothing   # sweeps whose sink is request-owned (quadrature storage)

# Sensitivity scatter targets. Deliberately not Ferrite.AbstractAssembler:
# their column/entry layout is the parameter space, not the dof space, so the
# celldofs-scatter convenience methods above must never match them.
struct ParameterJacobianAssembler{T, MT <: AbstractMatrix{T}, atomic}
    B::MT   # residual_size × nθ
end
function Ferrite.assemble!(assembler::ParameterJacobianAssembler{<:Any, <:Any, atomic}, cell::CellCache, Bₑ::AbstractMatrix) where {atomic}
    for j in axes(Bₑ, 2)
        for (i, dof) in enumerate(celldofs(cell))
            _accum!(Val(atomic), assembler.B, Bₑ[i, j], dof, j)
        end
    end
    return
end

struct ParameterVJPAssembler{T, VT <: AbstractVector{T}, atomic}
    g::VT   # length nθ
end
function Ferrite.assemble!(assembler::ParameterVJPAssembler{<:Any, <:Any, atomic}, cell::CellCache, gₑ::AbstractVector) where {atomic}
    for i in eachindex(gₑ)
        _accum!(Val(atomic), assembler.g, gₑ[i], i)
    end
    return
end

duplicate_for_device(device, a::ParameterJacobianAssembler) = a
duplicate_for_device(device, a::ParameterVJPAssembler) = a
finalize_assembly!(::ParameterJacobianAssembler) = nothing
finalize_assembly!(::ParameterVJPAssembler) = nothing

allocate_vector(::Vector{T}, dh) where T = zeros(T, ndofs(dh))
allocate_vector(::Type{Vector{T}}, dh) where T = zeros(T, ndofs(dh))
