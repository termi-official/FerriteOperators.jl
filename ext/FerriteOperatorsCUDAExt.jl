module FerriteOperatorsCUDAExt

using FerriteOperators
using CUDA
using Adapt

import Ferrite
import KernelAbstractions as KA
KA.backend(::CudaDevice) = CUDA.CUDABackend()

FerriteOperators.matrix_type(device::CudaDevice, ::FerriteOperators.StandardOperatorSpecification) = CUDA.CUSPARSE.CuSparseMatrixCSC{FerriteOperators.value_type(device), FerriteOperators.index_type(device)}
FerriteOperators.vector_type(device::CudaDevice) = CuVector{FerriteOperators.value_type(device)}

# FIXME upstream: Ferrite's DeviceCSCAssembler requires a residual vector, so the
# matrix-only `start_assemble(K::CuSparseMatrixCSC)` path (f === nothing) errors.
# Supply a throwaway residual buffer for matrix-only GPU assembly.
function Ferrite.start_assemble(::FerriteOperators.AbstractAssemblyStrategy, J::CUDA.CUSPARSE.CuSparseMatrixCSC{Tv}; fillzero::Bool = true) where {Tv}
    f = CUDA.zeros(Tv, size(J, 1))
    return Ferrite.start_assemble(J, f; fillzero)
end

# # FIXME: upstream — DeviceCSCAssembler does not subtype Ferrite.AbstractAssembler, so the
# # cell-based assemble! bridge in ferrite-addons/assembly.jl doesn't match.
# const DeviceCSCAssembler = Base.get_extension(Ferrite, :FerriteCudaExt).DeviceCSCAssembler
# const CellCacheType = FerriteOperators.CellCacheType
# Ferrite.assemble!(a::DeviceCSCAssembler, cell::CellCacheType, Ke::AbstractMatrix, fe::AbstractVector) = Ferrite.assemble!(a, Ferrite.celldofs(cell), Ke, fe)
# Ferrite.assemble!(a::DeviceCSCAssembler, cell::CellCacheType, Ke::AbstractMatrix) = Ferrite.assemble!(a, Ferrite.celldofs(cell), Ke)
# Ferrite.assemble!(a::DeviceCSCAssembler, cell::CellCacheType, fe::AbstractVector) = Ferrite.assemble!(a, Ferrite.celldofs(cell), fe)
# FerriteOperators.finalize_assembly!(::DeviceCSCAssembler) = nothing

end
