module FerriteOperatorsCUDAExt

using FerriteOperators
using CUDA
using Adapt

import KernelAbstractions as KA
KA.backend(::CudaDevice) = CUDA.CUDABackend()

Adapt.adapt_storage(::CUDA.CUDABackend, x::Array) = CuArray(x)

FerriteOperators.matrix_type(device::CudaDevice, ::FerriteOperators.StandardOperatorSpecification) = CUDA.CUSPARSE.CuSparseMatrixCSC{FerriteOperators.value_type(device), FerriteOperators.index_type(device)}
FerriteOperators.vector_type(device::CudaDevice) = CuVector{FerriteOperators.value_type(device)}

# FIXME: upstream — DeviceCSCAssembler does not subtype Ferrite.AbstractAssembler, so the
# cell-based assemble! bridge in ferrite-addons/assembly.jl doesn't match.
const DeviceCSCAssembler = Base.get_extension(Ferrite, :FerriteCudaExt).DeviceCSCAssembler
const CellCacheType = FerriteOperators.CellCacheType
Ferrite.assemble!(a::DeviceCSCAssembler, cell::CellCacheType, Ke::AbstractMatrix, fe::AbstractVector) = Ferrite.assemble!(a, Ferrite.celldofs(cell), Ke, fe)
Ferrite.assemble!(a::DeviceCSCAssembler, cell::CellCacheType, Ke::AbstractMatrix) = Ferrite.assemble!(a, Ferrite.celldofs(cell), Ke)
Ferrite.assemble!(a::DeviceCSCAssembler, cell::CellCacheType, fe::AbstractVector) = Ferrite.assemble!(a, Ferrite.celldofs(cell), fe)
FerriteOperators.finalize_assembly!(::DeviceCSCAssembler) = nothing

end
