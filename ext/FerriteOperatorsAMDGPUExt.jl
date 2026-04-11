module FerriteOperatorsAMDGPUExt

using FerriteOperators
using AMDGPU
using Adapt

import KernelAbstractions as KA
KA.backend(::RocDevice) = AMDGPU.ROCBackend()

FerriteOperators.matrix_type(device::RocDevice, ::FerriteOperators.StandardOperatorSpecification) = AMDGPU.rocSPARSE.ROCSparseMatrixCSC{FerriteOperators.value_type(device), FerriteOperators.index_type(device)}
FerriteOperators.vector_type(device::RocDevice) = ROCVector{FerriteOperators.value_type(device)}

end
