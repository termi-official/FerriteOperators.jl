module FerriteOperatorsAMDGPUExt

using FerriteOperators
using AMDGPU
using Adapt

import KernelAbstractions as KA
KA.backend(::RocDevice) = AMDGPU.ROCBackend()

Adapt.adapt_storage(::AMDGPU.ROCBackend, x::Array) = ROCArray(x)

end
