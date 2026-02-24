module FerriteOperatorsAMDGPUExt

using FerriteOperators
using AMDGPU
using Adapt

FerriteOperators.default_backend(::RocDevice) = AMDGPU.ROCBackend()

Adapt.adapt_storage(::AMDGPU.ROCBackend, x::Array) = ROCArray(x)

end
