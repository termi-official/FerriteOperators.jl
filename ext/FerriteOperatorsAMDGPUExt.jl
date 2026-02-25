module FerriteOperatorsAMDGPUExt

using FerriteOperators
using AMDGPU
using Adapt

FerriteOperators.default_backend(::RocDevice) = AMDGPU.ROCBackend()

Adapt.adapt_storage(::AMDGPU.ROCBackend, x::Array) = ROCArray(x)

function FerriteOperators.max_sharedmem_per_block(::RocDevice)
    dev = AMDGPU.device()
    return AMDGPU.HIP.attribute(dev, AMDGPU.HIP.hipDeviceAttributeMaxSharedMemoryPerBlock)
end

function FerriteOperators.max_registers_per_block(::RocDevice)
    dev = AMDGPU.device()
    return AMDGPU.HIP.attribute(dev, AMDGPU.HIP.hipDeviceAttributeMaxRegistersPerBlock)
end

end
