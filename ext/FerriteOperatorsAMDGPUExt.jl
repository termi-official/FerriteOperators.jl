module FerriteOperatorsAMDGPUExt

using FerriteOperators
using AMDGPU
using Adapt

FerriteOperators.default_backend(::RocDevice) = AMDGPU.ROCBackend()

Adapt.adapt_storage(::RocDevice, x::Array) = ROCArray(x)

@inline function FerriteOperators.max_auto_blocks(::RocDevice, ::AMDGPU.ROCBackend, threads_per_block::Ti) where {Ti <: Integer}
    dev = AMDGPU.HIP.device()
    sm_count = AMDGPU.HIP.attribute(dev, AMDGPU.HIP.hipDeviceAttributeMultiprocessorCount)
    max_threads_per_sm = AMDGPU.HIP.attribute(dev, AMDGPU.HIP.hipDeviceAttributeMaxThreadsPerMultiProcessor)
    blocks_per_sm = cld(max_threads_per_sm, max(Int(threads_per_block), 1))
    return convert(Ti, max(1, sm_count * blocks_per_sm))
end

end
