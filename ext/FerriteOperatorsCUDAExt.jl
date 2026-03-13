module FerriteOperatorsCUDAExt

using FerriteOperators
using CUDA
using Adapt

import KernelAbstractions as KA
KA.backend(::CudaDevice) = CUDA.CUDABackend()

Adapt.adapt_storage(::CUDA.CUDABackend, x::Array) = CuArray(x)

function FerriteOperators.max_sharedmem_per_block(::CudaDevice)
    dev = CUDA.device()
    return CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
end

function FerriteOperators.max_registers_per_block(::CudaDevice)
    dev = CUDA.device()
    return CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
end

end
