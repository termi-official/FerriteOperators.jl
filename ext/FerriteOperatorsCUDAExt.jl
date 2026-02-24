module FerriteOperatorsCUDAExt

using FerriteOperators
using CUDA
using Adapt

FerriteOperators.default_backend(::CudaDevice) = CUDA.CUDABackend()

Adapt.adapt_storage(::CudaDevice, x::Array) = CuArray(x)

@inline function FerriteOperators.max_auto_blocks(::CudaDevice, ::CUDA.CUDABackend, threads_per_block::Ti) where {Ti <: Integer}
    dev = CUDA.device()
    sm_count = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    max_threads_per_sm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    blocks_per_sm = cld(max_threads_per_sm, max(Int(threads_per_block), 1))
    return convert(Ti, max(1, sm_count * blocks_per_sm))
end

end
