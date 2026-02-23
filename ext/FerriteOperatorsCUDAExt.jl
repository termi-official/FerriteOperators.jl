module FerriteOperatorsCUDAExt

using FerriteOperators
using CUDA
using Adapt

FerriteOperators.default_backend(::CudaDevice) = CUDA.CUDABackend()

Adapt.adapt_storage(::CudaDevice, x::Array) = CuArray(x)

end
