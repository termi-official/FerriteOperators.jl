module FerriteOperatorsCUDAExt

using FerriteOperators
using CUDA
using Adapt

FerriteOperators.default_backend(::CudaDevice) = CUDA.CUDABackend()

Adapt.adapt_storage(::CUDA.CUDABackend, x::Array) = CuArray(x)

end
