module FerriteOperatorsCUDAExt

using FerriteOperators
using CUDA
using Adapt

import KernelAbstractions as KA
KA.backend(::CudaDevice) = CUDA.CUDABackend()

Adapt.adapt_storage(::CUDA.CUDABackend, x::Array) = CuArray(x)

end
