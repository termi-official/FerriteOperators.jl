# NVIDIA CUDA entry point for the GPU test suite.
# Run directly:  julia --project=test test/gpu/cuda.jl
using CUDA

include("gpu.jl")

@test CUDA.functional()
CUDA.allowscalar(false)

run_gpu_tests(
    CudaDevice{Float64, Int32}(Int32(64), Int32(4));
    testset_name = "FerriteOperators GPU — CUDA (Float64)",
)
