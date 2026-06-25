# AMD ROCm entry point for the GPU test suite.
# Run directly:  julia --project=test test/gpu/amd.jl
using AMDGPU

include("gpu.jl")

@test AMDGPU.functional()
AMDGPU.allowscalar(false)

run_gpu_tests(
    RocDevice(Int32(64), Int32(4));
    testset_name = "FerriteOperators GPU — AMD ROCm",
)
