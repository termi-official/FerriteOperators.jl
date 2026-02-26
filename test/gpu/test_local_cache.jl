using Test
using StaticArrays
using Adapt
import KernelAbstractions as KA
using FerriteOperators
import FerriteOperators:
    allocate_local_cache, materialize, _gpu_zeros, default_backend,
    GPURegisterMemoryType, GPUGlobalMemoryType,
    AbstractGPULocalCacheFactory, RegisterLocalCacheFactory, SharedLocalCacheFactory,
    AbstractGlobalLocalCacheFactory, BilinearGlobalLocalCacheFactory,
    BilinearBufferRequirement, NonlinearBufferRequirement, LinearBufferRequirement,
    BilinearLocalCache, NonlinearLocalCache, LinearLocalCache,
    get_Ke, get_ue, get_re, value_type, ndofs_per_cell

const NTHREADS = 64  # one wavefront / warp
const NDOFS = 4      # pretend 4 dofs per cell

# Mock: replaces sdh — only thing allocate_local_cache needs is ndofs_per_cell
struct MockSubDofHandler
    n::Int
end
ndofs_per_cell(m::MockSubDofHandler) = m.n

## CPU reference ##
function reference_trace(::Type{T}, N, tid) where T
    s = zero(T)
    for i in 1:N
        s += T(tid) * T(i + (i - 1) * N)
    end
    return s
end

function reference_matvec_sum(::Type{T}, N, tid) where T
    Ke = [T(i + (j - 1) * N) for i in 1:N, j in 1:N]
    ue = [T(tid) * T(i) for i in 1:N]
    return sum(Ke * ue)
end

## Unified kernels: device_cache is either GPULocalCacheConfig (Register) or pool (Global).
## materialize(device_cache, tid, local_tid, groupsize) creates the local cache uniformly.
KA.@kernel function _test_bilinear_kernel!(output, device_cache, ::Val{N}) where N
    tid = KA.@index(Global, Linear)
    local_tid = KA.@index(Local, Linear)
    groupsize = KA.@groupsize()[1]
    T = eltype(output)

    cache = materialize(device_cache, tid, local_tid, groupsize)
    Ke = get_Ke(cache)

    for j in 1:N, i in 1:N
        Ke[i, j] = T(tid) * T(i + (j - 1) * N)
    end
    s = zero(T)
    for i in 1:N
        s += Ke[i, i]
    end
    output[tid] = s
end

KA.@kernel function _test_nonlinear_kernel!(output, device_cache, ::Val{N}) where N
    tid = KA.@index(Global, Linear)
    local_tid = KA.@index(Local, Linear)
    groupsize = KA.@groupsize()[1]
    T = eltype(output)

    cache = materialize(device_cache, tid, local_tid, groupsize)
    Ke = get_Ke(cache)
    ue = get_ue(cache)
    re = get_re(cache)

    for j in 1:N, i in 1:N
        Ke[i, j] = T(i + (j - 1) * N)
    end
    for i in 1:N
        ue[i] = T(tid) * T(i)
    end
    for i in 1:N
        s = zero(T)
        for j in 1:N
            s += Ke[i, j] * ue[j]
        end
        re[i] = s
    end
    s = zero(T)
    for i in 1:N
        s += re[i]
    end
    output[tid] = s
end

## Pipeline test functions ##

function test_register_bilinear(backend, device, ::Type{T}) where T
    @testset "Register - Bilinear" begin
        mock_sdh = MockSubDofHandler(NDOFS)

        # allocate_local_cache returns a RegisterLocalCacheFactory (zero-size isbitstype)
        device_cache = allocate_local_cache(BilinearBufferRequirement(), GPURegisterMemoryType(), device, mock_sdh; total_nthreads=NTHREADS)
        @test device_cache isa RegisterLocalCacheFactory
        @test isbitstype(typeof(device_cache))

        # Launch kernel — device_cache is passed directly (not a vector)
        output_gpu = Adapt.adapt(backend, zeros(T, NTHREADS))
        kernel = _test_bilinear_kernel!(backend, NTHREADS)
        kernel(output_gpu, device_cache, Val(NDOFS); ndrange=NTHREADS)
        KA.synchronize(backend)

        output = Array(output_gpu)
        for tid in 1:NTHREADS
            @test output[tid] ≈ reference_trace(T, NDOFS, tid)
        end
    end
end

function test_register_nonlinear(backend, device, ::Type{T}) where T
    @testset "Register - Nonlinear" begin
        mock_sdh = MockSubDofHandler(NDOFS)

        device_cache = allocate_local_cache(NonlinearBufferRequirement(), GPURegisterMemoryType(), device, mock_sdh; total_nthreads=NTHREADS)
        @test device_cache isa RegisterLocalCacheFactory

        output_gpu = Adapt.adapt(backend, zeros(T, NTHREADS))
        kernel = _test_nonlinear_kernel!(backend, NTHREADS)
        kernel(output_gpu, device_cache, Val(NDOFS); ndrange=NTHREADS)
        KA.synchronize(backend)

        output = Array(output_gpu)
        for tid in 1:NTHREADS
            @test output[tid] ≈ reference_matvec_sum(T, NDOFS, tid)
        end
    end
end

function test_global_bilinear(backend, device, ::Type{T}) where T
    @testset "Global - Bilinear" begin
        mock_sdh = MockSubDofHandler(NDOFS)

        # allocate_local_cache returns a BilinearGlobalLocalCacheFactory (wraps GPU array pool)
        factory = allocate_local_cache(BilinearBufferRequirement(), GPUGlobalMemoryType(), device, mock_sdh; total_nthreads=NTHREADS)
        @test factory isa BilinearGlobalLocalCacheFactory
        @test !(factory.Ke_pool isa Array)  # should be a GPU array

        # Launch kernel — factory gets adapted by KA (ROCArray → ROCDeviceArray)
        output_gpu = Adapt.adapt(backend, zeros(T, NTHREADS))
        kernel = _test_bilinear_kernel!(backend, NTHREADS)
        kernel(output_gpu, factory, Val(NDOFS); ndrange=NTHREADS)
        KA.synchronize(backend)

        output = Array(output_gpu)
        for tid in 1:NTHREADS
            @test output[tid] ≈ reference_trace(T, NDOFS, tid)
        end
    end
end

## Top-level runner ##
function test_local_cache_pipeline(backend, device, ::Type{T}) where T
    test_register_bilinear(backend, device, T)
    test_register_nonlinear(backend, device, T)
    test_global_bilinear(backend, device, T)
end

## Detect backend and run ##
roc_available = try
    using AMDGPU
    include(joinpath(@__DIR__, "..", "..", "ext", "FerriteOperatorsAMDGPUExt.jl"))
    AMDGPU.functional()
catch; false end

cuda_available = try
    using CUDA
    include(joinpath(@__DIR__, "..", "..", "ext", "FerriteOperatorsCUDAExt.jl"))
    CUDA.functional()
catch; false end

@testset "Local Cache Pipeline" begin
    if roc_available
        @testset "ROC" begin
            test_local_cache_pipeline(AMDGPU.ROCBackend(), RocDevice(), Float64)
        end
    else
        @info "AMDGPU not available, skipping ROC"
    end

    if cuda_available
        @testset "CUDA" begin
            test_local_cache_pipeline(CUDA.CUDABackend(), CudaDevice(), Float32)
        end
    else
        @info "CUDA not available, skipping CUDA"
    end
end
