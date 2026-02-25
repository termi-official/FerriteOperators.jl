using Test
using StaticArrays
import KernelAbstractions as KA

# ---- Test: MArray (registers) inside a KA kernel ----
# Tests whether a struct with MMatrix/MVector fields works on GPU

## Test buffer: mimics SimpleAssemblyCache with static arrays ##
struct TestBuffer{T}
    Ke::MMatrix{4, 4, T, 16}
    ue::MVector{4, T}
    re::MVector{4, T}
end

## Kernel: each thread creates a TestBuffer, does a small computation, writes result ##
KA.@kernel function _test_marray_kernel!(output, scale::T) where T
    tid = KA.@index(Global, Linear)

    # Create MArray on the stack (registers on GPU)
    Ke = MMatrix{4, 4, T, 16}(undef)
    ue = MVector{4, T}(undef)
    re = MVector{4, T}(undef)

    # Fill Ke: simple pattern so we can verify
    for j in 1:4
        for i in 1:4
            Ke[i, j] = T(tid) * T(i + (j - 1) * 4) * scale
        end
    end

    # Fill ue
    for i in 1:4
        ue[i] = T(tid) * T(i)
    end

    # re = Ke * ue (small mat-vec in registers)
    for i in 1:4
        s = zero(T)
        for j in 1:4
            s += Ke[i, j] * ue[j]
        end
        re[i] = s
    end

    # Write result: store sum of re for this thread
    result = zero(T)
    for i in 1:4
        result += re[i]
    end
    output[tid] = result
end

## Also test with the struct passed in ##
KA.@kernel function _test_buffer_struct_kernel!(output, buffers, scale::T) where T
    tid = KA.@index(Global, Linear)

    buf = buffers[tid]
    Ke = buf.Ke
    ue = buf.ue

    # re = Ke * ue
    re = MVector{4, T}(undef)
    for i in 1:4
        s = zero(T)
        for j in 1:4
            s += Ke[i, j] * ue[j]
        end
        re[i] = s
    end

    output[tid] = re[1] + re[2] + re[3] + re[4]
end

## CPU reference ##
function reference_result(T, nthreads, scale)
    results = zeros(T, nthreads)
    for tid in 1:nthreads
        Ke = zeros(T, 4, 4)
        ue = zeros(T, 4)
        for j in 1:4, i in 1:4
            Ke[i, j] = T(tid) * T(i + (j - 1) * 4) * scale
        end
        for i in 1:4
            ue[i] = T(tid) * T(i)
        end
        re = Ke * ue
        results[tid] = sum(re)
    end
    return results
end

println("="^70)
println("Test: MArray (registers) inside KA kernel")
println("="^70)

## NOTE: KA v0.9.40 CPU backend has a bug with @index(Global, Linear) — skipping CPU test

## Test on ROC if available ##
roc_available = try
    using AMDGPU
    AMDGPU.functional()
catch
    false
end

if roc_available
    @testset "MArray in KA kernel - ROC backend" begin
        nthreads = 64  # one wavefront
        T = Float64
        scale = T(0.5)

        output_gpu = AMDGPU.zeros(T, nthreads)
        backend = AMDGPU.ROCBackend()

        kernel = _test_marray_kernel!(backend, nthreads)
        kernel(output_gpu, scale; ndrange=nthreads)
        KA.synchronize(backend)

        output = Array(output_gpu)
        ref = reference_result(T, nthreads, scale)
        @test output ≈ ref
        println("  ROC backend: PASS (output[1:4] = $(output[1:4]))")
    end
else
    @info "AMDGPU not available, skipping ROC MArray test"
end

## Test on CUDA if available ##
cuda_available = try
    using CUDA
    CUDA.functional()
catch
    false
end

if cuda_available
    @testset "MArray in KA kernel - CUDA backend" begin
        nthreads = 256
        T = Float32
        scale = T(0.5)

        output_gpu = CUDA.zeros(T, nthreads)
        backend = CUDA.CUDABackend()

        kernel = _test_marray_kernel!(backend, nthreads)
        kernel(output_gpu, scale; ndrange=nthreads)
        KA.synchronize(backend)

        output = Array(output_gpu)
        ref = reference_result(T, nthreads, scale)
        @test output ≈ ref
        println("  CUDA backend: PASS (output[1:4] = $(output[1:4]))")
    end
else
    @info "CUDA not available, skipping CUDA MArray test"
end

println("="^70)
