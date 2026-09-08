# Assembly throughput of a bilinear diffusion operator across devices.
#
#     julia --project=benchmarks -t auto benchmarks/gpu_assembly.jl [elements_per_side]
#
# Linear hexahedra with a 2-point rule are memory bound — a handful of flops per
# loaded shape gradient — so this measures the assembly machinery and the
# scatter, not the element kernel. Expect a modest GPU speedup at best; a
# compute-heavy element (higher order, a material law) is where the device wins.
#
# CUDA is optional: without a functional GPU the script reports the CPU devices
# alone.
using FerriteOperators, FerriteOperatorsExampleElements
using Printf
using Polyester
import KernelAbstractions as KA

const CUDA_AVAILABLE = try
    @eval using CUDA
    @eval import CUDA: CUSPARSE.CuSparseMatrixCSC
    CUDA.functional()
catch err
    @info "CUDA unavailable, benchmarking the CPU devices only" err
    false
end

const Tv = Float32
const Ti = Int32
const N  = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 40

function testbed(n)
    grid = generate_grid(Hexahedron, (n, n, n),
                         Vec{3}((-1.0f0, -1.0f0, -1.0f0)), Vec{3}((1.0f0, 1.0f0, 1.0f0)))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    return dh
end

# Min-of-N wall time after a warmup, plus the host allocations of one update.
function measure(op; samples = 10)
    update_operator!(op, nothing)
    times = [(@elapsed update_operator!(op, nothing)) for _ in 1:samples]
    allocations = @allocated update_operator!(op, nothing)
    return minimum(times), allocations
end

function main()
    dh  = testbed(N)
    qrc = QuadratureRuleCollection(Tv, 2)
    integrator = SimpleBilinearDiffusionIntegrator(2.5, qrc, :u)
    ncells = getncells(Ferrite.get_grid(dh))
    @printf("Bilinear diffusion, %d linear hexahedra, %d dofs, %s\n", ncells, ndofs(dh), Tv)

    results = Pair{String, Tuple{Float64, Int}}[]

    push!(results, "SequentialCPUDevice" =>
        measure(setup_operator(AssemblyStrategy(SequentialCPUDevice{Tv, Ti}()), integrator, dh)))

    push!(results, "PolyesterDevice ($(Threads.nthreads()) threads)" =>
        measure(setup_operator(
            AssemblyStrategy(PolyesterDevice{Tv, Ti}(32); scheduling = ColoredScheduling()),
            integrator, dh)))

    push!(results, "KernelAbstractionsDevice (KA.CPU)" =>
        measure(setup_operator(
            AssemblyStrategy(KernelAbstractionsDevice(KA.CPU(); value_type = Tv, index_type = Ti);
                             scheduling = ColoredScheduling()),
            integrator, dh)))

    if CUDA_AVAILABLE
        device   = KernelAbstractionsDevice(CUDABackend(); value_type = Tv, index_type = Ti)
        spec     = StandardOperatorSpecification(; matrix_type = CuSparseMatrixCSC{Tv, Ti})
        strategy = AssemblyStrategy(FullAssembly(spec), ColoredScheduling(), device)
        push!(results, "KernelAbstractionsDevice (CUDA)" => measure(setup_operator(strategy, integrator, dh)))
    end

    baseline = first(results).second[1]
    @printf("\n%-40s %12s %10s %14s\n", "device", "min time", "speedup", "host allocs")
    for (name, (time, allocations)) in results
        @printf("%-40s %10.2f ms %9.2fx %12d B\n", name, 1e3 * time, baseline / time, allocations)
    end
    return nothing
end

main()
