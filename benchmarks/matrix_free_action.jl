# Operator-action throughput: sum-factorized matrix-free against assembled SpMV.
#
#     julia --project=benchmarks -t auto benchmarks/matrix_free_action.jl [target_dofs]
#
# One bilinear diffusion form on distorted hexahedra, evaluated five ways: the
# matrix-free action under both element mappings on CUDA and on two CPU devices,
# and `mul!` on the assembled CuSparse matrix of the same form. The headline is
# the polynomial order at which the action beats the SpMV, and whether one
# workgroup per element beats one worker per element.
#
# The mesh is resized per order so every arm carries roughly `target_dofs`
# unknowns; Ferrite ships `Lagrange{RefHexahedron, order}` for order ≤ 3.
#
# CUDA is optional: without a functional GPU the script reports the CPU arms.
using FerriteOperators, FerriteOperatorsExampleElements
using LinearAlgebra
using Printf
using SparseArrays
using Polyester

const CUDA_AVAILABLE = try
    @eval using CUDA
    @eval import CUDA: CUSPARSE.CuSparseMatrixCSC
    CUDA.functional()
catch err
    @info "CUDA unavailable, benchmarking the CPU arms only" err
    false
end

const Tv = Float32
const Ti = Int32
const TARGET_DOFS = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 140_000
const ORDERS = 1:3

function distorted_testbed(order, n; distortion = 0.15f0)
    grid = generate_grid(Hexahedron, (n, n, n),
                         Vec{3}((-1.0f0, -1.0f0, -1.0f0)), Vec{3}((1.0f0, 1.0f0, 1.0f0)))
    h = 2.0f0 / n
    nodes = [Ferrite.Node(Vec{3, Tv}(ntuple(d -> node.x[d] +
                (all(abs.(node.x) .< 1 - 1.0f-4) ? distortion * h * Tv(sin(2.7 * i + 1.3 * d)) : 0.0f0), 3)))
             for (i, node) in enumerate(Ferrite.getnodes(grid))]
    dh = DofHandler(Grid(Ferrite.getcells(grid), nodes))
    add!(dh, :u, Lagrange{RefHexahedron, order}())
    close!(dh)
    return dh
end

# Min-of-N wall time of one action, plus the host allocations of one call.
function measure(action!; samples = 20)
    action!()
    times = [(@elapsed action!()) for _ in 1:samples]
    return minimum(times), (@allocated action!())
end

function run_order(order)
    n = max(2, round(Int, (TARGET_DOFS^(1 / 3) - 1) / order))
    dh = distorted_testbed(order, n)
    qrc = QuadratureRuleCollection(Tv, order + 1)
    integrator = SumFactorizedDiffusionIntegrator(Tv(2.5), qrc, :u)
    ncells, n_dofs = getncells(Ferrite.get_grid(dh)), ndofs(dh)
    @printf("\norder %d: %d hexahedra, %d dofs, %d quadrature points per cell\n",
            order, ncells, n_dofs, (order + 1)^3)

    results = Pair{String, Tuple{Float64, Int}}[]
    u = Tv[sin(Tv(4.9) * i + Tv(2.1)) for i in 1:n_dofs]
    y = zeros(Tv, n_dofs)

    for (name, device, scheduling) in (
            ("CPU sequential action", SequentialCPUDevice{Tv, Int}(), SequentialScheduling()),
            ("CPU Polyester action ($(Threads.nthreads()) threads)",
             PolyesterDevice{Tv, Int}(32), ColoredScheduling()))
        op = setup_operator(AssemblyStrategy(MatrixFreeAction(), scheduling, device), integrator, dh)
        push!(results, name => measure(() -> mul!(y, op, u)))
    end

    if CUDA_AVAILABLE
        ud = CuVector(u)
        yd = CUDA.zeros(Tv, n_dofs)
        for (name, mapping) in (("CUDA worker-per-element action", WorkerPerElement()),
                                ("CUDA cooperative action", CooperativeElement()))
            device = KernelAbstractionsDevice(CUDABackend(); value_type = Tv, index_type = Ti,
                                              items_per_worker = 2, max_workgroup_size = 256)
            op = setup_operator(AssemblyStrategy(MatrixFreeAction(; element_mapping = mapping),
                                                 SequentialScheduling(), device), integrator, dh)
            push!(results, name => measure(() -> mul!(yd, op, ud)))
        end

        # The assembled reference: the matrix is built once, and only the SpMV
        # is timed. Its memory is what limits the order this arm reaches.
        assembled = try
            device   = KernelAbstractionsDevice(CUDABackend(); value_type = Tv, index_type = Ti)
            spec     = StandardOperatorSpecification(; matrix_type = CuSparseMatrixCSC{Tv, Ti})
            strategy = AssemblyStrategy(FullAssembly(spec), ColoredScheduling(), device)
            op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
            update_operator!(op, nothing)
            op
        catch err
            @info "assembled arm skipped at order $order" err
            nothing
        end
        if assembled !== nothing
            A = assembled.A
            @printf("  assembled matrix: %d stored entries (%.0f MB on device)\n",
                    nnz(A), nnz(A) * (sizeof(Tv) + sizeof(Ti)) / 2^20)
            push!(results, "CUDA assembled SpMV" =>
                measure(() -> (mul!(yd, A, ud); CUDA.synchronize())))
        end
    end

    reference = findfirst(r -> first(r) == "CUDA assembled SpMV", results)
    baseline = reference === nothing ? results[1].second[1] : results[reference].second[1]
    @printf("  %-42s %12s %10s %14s\n", "arm", "min time", "vs base", "host allocs")
    for (name, (time, allocations)) in results
        @printf("  %-42s %10.3f ms %9.2fx %12d B\n", name, 1.0e3 * time, baseline / time, allocations)
    end
    return nothing
end

function main()
    @printf("Matrix-free diffusion action vs assembled SpMV, %s, target %d dofs\n", Tv, TARGET_DOFS)
    CUDA_AVAILABLE && @printf("device: %s\n", CUDA.name(CUDA.device()))
    for order in ORDERS
        run_order(order)
    end
    println("\n`vs base` is relative to the assembled SpMV where that arm ran, ",
            "and to the sequential CPU action otherwise.")
    return nothing
end

main()
