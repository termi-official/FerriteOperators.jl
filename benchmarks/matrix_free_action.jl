# Operator-action throughput: sum-factorized matrix-free against assembled SpMV.
#
#     julia --project=benchmarks -t auto benchmarks/matrix_free_action.jl [target_dofs]
#
# One bilinear diffusion form on distorted hexahedra, evaluated five ways: the
# matrix-free action under both element mappings on CUDA and on two CPU devices,
# and `mul!` on the assembled CuSparse matrix of the same form. Each matrix-free
# arm runs under all three `storage` elections — `Recompute()` (MFEM NONE:
# re-derive the geometry per action), `Stored()` (PARTIAL: precomputed
# per-quadrature-point factors) and `ElementAssembly()` (ELEMENT: the dense
# element matrices, worker-per-element only). The headline is the polynomial
# order at which the action beats the SpMV, whether either stored level moves
# that crossover, and whether one workgroup per element beats one worker per
# element.
#
# The mesh is resized per order so every arm carries roughly `target_dofs`
# unknowns; Ferrite ships `Lagrange{RefHexahedron, order}` for order ≤ 3.
#
# CUDA is optional: without a functional GPU the script reports the CPU arms.
#
# THE IDLE-CLOCK TRAP. A GPU at rest sits at its idle clocks (300 MHz SM /
# 405 MHz memory on an RTX 2080) and takes on the order of a SECOND of sustained
# load to ramp to its boost clocks — far longer than one action. A short warm-up
# therefore samples an arm mid-ramp, and min-of-N does not protect against it:
# every sample of the window is slow. Measured on this benchmark's own arms,
# back-to-back stock runs disagreed by up to 5x on the short ones (the assembled
# SpMV read 0.105 ms hot and 1.667 ms cold). Every arm here is therefore preceded
# by a SUSTAINED burn of that same arm (`WARMUP_SECONDS`) and repeated over
# `PASSES` independent burn+sample rounds, and the SM/memory clocks are queried
# around each arm so a cold measurement is visible rather than silent.
using FerriteOperators, FerriteOperatorsExampleElements
using LinearAlgebra
using Printf
using SparseArrays
using Polyester
import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index, @Const
import FerriteOperators: Atomix

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
const WARMUP_SECONDS = 1.5
const PASSES = 3

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

# The SM and memory clocks, as `nvidia-smi` reports them, or `nothing` where the
# query is unavailable. Reported per arm so a number taken on a ramping device is
# visible in the output instead of silently wrong (see the idle-clock trap above).
function clocks()
    try
        sm, mem = split(strip(read(`nvidia-smi --query-gpu=clocks.sm,clocks.mem
                                    --format=csv,noheader,nounits`, String)), ',')
        return (parse(Int, strip(sm)), parse(Int, strip(mem)))
    catch
        return nothing
    end
end

_clock_note(::Nothing) = ""
_clock_note((sm, mem)) = @sprintf(" [%d/%d MHz]", sm, mem)

# Min-of-N wall time of one action over `PASSES` independent rounds, plus the
# host allocations of one call.
#
# Each round BURNS the arm for `WARMUP_SECONDS` before sampling it, so no round
# is sampled while the device clocks are ramping, and the reported time is the
# minimum over rounds — a round that still caught a cold device is discarded by
# the outer minimum rather than averaged into the result.
function measure(action!; samples = 20, passes = PASSES, warmup = WARMUP_SECONDS)
    best = Inf
    for _ in 1:passes
        deadline = time() + warmup
        action!()
        while time() < deadline
            action!()
        end
        best = min(best, minimum((@elapsed action!()) for _ in 1:samples))
    end
    return best, (@allocated action!())
end

# The ELEMENT level with nothing around it: one worker per cell, the dof range
# read straight from the flat `cell_dofs`, `uₑ` gathered into registers, each row
# of `Kₑ·uₑ` accumulated in a register and scattered atomically. The dof array
# carries the device handler's own `Int`, so the index traffic is the one the
# engine pays.
@kernel function _floor_action!(y, @Const(u), @Const(K), @Const(cell_dofs),
        ::Val{ND}, ncells) where {ND}
    worker = @index(Global, Linear)
    stride = prod(KA.@ndrange())
    for cell in worker:stride:ncells
        base = (cell - 1) * ND
        uₑ = ntuple(b -> (@inbounds u[cell_dofs[base + b]]), Val(ND))
        for a in 1:ND
            acc = zero(eltype(K))
            for b in 1:ND
                @inbounds acc += K[cell, a, b] * uₑ[b]
            end
            @inbounds Atomix.@atomic y[cell_dofs[base + a]] += acc
        end
    end
end

element_floor_time!(args...) = nothing

if CUDA_AVAILABLE
    @eval function element_floor_time!(yd, ud, integrator, dh, nd, ncells)
        backend = CUDABackend()
        device = KernelAbstractionsDevice(backend; value_type = Tv, index_type = Ti,
                                          items_per_worker = 2, max_workgroup_size = 256)
        op = setup_operator(AssemblyStrategy(MatrixFreeAction(; storage = ElementAssembly()),
                                             SequentialScheduling(), device), integrator, dh)
        K = op.engine.subdomain_caches[1].device_cache.element.K
        cell_dofs = CuVector(dh.cell_dofs)
        workgroup, blocks = FerriteOperators.launch_geometry(device, ncells)
        kernel = _floor_action!(backend, workgroup)
        time = first(measure(() -> (kernel(yd, ud, K, cell_dofs, Val(nd), ncells;
                                           ndrange = workgroup * blocks);
                                    KA.synchronize(backend))))
        op = nothing
        K = nothing
        cell_dofs = nothing
        GC.gc()
        CUDA.reclaim()
        return time
    end
end

const STORAGE = (("Stored", Stored()), ("Recompute", Recompute()), ("EA", ElementAssembly()))
# The ELEMENT level maps one worker onto one dense product; it has no lattice
# for a workgroup to split.
const COOPERATIVE_STORAGE = (("Stored", Stored()), ("Recompute", Recompute()))

function run_order(order)
    n = max(2, round(Int, (TARGET_DOFS^(1 / 3) - 1) / order))
    dh = distorted_testbed(order, n)
    qrc = QuadratureRuleCollection(Tv, order + 1)
    integrator = SumFactorizedDiffusionIntegrator(Tv(2.5), qrc, :u)
    ncells, n_dofs = getncells(Ferrite.get_grid(dh)), ndofs(dh)
    nqp = (order + 1)^3
    @printf("\norder %d: %d hexahedra, %d dofs, %d quadrature points per cell\n",
            order, ncells, n_dofs, nqp)
    # What each level keeps. PARTIAL holds one symmetric-by-construction 3x3
    # factor per quadrature point, stored FULL so that level and NONE form it by
    # the same expression; ELEMENT holds a dense `nd x nd` per cell, the term
    # that grows fastest with the order.
    nd = (order + 1)^3                       # dofs per hexahedron
    pa_bytes = nqp * 9 * sizeof(Tv)          # one full 3x3 factor per quadrature point
    ea_bytes = nd * nd * sizeof(Tv)          # one dense element matrix per cell
    @printf("  storage per cell: NONE 0 B | PARTIAL %d B (%.0f MB) | ELEMENT %d B (%.0f MB)\n",
            pa_bytes, ncells * pa_bytes / 2^20, ea_bytes, ncells * ea_bytes / 2^20)
    @printf("  storage per dof:  NONE 0 B | PARTIAL %d B | ELEMENT %d B\n",
            (ncells * pa_bytes) ÷ n_dofs, (ncells * ea_bytes) ÷ n_dofs)

    results = Pair{String, Tuple{Float64, Int}}[]
    fills = Pair{String, Float64}[]
    # SM/memory clocks as each device arm finished sampling, so a cold arm is
    # visible in the table.
    clockstamps = Dict{String, Any}()
    u = Tv[sin(Tv(4.9) * i + Tv(2.1)) for i in 1:n_dofs]
    y = zeros(Tv, n_dofs)

    for (name, device, scheduling) in (
            ("CPU sequential action", SequentialCPUDevice{Tv, Int}(), SequentialScheduling()),
            ("CPU Polyester action ($(Threads.nthreads()) threads)",
             PolyesterDevice{Tv, Int}(32), ColoredScheduling()))
        for (label, storage) in STORAGE
            op = setup_operator(AssemblyStrategy(MatrixFreeAction(; storage), scheduling, device),
                                integrator, dh)
            push!(results, "$name [$label]" => measure(() -> mul!(y, op, u)))
            storage isa Recompute || push!(fills, "$name [$label]" =>
                first(measure(() -> update_operator!(op, nothing); samples = 5)))
        end
    end

    if CUDA_AVAILABLE
        ud = CuVector(u)
        yd = CUDA.zeros(Tv, n_dofs)
        # Every device arm is built, measured and RELEASED before the next one
        # allocates. The stored levels and the per-worker scratch run into
        # hundreds of megabytes between them, and an arm timed on an allocator
        # the previous arms fragmented measures the allocator: the assembled
        # SpMV at order 1 reads 6x its clean-device time that way.
        function device_arm!(build)
            op = build()
            time = measure(() -> mul!(yd, op, ud))
            fill = op isa MatrixFreeFerriteOperator ?
                first(measure(() -> (update_operator!(op, nothing); CUDA.synchronize()); samples = 5)) : nothing
            op = nothing
            GC.gc()
            CUDA.reclaim()
            return time, fill
        end

        # The assembled reference goes first, on the cleanest device state there
        # is; only the SpMV is timed, never the assembly.
        try
            device   = KernelAbstractionsDevice(CUDABackend(); value_type = Tv, index_type = Ti)
            spec     = StandardOperatorSpecification(; matrix_type = CuSparseMatrixCSC{Tv, Ti})
            strategy = AssemblyStrategy(FullAssembly(spec), ColoredScheduling(), device)
            op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
            update_operator!(op, nothing)
            A = op.A
            entry_bytes = nnz(A) * (sizeof(Tv) + sizeof(Ti))
            @printf("  assembled matrix: %d stored entries (%.0f MB on device), %d B per cell, %d B per dof\n",
                    nnz(A), entry_bytes / 2^20, entry_bytes ÷ ncells, entry_bytes ÷ n_dofs)
            push!(results, "CUDA assembled SpMV" => measure(() -> (mul!(yd, A, ud); CUDA.synchronize())))
            clockstamps["CUDA assembled SpMV"] = clocks()
            op = nothing
            A = nothing
            GC.gc()
            CUDA.reclaim()
        catch err
            @info "assembled arm skipped at order $order" err
        end

        for (name, mapping, levels) in (
                ("CUDA worker-per-element action", WorkerPerElement(), STORAGE),
                ("CUDA cooperative action", CooperativeElement(), COOPERATIVE_STORAGE)),
            (label, storage) in levels

            time, fill = device_arm!() do
                device = KernelAbstractionsDevice(CUDABackend(); value_type = Tv, index_type = Ti,
                                                  items_per_worker = 2, max_workgroup_size = 256)
                setup_operator(AssemblyStrategy(MatrixFreeAction(; element_mapping = mapping, storage),
                                                SequentialScheduling(), device), integrator, dh)
            end
            push!(results, "$name [$label]" => time)
            clockstamps["$name [$label]"] = clocks()
            storage isa Recompute || push!(fills, "$name [$label]" => fill)
        end

        # The MATH FLOOR of the ELEMENT level: the same dense `Kₑ·uₑ` per cell,
        # written as one hand-rolled kernel that reads the dof range straight out
        # of the flat `cell_dofs`, gathers into registers and scatters the rows
        # atomically. It carries no workspace, no element cache and no assembler,
        # so the gap between it and the `[EA]` arm above is what the engine's
        # per-item plumbing costs.
        floor_time = element_floor_time!(yd, ud, integrator, dh, nd, ncells)
        if floor_time !== nothing
            push!(results, "CUDA ELEMENT math floor (hand-rolled)" => (floor_time, 0))
            clockstamps["CUDA ELEMENT math floor (hand-rolled)"] = clocks()
        end
    end

    reference = findfirst(r -> first(r) == "CUDA assembled SpMV", results)
    baseline = reference === nothing ? results[1].second[1] : results[reference].second[1]
    @printf("  %-46s %12s %10s %14s\n", "arm", "min time", "vs base", "host allocs")
    for (name, (time, allocations)) in results
        @printf("  %-46s %10.3f ms %9.2fx %12d B%s\n", name, 1.0e3 * time, baseline / time,
                allocations, _clock_note(get(clockstamps, name, nothing)))
    end
    isempty(fills) || println("  storage fill (update_operator!, min of 5):")
    for (name, time) in fills
        @printf("  %-46s %10.3f ms\n", name, 1.0e3 * time)
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
            "and to the sequential CPU action otherwise. `[Recompute]` is the MFEM ",
            "NONE level, `[Stored]` PARTIAL, `[EA]` ELEMENT.")
    return nothing
end

main()
