# DG operator action: the condensed block-row store against the assembled SpMV.
#
#     julia --project=benchmarks -t auto benchmarks/dg_action.jl [n_p1 n_p2]
#
# One SIPG diffusion form (`SIPGDiffusionIntegrator`) over `DiscontinuousLagrange`
# hexahedra, applied on the device four ways and bounded two more. The question is
# whether the shipped matrix-free route — `MatrixFreeAction(storage =
# BlockRowAssembly())` under `LanesPerElement` and the one-colour
# `ColoredScheduling` partition — beats cuSPARSE on the assembled matrix of the
# SAME form, and by how much.
#
# Why it can. Over a discontinuous space the assembled matrix stores each cell's
# diagonal block once and each interior facet's coupling TWICE (both triangles);
# the block-row store keeps exactly those blocks and drops the per-entry CSC row
# index entirely — a cell-contiguous dof layout lets it DERIVE each gathered dof
# from the neighbour cell id it already reads — so it touches fewer bytes per dof.
# What it gives back is rate: cuSPARSE streams CSC/CSR faster than the block-row
# kernel streams its four-dimensional store. The speedup is that byte ratio times
# that rate ratio, and the table below reports both factors, so a result is
# attributable and not just a verdict.
#
# And why it is DETERMINISTIC. Every item scatters its own cell's `Nb` rows and
# nothing else (`element_scatter_length`), so the whole cell set is one valid
# colour — no colouring algorithm runs, no atomic is issued, and two runs of the
# same action on the same input agree BIT FOR BIT. That is asserted here, before
# either block-row arm is timed; it is the property a Krylov method needs and the
# facet-item action (atomic on both sides) does not have.
#
# Every timed arm is validated against the host-assembled `A * u` to `1e-4`
# relative BEFORE `measure` is called, and the actual error is printed: these arms
# read stores built by different routes, so a silent disagreement would publish a
# fast WRONG number.
#
# The block-row arms are quoted against a PURE-STREAM CEILING of their own access
# pattern (`_stream_block_row_lane!`/`_stream_block_row_worker!`): the same loads
# with the product and the scatter removed, so "x% of optimal" is a measured
# distance rather than a cuSPARSE-relative claim. As in `matrix_free_action.jl`
# these bound an ACCESS PATTERN, not the memory system, and can sit below a kernel
# that hides its latency behind arithmetic.
#
# THE IDLE-CLOCK TRAP. A GPU at rest sits at its idle clocks (300 MHz SM on an
# RTX 2080) and takes on the order of a SECOND of sustained load to reach boost —
# far longer than one action, so a short warm-up samples an arm mid-ramp and
# min-of-N does not protect against it. Every arm is therefore preceded by a
# SUSTAINED burn of that same arm (`WARMUP_SECONDS`), repeated over `PASSES`
# independent burn+sample rounds, and the SM/memory clocks are sampled MID-BURN
# (see `measure`) so a cold measurement is visible rather than silent.
#
# MEASURED — RTX 2080, min of 3 burn+sample passes x 20 samples, distorted
# hexahedra, launch policy `HOUSE_POLICY`. Every arm ran at 1710-1890 MHz SM (the
# card's rated boost is 1710) and 6800 MHz memory, and validated against the
# host-assembled `A * u` to 6e-7 or better. Two full runs agreed within 2.6%.
#
#   hex p=1, Nb=8, 15 625 cells, 125 000 dofs, 45 000 interior facets
#   assembled 53.0 MB (444 B/dof touched) | block-row 26.7 MB (230 B/dof touched)
#
#   | arm                                | min time | vs CSC | model GB/s | bitwise      |
#   |------------------------------------|---------:|-------:|-----------:|--------------|
#   | assembled CSC SpMV (cuSPARSE)      | 0.157 ms |  1.00x |        355 | no           |
#   | assembled CSR SpMV (cuSPARSE)      | 0.154 ms |  1.02x |        362 | yes          |
#   | block-row, LanesPerElement         | 0.125 ms |  1.26x |        231 | yes          |
#   | block-row, WorkerPerElement        | 0.158 ms |  0.99x |        182 | yes          |
#   | MatrixFreeAction[Recompute]        |        - |      - |          - | no route     |
#   | ceiling: block-row stream [lanes]  | 0.092 ms |  1.70x |        305 | -            |
#   | ceiling: block-row stream [worker] | 0.100 ms |  1.56x |        278 | -            |
#
#   hex p=2, Nb=27, 4 096 cells, 110 592 dofs, 11 520 interior facets
#   assembled 152.2 MB (1443 B/dof touched) | block-row 79.7 MB (725 B/dof touched)
#
#   | arm                                | min time | vs CSC | model GB/s | bitwise      |
#   |------------------------------------|---------:|-------:|-----------:|--------------|
#   | assembled CSC SpMV (cuSPARSE)      | 0.416 ms |  1.00x |        384 | no           |
#   | assembled CSR SpMV (cuSPARSE)      | 0.421 ms |  0.99x |        379 | yes          |
#   | block-row, LanesPerElement         | 0.533 ms |  0.78x |        151 | yes          |
#   | block-row, WorkerPerElement        | 0.739 ms |  0.56x |        109 | yes          |
#   | MatrixFreeAction[Recompute]        |        - |      - |          - | no route     |
#   | ceiling: block-row stream [lanes]  | 0.745 ms |  0.56x |        107 | not bounding |
#   | ceiling: block-row stream [worker] | 0.427 ms |  0.97x |        186 | -            |
#
# THE BLOCK-ROW ACTION BEATS cuSPARSE AT p=1 AND NOT AT p=2, and neither number is
# inside its band. Both factors of the byte model are measured above:
#
#   |      | byte ratio | rate ratio | product | measured |
#   | p=1  |      1.93x |       0.65 |   1.26x |   1.26x  |
#   | p=2  |      1.99x |       0.39 |   0.78x |   0.78x  |
#
# The gather window is no longer a materialized `(slot, k)` table: over a
# cell-contiguous dof layout the store DERIVES entry `(f, j)` as
# `(neighbour(f) - 1)·Nb + j` from the per-slot neighbour list it already reads to
# skip its boundary blocks. At p=1 that moved BOTH factors — bytes 1.57x -> 1.93x
# (283 -> 230 B/dof) and rate 0.56 -> 0.65, the rate because the `uₑ[j]` the
# kernel chases into `u` is now computed rather than fetched — and the arm went
# 0.180 -> 0.125 ms, 0.87x -> 1.26x.
#
# AT p=2 IT COST 18%, 0.453 -> 0.533 ms, 0.93x -> 0.78x, and that regression is
# the round's real finding: THE p=2 LANE KERNEL IS NOT BYTE-BOUND. Three controls,
# same harness, same session, headline arm, `vs CSC`:
#
#   |                        |    p=1    |    p=2    |
#   | materialized `Int`     | 0.180 ms  | 0.453 ms  |  (slice 5, the baseline)
#   | materialized `Int32`   | 0.168 ms  | 0.676 ms  |  strictly fewer bytes
#   | derived (shipped here) | 0.125 ms  | 0.533 ms  |  no index stream at all
#
# Narrowing the table to `Int32` REMOVES 3.5 MB of p=2 traffic and costs 49%. A
# store that touches strictly fewer bytes running half again slower is not a byte
# model missing a term; it is a kernel whose time is set by how NVVM schedules it,
# and at `Nb`=27 (189 gathers and 189 stored values per row) that kernel sits on a
# cliff. Two further variants measured the same way: doing the derivation's
# arithmetic in `Int32` moved p=1/p=2 by <1%/1%, and hoisting it to ONE base per
# BLOCK instead of a `divrem` per ENTRY — strictly less work on every axis — gave
# 0.145 ms / 0.872 ms, worse at both orders. The p=2 limiter remains RECORDED and
# unaddressed: the `(cell, facet, i, j)` store walks its inner `j` with a stride of
# `ncells * (1 + Nf) * Nb`, 442 KB at `Nb`=27.
#
# NOT occupancy. The launch-policy probe at the end of each case doubles the lane
# blocks (one element slot per cell instead of two) and moves p=1 from 0.126 to
# 0.123 ms — nothing. No policy measured reaches the acceptance band at either
# order, so the verdict is a property of the route and not of the launch.
#
# `model GB/s` is each arm's own DRAM-REALISTIC byte count over its min time: the
# stored values plus the index stream each arm actually reads, the `u` gather and
# the `y` write counted once. The lane mapping re-reads a block column once per
# row and those re-reads are served by L1/L2, so counting them would report an
# effective rate over cached traffic rather than a memory rate.
#
# `MatrixFreeAction[Recompute]` — FO's lowest storage rung — HAS NO ROUTE for this
# element, on any device, and the arm prints the rejection instead of a number.
# That rung is reached through `apply_element_action!`, which is an element's
# promise that it evaluates `Kₑ·uₑ` without forming `Kₑ`; an interior-penalty facet
# system has no such factored form to promise, so `SIPGDiffusionIntegrator`
# declares only the matrix and residual kernels. The condensed block-row store is
# what gives this element a matrix-free device action at all.
#
# A PRECISION CONSTRAINT WORTH KNOWING, met here rather than worked around: a
# two-sided element cannot run on a `Float32` FACET quadrature rule at all, because
# Ferrite 1.7 defines `transform_interface_points!` — the here → there reference
# point mapping every `InterfaceValues` reinit performs — for `Vec{dim, Float64}`
# points only. The facet rule below is therefore `Float64` while the volume rule,
# and with it the element's value type, the block-row store, the assembled matrix
# and every timed arm, stay `Float32`.
#
# CUDA is required: the arms are all device arms.
using FerriteOperators, FerriteOperatorsExampleElements
using LinearAlgebra
using Printf
using SparseArrays
import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index, @Const
import FerriteOperators: launch_geometry, lane_launch_geometry

const CUDA_AVAILABLE = try
    @eval using CUDA
    @eval import CUDA: CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR
    CUDA.functional()
catch err
    @info "CUDA unavailable; this benchmark has only device arms" err
    false
end

const Tv = Float32
const Ti = Int32
const WARMUP_SECONDS = 1.5
const PASSES = 3
const D_DIFFUSION = 1.3
const PENALTY_η = 4.0

# hex p=1: 25^3 cells x 8 dofs = 125 000; hex p=2: 16^3 x 27 = 110 592.
const CASES = length(ARGS) ≥ 2 ? ((1, parse(Int, ARGS[1])), (2, parse(Int, ARGS[2]))) :
    ((1, 25), (2, 16))

# The FACET rule is `Float64` where every other type here is `Float32`: Ferrite
# 1.7's `transform_interface_points!` — which maps the here-side quadrature points
# onto the there-side reference facet on every `InterfaceValues` reinit — has
# methods for `Vec{dim, Float64}` points alone. The element takes its VALUE type
# from the VOLUME rule, so the element matrix, the block-row store, the assembled
# matrix and the whole action stay `Float32`; only the reference-facet point
# mapping runs in double.
dg_integrator(order) = SIPGDiffusionIntegrator(
    D_DIFFUSION, PENALTY_η, QuadratureRuleCollection(Tv, order + 1),
    FerriteOperators.FacetQuadratureRuleCollection(Float64, order + 1), :u)

# The house testbed, over a discontinuous space: interior nodes pushed off the
# lattice, so every facet carries its own normal and measure and the penalty's
# `|F| / min(|K⁻|, |K⁺|)` is not a constant the arithmetic could fold away.
function distorted_dg_testbed(order, n; distortion = 0.15f0)
    grid = generate_grid(Hexahedron, (n, n, n),
                         Vec{3}((-1.0f0, -1.0f0, -1.0f0)), Vec{3}((1.0f0, 1.0f0, 1.0f0)))
    h = 2.0f0 / n
    nodes = [Ferrite.Node(Vec{3, Tv}(ntuple(d -> node.x[d] +
                (all(abs.(node.x) .< 1 - 1.0f-4) ? distortion * h * Tv(sin(2.7 * i + 1.3 * d)) : 0.0f0), 3)))
             for (i, node) in enumerate(Ferrite.getnodes(grid))]
    dh = DofHandler(Grid(Ferrite.getcells(grid), nodes))
    add!(dh, :u, DiscontinuousLagrange{RefHexahedron, order}())
    close!(dh)
    return dh
end

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

# Min-of-N wall time of one action over `PASSES` independent rounds. Each round
# BURNS the arm for `WARMUP_SECONDS` before sampling it, and the outer minimum
# discards a round that still caught a cold device rather than averaging it in.
#
# The clock stamp is taken MID-BURN, from a second thread, and not after the arm
# the way `matrix_free_action.jl` takes it: spawning `nvidia-smi` costs a fraction
# of a second, and a device that has gone idle in the meantime reports its idle
# clocks — which reads as a cold measurement of an arm that was in fact hot.
# Under load is the only instant at which the number answers the question asked.
function measure(action!; samples = 20, passes = PASSES, warmup = WARMUP_SECONDS)
    best = Inf
    stamp = nothing
    for pass in 1:passes
        deadline = time() + warmup
        probe = nothing
        action!()
        while time() < deadline
            action!()
            if pass == passes && probe === nothing && time() > deadline - warmup / 2
                probe = Threads.@spawn clocks()
            end
        end
        probe === nothing || (stamp = fetch(probe))
        best = min(best, minimum((@elapsed action!()) for _ in 1:samples))
    end
    return best, stamp
end

####################################
## Validation, before any timing
####################################

# Every arm reads a store built by its OWN route — cuSPARSE reads the assembled
# matrix, the block-row arms read a store condensed from the same element kernels
# facet by facet — so agreement is a real check on both routes and not a tautology.
function _check_action!(label, yd, run!, reference; rtol = 1.0f-4)
    fill!(yd, 0)
    run!()
    y = Array(yd)
    δ = maximum(abs, y .- reference) / max(maximum(abs, reference), eps(Tv))
    δ ≤ rtol || error("benchmark arm `$label` disagrees with the assembled reference by $δ " *
                      "(rtol $rtol); the published number for this arm would be meaningless.")
    return δ
end

# Two runs of the same action on the same input, compared EXACTLY. Every arm is
# observed; the block-row arms are REQUIRED to pass, because one colour and a
# plain `+=` leave nothing that could reorder a sum, and that determinism is the
# claim the shipped route makes over an atomic scatter. cuSPARSE promises no such
# thing for either layout, so its arms only report what they did.
function _bitwise_repeats(yd, run!)
    fill!(yd, 0); run!(); first_run = Array(yd)
    fill!(yd, 0); run!(); second_run = Array(yd)
    return first_run == second_run
end

function _require_bitwise(label, yd, run!)
    _bitwise_repeats(yd, run!) || error(
        "benchmark arm `$label` is NOT bitwise repeatable: two runs of the same action on the " *
        "same input differ. The one-colour partition promises a conflict-free scatter, so this " *
        "is a defect in the action and not a property of the measurement.")
    return true
end

# A stream ceiling computes no action, so all that can be asserted is that its
# loads were not eliminated.
function _check_ceiling(label, sink)
    s = Array(sink)
    (all(isfinite, s) && any(!iszero, s)) || error(
        "benchmark stream ceiling `$label` reduced to $(all(isfinite, s) ? "all zeros" : "a non-finite value"); " *
        "its loads were eliminated, so the number it would publish is not the access pattern's rate.")
    return nothing
end

####################################
## The pure-stream ceiling of the block-row access pattern
####################################

# `_block_row_dot`'s loads with the product, the `u` gather and the scatter
# removed: the diagonal block, one block per facet that HAS a neighbour, and the
# INDEX stream the gather window costs. That stream is whichever one the store
# actually reads — the `(slot, k)` table entry by entry where the layout forced
# one, and the single neighbour cell id per facet block where the window is
# derived arithmetically. `DERIVED` is a compile-time flag, so each arm compiles
# to its own kernel and neither pays for the other's branch. The stream folds
# into an INTEGER accumulator and is converted once per thread — a per-entry
# `Int`-to-`Float32` conversion made the equivalent arm in
# `matrix_free_action.jl` slower than the kernel it bounds.
@inline function _stream_window_block(windows, slot, f, ::Val{NB}, ::Val{false}) where {NB}
    s = 0
    offset = f * NB
    for j in 1:NB
        s += Int(@inbounds windows[slot, offset + j])
    end
    return s
end
@inline _stream_window_block(windows, slot, f, ::Val{NB}, ::Val{true}) where {NB} =
    f == 0 ? 0 : Int(@inbounds windows[slot, f])

@inline function _stream_block_row(K, neighbours, windows, slot, i, ::Val{NB}, ::Val{NF},
        derived::Val) where {NB, NF}
    row = zero(eltype(K))
    idx = _stream_window_block(windows, slot, 0, Val(NB), derived)
    for j in 1:NB
        @inbounds row += K[slot, 1, i, j]
    end
    for f in 1:NF
        (@inbounds neighbours[slot, f]) == 0 && continue
        idx += _stream_window_block(windows, slot, f, Val(NB), derived)
        for j in 1:NB
            @inbounds row += K[slot, 1 + f, i, j]
        end
    end
    return row, idx
end

# The LANE launch: one cell slot per grid-stride step, lane `l` taking rows
# `l:nlanes:NB`. The thread → (slot, lane) map is `_lane_position`'s, so this arm
# streams what the mapping's kernel streams.
@kernel function _stream_block_row_lane!(sink, @Const(K), @Const(neighbours), @Const(windows),
        @Const(slots), @Const(items), ::Val{NB}, ::Val{NF}, ::Val{NLANES}, ::Val{PER},
        derived::Val, n_slots, n_items) where {NB, NF, NLANES, PER}
    thread = @index(Global, Linear)
    group, local_index = divrem(Int(thread) - 1, NLANES * PER)
    lane, block = divrem(local_index, PER)
    first_slot = group * PER + block + 1
    acc = zero(eltype(K))
    idx = 0
    if first_slot ≤ n_slots
        for s in first_slot:n_slots:n_items
            slot = Int(@inbounds slots[@inbounds items[s]])
            for i in (lane + 1):NLANES:NB
                row, window = _stream_block_row(K, neighbours, windows, slot, i, Val(NB), Val(NF), derived)
                acc += row
                idx += window
            end
        end
    end
    @inbounds sink[thread] = acc + eltype(K)(idx % 2)
end

# The WORKER launch: one grid-stride worker per cell, all `NB` rows of it.
@kernel function _stream_block_row_worker!(sink, @Const(K), @Const(neighbours), @Const(windows),
        @Const(slots), @Const(items), ::Val{NB}, ::Val{NF}, derived::Val, n_items) where {NB, NF}
    worker = @index(Global, Linear)
    stride = prod(KA.@ndrange())
    acc = zero(eltype(K))
    idx = 0
    for s in worker:stride:n_items
        slot = Int(@inbounds slots[@inbounds items[s]])
        for i in 1:NB
            row, window = _stream_block_row(K, neighbours, windows, slot, i, Val(NB), Val(NF), derived)
            acc += row
            idx += window
        end
    end
    @inbounds sink[worker] = acc + eltype(K)(idx % 2)
end

# Where a ceiling sits BELOW its own action it bounds nothing, and the row says so
# rather than leaving the caveat in a source comment. It happens: the ceiling folds
# its index stream into ONE integer accumulator per lane, a longer serial chain
# than the per-row float accumulators of the action it bounds.
const CEILING_ARMS = Dict("ceiling: block-row stream [lanes]"  => "block-row [LanesPerElement]",
                          "ceiling: block-row stream [worker]" => "block-row [WorkerPerElement]")

function _ceiling_note(name, time, results)
    arm = get(CEILING_ARMS, name, nothing)
    arm === nothing && return ""
    index = findfirst(r -> first(r) == arm, results)
    index === nothing && return ""
    action = results[index].second
    return time > action ?
        @sprintf("  <- NOT BOUNDING: above its action (%.3f ms)", 1.0e3 * action) : ""
end

####################################
## Byte models
####################################

# The assembled SpMV: one value and one row index per stored entry, the column
# pointers, and `u`/`y` once each.
_spmv_bytes(A, n_dofs) = nnz(A) * (sizeof(Tv) + sizeof(Ti)) +
    (size(A, 2) + 1) * sizeof(Ti) + 2 * n_dofs * sizeof(Tv)

# The block-row action: the NON-ZERO blocks (a boundary facet's block is stored
# but skipped), the index stream the gather window costs, the neighbour table the
# action reads to skip its boundary blocks, the slot map, and `u`/`y` once each.
_block_row_bytes(n_blocks, nb, ncells, nf, windows, n_dofs) =
    n_blocks * nb * nb * sizeof(Tv) + _window_bytes(windows, ncells, nb, nf) +
    ncells * nf * sizeof(Int32) + ncells * sizeof(Int32) + 2 * n_dofs * sizeof(Tv)

# A materialized table costs one entry per gather dof. The arithmetic window
# costs the cursor's OWN copy of the neighbour table instead — the cache's and
# the cursor's are adapted onto the device separately, so the kernel streams that
# table twice and this model says so.
_window_bytes(windows::AbstractMatrix, ncells, nb, nf) =
    ncells * (1 + nf) * nb * sizeof(eltype(windows))
_window_bytes(::Any, ncells, nb, nf) = ncells * nf * sizeof(Int32)

# What the ceiling arms stream in place of the action's window reads, and the
# compile-time flag that tells the two apart.
_ceiling_windows(windows::AbstractMatrix) = (windows, Val(false))
_ceiling_windows(windows) = (windows.neighbours, Val(true))

_rate(bytes, time) = bytes / time / 1.0e9

####################################
## The run
####################################

# The launch policy is `matrix_free_action.jl`'s for every one of its CUDA arms,
# fixed before any number was seen. It is a user-facing knob rather than a
# property of the route, so the probe at the end of each case reports what two
# other settings of it do — beside the headline, never in place of it.
const HOUSE_POLICY = (items_per_worker = 2, max_workgroup_size = 256)

_device(policy = HOUSE_POLICY) = KernelAbstractionsDevice(
    CUDABackend(); value_type = Tv, index_type = Ti,
    items_per_worker = policy.items_per_worker, max_workgroup_size = policy.max_workgroup_size)

# The store the block-row kernels read, reached through the engine's own device
# cache so the ceiling streams the very bytes the headline arm streamed.
_block_row_store(op) = let sc = first(get_subdomain_caches(op))
    FerriteOperators._block_row_cache(FerriteOperators._alternate_fill_element(sc.device_cache))
end

function run_case(order, n)
    dh = distorted_dg_testbed(order, n)
    integrator = dg_integrator(order)
    grid = Ferrite.get_grid(dh)
    ncells, n_dofs = getncells(grid), ndofs(dh)
    nb = ndofs(dh) ÷ ncells
    nf = Ferrite.nfacets(getcells(grid, 1))

    # The reference: the SAME form assembled on the HOST, then handed to the device
    # as CSC and as CSR. Host, because the assembling sweep rides the element's
    # two-sided facet traversal and reads `InterfaceValues` that a real GPU does not
    # have — which is why the shipped element fills its block-row store from a host
    # mirror too, and why the CSR arm is a conversion of this matrix rather than a
    # second assembly. Both are setup costs and neither is timed.
    #
    # The interior-facet coupling is not in the `DofHandler`'s cell pattern, so the
    # specification declares it.
    spec = StandardOperatorSpecification(; sparsity_entries = interior_facet_entries!)
    A = let op = setup_operator(AssemblyStrategy(SequentialCPUDevice{Tv, Ti}();
                                                 form = FullAssembly(spec)), integrator, dh)
        update_operator!(op, nothing)
        op.A
    end
    u = Tv[sin(Tv(4.9) * i + Tv(2.1)) for i in 1:n_dofs]
    reference = A * u

    @printf("\nhex p=%d: %d cells, %d dofs, Nb=%d, %d stored entries\n",
            order, ncells, n_dofs, nb, nnz(A))
    assembled_bytes = _spmv_bytes(A, n_dofs)
    @printf("  assembled: %.1f MB on device (%d B per dof)\n",
            assembled_bytes / 2^20, assembled_bytes ÷ n_dofs)

    results = Pair{String, Float64}[]
    clockstamps = Dict{String, Any}()
    errors = Dict{String, Any}()
    bytes = Dict{String, Float64}()
    bitwise = Dict{String, Bool}()

    ud = CuVector(u)
    yd = CUDA.zeros(Tv, n_dofs)

    # Every arm is built, measured and RELEASED before the next allocates: an arm
    # timed on an allocator the previous arms fragmented measures the allocator.
    function record!(name, run!; byte_count = nothing, require_bitwise = false)
        errors[name] = _check_action!(name, yd, run!, reference)
        bitwise[name] = require_bitwise ? _require_bitwise(name, yd, run!) : _bitwise_repeats(yd, run!)
        time, stamp = measure(run!)
        push!(results, name => time)
        clockstamps[name] = stamp
        byte_count === nothing || (bytes[name] = byte_count)
        return nothing
    end

    for (name, T) in ("assembled CSC SpMV" => CuSparseMatrixCSC, "assembled CSR SpMV" => CuSparseMatrixCSR)
        Ad = T(A)
        record!(name, () -> (mul!(yd, Ad, ud); CUDA.synchronize()); byte_count = assembled_bytes)
        Ad = nothing
        GC.gc(); CUDA.reclaim()
    end

    # THE HEADLINE and its contrast: the same condensed store, the same one-colour
    # partition, two element mappings.
    store = nothing
    for (name, mapping) in ("block-row [LanesPerElement]" => LanesPerElement(),
                            "block-row [WorkerPerElement]" => WorkerPerElement())
        op = setup_operator(AssemblyStrategy(MatrixFreeAction(; element_mapping = mapping,
                                                              storage = BlockRowAssembly()),
                                             ColoredScheduling(), _device()), integrator, dh)
        # One colour of every cell — the scatter-disjointness promise, not a
        # colouring algorithm's result.
        partition = first(get_subdomain_caches(op)).partition
        (length(partition) == 1 && length(only(partition)) == ncells) || error(
            "the block-row action resolved $(length(partition)) chunks over $ncells cells; " *
            "the bitwise-repeat claim rests on the ONE-colour partition.")
        if store === nothing
            store = _block_row_store(op)
            n_blocks = ncells + count(!iszero, Array(store.neighbours))
            @printf("  block-row: %.1f MB of blocks (%d of %d stored non-zero), %d B per dof, %s window\n",
                    length(store.K) * sizeof(Tv) / 2^20, n_blocks, ncells * (1 + nf),
                    _block_row_bytes(n_blocks, nb, ncells, nf, store.windows, n_dofs) ÷ n_dofs,
                    store.windows isa AbstractMatrix ?
                        "materialized $(eltype(store.windows))" : "arithmetic")
        end
        n_blocks = ncells + count(!iszero, Array(store.neighbours))
        record!(name, () -> (mul!(yd, op, ud); CUDA.synchronize()),
                byte_count = _block_row_bytes(n_blocks, nb, ncells, nf, store.windows, n_dofs),
                require_bitwise = true)
        op = nothing
        GC.gc(); CUDA.reclaim()
    end

    # FO's shipped ladder over the element's OWN facet family, for the record.
    # `Recompute()` needs an `apply_element_action!` — the element's promise that
    # it can evaluate `Kₑ·uₑ` WITHOUT forming `Kₑ` — and an interior-penalty facet
    # system has no such factored form to promise. The rejection below is that
    # declaration missing, not a device limitation.
    try
        op = setup_operator(AssemblyStrategy(MatrixFreeAction(; element_mapping = WorkerPerElement(),
                                                              storage = Recompute()),
                                             SequentialScheduling(), _device()), integrator, dh)
        record!("MatrixFreeAction[Recompute]", () -> (mul!(yd, op, ud); CUDA.synchronize()))
        op = nothing
        GC.gc(); CUDA.reclaim()
    catch err
        println("  MatrixFreeAction[Recompute] over the facet items is not available:")
        println("    ", first(split(sprint(showerror, err), '\n')))
        GC.gc(); CUDA.reclaim()
    end

    # The ceilings, over the headline arm's own store.
    let backend = CUDABackend(), device = _device()
        K, neighbours, slots = store.K, store.neighbours, store.slots
        windows, derived = _ceiling_windows(store.windows)
        items = CuVector{Ti}(1:ncells)
        n_blocks = ncells + count(!iszero, Array(neighbours))
        ceiling_bytes = _block_row_bytes(n_blocks, nb, ncells, nf,
                                         store.windows, n_dofs) - 2 * n_dofs * sizeof(Tv)

        nlanes = min(nb, device.max_workgroup_size)
        lane_wg, lane_blocks, n_slots = lane_launch_geometry(device, nlanes, ncells)
        sink_lane = CUDA.zeros(Tv, lane_wg * lane_blocks)
        lane_run!() = (_stream_block_row_lane!(backend, lane_wg)(
                           sink_lane, K, neighbours, windows, slots, items, Val(nb), Val(nf),
                           Val(nlanes), Val(lane_wg ÷ nlanes), derived, n_slots, ncells;
                           ndrange = lane_wg * lane_blocks);
                       KA.synchronize(backend))
        fill!(sink_lane, 0); lane_run!(); _check_ceiling("ceiling [lanes]", sink_lane)
        time, stamp = measure(lane_run!)
        push!(results, "ceiling: block-row stream [lanes]" => time)
        clockstamps["ceiling: block-row stream [lanes]"] = stamp
        bytes["ceiling: block-row stream [lanes]"] = ceiling_bytes + length(sink_lane) * sizeof(Tv)

        workgroup, blocks = launch_geometry(device, ncells)
        sink_worker = CUDA.zeros(Tv, workgroup * blocks)
        worker_run!() = (_stream_block_row_worker!(backend, workgroup)(
                             sink_worker, K, neighbours, windows, slots, items, Val(nb), Val(nf),
                             derived, ncells;
                             ndrange = workgroup * blocks);
                         KA.synchronize(backend))
        fill!(sink_worker, 0); worker_run!(); _check_ceiling("ceiling [worker]", sink_worker)
        time, stamp = measure(worker_run!)
        push!(results, "ceiling: block-row stream [worker]" => time)
        clockstamps["ceiling: block-row stream [worker]"] = stamp
        bytes["ceiling: block-row stream [worker]"] = ceiling_bytes + length(sink_worker) * sizeof(Tv)
    end

    baseline = results[findfirst(r -> first(r) == "assembled CSC SpMV", results)].second
    headline = results[findfirst(r -> first(r) == "block-row [LanesPerElement]", results)].second
    @printf("  %-36s %12s %10s %12s %10s %9s\n",
            "arm", "min time", "vs CSC", "model GB/s", "bitwise", "rel err")
    for (name, time) in results
        @printf("  %-36s %10.3f ms %9.2fx %11s %10s %9s%s%s\n", name, 1.0e3 * time, baseline / time,
                haskey(bytes, name) ? @sprintf("%.0f", _rate(bytes[name], time)) : "-",
                haskey(bitwise, name) ? (bitwise[name] ? "yes" : "no") : "-",
                haskey(errors, name) ? @sprintf("%.1e", errors[name]) : "-",
                _clock_note(get(clockstamps, name, nothing)),
                _ceiling_note(name, time, results))
    end

    band = order == 1 ? 1.30 : 1.00
    @printf("  headline (block-row lanes) %.2fx the assembled CSC SpMV — band >= %.2fx: %s\n",
            baseline / headline, band, baseline / headline ≥ band ? "CONFIRMED" : "MISSED")

    # ATTRIBUTION, NOT THE HEADLINE. The lane block is one ELEMENT SLOT, and the
    # policy above gives a slot two cells, so the launch covers the grid with half
    # the lane blocks a slot-per-cell launch would. The prototype this round is
    # measured against launched one block per cell. Whether that accounts for the
    # distance from the ceiling is a question the maintainer needs answered before
    # deciding what to change, so it is answered here — and reported as its own
    # line, with the verdict above computed from the house policy alone.
    println("  launch-policy sensitivity of the headline arm (attribution only):")
    for (label, policy) in ("house (2 items/worker, group 256)" => HOUSE_POLICY,
                            "slot per cell (1, 256)" => (items_per_worker = 1, max_workgroup_size = 256),
                            "library default (2, 64)" => (items_per_worker = 2, max_workgroup_size = 64))
        device = _device(policy)
        op = setup_operator(AssemblyStrategy(MatrixFreeAction(; element_mapping = LanesPerElement(),
                                                              storage = BlockRowAssembly()),
                                             ColoredScheduling(), device), integrator, dh)
        run!() = (mul!(yd, op, ud); CUDA.synchronize())
        _check_action!("policy $label", yd, run!, reference)
        lane_wg, lane_blocks, _ = lane_launch_geometry(device, min(nb, policy.max_workgroup_size), ncells)
        time, stamp = measure(run!)
        @printf("  %-36s %10.3f ms %9.2fx %7d threads%s\n",
                label, 1.0e3 * time, baseline / time, lane_wg * lane_blocks, _clock_note(stamp))
        op = nothing
        GC.gc(); CUDA.reclaim()
    end
    return nothing
end

function main()
    CUDA_AVAILABLE || error("benchmarks/dg_action.jl has only device arms; CUDA is not functional here.")
    @printf("SIPG DG action vs assembled SpMV, %s/%s, device %s\n", Tv, Ti, CUDA.name(CUDA.device()))
    for (order, n) in CASES
        run_case(order, n)
    end
    println("\n`vs CSC` is relative to the assembled CSC SpMV of the same form. `model GB/s` is ",
            "each arm's DRAM-realistic byte count over its min time, so the arms are comparable; ",
            "the ceilings bound an ACCESS PATTERN, not the memory system.")
    return nothing
end

main()
