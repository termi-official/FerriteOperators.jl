# G3 — `BlockRowAssembly()`: the condensed block-row store and the
# cell-with-neighbours item. The fill source is the two-sided demo family of
# `custom_iterator_family.jl` (the same one `test_custom_iterator.jl` and
# `test/gpu/` run), so the store is filled through the element entry point a
# downstream two-sided element already implements and no fixture of its own.
#
# The reference is that family's closed-form global matrix: the action of a
# store condensed from pair items must equal the assembled operator's, item
# shapes notwithstanding.

include(joinpath(@__DIR__, "custom_iterator_family.jl"))

using FerriteOperatorsExampleElements
using Test
using LinearAlgebra
import FerriteOperators: iterator_dofs, iterator_scatter_address
# FerriteKAExt — the device handler, `distribute_to_workers` and every `Adapt`
# rule the device kernel builds on — is triggered by these four together.
import Adapt, GPUArrays, GPUArraysCore
import KernelAbstractions as KA

const BLOCK_ROW_ρ = 1.7

block_row_testbed() = jump_testbed((4, 3))

block_row_probe(dh) = Float64[sin(0.7i) + 0.2cos(1.3i) for i in 1:ndofs(dh)]

sequential_arm(storage; scheduling = SequentialScheduling()) =
    AssemblyStrategy(MatrixFreeAction(; storage), scheduling, SequentialCPUDevice())

ka_arm(mapping, storage; scheduling = SequentialScheduling()) = AssemblyStrategy(
    MatrixFreeAction(; element_mapping = mapping, storage), scheduling,
    KernelAbstractionsDevice(KA.CPU(); value_type = Float64, index_type = Int,
                             items_per_worker = 2, max_workgroup_size = 8))

block_row_arms(storage; scheduling = SequentialScheduling()) = (
    "SequentialCPUDevice"          => sequential_arm(storage; scheduling),
    "KA.CPU WorkerPerElement"      => ka_arm(WorkerPerElement(), storage; scheduling),
    "KA.CPU LanesPerElement"       => ka_arm(LanesPerElement(), storage; scheduling),
)

action(op, u) = (y = zeros(length(u)); mul!(y, op, u); y)

@testset "G3: BlockRowAssembly()" begin
    tb = block_row_testbed()
    n  = ndofs_per_cell(first(tb.dh.subdofhandlers))
    u  = block_row_probe(tb.dh)
    A  = reference_matrix(tb.dh, tb.prs, n)
    reference = A * u

    @testset "the action matches the assembled reference — $label, $sched" for
            (sched, scheduling) in ("SequentialScheduling" => SequentialScheduling(),
                                    "ColoredScheduling" => ColoredScheduling()),
            (label, strategy) in block_row_arms(BlockRowAssembly(); scheduling)

        op = setup_operator(strategy, tb.integrator, tb.dh)
        @test action(op, u) ≈ reference rtol = 1.0e-12
    end

    @testset "the store is slot-leading, facet-blocked and topology-derived" begin
        op = setup_operator(sequential_arm(BlockRowAssembly()), tb.integrator, tb.dh)
        cache = first(get_subdomain_caches(op)).domain.element
        # 12 cells, 4 facets each, `n`-square blocks; `slot` LEADS and is
        # therefore stride-1, which is what makes consecutive workers read
        # adjacent addresses.
        @test size(cache.K) == (getncells(tb.grid), 5, n, n)
        # The grid's boundary facets carry no neighbour, and only those.
        @test count(iszero, cache.neighbours) == 2 * (4 + 3)
    end

    @testset "the fill rides the element's two-sided items, the action the cells" begin
        Threads.atomic_xchg!(VISITED, 0)
        op = setup_operator(sequential_arm(BlockRowAssembly()), tb.integrator, tb.dh)
        # 9 horizontal pairs, not 12 cells: the fill visited the element's own
        # item list while the action's partition is one colour of every cell.
        @test VISITED[] == length(tb.prs)
        @test length(first(get_subdomain_caches(op)).partition) == 1
        @test length(first(get_subdomain_caches(op)).partition[1]) == getncells(tb.grid)

        # A refill overwrites rather than accumulating.
        update_operator!(op, nothing)
        @test action(op, u) ≈ reference rtol = 1.0e-12
    end

    @testset "additional_iteration_kinds is BlockRowAssemblyCache's OWN declaration" begin
        op = setup_operator(sequential_arm(BlockRowAssembly()), tb.integrator, tb.dh)
        cache = first(get_subdomain_caches(op)).domain.element
        @test FerriteOperators.additional_iteration_kinds(MatrixFreeAction(), cache) == (QuadratureDataKind(),)
        # Without `BlockRowAssemblyCache`'s own method, the decorator's blanket
        # forward would hand back the WRAPPED element's answer instead — `()`,
        # since the demo pair family declares nothing extra of its own.
        @test FerriteOperators.additional_iteration_kinds(MatrixFreeAction(), cache.inner) == ()
    end

    @testset "the fill runs engine-driven on a CPU-resident device" begin
        # `KA.CPU()` shares `AbstractGPUDevice`'s launch machinery but is
        # HOST-resident, so — like `SequentialCPUDevice` — its fill prefers the
        # engine sweep over the host mirror (`update_operator!`'s docstring).
        # `SequentialScheduling` on the ACTION is deliberate: the fill's own
        # partition (`BlockRowFillItems`) is ALWAYS coloured, independent of
        # what the action was set up with.
        strategy = ka_arm(WorkerPerElement(), BlockRowAssembly(); scheduling = SequentialScheduling())
        op = setup_operator(strategy, tb.integrator, tb.dh)
        sc = first(get_subdomain_caches(op))

        # It resolved a traversal of its own, and that traversal's item count
        # is the PAIR count — not the action's cell count.
        @test sc.alternates !== nothing
        _, alt_partition = FerriteOperators._kind_caches(sc, QuadratureDataKind())
        @test sum(length, alt_partition) == length(tb.prs)
        _, primary_partition = FerriteOperators._kind_caches(sc, MatrixFreeActionKind())
        @test primary_partition === sc.partition
        @test sum(length, primary_partition) == getncells(tb.grid)

        # Same store S2's host-path fill (`fill_block_rows!`) builds, up to
        # summation order: the engine sweep groups pair items by COLOUR
        # (race-safety on a multi-worker device), the host mirror walks them in
        # LINEAR pair order, so a cell touched by more than one item accumulates
        # in a different order — `≈`, not `==`.
        cache = sc.domain.element
        reference_cache = deepcopy(cache)
        fill!(reference_cache.K, 0.0)
        FerriteOperators.fill_block_rows!(reference_cache, nothing, nothing)
        @test cache.K ≈ reference_cache.K rtol = 1.0e-12

        @test action(op, u) ≈ reference rtol = 1.0e-12
    end

    @testset "the one colour is disjoint on SCATTER dofs and not on gather dofs" begin
        op = setup_operator(sequential_arm(BlockRowAssembly(); scheduling = ColoredScheduling()),
                            tb.integrator, tb.dh)
        sc = first(get_subdomain_caches(op))
        it = assembly_iterator(MatrixFreeActionKind(), sc.domain.element, sc.domain.sdh)
        scattered = [Set(iterator_scatter_address(position_iterator(it, c, Ferrite.UpdateFlags())))
                     for c in only(sc.partition)]
        gathered = [Set(iterator_dofs(position_iterator(it, c, Ferrite.UpdateFlags())))
                    for c in only(sc.partition)]
        @test all(isdisjoint(scattered[i], scattered[j])
                  for i in eachindex(scattered) for j in eachindex(scattered) if i != j)
        # Deliberately NOT disjoint: neighbours share the gather window, which is
        # the whole point of the item and is read-only.
        @test any(!isdisjoint(gathered[i], gathered[j])
                  for i in eachindex(gathered) for j in eachindex(gathered) if i != j)
    end

    @testset "the colored action repeats bit for bit — $label" for
            (label, strategy) in block_row_arms(BlockRowAssembly(); scheduling = ColoredScheduling())

        op = setup_operator(strategy, tb.integrator, tb.dh)
        y1, y2 = zeros(ndofs(tb.dh)), zeros(ndofs(tb.dh))
        mul!(y1, op, u)
        mul!(y2, op, u)
        @test y1 == y2
    end

    @testset "the sequential arm is allocation-free per mul!" begin
        op = setup_operator(sequential_arm(BlockRowAssembly()), tb.integrator, tb.dh)
        y = zeros(ndofs(tb.dh))
        measure(op, y, u) = (mul!(y, op, u); mul!(y, op, u); @allocated mul!(y, op, u))
        @test measure(op, y, u) == 0
    end

    @testset "premultiply_inverse_mass fuses M⁻¹A into the store" begin
        mass = SimpleBilinearMassIntegrator(BLOCK_ROW_ρ, QuadratureRuleCollection(2), :u)
        M = let op = setup_operator(AssemblyStrategy(SequentialCPUDevice()), mass, tb.dh)
            update_operator!(op, nothing)
            op.A
        end
        fused = M \ reference

        @testset "$label" for (label, strategy) in
                block_row_arms(BlockRowAssembly(; premultiply_inverse_mass = mass))
            op = setup_operator(strategy, tb.integrator, tb.dh)
            @test action(op, u) ≈ fused rtol = 1.0e-10
            # The unfused store is a different operator — the fusion is not a
            # no-op that the tolerance above would hide.
            @test !isapprox(action(op, u), reference; rtol = 1.0e-3)
        end

        @testset "a refill re-fuses rather than fusing twice" begin
            op = setup_operator(sequential_arm(
                BlockRowAssembly(; premultiply_inverse_mass = mass)), tb.integrator, tb.dh)
            update_operator!(op, nothing)
            @test action(op, u) ≈ fused rtol = 1.0e-10
        end
    end

    @testset "ElementAssembly() over a multi-cell item family is rejected" begin
        err = @test_throws ArgumentError setup_operator(
            sequential_arm(ElementAssembly()), tb.integrator, tb.dh)
        msg = err.value.msg
        @test occursin("BlockRowAssembly()", msg)
        @test occursin("cellid", msg)
        @test occursin("JumpPenaltyCache", msg)
    end

    @testset "CooperativeElement over BlockRowAssembly is rejected" begin
        strategy = AssemblyStrategy(
            MatrixFreeAction(; element_mapping = CooperativeElement(), storage = BlockRowAssembly()),
            SequentialScheduling(), KernelAbstractionsDevice(KA.CPU()))
        err = @test_throws ArgumentError setup_operator(strategy, tb.integrator, tb.dh)
        @test occursin("BlockRowAssembly", err.value.msg)
        @test occursin("lattice", err.value.msg)
    end

    @testset "a cell-square element cannot elect BlockRowAssembly" begin
        dh = let grid = generate_grid(Quadrilateral, (2, 2))
            handler = DofHandler(grid)
            add!(handler, :u, Lagrange{RefQuadrilateral, 1}())
            close!(handler)
        end
        err = @test_throws ArgumentError setup_operator(
            sequential_arm(BlockRowAssembly()),
            SimpleBilinearMassIntegrator(BLOCK_ROW_ρ, QuadratureRuleCollection(2), :u), dh)
        @test occursin("TWO-SIDED", err.value.msg)
        @test occursin("ElementAssembly()", err.value.msg)
    end
end
