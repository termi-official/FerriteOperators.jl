# A work-item iterator written entirely OUTSIDE the package: a two-sided
# interface traversal whose item is a PAIR of neighbouring cells, positioned by
# an iterator of its own and enumerated by a provider of its own. Nothing in
# `src/` knows this family exists; it reaches the engine through the two
# protocol seams and the three required accessors and nothing else.
#
# PUBLIC-SURFACE COUNT. Unexported FerriteOperators names used: 0. The `import`
# below exists only because Julia requires it to ADD METHODS. Two of Ferrite's
# own unexported names are used (`Ferrite.get_grid`, `Ferrite.add_entry!`).
#
# What the family costs its author, counted on this file:
#   2 protocol methods   — `assembly_iterator`, `item_provider`
#   3 required accessors — `Ferrite.cellid`, `iterator_dofs`, `iterator_handler`
#   2 partition methods  — `compute_partition` for the two scheduling policies
#   1 duplication method — `duplicate_for_device`, for the threaded CPU route
#   1 declaration        — `sparsity_entries`, the entries a two-sided item
#                          couples and the cell pattern does not carry
#   1 family marker      — `JumpPairFamily`, the type the registration dispatches on
#   2 registration methods — `item_families`, `setup_family_caches`
# everything else here is the ELEMENT axis and the assertions.
#
# REGISTRATION is the last three, and it is optional: a family whose items ride
# the cell route needs only the nine above. It is written here because
# registering is what proves the seam.
#
# THE MATRIX-FREE ARM (optional) costs ONE more ELEMENT-axis method
# (`apply_element_action!`) and no protocol method: the action's own iterator
# default sits below the cache declarations, so the two above still win.
#
# THE DEVICE HALF (optional; `KernelAbstractionsDevice(KA.CPU())`) is a separate
# device-resident iterator (`DevicePairCursor`) plus 4 methods —
# `device_assembly_iterator`, `setup_device_instances` × 2, `device_worker_view`
# × 2 — and one `reinit_values!` overload for it. Not counted above: an author
# who never targets `KernelAbstractionsDevice` writes none of it.

using FerriteOperators
using Test
using SparseArrays
using LinearAlgebra
using Polyester
# FerriteKAExt — the device handler, `distribute_to_workers` and every `Adapt`
# rule the device kernel builds on — is triggered by these four together, not by
# KernelAbstractions alone. No CUDA: the device arm below targets
# `KernelAbstractionsDevice(KA.CPU())` only. `test/gpu/` runs the same family on
# CUDA and triggers the extension through `using CUDA` instead.
import Adapt, GPUArrays, GPUArraysCore
import KernelAbstractions as KA

# Types, protocol methods and the reference assembler — reused as-is by the
# CUDA arm in `test/gpu/`.
include(joinpath(@__DIR__, "custom_iterator_family.jl"))

@testset "custom item iterator (two-sided interface family)" begin
    tb = jump_testbed()
    n  = ndofs_per_cell(first(tb.dh.subdofhandlers))

    # Setup validation probes the RESOLVED iterator type. The cache annotates
    # `reinit_values!` against `PairCache` and nothing else, so a probe
    # hard-wired to `CellCache` would reject it by name here.
    @test !hasmethod(reinit_values!, Tuple{JumpPenaltyCache, Ferrite.CellCache})
    op = setup_operator(strategy_for(tb.spec, SequentialCPUDevice()), tb.integrator, tb.dh)
    element = first(get_subdomain_caches(op)).domain.element
    sdh     = first(tb.dh.subdofhandlers)
    @test assembly_iterator(nothing, element, sdh) isa PairCache
    @test hasmethod(reinit_values!, Tuple{JumpPenaltyCache, PairCache})

    @testset "A2 — the sweep visits the PROVIDER's items, not the cells" begin
        @test length(tb.prs) == 9
        @test getncells(tb.grid) == 12
        partition = first(get_subdomain_caches(op)).partition
        @test length(partition[1]) == 9
        sweep!(op)
        @test VISITED[] == 9
    end

    @testset "A3 — the assembled matrix equals the hand-built reference" begin
        A = Matrix(sweep!(op))
        @test A ≈ reference_matrix(tb.dh, tb.prs, n) rtol = 1.0e-12
        # Side-asymmetry: with α ≠ 1 the block is not invariant under swapping
        # the two sides, so a left/right mix-up would show here.
        @test !(A ≈ reference_matrix(tb.dh, [(r, l) for (l, r) in tb.prs], n))
    end

    @testset "A4 — the traversal really is two-sided" begin
        A = sweep!(op)
        cells = allocate_matrix(tb.dh)          # the CELL pattern alone
        d1, d2 = celldofs(tb.dh, 1), celldofs(tb.dh, 2)
        @test isempty(intersect(d1, d2))        # nothing derivable from one cell
        for i in d1, j in d2
            @test A[i, j] != 0                  # coupled by the pair item
            @test !(i in view(rowvals(cells), nzrange(cells, j)))  # in no cell's block
        end
    end

    @testset "A5 — the same family under a threaded, coloured device" begin
        seq = copy(sweep!(op))
        colored = setup_operator(
            strategy_for(tb.spec, PolyesterDevice(; min_items_per_worker = 1), ColoredScheduling()),
            tb.integrator, tb.dh)
        @test length(first(get_subdomain_caches(colored)).device_cache) > 1  # really per-worker
        par = copy(sweep!(colored))
        @test VISITED[] == 9
        @test Matrix(par) ≈ Matrix(seq) rtol = 1.0e-12
        # A repeat of the same configuration reproduces itself exactly; the
        # cross-device comparison above is `≈` only because the summation order
        # differs.
        @test sweep!(colored) == par
    end

    @testset "A7 — the device half, landed: KernelAbstractionsDevice(KA.CPU())" begin
        seq = copy(sweep!(op))
        device_op = setup_operator(
            strategy_for(tb.spec, KernelAbstractionsDevice(KA.CPU(); items_per_worker = 1),
                ColoredScheduling()),
            tb.integrator, tb.dh)
        # A low `items_per_worker` for the same reason the Polyester arm lowers
        # `min_items_per_worker`: a 9-item set at the default (2) would still
        # engage several workers, but this makes it explicit.
        @test size(first(get_subdomain_caches(device_op)).device_cache.Ke, 1) > 1
        dev = copy(sweep!(device_op))
        # `pairs` is dropped for a `KernelAbstractionsDevice` cache (the fix
        # for the CUDA arm below), so this sweep no longer increments VISITED;
        # the pair-item count is asserted from the partition instead.
        @test sum(length, first(get_subdomain_caches(device_op)).partition) == 9
        # Host-vs-device exactness, not a tolerance: the device sweep's dof
        # windows and element math are the same arithmetic as the host's.
        @test maximum(abs, Matrix(dev) .- Matrix(seq)) == 0.0
    end

    @testset "the recipe spelling resolves under MatrixFreeAction too" begin
        # Both protocol methods above are spelled the way `devdocs/design.md`
        # prescribes — kind OPEN, cache narrow — and `MatrixFreeActionKind` is
        # the one sweep kind the package declares a default iterator for. That
        # default sits BELOW the cache declarations
        # (`default_assembly_iterator`), so this family keeps its own iterators
        # under the action; a kind-narrow/cache-open default would tie instead.
        reference = reference_matrix(tb.dh, tb.prs, n)
        u = Float64[sin(3.1 * i) + 0.2cos(i) for i in 1:ndofs(tb.dh)]
        expected = reference * u
        sdh = first(tb.dh.subdofhandlers)

        for (label, device) in ("sequential CPU" => SequentialCPUDevice(),
                                "KA.CPU device" => KernelAbstractionsDevice(KA.CPU(); items_per_worker = 1))
            for storage in (Stored(), Recompute())
                op = setup_operator(
                    AssemblyStrategy(MatrixFreeAction(; storage), SequentialScheduling(), device),
                    tb.integrator, tb.dh)
                element = first(get_subdomain_caches(op)).domain.element
                # The declaration wins over the action's own default on BOTH
                # shapes of the seam: the host iterator and the device one.
                @test assembly_iterator(MatrixFreeActionKind(), element, sdh) isa PairCache
                @test device_assembly_iterator(MatrixFreeActionKind(), element, sdh, sdh) isa DevicePairCursor
                y = zeros(ndofs(tb.dh))
                Threads.atomic_xchg!(VISITED, 0)
                mul!(y, op, u)
                @test y ≈ expected rtol = 1.0e-12
                # The action swept the 9 PAIR items, not the 12 cells — the same
                # count assertion the assembling arms make.
                @test VISITED[] == 0        # the action kernel, not the matrix kernel
                @test length(first(get_subdomain_caches(op)).partition[1]) == 9
            end
        end
    end

    @testset "A6 — the colouring promise the provider makes, asserted directly" begin
        provider = item_provider(nothing, element, sdh)
        @test provider isa PairItems
        partition = compute_partition(ColoredScheduling(), provider)
        @test length(partition) > 1
        window(i) = Set{Int}(vcat(celldofs(tb.dh, tb.prs[i][1]), celldofs(tb.dh, tb.prs[i][2])))
        for color in partition, i in color, j in color
            i == j && continue
            @test isdisjoint(window(i), window(j))     # no shared SCATTER DOF
        end
        @test sort!(reduce(vcat, partition)) == collect(1:9)
    end

    @testset "the conventional accessor half is the iterator's own" begin
        pc = assembly_iterator(nothing, element, sdh)
        Ferrite.reinit!(pc, 5)
        l, r = tb.prs[5]
        @test cellid(pc) == l
        @test iterator_handler(pc) === sdh
        @test iterator_dofs(pc) == [celldofs(tb.dh, l); celldofs(tb.dh, r)]
        # Ferrite's assembler addresses a scatter through this window and needs
        # it duplicate free — the reason the demo's space is discontinuous.
        @test allunique(iterator_dofs(pc))
        @test pc.coords_left == getcoordinates(tb.grid, l)
        @test pc.coords_right == getcoordinates(tb.grid, r)
    end

    @testset "the registration seam: a family declared entirely from test code" begin
        # Every operator above was built through THIS file's
        # `setup_family_caches` method, reached by dispatch on the declared
        # marker and by nothing else.
        before = FAMILY_SETUPS[]
        fresh  = setup_operator(strategy_for(tb.spec, SequentialCPUDevice()), tb.integrator, tb.dh)
        @test FAMILY_SETUPS[] == before + 1
        @test item_families(tb.integrator, tb.dh) === (JumpPairFamily(),)
        # The cell family is NOT registered, so nothing swept the 12 cells: the
        # 9 pair items are the whole traversal.
        @test length(get_subdomain_caches(fresh)) == length(tb.dh.subdofhandlers)
        @test length(first(get_subdomain_caches(fresh)).partition[1]) == 9
        @test Matrix(sweep!(fresh)) ≈ reference_matrix(tb.dh, tb.prs, n) rtol = 1.0e-12
        # The three shipped families answer the SAME generic function this one
        # does — the dogfood claim, asserted from outside `src/`.
        for family in (CellFamily(), FacetItemFamily(), AlgebraicItemFamily(), JumpPairFamily())
            @test hasmethod(setup_family_caches, Tuple{typeof(family), Any, Any, Any, Any})
        end
    end

    @testset "the defaults are unchanged for every other family" begin
        @test item_provider(nothing, nothing, sdh) isa CellItems
        @test item_families(nothing, tb.dh) === (CellFamily(),)
    end
end
