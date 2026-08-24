using FerriteOperators
using Test
using SparseArrays
using Polyester

# Items with no mesh support need a DofHandler that can carry dofs outside the
# mesh, which is Ferrite's mesh-free algebraic variables.
if !isdefined(Ferrite, :AlgebraicVariable)
    @info "Skipping the algebraic-item tests: this Ferrite has no `AlgebraicVariable`, " *
          "so a DofHandler cannot carry dofs outside the mesh."
else

    include(joinpath(@__DIR__, "fixture_elements.jl"))

    sequential_strategy(spec) = AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), SequentialCPUDevice())

    # Declaration doubles for the setup-time validation, over a subdomain that
    # assembles nothing. `with_cache` decides whether the mandatory
    # `setup_algebraic_cache` exists at all.
    struct NoopAlgebraicCache end
    FerriteOperators.assemble_algebraic!(::ResidualRequest, ::NoopAlgebraicCache, args) = nothing
    FerriteOperators.duplicate_for_device(device, c::NoopAlgebraicCache) = c

    struct DeclaredAlgebraicItems{with_cache} <: AbstractNonlinearIntegrator
        items::Vector{Vector{Int}}
    end
    DeclaredAlgebraicItems(items; with_cache = true) = DeclaredAlgebraicItems{with_cache}(items)
    FerriteOperators.algebraic_items(m::DeclaredAlgebraicItems, dh) = m.items
    FerriteOperators.setup_element_cache(::DeclaredAlgebraicItems, ::SubDofHandler) =
        FerriteOperators.EmptyVolumetricElementCache()
    FerriteOperators.setup_algebraic_cache(::DeclaredAlgebraicItems{true}, dh) = NoopAlgebraicCache()

    # A cache that claims condensed internal state without an analytic
    # `Consistent` kernel — the combination the AD decorator serves through
    # `condensed_corrector`, which is field-space sized.
    struct CondensedProbeCache <: FerriteOperators.AbstractVolumetricElementCache end
    FerriteOperators.has_internal_state(::Type{CondensedProbeCache}) = true
    FerriteOperators.assemble_cell!(::ResidualRequest, ::CondensedProbeCache, args) = nothing
    FerriteOperators.reinit_values!(::CondensedProbeCache, cell) = nothing
    Ferrite.getnquadpoints(::CondensedProbeCache) = 2

    # A functional BOTH item families contribute to, under its own tag so the
    # reservoir fixture's `nothing`-contributing tag stays what it is.
    FerriteOperators.functional_value_type(::FunctionalKind{:reservoir_mixed}) = Float64
    FerriteOperators.evaluate_cell_functional(::FunctionalKind{:reservoir_mixed}, c::ReservoirCellCache, args) =
        FerriteOperators.evaluate_cell_functional(FunctionalKind(:reservoir_volume), c, args)
    FerriteOperators.evaluate_algebraic_functional(::FunctionalKind{:reservoir_mixed}, ::ReservoirItemCache, args) =
        10.0 * args.item.index

    @testset "Algebraic items" begin
        testbed = reservoir_testbed((2, 2))
        (; dh, cell_coupling, item_coupling, item_dofs) = testbed
        spec  = StandardOperatorSpecification(; algebraic_couplings = (cell_coupling, item_coupling))
        n     = ndofs(dh)
        u     = 0.1 .* sin.(0.9 .* (1:n))
        θ     = 1.3
        m     = ReservoirIntegrator()
        Kref, rref = reservoir_reference(testbed, m, u, θ)

        @testset "declaration and derived partition" begin
            items = algebraic_items(m, dh)
            # Both 0D rows sit on the same two lumped dofs — the sharing that
            # decides how the family can be scheduled.
            @test length(items) == 2
            @test all(item -> item == item_dofs, items)
            provider = FerriteOperators.AlgebraicItems(items)
            @test FerriteOperators.compute_partition(SequentialScheduling(), provider) == ([1, 2],)
            @test FerriteOperators.compute_partition(ColoredScheduling(), provider) == [[1], [2]]
            # The engine appends the family after the cell subdomains.
            op = setup_operator(sequential_strategy(spec), m, dh)
            @test length(op.engine.subdomain_caches) == length(dh.subdofhandlers) + 1
            @test first(op.engine.subdomain_caches).domain isa FerriteOperators.AssemblyDomain
            @test last(op.engine.subdomain_caches).domain isa FerriteOperators.AlgebraicDomain
        end

        @testset "analytic kernels against the Ferrite reference" begin
            op = setup_operator(sequential_strategy(spec), m, dh)
            r  = zeros(n)
            update_linearization!(op, r, u, θ)
            @test op.J ≈ Kref
            @test r ≈ rref
            r2 = zeros(n)
            evaluate!(op, r2, u, θ)
            @test r2 ≈ rref
            # The 0D rows are what the item family contributed, and they are
            # not empty.
            @test maximum(abs, Kref[item_dofs, item_dofs]) > 0
            @test maximum(abs, rref[item_dofs]) > 0
        end

        @testset "AD algebraic kernel reproduces the analytic one" begin
            # Only the residual kernel is analytic here, so the whole 2×2 item
            # block comes out of the decorator's ForwardDiff seeding.
            opad = setup_operator(sequential_strategy(spec), ReservoirIntegrator(; analytic = false), dh)
            r = zeros(n)
            update_linearization!(opad, r, u, θ)
            @test opad.J ≈ Kref
            @test r ≈ rref
            @test opad.J[item_dofs, item_dofs] ≈ Kref[item_dofs, item_dofs]
        end

        @testset "sparsity declaration is the caller's" begin
            # Without the `AlgebraicCoupling` the (p₁, p₂) off-diagonal has no
            # entry and Ferrite rejects the item's first scatter.
            plain = setup_operator(
                sequential_strategy(StandardOperatorSpecification(; algebraic_couplings = (cell_coupling,))), m, dh)
            @test_throws Exception update_linearization!(plain, zeros(n), u, θ)
        end

        @testset "parallel device with atomic scatter" begin
            # Chunk size 1 puts each of the two colliding items on its own
            # worker, which is exactly what the atomic scatter has to resolve.
            opseq = setup_operator(sequential_strategy(spec), m, dh)
            oppar = setup_operator(AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), PolyesterDevice(1)), m, dh)
            rs = zeros(n); update_linearization!(opseq, rs, u, θ)
            rp = zeros(n); update_linearization!(oppar, rp, u, θ)
            @test oppar.J ≈ opseq.J
            @test rp ≈ rs
        end

        @testset "colored scheduling gives one item per barrier" begin
            # Coloring is unavailable to a `global_dofs` declaration, so the
            # colored run uses the uncoupled cell term.
            mu = ReservoirIntegrator(; coupled = false)
            @test isempty(global_dofs(mu, dh.subdofhandlers[1]))
            ops = setup_operator(sequential_strategy(spec), mu, dh)
            opc = setup_operator(AssemblyStrategy(FullAssembly(spec), ColoredScheduling(), PolyesterDevice(1)), mu, dh)
            rs = zeros(n); update_linearization!(ops, rs, u, θ)
            rc = zeros(n); update_linearization!(opc, rc, u, θ)
            @test opc.J ≈ ops.J
            @test rc ≈ rs
            Kuref, ruref = reservoir_reference(testbed, mu, u, θ)
            @test ops.J ≈ Kuref
            @test rs ≈ ruref
            @test last(opc.engine.subdomain_caches).partition == [[1], [2]]
        end

        @testset "parameter sensitivity of the 0D rows" begin
            # ∂F/∂θ is carried entirely by the item sources: −s_k on p₁, +s_k on
            # p₂, and nothing anywhere else.
            opad = setup_operator(sequential_strategy(spec), ReservoirIntegrator(; analytic = false), dh)
            B = zeros(n, 1)
            update_parameter_jacobian!(B, opad, u, θ)
            @test B[item_dofs[1], 1] ≈ -sum(m.sources)
            @test B[item_dofs[2], 1] ≈ sum(m.sources)
            @test all(≈(0.0; atol = 1.0e-14), B[setdiff(1:n, item_dofs), 1])
        end

        @testset "reductions reach the family and default to no contribution" begin
            op = setup_operator(sequential_strategy(spec), m, dh)
            # The items run and answer `nothing`, so the domain integral is the
            # cell sum — the grid's area.
            @test evaluate_functional(op, FunctionalKind(:reservoir_volume), (u = u,), θ) ≈ 4.0
        end

        @testset "an algebraic contribution folds with the cell sum, decorated or not" begin
            decorated   = setup_operator(sequential_strategy(spec), m, dh)
            undecorated = setup_operator(sequential_strategy(spec), m, dh; ad_backend = nothing)
            @test last(decorated.engine.subdomain_caches).domain.element isa ADElementCache
            @test !(last(undecorated.engine.subdomain_caches).domain.element isa ADElementCache)

            expected = 4.0 + 10.0 * 1 + 10.0 * 2   # grid area + both items
            @test evaluate_functional(decorated, FunctionalKind(:reservoir_mixed), (u = u,), θ) ≈ expected
            @test evaluate_functional(undecorated, FunctionalKind(:reservoir_mixed), (u = u,), θ) ≈ expected
        end

        @testset "declaration validation" begin
            strategy = sequential_strategy(spec)
            # No `setup_algebraic_cache` for a declaration that has items.
            @test_throws ArgumentError setup_operator(strategy, DeclaredAlgebraicItems([[1]]; with_cache = false), dh)
            # Items of different sizes cannot share one set of local buffers.
            @test_throws ArgumentError setup_operator(strategy, DeclaredAlgebraicItems([[1, 2], [3]]), dh)
            # Out of bounds, repeated within an item, or empty.
            @test_throws ArgumentError setup_operator(strategy, DeclaredAlgebraicItems([[n + 1]]), dh)
            @test_throws ArgumentError setup_operator(strategy, DeclaredAlgebraicItems([[1, 1]]), dh)
            @test_throws ArgumentError setup_operator(strategy, DeclaredAlgebraicItems([Int[]]), dh)
            # A well-formed declaration passes the same route.
            @test setup_operator(strategy, DeclaredAlgebraicItems([[1, 2], [2, 3]]), dh) isa
                FerriteOperators.AbstractNonlinearOperator
            # A cache without the mandatory residual kernel.
            @test_throws ArgumentError FerriteOperators.validate_algebraic_cache(CondensedProbeCache())
        end

        @testset "AffineRate slots reconstruct over the item's dofs" begin
            # The 0D rows READ the rate slot here, so the reconstruction is
            # visible in the residual and not only in a workspace buffer.
            op = setup_operator(sequential_strategy(spec), ReservoirIntegrator(; reads_du = true), dh;
                                slots = (:u, :du))
            uprev = 0.05 .* cos.(1:n)

            rate = zeros(n)
            update_linearization!(op, rate, (u = u, du = AffineRate(2.0, uprev)), θ, nothing)
            materialized = zeros(n)
            update_linearization!(op, materialized, (u = u, du = 2.0 .* (u .- uprev)), θ, nothing)
            @test rate ≈ materialized

            # Negative control: a different slope is a different residual, so
            # the agreement above is the reconstruction and not a slot the
            # kernels quietly ignore.
            other = zeros(n)
            update_linearization!(op, other, (u = u, du = AffineRate(3.0, uprev)), θ, nothing)
            @test !(rate ≈ other)
            # …and the rate reaches the item rows specifically.
            @test rate[item_dofs] ≉ rref[item_dofs]
        end

        @testset "InternalSource slots gather empty on an algebraic item" begin
            # An algebraic item owns no condensed internal dofs, so its slice
            # of an `InternalSource` is the empty vector — the same states
            # NamedTuple serves condensed cell elements and algebraic items in
            # one operator.
            op = setup_operator(sequential_strategy(spec), m, dh; slots = (:u, :q))
            r = zeros(n)
            update_linearization!(op, r, (u = u, q = InternalSource(u)), θ, nothing)
            @test r ≈ rref
            ws = first(last(op.engine.subdomain_caches).device_cache)
            @test isempty(ws.slot_buffers.q)
            @test evaluate_functional(
                op, FunctionalKind(:reservoir_volume), (u = u, q = InternalSource(u)), θ) ≈
                evaluate_functional(op, FunctionalKind(:reservoir_volume), (u = u, q = zeros(n)), θ)
        end
    end

    @testset "Guards on element caches with declared global dofs" begin
        sd = stress_driven_testbed((2, 2))
        sdh = sd.dh.subdofhandlers[1]

        @testset "condensed cache plus global dofs is rejected at setup" begin
            # The generic `Consistent` route would multiply a padded ∂F/∂q block
            # by a field-space corrector; caught where the decorator is built.
            @test_throws ArgumentError ADElementCache(CondensedProbeCache(), sdh; n_global_dofs = 1)
            @test_throws ArgumentError decorate_element_cache(CondensedProbeCache(), sdh, ForwardDiffAD(), 1)
            # Without the declaration the same cache is decorated as usual.
            @test ADElementCache(CondensedProbeCache(), sdh) isa ADElementCache
        end

        @testset "patch assembly rejects declared global dofs" begin
            op = setup_operator(
                AssemblyStrategy(FullAssembly(StandardOperatorSpecification(; algebraic_couplings = (sd.coupling,))),
                                 SequentialScheduling(), SequentialCPUDevice()),
                StressDrivenIntegrator(sd.var, sd.E, sd.σ̄), sd.dh)
            provider = PatchItems(sdh, [[1, 2]])
            @test_throws ArgumentError patch_workspace(op, provider)
            @test_throws ArgumentError assemble_patch_matrices!(
                [zeros(patch_ndofs(provider, 1), patch_ndofs(provider, 1))],
                op, provider, zeros(ndofs(sd.dh)), nothing)
        end
    end

end
