using FerriteOperators
using FerriteOperatorsExampleElements
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester
using TimerOutputs

include(joinpath(@__DIR__, "fixture_elements.jl"))

@testset "Operators" begin
    reset_timer!()
    @testset "Element Assembly Matrix" begin
        Aₑ = [1.0 -1.0; -1.0 1.0]
        Aₑflat = [1.0, -1.0, -1.0, 1.0]
        N = 10

        # Assemble reference
        A = zeros(N,N)
        for i in 1:N-1
            A[i:i+1,i:i+1] .+= Aₑ
        end
        x = collect(1.0:N).^2
        yref = A*x

        # Generic action of H1 discretization
        vindices = FerriteOperators.GenericIndexedData(
            [1+(i ÷ 2) for i in 1:2N],
            [FerriteOperators.GenericEAVectorIndex(2i-1, 2) for i in 1:N],
        )
        mindices = [
            FerriteOperators.GenericEAMatrixIndex(4i-3, 2, 2) for i in 1:(N-1)
        ]

        # 
        op = FerriteOperators.EAOperator(
            SequentialCPUDevice(),
            FerriteOperators.EAViewCache(),
            FerriteOperators.GenericIndexedData(
                repeat(Aₑflat, N),
                mindices,
            ),
            vindices,
            vindices,
        )
        y = zeros(N)
        mul!(y, op, x)
        @test y ≈ yref

        op = FerriteOperators.EAOperator(
            PolyesterDevice(1),
            FerriteOperators.EAViewCache(),
            FerriteOperators.GenericIndexedData(
                repeat(Aₑflat, N),
                mindices,
            ),
            vindices,
            vindices,
        )
        y = zeros(N)
        mul!(y, op, x)

        @test y ≈ yref
    end

    @testset "Actions" begin
        vin = ones(5)
        vout = ones(5)

        nullop = NullOperator{Float64,5,5}()
        @test eltype(nullop) == Float64
        @test length(vin)  == size(nullop, 1)
        @test length(vout) == size(nullop, 2)

        mul!(vout, nullop, vin)
        @test vout == zeros(5)

        vout .= ones(5)
        mul!(vout, nullop, vin, 2.0, 1.0)
        @test vout == ones(5)

        @test get_matrix(nullop) ≈ zeros(5,5)


        diagop = DiagonalOperator([1.0, 2.0, 3.0, 4.0, 5.0])
        @test length(vin)  == size(diagop, 1)
        @test length(vout) == size(diagop, 2)
        mul!(vout, diagop, vin)
        @test vout == [1.0, 2.0, 3.0, 4.0, 5.0]

        # in and out must be distinguishable to catch out-aliasing mistakes
        vres = zeros(5)
        mul!(vres, diagop, [1.0, 1.0, 2.0, 2.0, 3.0])
        @test vres == [1.0, 2.0, 6.0, 8.0, 15.0]

        # Bilinear operators with constant linearization support both update forms
        update_linearization!(diagop, vin, nothing)
        update_linearization!(diagop, vres, vin, nothing)
        update_linearization!(nullop, vin, nothing)
        update_linearization!(nullop, vres, vin, nothing)

        mul!(vout, diagop, vin, 1.0, 1.0)
        @test vout == 2.0 .* [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, -2.0, 1.0)
        @test vout == zeros(5)

        @test get_matrix(diagop) ≈ spdiagm([1.0, 2.0, 3.0, 4.0, 5.0])


        vin = ones(4)
        vout .= ones(5)
        nullop_rect = NullOperator{Float64,4,5}()

        @test length(vin)  == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)

        @test get_matrix(nullop_rect) ≈ zeros(4,5)
    end

    @testset "Bilinear" begin
        # Setup
        grid = generate_grid(Quadrilateral, (10,9))
        Ferrite.transform_coordinates!(grid, x->Vec{2}(sign.(x.-0.5) .* (x.-0.5).^2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral,1}())
        close!(dh)
        qrc = QuadratureRuleCollection{2}()

        for integrator in [
            SimpleBilinearDiffusionIntegrator(
                1.0,
                QuadratureRuleCollection(2),
                :u
            ),
            SimpleBilinearMassIntegrator(
                1.0,
                QuadratureRuleCollection(1),
                :u
            )
        ]
            bilinop_base = setup_operator(SequentialAssemblyStrategy(SequentialCPUDevice()), integrator, dh)
            # Check that assembly works
            @test norm(bilinop_base.A) ≈ 0.0
            update_operator!(bilinop_base, 0.0)
            norm_baseline = norm(bilinop_base.A)
            @test norm_baseline > 0.0
            # Idempotency
            update_operator!(bilinop_base, 0.0)
            @test norm_baseline == norm(bilinop_base.A)

            # The bilinear form induces a linear operator, so the residual
            # entry point must reproduce its action.
            ub = sin.(0.4 .* (1:ndofs(dh)))
            rb = zeros(ndofs(dh))
            evaluate!(bilinop_base, rb, ub, 0.0)
            @test rb ≈ bilinop_base.A * ub rtol = 1e-13

            @testset "Strategy $strategy" for strategy in (
                    PerColorAssemblyStrategy(SequentialCPUDevice()),
                    PerColorAssemblyStrategy(PolyesterDevice(1)),
                    PerColorAssemblyStrategy(PolyesterDevice(2)),
            )
                bilinop = setup_operator(strategy, integrator, dh)
                # Consistency
                update_operator!(bilinop, 0.0)
                @test bilinop.A ≈ bilinop_base.A
                # Idempotency
                update_operator!(bilinop, 0.0)
                @test bilinop.A ≈ bilinop_base.A
            end
        end
    end

    @testset "Nonlinear" begin
        # Setup
        grid = generate_grid(Hexahedron, (3,3,3))
        Ferrite.transform_coordinates!(grid, x->Vec{3}(sign.(x.-0.5) .* (x.-0.5).^2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
        close!(dh)

        residual = zeros(ndofs(dh))
        u = zeros(ndofs(dh))
        apply_analytical!(u, dh, :u, x->0.01x.^2)

        for integrator in [
            SimpleHyperelasticityIntegrator(
                NeoHookean(10.0, 0.3),
                QuadratureRuleCollection(2),
                :u
            ),
        ]
            nlop_base = setup_operator(SequentialAssemblyStrategy(SequentialCPUDevice()), integrator, dh)

            # Check that assembly works
            @test norm(nlop_base.J) ≈ 0.0
            nlop_base.J .= NaN
            update_linearization!(nlop_base, u, 0.0)
            Jnorm_baseline = norm(nlop_base.J)
            @test Jnorm_baseline > 0.0
            yref = zero(u)
            mul!(yref, nlop_base.J, u)

            # Also querying the residual should not change the outcome
            residual .= NaN
            nlop_base.J .= NaN
            update_linearization!(nlop_base, residual, u, 0.0)
            @test Jnorm_baseline ≈ norm(nlop_base.J)
            rnorm_baseline = norm(residual)
            @test rnorm_baseline > 0.0

            # Now just the residual
            residual .= NaN
            nlop_base(residual, u, 0.0)
            @test rnorm_baseline ≈ norm(residual)

            # Idempotency
            update_linearization!(nlop_base, u, 0.0)
            @test Jnorm_baseline ≈ norm(nlop_base.J)
            nlop_base(residual, u, 0.0)
            @test Jnorm_baseline ≈ norm(nlop_base.J)
            @test rnorm_baseline ≈ norm(residual)
            residual_baseline = copy(residual)

            @testset "Full Assembly Strategy $strategy" for strategy in (
                SequentialAssemblyStrategy(SequentialCPUDevice()),
                PerColorAssemblyStrategy(SequentialCPUDevice()),
                PerColorAssemblyStrategy(PolyesterDevice(1)),
                PerColorAssemblyStrategy(PolyesterDevice(2)),
            )
                nlop = setup_operator(strategy, integrator, dh)
                # Consistency and Idempotency
                for i in 1:2
                    nlop.J .= NaN
                    update_linearization!(nlop, u, 0.0)
                    @test nlop.J ≈ nlop_base.J

                    nlop.J .= NaN
                    residual .= NaN
                    update_linearization!(nlop, residual, u, 0.0)
                    @test nlop.J ≈ nlop_base.J
                    @test residual ≈ residual_baseline

                    residual .= NaN
                    nlop(residual, u, 0.0)
                    @test residual ≈ residual_baseline
                end
            end

            @testset "Element Assembly Strategy $strategy" for strategy in (
                ElementAssemblyStrategy(SequentialCPUDevice()),
                ElementAssemblyStrategy(PolyesterDevice(1)),
                ElementAssemblyStrategy(PolyesterDevice(2)),
            )
                nlop = setup_operator(strategy, integrator, dh)
                # Consistency and Idempotency
                for i in 1:2
                    update_linearization!(nlop, u, 0.0)
                    y = zero(u)
                    mul!(y, nlop.J, u)
                    @test yref ≈ y
                    mul!(y, nlop.J, u)
                    @test yref ≈ y

                    residual .= NaN
                    update_linearization!(nlop, residual, u, 0.0)
                    mul!(y, nlop.J, u)
                    @test yref ≈ y
                    @test residual ≈ residual_baseline

                    residual .= NaN
                    nlop(residual, u, 0.0)
                    mul!(y, nlop.J, u)
                    @test yref ≈ y
                    @test residual ≈ residual_baseline
                end
            end
        end
    end

    @testset "Condensed Elements" begin
        qrc = QuadratureRuleCollection(2)
        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
        tb = visco_testbed(strategy, qrc, (3, 3, 3);
                           transform = x->Vec{3}(sign.(x.-0.5) .* (x.-0.5).^2))
        nlop, dh, grid = tb.op, tb.dh, tb.grid

        ch = ConstraintHandler(dh);
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> (0,0,0)));
        add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> (0.01,0,0)));
        close!(ch)

        ctx  = TimeIntegrationContext(0.0, π, π)   # backward-Euler local stage: γ̃ = Δt

        residual = zeros(residual_size(nlop))
        u        = zeros(unknown_size(nlop))
        uprev    = zeros(unknown_size(nlop))
        states   = condensed_states(u, uprev)
        apply_analytical!(u, dh, :u, x->0.01x.^2 .+ 0.01)
        @test length(residual) == 3 * (3+1)*(3+1)*(3+1)
        @test length(u)        == 3 * (3+1)*(3+1)*(3+1) + 6 * 8 * 3*3*3 # vdim=3, 4 nodes in each dim, 8 quadrature points, 6 unknowns for the symmetric viscosity tensor, 3*3*3 elements

        apply!(u, ch)
        condense_internal!(nlop, states, nothing, ctx)
        update_linearization!(nlop, residual, states, nothing, ctx)

        apply!(u, ch)
        apply_zero!(nlop.J, residual, ch)
        Δd = nlop.J \ residual
        d = @view u[1:ndofs(dh)]
        d .-= Δd

        condense_internal!(nlop, states, nothing, ctx)
        update_linearization!(nlop, residual, states, nothing, ctx)

        apply_zero!(nlop.J, residual, ch)
        Δd = nlop.J \ residual
        @test norm(Δd)/length(Δd) ≈ 0.0 atol=1e-12

        @test norm(d) ≈ 0.059623465672897884
        @test norm(u[ndofs(dh)+1:end]) ≈ 0.062203435313135984
        @test norm(uprev) ≈ 0.0
    end

    @testset "Transfer with different Dof Handlers" begin
        grid = generate_grid(Hexahedron, (1,1,1))
        Ferrite.transform_coordinates!(grid, x->Vec{3}(sign.(x.-0.5) .* (x.-0.5).^2))

        dh2 = DofHandler(grid)
        add!(dh2, :u, Lagrange{RefHexahedron,2}())
        close!(dh2)

        dh1 = DofHandler(grid)
        add!(dh1, :u, Lagrange{RefHexahedron,1}())
        close!(dh1)

        integrator = FerriteOperators.MassProlongatorIntegrator(QuadratureRuleCollection(4), :u)
        strategy   = SequentialAssemblyStrategy(SequentialCPUDevice())
        op         = setup_transfer_operator(strategy, integrator, dh2, dh1)
        update_operator!(op, nothing)

        u1 = zeros(ndofs(dh1))
        u2 = zeros(ndofs(dh2))
        apply_analytical!(u1, dh1, :u, x->1.0)
        apply_analytical!(u2, dh2, :u, x->1.0)

        @test u2 ≈ op.P * u1
    end

    @testset "Operator sizes" begin
        grid = generate_grid(Quadrilateral, (3,3))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral,1}())
        close!(dh)
        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
        n = ndofs(dh)

        bilin_op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u), dh)
        @test size(bilin_op) == (n, n)
        @test size(bilin_op, 1) == n
        @test size(bilin_op, 2) == n

        nl_op = setup_operator(strategy, SimpleHyperelasticityIntegrator(NeoHookean(210e3, 0.3), QuadratureRuleCollection(2), :u), dh)
        @test size(nl_op) == (n, n)
        @test size(nl_op, 1) == n
        @test size(nl_op, 2) == n

        lin_op = setup_operator(strategy, SimpleLinearIntegrator(1.0, QuadratureRuleCollection(2), :u), dh)
        @test size(lin_op) == (n,)
    end

    @testset "GPU device validation" begin
        @test_throws ArgumentError FerriteOperators.setup_device_instances(CudaDevice(), FerriteOperators.EAIndexWorkspace(0), 1)
        @test_throws ArgumentError FerriteOperators.n_workers(SequentialAssemblyStrategy(CudaDevice()), CudaDevice(), [1:5])
        @test_throws ArgumentError FerriteOperators.execute_on_device!(nothing, CudaDevice(), nothing, [])
    end

    @testset "Generic setup_device_instances" begin
        # setup_device_instances should work on any duplicable object, not just AbstractWorkspace
        struct _TestDuplicable
            x::Int
        end
        FerriteOperators.duplicate_for_device(::FerriteOperators.AbstractCPUDevice, d::_TestDuplicable) = _TestDuplicable(d.x)

        seq = SequentialCPUDevice()
        dc_seq = FerriteOperators.setup_device_instances(seq, _TestDuplicable(7), 1)
        @test length(dc_seq) == 1
        @test dc_seq[1].x == 7

        poly = PolyesterDevice()
        dc_poly = FerriteOperators.setup_device_instances(poly, _TestDuplicable(7), 3)
        @test length(dc_poly) == 3
        @test all(d -> d.x == 7, dc_poly)
    end

    @testset "Transfer setup validation" begin
        grid = generate_grid(Hexahedron, (1,1,1))
        dh2 = DofHandler(grid)
        add!(dh2, :u, Lagrange{RefHexahedron,2}())
        close!(dh2)
        dh1 = DofHandler(grid)
        add!(dh1, :u, Lagrange{RefHexahedron,1}())
        close!(dh1)
        integrator = FerriteOperators.MassProlongatorIntegrator(QuadratureRuleCollection(4), :u)
        @test_throws ArgumentError setup_transfer_operator(PerColorAssemblyStrategy(SequentialCPUDevice()), integrator, dh2, dh1)
    end

    @testset "Dummy Multi-Physics" begin
        grid = generate_grid(Hexahedron, ntuple(_ -> 4, 3))
        addcellset!(grid, "right_cells", x -> x[1] ≥ 0.0)
        addcellset!(grid, "left_cells", x -> x[1] ≤ 0.0)

        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())

        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, getcellset(grid, "right_cells"))
        add!(sdh1, :u, Lagrange{RefHexahedron, 1}())
        sdh2 = SubDofHandler(dh, getcellset(grid, "left_cells"))
        add!(sdh2, :u, Lagrange{RefHexahedron, 1}())
        close!(dh)

        n = 5^3
        qrc = QuadratureRuleCollection(2)

        assemble_linear(subintegrators) = begin
            op = setup_operator(strategy, LinearMultiDomainIntegrator(subintegrators), dh)
            update_operator!(op, nothing)
            op
        end

        # Linear case
        lin_op = assemble_linear(Dict(
            "right_cells" => SimpleLinearIntegrator( 1.0, qrc, :u),
            "left_cells"  => SimpleLinearIntegrator(-1.0, qrc, :u)
        ))
        @test size(lin_op) == (n,)

        # The name→subdomain map is the geometric one: loading only
        # "right_cells" must leave every dof outside that subdomain untouched.
        right_dofs = Set{Int}()
        for cc in CellIterator(sdh1)
            union!(right_dofs, celldofs(cc))
        end
        right_only = assemble_linear(Dict(
            "right_cells" => SimpleLinearIntegrator(1.0, qrc, :u),
            "left_cells"  => SimpleLinearIntegrator(0.0, qrc, :u)
        ))
        @test all(iszero, right_only.b[setdiff(1:n, right_dofs)])
        @test !all(iszero, right_only.b[collect(right_dofs)])

        # Routing is per-domain, so the two assignments of one integrator pair
        # differ, and swapping them sums to the uniform assignment.
        bilin(D_right, D_left) = begin
            op = setup_operator(strategy, BilinearMultiDomainIntegrator(Dict(
                "right_cells" => SimpleBilinearDiffusionIntegrator(D_right, qrc, :u),
                "left_cells"  => SimpleBilinearDiffusionIntegrator(D_left, qrc, :u)
            )), dh)
            update_operator!(op, nothing)
            op.A
        end
        Ka = bilin(1.0, 2.0)
        Kb = bilin(2.0, 1.0)
        @test Ka != Kb
        @test Ka + Kb ≈ bilin(3.0, 3.0)

        # A weak boundary term is resolved through the *volumetric* name of
        # the subdomain carrying it, never through a facetset namespace.
        t̄ = 3.25
        neumann_op = assemble_linear(Dict(
            "right_cells" => LinearNeumannProbe(t̄, :u, Set(getfacetset(grid, "right"))),
            "left_cells"  => SimpleLinearIntegrator(0.0, qrc, :u)
        ))
        @test sum(neumann_op.b) ≈ t̄ * 4.0 rtol = 1e-12

        # Message of the ArgumentError `f` raises, so the cell-exact content of
        # a rejection can be asserted and not just its type.
        function rejection_message(f)
            try
                f()
            catch err
                err isa ArgumentError && return err.msg
                rethrow()
            end
            error("expected an ArgumentError")
        end

        @testset "validation in both modes" begin
            L(v) = SimpleLinearIntegrator(v, qrc, :u)
            addcellset!(grid, "all_cells", x -> true)
            addcellset!(grid, "one_cell", Set([1]))

            # These four classes are rejected regardless of resolution mode.
            for mode in (Val(:sample), Val(:full))
                # a declared name that is not a cellset of the grid
                @test_throws ArgumentError FerriteOperators.resolve_subdomain_claims(
                    Dict("right_cells" => L(1.0), "left_cell" => L(1.0)), dh, mode)
                # a subdomain owned by no declared name
                @test_throws ArgumentError FerriteOperators.resolve_subdomain_claims(
                    Dict("right_cells" => L(1.0)), dh, mode)
                # a subdomain claimed by more than one declared name
                @test_throws ArgumentError FerriteOperators.resolve_subdomain_claims(
                    Dict("all_cells" => L(1.0), "right_cells" => L(1.0), "left_cells" => L(1.0)), dh, mode)
                # a declared name claiming no subdomain (disjoint from every subdomain)
                sub_grid = generate_grid(Hexahedron, (3, 1, 1))
                addcellset!(sub_grid, "a", Set([1]))
                addcellset!(sub_grid, "b", Set([2]))
                addcellset!(sub_grid, "c", Set([3]))
                sub_dh = DofHandler(sub_grid)
                sa = SubDofHandler(sub_dh, getcellset(sub_grid, "a")); add!(sa, :u, Lagrange{RefHexahedron, 1}())
                sb = SubDofHandler(sub_dh, getcellset(sub_grid, "b")); add!(sb, :u, Lagrange{RefHexahedron, 1}())
                close!(sub_dh)
                @test FerriteOperators.resolve_subdomain_claims(
                    Dict("a" => L(1.0), "b" => L(1.0)), sub_dh, mode) == ["a", "b"]
                @test_throws ArgumentError FerriteOperators.resolve_subdomain_claims(
                    Dict("a" => L(1.0), "b" => L(1.0), "c" => L(1.0)), sub_dh, mode)
            end

            # Production sampling reads the first cell only, so a subdomain that
            # straddles two declared cellsets resolves by that cell alone; debug
            # mode rejects it, naming the cells at the mismatch.
            addcellset!(grid, "right_head", Set([first(sdh1.cellset)]))
            straddle = Dict("right_head" => L(1.0), "left_cells" => L(1.0))
            @test FerriteOperators.resolve_subdomain_claims(straddle, dh, Val(:sample)) ==
                ["right_head", "left_cells"]
            straddle_msg = rejection_message(
                () -> FerriteOperators.resolve_subdomain_claims(straddle, dh, Val(:full)))
            @test occursin("owned by no declared name", straddle_msg)
            @test occursin("right_head", straddle_msg)

            # Debug mode rejects overlapping DECLARED cellsets at fill time,
            # naming the exact cell, even where no subdomain is ambiguous.
            overlap_msg = rejection_message(() -> FerriteOperators.resolve_subdomain_claims(
                Dict("all_cells" => L(1.0), "one_cell" => L(1.0)), dh, Val(:full)))
            @test occursin("Cell 1 lies in both", overlap_msg)

            # The operator entry point routes through the compile-time mode.
            @test_throws ArgumentError setup_operator(strategy, LinearMultiDomainIntegrator(
                Dict("right_cells" => L(1.0))), dh)
        end

        @testset "resolution on a large grid" begin
            n_side = 400                      # 160_000 cells
            big_grid = generate_grid(Quadrilateral, (n_side, n_side))
            half = n_side * n_side ÷ 2
            addcellset!(big_grid, "lower", Set(1:half))
            addcellset!(big_grid, "upper", Set((half + 1):(n_side * n_side)))
            big_dh = DofHandler(big_grid)
            lo = SubDofHandler(big_dh, getcellset(big_grid, "lower")); add!(lo, :u, Lagrange{RefQuadrilateral, 1}())
            hi = SubDofHandler(big_dh, getcellset(big_grid, "upper")); add!(hi, :u, Lagrange{RefQuadrilateral, 1}())
            close!(big_dh)
            big_qrc = QuadratureRuleCollection(2)
            subs = Dict("lower" => SimpleLinearIntegrator(1.0, big_qrc, :u),
                        "upper" => SimpleLinearIntegrator(2.0, big_qrc, :u))

            @test FerriteOperators.resolve_subdomain_claims(subs, big_dh, Val(:full)) == ["lower", "upper"]

            addcellset!(big_grid, "seam", Set([half, half + 1]))
            seam_msg = rejection_message(() -> FerriteOperators.resolve_subdomain_claims(
                merge(subs, Dict("seam" => SimpleLinearIntegrator(0.0, big_qrc, :u))), big_dh, Val(:full)))
            @test occursin("Cell $half lies in both", seam_msg)
        end

        # Routing outer, composition inner: a named subdomain whose term is a
        # composite of two bilinear forms.
        composed = setup_operator(strategy, BilinearMultiDomainIntegrator(Dict(
            "right_cells" => BilinearCompositeIntegrator(
                SimpleBilinearDiffusionIntegrator(1.0, qrc, :u),
                SimpleBilinearMassIntegrator(2.0, qrc, :u)),
            "left_cells"  => SimpleBilinearDiffusionIntegrator(1.0, qrc, :u)
        )), dh)
        update_operator!(composed, nothing)
        mass_only = setup_operator(strategy, BilinearMultiDomainIntegrator(Dict(
            "right_cells" => SimpleBilinearMassIntegrator(2.0, qrc, :u),
            "left_cells"  => SimpleBilinearMassIntegrator(0.0, qrc, :u)
        )), dh)
        update_operator!(mass_only, nothing)
        @test composed.A ≈ bilin(1.0, 1.0) + mass_only.A rtol = 1e-12

        # Nonlinear case
        dhv = DofHandler(grid)
        sdhv1 = SubDofHandler(dhv, getcellset(grid, "right_cells"))
        add!(sdhv1, :u, Lagrange{RefHexahedron, 1}()^3)
        sdhv2 = SubDofHandler(dhv, getcellset(grid, "left_cells"))
        add!(sdhv2, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(dhv)
        nl_multi = NonlinearMultiDomainIntegrator(Dict(
            "right_cells" => SimpleHyperelasticityIntegrator(NeoHookean(210e3, 0.30), qrc, :u),
            "left_cells"  => SimpleHyperelasticityIntegrator(NeoHookean(180e3, 0.35), qrc, :u)
        ))
        nl_op = setup_operator(strategy, nl_multi, dhv)
        u = zeros(ndofs(dhv))
        apply_analytical!(u, dhv, :u, x->0.01x.^2)
        update_linearization!(nl_op, u, nothing)
        @test size(nl_op) == (3n, 3n)
        @test size(nl_op, 1) == 3n
        @test size(nl_op, 2) == 3n
    end

    print_timer()
end
