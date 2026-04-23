using FerriteOperators
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester


@testset "Element API" begin
    import FerriteOperators: assemble_element!, assemble_facet!
    import FerriteOperators: setup_element_cache, setup_boundary_cache
    import FerriteOperators: SimpleBilinearMassIntegrator, SimpleBilinearDiffusionIntegrator, SimpleLinearIntegrator
    import FerriteOperators

    setup_test_cache(kwargs...) =
        FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
    function setup_test_composite_volume_cache(kwargs...)
        element_cache =
            FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeVolumetricElementCache((element_cache, element_cache)),
        )
    end
    function setup_test_composite_surface_cache(kwargs...)
        element_cache =
            FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_boundary_cache(kwargs...))
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeSurfaceElementCache((element_cache, element_cache)),
        )
    end

    grid = generate_grid(Hexahedron, (1, 1, 1))
    qrc  = QuadratureRuleCollection(3)
    qr   = QuadratureRule{RefHexahedron}(3)
    qrcf = QuadratureRuleCollection(3)
    qrf  = FacetQuadratureRule{RefHexahedron}(3)
    ip   = Lagrange{RefHexahedron, 1}()

    dhs = DofHandler(grid)
    add!(dhs, :u, ip)
    close!(dhs)
    sdhs = first(dhs.subdofhandlers)
    cell_cache_s = Ferrite.CellCache(sdhs)
    Ferrite.reinit!(cell_cache_s, 1)
    uₑs = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0] .* 1e-4

    ipv = ip^3
    dhv = DofHandler(grid)
    add!(dhv, :u, ipv)
    close!(dhv)
    sdhv = first(dhv.subdofhandlers)
    cell_cache_v = Ferrite.CellCache(sdhv)
    Ferrite.reinit!(cell_cache_v, 1)
    uₑv =
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ] .* 1e-4

    # We check for pairwise consistency of the assembly operations
    # First we check if the empty caches work correctly
    @testset "Empty caches" begin
        rₑ¹ = zeros(ndofs(dhs))
        rₑ² = zeros(ndofs(dhs))
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        # Volume
        assemble_element!(
            Kₑ¹,
            rₑ¹,
            uₑs,
            cell_cache_s,
            FerriteOperators.EmptyVolumetricElementCache(),
            0.0,
        )
        @test iszero(Kₑ¹)
        @test iszero(rₑ¹)

        assemble_element!(rₑ², uₑs, cell_cache_s, FerriteOperators.EmptyVolumetricElementCache(), 0.0)
        @test iszero(rₑ²)

        assemble_element!(Kₑ², uₑs, cell_cache_s, FerriteOperators.EmptyVolumetricElementCache(), 0.0)
        @test iszero(Kₑ²)

        # Surface
        for local_facet_index = 1:nfacets(cell_cache_s)
            assemble_facet!(
                Kₑ¹,
                rₑ¹,
                uₑs,
                cell_cache_s,
                local_facet_index,
                FerriteOperators.EmptySurfaceElementCache(),
                0.0,
            )
            @test iszero(Kₑ¹)
            @test iszero(rₑ¹)

            assemble_facet!(
                rₑ²,
                uₑs,
                cell_cache_s,
                local_facet_index,
                FerriteOperators.EmptySurfaceElementCache(),
                0.0,
            )
            @test iszero(rₑ²)

            assemble_facet!(
                Kₑ²,
                uₑs,
                cell_cache_s,
                local_facet_index,
                FerriteOperators.EmptySurfaceElementCache(),
                0.0,
            )
            @test iszero(Kₑ²)
        end
    end

    @testset "Scalar volumetric bilinear composite elements: $model" for model in (
        SimpleBilinearMassIntegrator(1.0, qrc, :u),
        SimpleBilinearDiffusionIntegrator(1.0, qrc, :u),
    )
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        assemble_element!(Kₑ¹, cell_cache_s, element_cache, 0.0)
        @test !iszero(Kₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        assemble_element!(Kₑ², cell_cache_s, composite_element_cache, 0.0)
        @test 2Kₑ¹ ≈ Kₑ²
    end

    @testset "Scalar linear composite elements: $model" for model in (
        SimpleLinearIntegrator(1.0, qrc, :u),
    )
        bₑ¹ = zeros(ndofs(dhs))
        bₑ² = zeros(ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        assemble_element!(bₑ¹, cell_cache_s, element_cache, 0.0)
        @test !iszero(bₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        assemble_element!(bₑ², cell_cache_s, composite_element_cache, 0.0)
        @test 2bₑ¹ ≈ bₑ²
    end
end

@testset "Operators" begin
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

        @test length(vin)  == size(nullop, 1)
        @test length(vout) == size(nullop, 2)
        
        @test get_matrix(nullop) ≈ zeros(5,5)


        diagop = DiagonalOperator([1.0, 2.0, 3.0, 4.0, 5.0])
        @test length(vin)  == size(diagop, 1)
        @test length(vout) == size(diagop, 2)
        mul!(vout, diagop, vin)
        @test vout == [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, 1.0, 1.0)
        @test vout == 2.0 .* [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, -2.0, 1.0)
        @test vout == zeros(5)
        @test length(vin)  == size(diagop, 1)
        @test length(vout) == size(diagop, 2)

        @test get_matrix(diagop) ≈ spdiagm([1.0, 2.0, 3.0, 4.0, 5.0])


        vin = ones(4)
        vout .= ones(5)
        nullop_rect = NullOperator{Float64,4,5}()

        @test length(vin)  == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)
        @test vout == vout
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
            FerriteOperators.SimpleBilinearDiffusionIntegrator(
                1.0,
                QuadratureRuleCollection(2),
                :u
            ),
            FerriteOperators.SimpleBilinearMassIntegrator(
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

            @testset "Strategy $strategy" for strategy in (
                    PerColorAssemblyStrategy(SequentialCPUDevice()),
                    PerColorAssemblyStrategy(PolyesterDevice(1)),
                    PerColorAssemblyStrategy(PolyesterDevice(2)),
                    PerColorAssemblyStrategy(PolyesterDevice(3)),
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
        struct NeoHookean
            E::Float64
            ν::Float64
        end
        function (p::NeoHookean)(F)
            (; E, ν) = p
            μ = E / (2(1 + ν))
            λ = (E * ν) / ((1 + ν) * (1 - 2ν))
            C = tdot(F)
            Ic = tr(C)
            J = sqrt(det(C))
            return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
        end

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
            FerriteOperators.SimpleHyperelasticityIntegrator(
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
                PerColorAssemblyStrategy(PolyesterDevice(3)),
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
                ElementAssemblyStrategy(PolyesterDevice(3)),
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
        integrator = FerriteOperators.SimpleCondensedLinearViscoelasticity(
            FerriteOperators.MaxwellParameters(),
            qrc,
            :u,
            :εᵛ,
        )

        grid = generate_grid(Hexahedron, (3,3,3))
        Ferrite.transform_coordinates!(grid, x->Vec{3}(sign.(x.-0.5) .* (x.-0.5).^2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
        close!(dh)

        ch = ConstraintHandler(dh);
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> (0,0,0)));
        add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> (0.01,0,0)));
        close!(ch)

        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
        nlop = setup_operator(strategy, integrator, dh)

        residual = zeros(residual_size(nlop))
        u        = zeros(unknown_size(nlop))
        uprev    = zeros(unknown_size(nlop))
        apply_analytical!(u, dh, :u, x->0.01x.^2 .+ 0.01)
        @test length(residual) == 3 * (3+1)*(3+1)*(3+1)
        @test length(u)        == 3 * (3+1)*(3+1)*(3+1) + 6 * 8 * 3*3*3 # vdim=3, 4 nodes in each dim, 8 quadrature points, 6 unknowns for the symmetric viscosity tensor, 3*3*3 elements

        apply!(u, ch)
        update_linearization!(nlop, residual, u, FerriteOperators.GenericFirstOrderTimeParameters(nothing, 0.0, π, uprev))

        apply!(u, ch)
        apply_zero!(nlop.J, residual, ch)
        Δd = nlop.J \ residual
        d = @view u[1:ndofs(dh)]
        d .-= Δd

        update_linearization!(nlop, residual, u, FerriteOperators.GenericFirstOrderTimeParameters(nothing, 0.0, π, uprev))

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

        bilin_op = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u), dh)
        @test size(bilin_op) == (n, n)
        @test size(bilin_op, 1) == n
        @test size(bilin_op, 2) == n

        nl_op = setup_operator(strategy, FerriteOperators.SimpleHyperelasticityIntegrator(NeoHookean(210e3, 0.3), QuadratureRuleCollection(2), :u), dh)
        @test size(nl_op) == (n, n)
        @test size(nl_op, 1) == n
        @test size(nl_op, 2) == n

        lin_op = setup_operator(strategy, FerriteOperators.SimpleLinearIntegrator(1.0, QuadratureRuleCollection(2), :u), dh)
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

        # Linear case
        lin_multi = LinearMultiDomainIntegrator(Dict(
            sdh1 => FerriteOperators.SimpleLinearIntegrator( 1.0, QuadratureRuleCollection(2), :u),
            sdh2 => FerriteOperators.SimpleLinearIntegrator(-1.0, QuadratureRuleCollection(2), :u)
        ))
        lin_op = setup_operator(strategy, lin_multi, dh)
        update_operator!(lin_op, nothing)
        @test size(lin_op) == (n,)

        # Bilinear case
        bilin_multi = BilinearMultiDomainIntegrator(Dict(
            sdh1 => FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u),
            sdh2 => FerriteOperators.SimpleBilinearDiffusionIntegrator(2.0, QuadratureRuleCollection(2), :u)
        ))
        bilin_op = setup_operator(strategy, bilin_multi, dh)
        update_operator!(bilin_op, nothing)
        @test size(bilin_op) == (n, n)
        @test size(bilin_op, 1) == n
        @test size(bilin_op, 2) == n

        # Nonlinear case
        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, getcellset(grid, "right_cells"))
        add!(sdh1, :u, Lagrange{RefHexahedron, 1}()^3)
        sdh2 = SubDofHandler(dh, getcellset(grid, "left_cells"))
        add!(sdh2, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(dh)
        nl_multi = NonlinearMultiDomainIntegrator(Dict(
            sdh1 => FerriteOperators.SimpleHyperelasticityIntegrator(NeoHookean(210e3, 0.30), QuadratureRuleCollection(2), :u),
            sdh2 => FerriteOperators.SimpleHyperelasticityIntegrator(NeoHookean(180e3, 0.35), QuadratureRuleCollection(2), :u)
        ))
        nl_op = setup_operator(strategy, nl_multi, dh)
        u = zeros(ndofs(dh))
        apply_analytical!(u, dh, :u, x->0.01x.^2)
        update_linearization!(nl_op, u, nothing)
        @test size(nl_op) == (3n, 3n)
        @test size(nl_op, 1) == 3n
        @test size(nl_op, 2) == 3n
    end
end
