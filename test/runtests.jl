using FerriteOperators
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester

@testset "FerriteOperators.jl" begin
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

        lin_op = FerriteOperators.LinearFerriteOperator(zeros(n), strategy, FerriteOperators.SubdomainCache[])
        @test size(lin_op) == (n,)
    end

    @testset "GPU device validation" begin
        @test_throws ArgumentError FerriteOperators.setup_device_cache(CudaDevice(), FerriteOperators.EAIndexWorkspace(0), 1)
        @test_throws ArgumentError FerriteOperators.n_workers(SequentialAssemblyStrategy(CudaDevice()), CudaDevice(), [1:5])
        @test_throws ArgumentError FerriteOperators.execute_on_device!(nothing, CudaDevice(), nothing, [])
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
end
