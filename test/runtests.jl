using FerriteOperators
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays

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
end
