using FerriteOperators
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays

@testset "FerriteOperators.jl" begin
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

        integrator = FerriteOperators.SimpleBilinearDiffusionIntegrator(
            1.0,
            QuadratureRuleCollection(2),
            :u
        )
        bilinop_base = setup_assembled_operator(SequentialAssemblyStrategy(SequentialCPUDevice()), integrator, dh)
        # Check that assembly works
        @test norm(bilinop_base.A) ≈ 0.0
        update_operator!(bilinop_base,0.0)
        norm_baseline = norm(bilinop_base.A)
        @test norm_baseline > 0.0
        # Idempotency
        update_operator!(bilinop_base,0.0)
        @test norm_baseline == norm(bilinop_base.A)

        @testset "Strategy $strategy" for strategy in (
                PerColorAssemblyStrategy(SequentialCPUDevice()),
                PerColorAssemblyStrategy(PolyesterDevice(1)),
                PerColorAssemblyStrategy(PolyesterDevice(2)),
                PerColorAssemblyStrategy(PolyesterDevice(3)),
        )
            bilinop = setup_assembled_operator(strategy, integrator, dh)
            # Consistency
            update_operator!(bilinop,0.0)
            @test bilinop.A ≈ bilinop_base.A
            # Idempotency
            update_operator!(bilinop,0.0)
            @test bilinop.A ≈ bilinop_base.A
        end
    end
end
