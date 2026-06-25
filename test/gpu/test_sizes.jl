## Operator sizes (GPU) — mirror of the CPU "Operator sizes" testset. ##
# Pure size/eltype queries on the GPU operator (no assembly, no kernel launch).
# Building the operators here exercises on-device setup: element-cache duplication
# and (for linear) on-device system-vector allocation.
function run_sizes_tests(device)
    @testset "Operator sizes (GPU)" begin
        grid = generate_grid(Quadrilateral, (3, 3))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        n = ndofs(dh)
        strategy = SequentialAssemblyStrategy(device)

        @testset "Bilinear" begin
            bilin_op = setup_operator(
                strategy,
                FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u),
                dh,
            )
            @test size(bilin_op) == (n, n)
            @test size(bilin_op, 1) == n
            @test size(bilin_op, 2) == n
        end

        @testset "Nonlinear" begin
            nl_op = setup_operator(
                strategy,
                FerriteOperators.SimpleHyperelasticityIntegrator(GPUTestNeoHookean(210e3, 0.3), QuadratureRuleCollection(2), :u),
                dh,
            )
            @test size(nl_op) == (n, n)
            @test size(nl_op, 1) == n
            @test size(nl_op, 2) == n
        end

        @testset "Linear" begin
            lin_op = setup_operator(
                strategy,
                FerriteOperators.SimpleLinearIntegrator(1.0, QuadratureRuleCollection(2), :u),
                dh,
            )
            @test size(lin_op) == (n,)
        end
    end
end
