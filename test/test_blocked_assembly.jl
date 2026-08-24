using FerriteOperators
using Test
using SparseArrays
using BlockArrays
using SparseMatricesCSR

# Assembling into a `BlockMatrix` of CSR blocks needs Ferrite's block/CSR
# constraint-application layer (`Ferrite.addindex!(::SparseMatrixCSR, …)`),
# which the blocked assembler reaches through for every entry.
const _blocked_csr_supported = isdefined(Ferrite, :AlgebraicVariable) && try
    bsp = BlockSparsityPattern([2, 1])
    for i in 1:3, j in 1:3
        Ferrite.add_entry!(bsp, i, j)
    end
    K = allocate_matrix(BlockMatrix{Float64, Matrix{SparseMatrixCSR{1, Float64, Int}}}, bsp)
    assemble!(start_assemble(K, zeros(3)), [1, 3], ones(2, 2), ones(2))
    true
catch
    false
end

if !_blocked_csr_supported
    @info "Skipping the blocked CSR assembly tests: this Ferrite cannot allocate or " *
          "assemble a `BlockMatrix` of `SparseMatrixCSR` blocks (needs the block/CSR " *
          "`addindex!` layer, and `AlgebraicVariable` for the testbed)."
else

    include(joinpath(@__DIR__, "fixture_elements.jl"))

    @testset "Blocked assembly with CSR blocks" begin
        testbed = stress_driven_testbed((3, 3))
        (; dh, var, coupling, E, σ̄) = testbed
        n     = ndofs(dh)
        nalg  = length(algebraic_dofs(dh, :εbar))
        nu    = n - nalg
        u     = 0.01 .* sin.(0.7 .* (1:n))

        monolithic = AssemblyStrategy(
            FullAssembly(StandardOperatorSpecification(; algebraic_couplings = (coupling,))),
            SequentialScheduling(), SequentialCPUDevice())
        blocked_spec = BlockedOperatorSpecification(
            [nu, nalg], BlockMatrix{Float64, Matrix{SparseMatrixCSR{1, Float64, Int}}};
            algebraic_couplings = (coupling,))
        blocked = AssemblyStrategy(FullAssembly(blocked_spec), SequentialScheduling(), SequentialCPUDevice())

        integrator = StressDrivenIntegrator(var, E, σ̄)
        opmono = setup_operator(monolithic, integrator, dh)
        opblk  = setup_operator(blocked, integrator, dh)

        @testset "matrix type and block layout" begin
            @test opblk.J isa BlockMatrix{Float64, Matrix{SparseMatrixCSR{1, Float64, Int}}}
            @test size(opblk.J) == (n, n)
            @test blocksize(opblk.J) == (2, 2)
            @test blocklengths(axes(opblk.J, 1)) == [nu, nalg]
            # The residual stays a plain vector.
            @test FerriteOperators.create_system_vector(opblk.engine.strategy, dh) isa Vector{Float64}
        end

        @testset "fused Jacobian and residual match the monolithic assembly" begin
            rm = zeros(n); update_linearization!(opmono, rm, u, nothing)
            rb = zeros(n); update_linearization!(opblk, rb, u, nothing)
            @test rb ≈ rm
            @test Matrix(opblk.J) ≈ Matrix(opmono.J)
        end

        @testset "residual-only sweep" begin
            rm = zeros(n); evaluate!(opmono, rm, u, nothing)
            rb = zeros(n); evaluate!(opblk, rb, u, nothing)
            @test rb ≈ rm
        end

        @testset "matrix-only sweep" begin
            update_linearization!(opmono, u, nothing)
            update_linearization!(opblk, u, nothing)
            @test Matrix(opblk.J) ≈ Matrix(opmono.J)
        end

        @testset "bilinear operator on a blocked target" begin
            bilinear = SimpleBilinearMassIntegrator(1.0, QuadratureRuleCollection(2), :u)
            opbmono = setup_operator(monolithic, bilinear, dh); update_operator!(opbmono, nothing)
            opbblk  = setup_operator(blocked, bilinear, dh);    update_operator!(opbblk, nothing)
            @test Matrix(opbblk.A) ≈ Matrix(opbmono.A)
        end

        @testset "a linear operator holds no matrix to lay out" begin
            linear = LinearNeumannProbe(1.0, :u, Set(FacetIndex[]))
            @test_throws ArgumentError setup_operator(blocked, linear, dh)
        end
    end

end
