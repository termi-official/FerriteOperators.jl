using FerriteOperators
using Test
using SparseArrays
using Polyester

# Elements whose local system carries dofs that are not in `celldofs` need
# Ferrite's mesh-free algebraic variables to have such dofs at all.
if !isdefined(Ferrite, :AlgebraicVariable)
    @info "Skipping the global-dof tests: this Ferrite has no `AlgebraicVariable`, " *
          "so a DofHandler cannot carry dofs outside the mesh."
else

    include(joinpath(@__DIR__, "fixture_elements.jl"))

    sequential_strategy(spec) = AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), SequentialCPUDevice())

    # Declaration double for the setup-time validation: whatever dof vector it
    # is handed, over a subdomain that assembles nothing.
    struct DeclaredGlobalDofs <: AbstractNonlinearIntegrator
        dofs::Vector{Int}
    end
    FerriteOperators.global_dofs(m::DeclaredGlobalDofs, sdh::SubDofHandler) = m.dofs
    FerriteOperators.setup_element_cache(::DeclaredGlobalDofs, ::SubDofHandler) =
        FerriteOperators.EmptyVolumetricElementCache()

    @testset "Elements with global dofs" begin
        testbed = stress_driven_testbed((3, 3))
        (; dh, var, coupling, E, σ̄) = testbed
        spec  = StandardOperatorSpecification(; algebraic_couplings = (coupling,))
        n     = ndofs(dh)
        gdofs = algebraic_dofs(dh, :εbar)
        Kref, fref = stress_driven_reference(testbed)
        u = 0.01 .* sin.(0.7 .* (1:n))

        @testset "declaration and local layout" begin
            m   = StressDrivenIntegrator(var, E, σ̄)
            sdh = dh.subdofhandlers[1]
            @test global_dofs(m, sdh) == gdofs
            @test global_dof_range(m, sdh) == ndofs_per_cell(sdh) .+ (1:length(gdofs))
            # The declaration is invisible to the mesh side of the handler.
            @test isempty(intersect(celldofs(dh, 1), gdofs))
        end

        @testset "analytic element against the Ferrite reference" begin
            op = setup_operator(sequential_strategy(spec), StressDrivenIntegrator(var, E, σ̄), dh)
            r  = zeros(n)
            update_linearization!(op, r, u, nothing)
            @test op.J ≈ Kref
            @test r ≈ Kref * u - fref
            r2 = zeros(n)
            evaluate!(op, r2, u, nothing)
            @test r2 ≈ r
            # The assembled system solves the RVE problem: the algebraic dofs
            # come out as the macroscopic strain answering σ̄.
            @test algebraic_value(dh, op.J \ fref, :εbar) ≈ algebraic_value(dh, Kref \ fref, :εbar)
        end

        @testset "AD element reproduces the augmented Jacobian" begin
            # Only the residual kernel is analytic here, so every entry of the
            # augmented Jacobian — the tail block included — comes out of the
            # padded ForwardDiff buffers.
            opad = setup_operator(sequential_strategy(spec),
                                  StressDrivenIntegrator(var, E, σ̄; analytic = false), dh)
            r = zeros(n)
            update_linearization!(opad, r, u, nothing)
            @test opad.J ≈ Kref
            @test r ≈ Kref * u - fref
            @test opad.J[gdofs, gdofs] ≈ Kref[gdofs, gdofs]
            @test maximum(abs, Kref[gdofs, gdofs]) > 0
        end

        @testset "parallel device with atomic scatter" begin
            opseq = setup_operator(sequential_strategy(spec), StressDrivenIntegrator(var, E, σ̄), dh)
            oppar = setup_operator(AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), PolyesterDevice(min_items_per_worker = 2)),
                                   StressDrivenIntegrator(var, E, σ̄), dh)
            rs = zeros(n); update_linearization!(opseq, rs, u, nothing)
            rp = zeros(n); update_linearization!(oppar, rp, u, nothing)
            @test oppar.J ≈ opseq.J
            @test rp ≈ rs
        end

        @testset "rejected strategies" begin
            m = StressDrivenIntegrator(var, E, σ̄)
            @test_throws ArgumentError setup_operator(
                AssemblyStrategy(FullAssembly(spec), ColoredScheduling(), SequentialCPUDevice()), m, dh)
        end

        @testset "declaration validation" begin
            # Checked against the handler before any cache is built: out of
            # bounds, repeated, or overlapping the cell dofs.
            strategy = sequential_strategy(spec)
            @test_throws ArgumentError setup_operator(strategy, DeclaredGlobalDofs([n + 1]), dh)
            @test_throws ArgumentError setup_operator(strategy, DeclaredGlobalDofs([0]), dh)
            @test_throws ArgumentError setup_operator(strategy, DeclaredGlobalDofs([n, n]), dh)
            @test_throws ArgumentError setup_operator(strategy, DeclaredGlobalDofs([celldofs(dh, 1)[1]]), dh)
            # A well-formed declaration passes the same route.
            @test setup_operator(strategy, DeclaredGlobalDofs(collect(gdofs)), dh) isa
                FerriteOperators.AbstractNonlinearOperator
        end

        @testset "composite integrators forward the declaration" begin
            m   = StressDrivenIntegrator(var, E, σ̄)
            m2  = StressDrivenIntegrator(var, E, σ̄; analytic = false)
            sdh = dh.subdofhandlers[1]
            @test global_dofs(NonlinearCompositeIntegrator(m, m2), sdh) == gdofs
            # Two inners must agree — the tail of one local system cannot be
            # two different dof vectors.
            @test_throws ArgumentError global_dofs(
                NonlinearCompositeIntegrator(m, DeclaredGlobalDofs([1, 2, 3])), sdh)
        end

        @testset "sparsity declaration is the caller's" begin
            # Without the coupling descriptor the algebraic rows carry no
            # off-diagonal entries and Ferrite rejects the first scatter.
            plain = setup_operator(sequential_strategy(StandardOperatorSpecification()),
                                   StressDrivenIntegrator(var, E, σ̄), dh)
            @test_throws Exception update_linearization!(plain, zeros(n), u, nothing)
        end
    end

end
