# The shipped DG consumer on a real device (Slice 4). What only a CUDA run
# proves: `SIPGDiffusionElementCache` crosses the ACTION launch with its host
# `InterfaceValues` DROPPED, the block-row store reaches the kernel already
# filled from the host mirror, and the one-colour cell partition makes the
# action bitwise repeatable.
#
# The reference is the same element ASSEMBLED on the host — a different traversal
# and a different storage; the hand-built SIPG comparison runs in the main suite.

dg_cuda_integrator(order) = SIPGDiffusionIntegrator(
    1.3, 4.0, QuadratureRuleCollection(order + 1),
    FerriteOperators.FacetQuadratureRuleCollection(order + 1), :u)

function dg_cuda_handler(celltype, dims, order)
    grid = generate_grid(celltype, dims)
    dh = DofHandler(grid)
    add!(dh, :u, DiscontinuousLagrange{Ferrite.getrefshape(getcells(grid, 1)), order}())
    close!(dh)
    return dh
end

dg_cuda_assembled(integrator, dh) = let op = setup_operator(
        AssemblyStrategy(SequentialCPUDevice();
            form = FullAssembly(StandardOperatorSpecification(;
                sparsity_entries = interior_facet_entries!))), integrator, dh)
    update_operator!(op, nothing)
    op.A
end

@testset "CUDA DG diffusion (SIPG over BlockRowAssembly)" begin
    device = KernelAbstractionsDevice(CUDABackend(); index_type = Ti, items_per_worker = 1)

    @testset "$label vs the host-assembled reference" for (label, celltype, dims, order) in
            (("quad p=1", Quadrilateral, (3, 3), 1),
             ("quad p=2", Quadrilateral, (3, 3), 2),
             ("hex p=1",  Hexahedron, (2, 2, 2), 1))

        dh  = dg_cuda_handler(celltype, dims, order)
        itg = dg_cuda_integrator(order)
        A   = dg_cuda_assembled(itg, dh)
        u   = Float64[sin(3.1i) + 0.2cos(i) for i in 1:ndofs(dh)]
        expected = A * u

        @testset "$(nameof(typeof(mapping)))" for mapping in (WorkerPerElement(), LanesPerElement())
            op = setup_operator(
                AssemblyStrategy(MatrixFreeAction(; element_mapping = mapping,
                                                  storage = BlockRowAssembly()),
                                 ColoredScheduling(), device), itg, dh)
            # One colour of every cell — the scatter-disjointness promise, not a
            # colouring algorithm's result.
            @test length(first(get_subdomain_caches(op)).partition) == 1
            @test length(only(first(get_subdomain_caches(op)).partition)) == getncells(Ferrite.get_grid(dh))

            y = CUDA.zeros(Float64, ndofs(dh))
            mul!(y, op, CuVector(u))
            @test Array(y) ≈ expected rtol = 1.0e-8

            # No atomics under the one colour, so the device action repeats exactly.
            z = CUDA.zeros(Float64, ndofs(dh))
            mul!(z, op, CuVector(u))
            @test Array(y) == Array(z)

            # The refill runs the HOST mirror — a real GPU has no device
            # `InterfaceValues` — and copies the store back down.
            update_operator!(op, nothing)
            fill!(y, 0.0)
            mul!(y, op, CuVector(u))
            @test Array(y) ≈ expected rtol = 1.0e-8
        end
    end

    @testset "a RateFormIntegrator fuses M⁻¹K on the device" begin
        dh  = dg_cuda_handler(Quadrilateral, (3, 3), 1)
        itg = dg_cuda_integrator(1)
        u   = Float64[sin(3.1i) + 0.2cos(i) for i in 1:ndofs(dh)]
        expected = dg_cuda_assembled(itg, dh) * u
        mass = SimpleBilinearMassIntegrator(1.7, QuadratureRuleCollection(2), :u)
        M = let op = setup_operator(AssemblyStrategy(SequentialCPUDevice()), mass, dh)
            update_operator!(op, nothing)
            op.A
        end
        fused = M \ expected
        strategy = AssemblyStrategy(MatrixFreeAction(; storage = BlockRowAssembly()),
                                    ColoredScheduling(), device)

        # The DENSE mass, block-inverted per cell into the store.
        op = setup_operator(strategy, RateFormIntegrator(itg, mass), dh)
        y = CUDA.zeros(Float64, ndofs(dh))
        mul!(y, op, CuVector(u))
        @test Array(y) ≈ fused rtol = 1.0e-8
        @test !isapprox(Array(y), expected; rtol = 1.0e-3)

        # The refill runs the HOST mirror and re-fuses before the upload.
        update_operator!(op, nothing)
        fill!(y, 0.0)
        mul!(y, op, CuVector(u))
        @test Array(y) ≈ fused rtol = 1.0e-8

        # The LUMPED mass: a diagonal element matrix, so the same store is
        # scaled row by row instead of block-solved.
        lumped = let lop = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                          RowSumLumped(mass), dh)
            update_operator!(lop, nothing)
            get_matrix(lop) \ expected
        end
        oplumped = setup_operator(strategy, RateFormIntegrator(itg, RowSumLumped(mass)), dh)
        z = CUDA.zeros(Float64, ndofs(dh))
        mul!(z, oplumped, CuVector(u))
        @test Array(z) ≈ lumped rtol = 1.0e-8

        # The per-cell inverse blocks as a device operator of their own — the
        # Cholesky inverse runs inside the fill kernel.
        element_arm = AssemblyStrategy(MatrixFreeAction(; storage = ElementAssembly()), ColoredScheduling(), device)
        minv = setup_operator(element_arm, ElementInverse(mass), dh)
        w = CUDA.zeros(Float64, ndofs(dh))
        mul!(w, minv, CuVector(u))
        @test Array(w) ≈ M \ u rtol = 1.0e-8

        # A dense mass composed with a cell-local rhs, both at the ELEMENT level.
        cell_rhs = SimpleBilinearDiffusionIntegrator(1.3, QuadratureRuleCollection(2), :u)
        Kcell = let op = setup_operator(AssemblyStrategy(SequentialCPUDevice()), cell_rhs, dh)
            update_operator!(op, nothing)
            get_matrix(op)
        end
        composed = setup_operator(element_arm, RateFormIntegrator(cell_rhs, mass), dh)
        fill!(w, 0.0)
        mul!(w, composed, CuVector(u))
        @test Array(w) ≈ M \ (Kcell * u) rtol = 1.0e-8
    end
end
