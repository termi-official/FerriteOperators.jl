# `BlockRowAssembly()`'s CUDA arm (Slice 2 / G3). What only a real device run
# proves: the cursor's window/slot tables reach the kernel as device arrays
# (`Adapt.@adapt_structure CellNeighbourCursor` — invisible on `KA.CPU()`, where
# `adapt` is the identity), and the host mirror's `copyto!` refill, which on a
# CPU device is a no-op because the store is shared.
#
# The two-sided demo family is the fill source, `include`d once by
# `test_custom_iterator_family.jl`, which runs before this file.

@testset "CUDA block-row storage (BlockRowAssembly)" begin
    tb = jump_testbed()
    n  = ndofs_per_cell(first(tb.dh.subdofhandlers))
    u  = Float64[sin(3.1 * i) + 0.2cos(i) for i in 1:ndofs(tb.dh)]
    expected = reference_matrix(tb.dh, tb.prs, n) * u
    device = KernelAbstractionsDevice(CUDABackend(); index_type = Int32, items_per_worker = 1)

    @testset "$(nameof(typeof(mapping))) vs the host reference" for mapping in
            (WorkerPerElement(), LanesPerElement())

        op = setup_operator(
            AssemblyStrategy(MatrixFreeAction(; element_mapping = mapping,
                                              storage = BlockRowAssembly()),
                             ColoredScheduling(), device),
            tb.integrator, tb.dh)
        # One colour of every cell — the scatter-disjointness promise, not a
        # colouring algorithm's result.
        @test length(first(get_subdomain_caches(op)).partition) == 1
        @test length(first(get_subdomain_caches(op)).partition[1]) == getncells(tb.grid)

        y = CUDA.zeros(Float64, ndofs(tb.dh))
        mul!(y, op, CuVector(u))
        @test Array(y) ≈ expected rtol = 1.0e-5

        # No atomics under the one colour, so the device action repeats exactly.
        z = CUDA.zeros(Float64, ndofs(tb.dh))
        mul!(z, op, CuVector(u))
        @test Array(y) == Array(z)

        # The refill fills the HOST mirror and copies the store back to the
        # device; a refill that never reached the device would leave `y` right
        # only because the setup fill already did.
        update_operator!(op, nothing)
        fill!(y, 0.0)
        mul!(y, op, CuVector(u))
        @test Array(y) ≈ expected rtol = 1.0e-5
    end
end
