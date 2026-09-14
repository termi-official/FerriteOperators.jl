# The custom-iterator demo's CUDA arm (Slice 0 / G5): both device leaks fixed
# in `../custom_iterator_family.jl` — `Adapt.@adapt_structure DevicePairCursor`
# and `JumpPenaltyCache` dropping its host `pairs` field — are invisible to
# `KernelAbstractionsDevice(KA.CPU())`, where `adapt` is the identity; only a
# real CUDA run proves either one.
include(joinpath(@__DIR__, "..", "custom_iterator_family.jl"))

@testset "CUDA custom item iterator (two-sided interface family)" begin
    tb = jump_testbed()
    n  = ndofs_per_cell(first(tb.dh.subdofhandlers))
    device = KernelAbstractionsDevice(CUDABackend(); index_type = Int32, items_per_worker = 1)
    reference = reference_matrix(tb.dh, tb.prs, n)

    @testset "assembling arm: FullAssembly -> CuSparseMatrixCSC" begin
        spec = StandardOperatorSpecification(; sparsity_entries = pair_sparsity(tb.prs),
            matrix_type = CuSparseMatrixCSC{Float64, Int32})
        op = setup_operator(
            AssemblyStrategy(device; form = FullAssembly(spec), scheduling = ColoredScheduling()),
            tb.integrator, tb.dh)
        @test op.A isa CuSparseMatrixCSC
        @test sum(length, first(get_subdomain_caches(op)).partition) == 9

        update_operator!(op, nothing)
        @test SparseMatrixCSC(op.A) ≈ reference rtol = 1.0e-5
    end

    @testset "matrix-free arm: MatrixFreeAction[Recompute]" begin
        u = Float64[sin(3.1 * i) + 0.2cos(i) for i in 1:ndofs(tb.dh)]
        expected = reference * u

        op = setup_operator(
            AssemblyStrategy(MatrixFreeAction(; storage = Recompute()), SequentialScheduling(), device),
            tb.integrator, tb.dh)
        @test length(first(get_subdomain_caches(op)).partition[1]) == 9

        y = CUDA.zeros(Float64, ndofs(tb.dh))
        mul!(y, op, CuVector(u))
        @test Array(y) ≈ expected rtol = 1.0e-5
    end
end
