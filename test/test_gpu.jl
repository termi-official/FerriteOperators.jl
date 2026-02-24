using FerriteOperators
using Test
using SparseArrays
import LinearAlgebra: mul!, norm

# ---- Ferrite-native reference assembly (mirrors the classic heat equation tutorial) ----
function assemble_reference_K(cellvalues, K, dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    f  = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(Ke, 0)
        fill!(fe, 0)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                δu  = shape_value(cellvalues, q_point, i)
                ∇δu = shape_gradient(cellvalues, q_point, i)
                fe[i] += δu * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

# ---- Problem setup (same as classic Ferrite heat equation) ----
grid = generate_grid(Quadrilateral, (20, 20))

ip = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
cellvalues = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
∂Ω = union(
    getfacetset(grid, "left"),
    getfacetset(grid, "right"),
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
)
add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> 0))
close!(ch)

# ---- Reference: assemble K and f the classic Ferrite way ----
K_ref = allocate_matrix(dh)
K_ref, f_ref = assemble_reference_K(cellvalues, K_ref, dh)

@testset "Diffusion - CPU strategies" begin
    integrator = FerriteOperators.SimpleBilinearDiffusionIntegrator(
        1.0,
        QuadratureRuleCollection(2),
        :u,
    )

    # --- Sequential CPU ---
    @testset "SequentialAssemblyStrategy" begin
        op = setup_operator(SequentialAssemblyStrategy(SequentialCPUDevice()), integrator, dh)
        update_operator!(op, 0.0)
        @test op.A ≈ K_ref
    end

    # --- Per-color with Polyester(2) ---
    @testset "PerColorAssemblyStrategy(PolyesterDevice(2))" begin
        op = setup_operator(PerColorAssemblyStrategy(PolyesterDevice(2)), integrator, dh)
        update_operator!(op, 0.0)
        @test op.A ≈ K_ref
    end

    # --- Element Assembly (matrix-free) on CPU ---
    @testset "ElementAssemblyStrategy(SequentialCPUDevice())" begin
        op = setup_operator(ElementAssemblyStrategy(SequentialCPUDevice()), integrator, dh)
        update_operator!(op, 0.0)

        # op.A is an EAOperator (not a sparse matrix), so test via mul!
        x = collect(Float64, 1:ndofs(dh))
        y_ref = K_ref * x
        y = zeros(ndofs(dh))
        mul!(y, op.A, x)
        @test y ≈ y_ref
    end
end

# ---- GPU tests (requires CUDA.jl) ----
gpu_available = try
    using CUDA
    CUDA.functional()
catch
    false
end

if gpu_available
    @testset "Diffusion - GPU ElementAssembly" begin
        integrator = FerriteOperators.SimpleBilinearDiffusionIntegrator(
            1.0,
            QuadratureRuleCollection(2),
            :u,
        )

        @testset "ElementAssemblyStrategy(CudaDevice())" begin
            op = setup_operator(ElementAssemblyStrategy(CudaDevice()), integrator, dh)
            update_operator!(op, 0.0)  # assembly runs on GPU via KA kernel

            x = collect(Float64, 1:ndofs(dh))
            y_ref = K_ref * x

            x_gpu = CuArray(x)
            y_gpu = CUDA.zeros(Float64, ndofs(dh))
            mul!(y_gpu, op.A, x_gpu)
            @test Array(y_gpu) ≈ y_ref
        end
    end
else
    @info "CUDA not available, skipping CUDA GPU tests"
end

# ---- ROC/AMDGPU tests ----
roc_available = try
    using AMDGPU
    AMDGPU.functional()
catch
    false
end

if roc_available
    @testset "Diffusion - ROC ElementAssembly" begin
        integrator = FerriteOperators.SimpleBilinearDiffusionIntegrator(
            1.0,
            QuadratureRuleCollection(2),
            :u,
        )

        @testset "ElementAssemblyStrategy(RocDevice())" begin
            op = setup_operator(ElementAssemblyStrategy(RocDevice()), integrator, dh)
            update_operator!(op, 0.0)  # assembly runs on GPU via KA kernel

            x = collect(Float64, 1:ndofs(dh))
            y_ref = K_ref * x

            x_gpu = ROCArray(x)
            y_gpu = AMDGPU.zeros(Float64, ndofs(dh))
            mul!(y_gpu, op.A, x_gpu)
            @test Array(y_gpu) ≈ y_ref
        end
    end
else
    @info "AMDGPU not available, skipping ROC GPU tests"
end

# ---- Debug: 2×2 grid CPU ElementAssembly walkthrough ----
# Run with: julia --project -e 'include("test/test_gpu.jl")'
# Then inspect the variables printed below.

println("\n" * "="^70)
println("DEBUG: 2×2 grid CPU ElementAssembly walkthrough")
println("="^70)

# 1. Small grid setup
grid2 = generate_grid(Quadrilateral, (2, 2))
ip2 = Lagrange{RefQuadrilateral, 1}()
qr2 = QuadratureRule{RefQuadrilateral}(2)
cv2 = CellValues(qr2, ip2)

dh2 = DofHandler(grid2)
add!(dh2, :u, ip2)
close!(dh2)

println("\n--- Grid ---")
println("  ncells  = ", getncells(grid2))
println("  nnodes  = ", getnnodes(grid2))
println("  ndofs   = ", ndofs(dh2))

println("\n--- Cell DOFs (cell_id → global dof indices) ---")
for ci in 1:getncells(grid2)
    println("  cell $ci: dofs = ", celldofs(dh2, ci))
end

println("\n--- CellValues dimensions ---")
println("  nbasefuncs (nbf) = ", getnbasefunctions(cv2))
println("  nquadpoints (nqp) = ", getnquadpoints(cv2))
println("  ngeo (geometric basefuncs) = ", size(cv2.geo_mapping.dMdξ, 1))

println("\n--- CellValues shared immutable data ---")
println("  dNdξ  size = ", size(cv2.fun_values.dNdξ), "  (nbf × nqp)")
println("  dMdξ  size = ", size(cv2.geo_mapping.dMdξ), "  (ngeo × nqp)")
println("  weights    = ", cv2.qr.weights)

# 2. Reference assembly (standard Ferrite)
K2_ref = allocate_matrix(dh2)
K2_ref, _ = assemble_reference_K(cv2, K2_ref, dh2)
println("\n--- Reference K (9×9 sparse) ---")
println("  size = ", size(K2_ref))
println("  nnz  = ", nnz(K2_ref))

# 3. ElementAssembly operator setup
integrator2 = FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u)
op2 = setup_operator(ElementAssemblyStrategy(SequentialCPUDevice()), integrator2, dh2)

ea = op2.A  # EAOperator
println("\n--- EAOperator ---")
println("  nelements = ", FerriteOperators.getnelements(ea))

println("\n--- element_matrices (flat Kₑ storage) ---")
println("  data length = ", length(ea.element_matrices.data), "  (4 cells × 4×4 = 64)")
for ci in 1:4
    idx = ea.element_matrices.index_structure[ci]
    println("  cell $ci: offset=$(idx.offset), nrows=$(idx.nrows), ncols=$(idx.ncols)")
end

println("\n--- vector_element_map (DOF mapping) ---")
println("  data = ", ea.vector_element_map.data[1:16], "  (cell_dofs flat)")
for ci in 1:4
    idx = ea.vector_element_map.index_structure[ci]
    dofs = ea.vector_element_map.data[idx.offset:idx.offset+idx.length-1]
    println("  cell $ci: offset=$(idx.offset), len=$(idx.length) → dofs=$dofs")
end

# 4. Assembly
println("\n--- Assembly (update_operator!) ---")
update_operator!(op2, 0.0)
println("  element_matrices.data (first 16 = Kₑ of cell 1):")
Ke1_flat = ea.element_matrices.data[1:16]
Ke1 = reshape(copy(Ke1_flat), (4, 4))
display(Ke1)

println("\n  element_matrices.data (next 16 = Kₑ of cell 2):")
Ke2_flat = ea.element_matrices.data[17:32]
Ke2 = reshape(copy(Ke2_flat), (4, 4))
display(Ke2)

# 5. mul! test
x2 = collect(Float64, 1:ndofs(dh2))
y2_ref = K2_ref * x2
y2 = zeros(ndofs(dh2))
mul!(y2, op2.A, x2)

println("\n--- mul!(y, A, x) ---")
println("  x       = ", x2)
println("  y (EA)  = ", y2)
println("  y (ref) = ", y2_ref)
println("  match?  = ", y2 ≈ y2_ref)

println("\n" * "="^70)

# ---- Debug: trace execute_task_on_device! internals ----
println("\n" * "="^70)
println("DEBUG: execute_task_on_device! internals (Sequential CPU, ElementAssembly)")
println("="^70)

# Reconstruct the objects that execute_task_on_device! sees
strategy2 = op2.strategy
assembler2 = Ferrite.start_assemble(strategy2, op2.A)
task2 = FerriteOperators.AssembleBilinearTerm(assembler2)

subdomain_cache2 = op2.subdomain_caches[1]
task_cache2 = FerriteOperators.SubdomainAssemblyTaskBuffer(nothing, 0.0, subdomain_cache2)

# 1. get_items — what cells to iterate over
itemsets2 = FerriteOperators.get_items(task2, task_cache2)
println("\n--- get_items ---")
println("  type     = ", typeof(itemsets2))
println("  ngroups  = ", length(itemsets2))
for (gi, group) in enumerate(itemsets2)
    println("  group $gi: $(length(group)) cells → ", collect(group))
end

# 2. get_task_buffer — the per-thread scratch workspace
task_buffer2 = FerriteOperators.get_task_buffer(task2, task_cache2, 1)
println("\n--- get_task_buffer (chunkid=1) ---")
println("  type = ", typeof(task_buffer2))
println("  fields = ", propertynames(task_buffer2))
println("  Ke size   = ", size(FerriteOperators.query_element_matrix(task_buffer2)))
println("  element   = ", typeof(FerriteOperators.query_element(task_buffer2)))
println("  geometry  = ", typeof(FerriteOperators.query_geometry_cache(task_buffer2)))

# 3. Walk through the first 2 cells manually
println("\n--- Manual cell loop (first 2 cells) ---")
for taskid in collect(first(itemsets2))[1:2]
    reinit!(task_buffer2, taskid)
    cell = FerriteOperators.query_geometry_cache(task_buffer2)
    println("  cell $taskid: dofs = ", celldofs(cell), ", coords[1] = ", getcoordinates(cell)[1])
end

println("\n" * "="^70)
