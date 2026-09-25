# The shipped DG consumer: `SIPGDiffusionIntegrator`, the element that puts the
# two-window scatter (G2), `BlockRowAssembly()` (G3) and per-kind traversal
# resolution (G1) to work on a real interior-penalty form.
#
# The reference is built INDEPENDENTLY of the element: cells through
# `CellIterator`, interior facets through Ferrite's own `InterfaceIterator`, and
# each cell's volume term added once and whole. It therefore checks the SIPG
# terms and the `1/n_interior_facets` apportionment at the same time.

using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using LinearAlgebra
# FerriteKAExt — the device handler, `distribute_to_workers` and every `Adapt`
# rule the device kernel builds on — is triggered by these four together.
import Adapt, GPUArrays, GPUArraysCore
import KernelAbstractions as KA

const DG_D = 1.3
const DG_η = 4.0
const DG_ρ = 1.7

dg_integrator(order) = SIPGDiffusionIntegrator(
    DG_D, DG_η, QuadratureRuleCollection(order + 1),
    FerriteOperators.FacetQuadratureRuleCollection(order + 1), :u)

# Interior nodes pushed off the lattice: the cells are no longer parallelograms,
# so every facet of the mesh carries its own normal and measure.
function dg_handler(celltype, dims, order; distortion = 0.0)
    grid = generate_grid(celltype, dims)
    dim  = length(dims)
    h    = 2.0 / maximum(dims)
    if !iszero(distortion)
        nodes = [Ferrite.Node(Vec{dim}(ntuple(d -> node.x[d] +
                    (all(abs.(node.x) .< 1 - 1.0e-8) ? distortion * h * sin(2.7i + 1.3d) : 0.0), dim)))
                 for (i, node) in enumerate(Ferrite.getnodes(grid))]
        grid = Grid(Ferrite.getcells(grid), nodes)
    end
    dh = DofHandler(grid)
    add!(dh, :u, DiscontinuousLagrange{Ferrite.getrefshape(getcells(grid, 1)), order}())
    close!(dh)
    return dh
end

# The independent SIPG assembly. `order`/`qorder` are the element's own, so the
# only thing shared with it is the FORM — not one line of code.
function reference_sipg(dh, order, qorder)
    grid = Ferrite.get_grid(dh)
    ip   = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :u)
    rs   = Ferrite.getrefshape(ip)
    ipg  = Ferrite.geometric_interpolation(typeof(getcells(grid, 1)))
    cv   = CellValues(QuadratureRule{rs}(qorder), ip, ipg)
    iv   = InterfaceValues(FacetQuadratureRule{rs}(qorder), ip, ipg)
    d    = Ferrite.getrefdim(ip)
    A    = zeros(ndofs(dh), ndofs(dh))
    vol  = zeros(getncells(grid))

    for cell in CellIterator(dh)
        reinit!(cv, cell)
        dofs = celldofs(cell)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            vol[cellid(cell)] += dΩ
            for i in 1:getnbasefunctions(cv), j in 1:getnbasefunctions(cv)
                A[dofs[i], dofs[j]] += DG_D * (shape_gradient(cv, qp, j) ⋅ shape_gradient(cv, qp, i)) * dΩ
            end
        end
    end

    for ic in InterfaceIterator(dh, ExclusiveTopology(grid))
        reinit!(iv, ic)
        dofs = interfacedofs(ic)
        area = sum(getdetJdV(iv, qp) for qp in 1:getnquadpoints(iv))
        σ = DG_η * (order + 1) * (order + d) / d * area / min(vol[cellid(ic.a)], vol[cellid(ic.b)])
        for qp in 1:getnquadpoints(iv)
            dΓ = getdetJdV(iv, qp)
            n  = getnormal(iv, qp)
            for i in 1:getnbasefunctions(iv), j in 1:getnbasefunctions(iv)
                jᵢ = shape_value_jump(iv, qp, i) * (-n)
                aᵢ = shape_gradient_average(iv, qp, i)
                jⱼ = shape_value_jump(iv, qp, j) * (-n)
                aⱼ = shape_gradient_average(iv, qp, j)
                A[dofs[i], dofs[j]] += (σ * (jᵢ ⋅ jⱼ) - DG_D * (jᵢ ⋅ aⱼ) - DG_D * (aᵢ ⋅ jⱼ)) * dΓ
            end
        end
    end
    return A
end

dg_probe(dh) = Float64[sin(0.7i) + 0.2cos(1.3i) for i in 1:ndofs(dh)]
dg_action(op, u) = (y = zeros(length(u)); mul!(y, op, u); y)

dg_assembling_strategy() = AssemblyStrategy(SequentialCPUDevice();
    form = FullAssembly(StandardOperatorSpecification(; sparsity_entries = interior_facet_entries!)))

dg_sequential(storage; scheduling = ColoredScheduling()) =
    AssemblyStrategy(MatrixFreeAction(; storage), scheduling, SequentialCPUDevice())

dg_ka(mapping, storage; scheduling = ColoredScheduling()) = AssemblyStrategy(
    MatrixFreeAction(; element_mapping = mapping, storage), scheduling,
    KernelAbstractionsDevice(KA.CPU(); value_type = Float64, index_type = Int,
                             items_per_worker = 2, max_workgroup_size = 8))

dg_arms(storage; scheduling = ColoredScheduling()) = (
    "SequentialCPUDevice"     => dg_sequential(storage; scheduling),
    "KA.CPU WorkerPerElement" => dg_ka(WorkerPerElement(), storage; scheduling),
    "KA.CPU LanesPerElement"  => dg_ka(LanesPerElement(), storage; scheduling),
)

# `(label, DofHandler, order, quadrature order)` — 2D and 3D, affine and
# distorted, p = 1 and the p = 2 space `DiscontinuousLagrange` also provides.
dg_meshes() = (
    ("distorted quad p=1", dg_handler(Quadrilateral, (3, 3), 1; distortion = 0.18), 1, 2),
    ("quad p=2",           dg_handler(Quadrilateral, (3, 3), 2), 2, 3),
    ("distorted hex p=1",  dg_handler(Hexahedron, (2, 2, 2), 1; distortion = 0.18), 1, 2),
)

@testset "The DG diffusion element (SIPG)" begin
    meshes = dg_meshes()

    @testset "the assembled operator is the SIPG matrix — $label" for (label, dh, order, qorder) in meshes
        op = setup_operator(dg_assembling_strategy(), dg_integrator(order), dh)
        update_operator!(op, nothing)
        A = Matrix(op.A)
        @test A ≈ reference_sipg(dh, order, qorder) rtol = 1.0e-10
        # SIPG is symmetric, and a diffusion operator annihilates constants:
        # both volume and jump terms vanish on `u ≡ 1`.
        @test A ≈ A' rtol = 1.0e-12
        @test norm(A * ones(ndofs(dh))) < 1.0e-9 * norm(A)

        # The induced residual is the same form contracted with `u`, and it runs
        # the element's OTHER mandatory kernel.
        r = zeros(ndofs(dh))
        evaluate!(op, r, dg_probe(dh), nothing)
        @test r ≈ A * dg_probe(dh) rtol = 1.0e-10
    end

    @testset "the action matches the assembled reference — $label, $arm" for
            (label, dh, order, qorder) in meshes, (arm, strategy) in dg_arms(BlockRowAssembly())

        reference = reference_sipg(dh, order, qorder) * dg_probe(dh)
        op = setup_operator(strategy, dg_integrator(order), dh)
        @test dg_action(op, dg_probe(dh)) ≈ reference rtol = 1.0e-10
    end

    @testset "SequentialScheduling resolves the same action — $arm" for (arm, strategy) in
            dg_arms(BlockRowAssembly(); scheduling = SequentialScheduling())

        label, dh, order, qorder = first(meshes)
        reference = reference_sipg(dh, order, qorder) * dg_probe(dh)
        op = setup_operator(strategy, dg_integrator(order), dh)
        @test dg_action(op, dg_probe(dh)) ≈ reference rtol = 1.0e-10
    end

    @testset "the fill rides the interior facets, the action the cells" begin
        _, dh, order, _ = first(meshes)
        op = setup_operator(dg_sequential(BlockRowAssembly()), dg_integrator(order), dh)
        sc = first(get_subdomain_caches(op))
        # A 3×3 quad mesh has 12 interior facets against 9 cells; the action's
        # partition is ONE colour of every cell.
        @test sum(length, compute_partition(SequentialScheduling(),
                  item_provider(QuadratureDataKind(), sc.domain.element, sc.domain.sdh))) == 12
        @test length(sc.partition) == 1
        @test length(only(sc.partition)) == getncells(Ferrite.get_grid(dh))

        # A refill overwrites rather than accumulating.
        reference = reference_sipg(dh, order, 2) * dg_probe(dh)
        update_operator!(op, nothing)
        @test dg_action(op, dg_probe(dh)) ≈ reference rtol = 1.0e-10
    end

    @testset "the colored action repeats bit for bit — $arm" for (arm, strategy) in
            dg_arms(BlockRowAssembly())

        _, dh, order, _ = first(meshes)
        op = setup_operator(strategy, dg_integrator(order), dh)
        u = dg_probe(dh)
        y1, y2 = zeros(ndofs(dh)), zeros(ndofs(dh))
        mul!(y1, op, u)
        mul!(y2, op, u)
        @test y1 == y2
    end

    @testset "a RateFormIntegrator fuses M⁻¹K into the store" begin
        _, dh, order, qorder = first(meshes)
        u = dg_probe(dh)
        reference = reference_sipg(dh, order, qorder) * u
        mass = SimpleBilinearMassIntegrator(DG_ρ, QuadratureRuleCollection(qorder), :u)
        M = let op = setup_operator(AssemblyStrategy(SequentialCPUDevice()), mass, dh)
            update_operator!(op, nothing)
            op.A
        end
        fused = M \ reference
        rate  = RateFormIntegrator(dg_integrator(order), mass)

        @testset "$arm" for (arm, strategy) in dg_arms(BlockRowAssembly())
            op = setup_operator(strategy, rate, dh)
            @test dg_action(op, u) ≈ fused rtol = 1.0e-9
            # The unfused store is a different operator — the fusion is not a
            # no-op the tolerance above would hide.
            @test !isapprox(dg_action(op, u), reference; rtol = 1.0e-3)
        end

        # The DENSE mass is invertible cell by cell over this discontinuous
        # space, so the assembled arm block-solves the same rows and lands on
        # the same operator — a different realization, not a different term.
        @testset "FullAssembly block-solves the same rows" begin
            op = setup_operator(dg_assembling_strategy(), rate, dh)
            update_operator!(op, nothing)
            @test get_matrix(op) * u ≈ fused rtol = 1.0e-9
        end
    end

    @testset "a cell with no interior facet is rejected" begin
        dh = dg_handler(Quadrilateral, (1, 1), 1)
        err = @test_throws ArgumentError setup_operator(dg_assembling_strategy(), dg_integrator(1), dh)
        @test occursin("no interior facet", err.value.msg)
        @test occursin("VOLUME", err.value.msg)
    end

    # P0-1 (do/gpu-dg adversarial review): a subdomain-crossing interior facet
    # used to be silently treated as a boundary — dropped from BOTH sides, with
    # the wrong operator still symmetric and constant-annihilating, undetectable
    # from the matrix alone. Two `SubDofHandler`s over one DG field must now
    # reject loudly at setup instead.
    @testset "an interior facet whose neighbour lies outside the subdomain is rejected" begin
        grid = generate_grid(Quadrilateral, (4, 2))
        addcellset!(grid, "left",  x -> x[1] < 0.0)
        addcellset!(grid, "right", x -> x[1] ≥ 0.0)
        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, getcellset(grid, "left"))
        add!(sdh1, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
        sdh2 = SubDofHandler(dh, getcellset(grid, "right"))
        add!(sdh2, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
        close!(dh)

        err = @test_throws ArgumentError setup_operator(dg_assembling_strategy(), dg_integrator(1), dh)
        @test occursin("SubDofHandler", err.value.msg)
        @test occursin("subdomain-crossing", err.value.msg)

        # The single-subdomain case (every shipped test) is unaffected.
        single = dg_handler(Quadrilateral, (4, 2), 1)
        op = setup_operator(dg_assembling_strategy(), dg_integrator(1), single)
        update_operator!(op, nothing)
        @test op.A !== nothing
    end
end

####################################
## Convergence
####################################

# `-D Δu + u = f` on the unit square with homogeneous Neumann data, so the
# interior-penalty form needs no boundary term and the problem is nonsingular.
# `u = cos(πx) cos(πy)` satisfies `∇u ⋅ n = 0` on every side.
dg_exact(x) = cos(π * x[1]) * cos(π * x[2])
dg_source(x) = (2 * DG_D * π^2 + 1) * dg_exact(x)

function dg_l2_error(n, order)
    grid = generate_grid(Quadrilateral, (n, n), Vec((0.0, 0.0)), Vec((1.0, 1.0)))
    dh   = DofHandler(grid)
    add!(dh, :u, DiscontinuousLagrange{RefQuadrilateral, order}())
    close!(dh)
    qorder = 2order + 2

    K = let op = setup_operator(dg_assembling_strategy(), dg_integrator(order), dh)
        update_operator!(op, nothing); op.A
    end
    M = let op = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                SimpleBilinearMassIntegrator(1.0, QuadratureRuleCollection(qorder), :u), dh)
        update_operator!(op, nothing); op.A
    end

    ip  = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :u)
    cv  = CellValues(QuadratureRule{RefQuadrilateral}(qorder), ip,
                     Ferrite.geometric_interpolation(Quadrilateral))
    f = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        dofs = celldofs(cell)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            fq = dg_source(spatial_coordinate(cv, qp, getcoordinates(cell)))
            for i in 1:getnbasefunctions(cv)
                f[dofs[i]] += fq * shape_value(cv, qp, i) * dΩ
            end
        end
    end

    uₕ = (K + M) \ f
    err = 0.0
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        uₑ = uₕ[celldofs(cell)]
        for qp in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, qp, getcoordinates(cell))
            err += (function_value(cv, qp, uₑ) - dg_exact(x))^2 * getdetJdV(cv, qp)
        end
    end
    return sqrt(err)
end

@testset "SIPG converges at O(h^(p+1)) in L2 — p = $order" for order in 1:2
    errors = [dg_l2_error(n, order) for n in (4, 8, 16)]
    orders = log2.(errors[1:(end - 1)] ./ errors[2:end])
    # A coarse sanity check with a loose gate, not a rigorous study: the
    # asymptotic rate is `p + 1`, and a wrong penalty sign or a lost face term
    # costs at least a full order.
    @test all(>(order + 0.6), orders)
end
