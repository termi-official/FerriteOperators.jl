using FerriteOperators
using FerriteOperatorsExampleElements
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester
using TimerOutputs

include(joinpath(@__DIR__, "fixture_elements.jl"))

@testset "Quadrature Data Processing" begin

    grid       = generate_grid(Hexahedron, (2, 2, 2))
    dh         = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    qrc        = QuadratureRuleCollection(2)
    integrator = SimpleBilinearDiffusionIntegrator(1.0, qrc, :u)
    strategy   = AssemblyStrategy(SequentialCPUDevice())

    # --- Result is consistent with manual per-cell indexing ---
    @testset "evaluate_quadrature! consistent with per-cell access" begin
        qop = setup_operator(strategy, integrator, dh)
        q   = setup_qvector(Float64, dh, qrc)
        u   = zeros(ndofs(dh))

        # f stores QP index (1..nqp) in each slot, so the probe sees the
        # within-cell ordering and not just the value it stored
        evaluate_quadrature!(q, qop, u, nothing,
            (ue, qp, cell, element_cache, pe, ctx) -> qp
        )

        nqp = getnquadpoints(QuadratureRule{RefHexahedron}(2))
        for cellid in 1:getncells(grid)
            @test get_range_for_cell(q, cellid) == collect(1:nqp)
        end
    end

    # --- The sweep's context reaches the per-quadrature-point kernel ---
    @testset "evaluate_quadrature! carries the context" begin
        qop = setup_operator(strategy, integrator, dh)
        q   = setup_qvector(Float64, dh, qrc)
        u   = zeros(ndofs(dh))
        f   = (ue, qp, cell, element_cache, pe, ctx) -> ctx === nothing ? -1.0 : evaluation_time(ctx)

        # Without one the kernel sees `nothing` — the stationary evaluation.
        evaluate_quadrature!(q, qop, u, nothing, f)
        @test all(==(-1.0), q.data)

        # With one it reads the time exactly as an element kernel does.
        evaluate_quadrature!(q, qop, u, nothing, f; ctx = TimeIntegrationContext(2.5, 0.1, 1.0))
        @test all(==(2.5), q.data)

        # The cell-set restriction and the context are independent.
        evaluate_quadrature!(q, qop, u, nothing, f, Set([1]); ctx = TimeIntegrationContext(4.0, 0.1, 1.0))
        @test all(==(4.0), get_range_for_cell(q, 1))
        @test all(==(2.5), get_range_for_cell(q, 2))
    end

    # --- Polyester (threaded) device produces the same result ---
    @testset "PolyesterDevice consistency" begin
        strategy_seq = AssemblyStrategy(SequentialCPUDevice())
        strategy_par = AssemblyStrategy(PolyesterDevice(4); scheduling = ColoredScheduling())
        qop_seq = setup_operator(strategy_seq, integrator, dh)
        qop_par = setup_operator(strategy_par, integrator, dh)
        q_seq   = setup_qvector(Float64, dh, qrc)
        q_par   = setup_qvector(Float64, dh, qrc)
        u       = zeros(ndofs(dh))

        evaluate_quadrature!(q_seq, qop_seq, u, nothing,
            (ue, qp, cell, element_cache, pe, ctx) -> Float64(cellid(cell)))
        evaluate_quadrature!(q_par, qop_par, u, nothing,
            (ue, qp, cell, element_cache, pe, ctx) -> Float64(cellid(cell)))
        @test q_seq == q_par
    end
end

@testset "VTKQuadratureFile" begin
    import FerriteOperators: VTKQuadratureFile, VTKQuadratureGrid, write_quadrature_data,
                              QuadratureDataQuery, prepare_quadrature_query, process_query!

    grid       = generate_grid(Hexahedron, (2, 2, 2))
    dh         = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    qrc        = QuadratureRuleCollection(2)
    integrator = SimpleHyperelasticityIntegrator(NeoHookean(10.0, 0.3), qrc, :u)
    strategy   = AssemblyStrategy(SequentialCPUDevice())
    qop        = setup_operator(strategy, integrator, dh)
    q          = setup_qvector(Float64, dh, qrc)
    u          = zeros(ndofs(dh))
    evaluate_quadrature!(q, qop, u, nothing,
        (ue, qp, cell, element_cache, pe, ctx) -> Float64(cellid(cell)))

    # --- VTKQuadratureGrid is constructable from (dh, qrc) ---
    @testset "VTKQuadratureGrid construction" begin
        qgrid = VTKQuadratureGrid(dh, qrc)
        @test Ferrite.getnnodes(qgrid) == length(q)   # one "node" per QP
    end

    mktempdir() do tmpdir
        qgrid = VTKQuadratureGrid(dh, qrc)
        path  = joinpath(tmpdir, "test_qp")

        # --- do-block syntax, analogous to VTKGridFile ---
        @testset "do-block creates and closes file" begin
            VTKQuadratureFile(path, qgrid) do vtk
                write_quadrature_data(vtk, q, "cell_id")
            end
            @test isfile(path * ".vtu")
        end

        # --- write_quadrature_data accepts Vec{3} data ---
        @testset "write_quadrature_data with Vec{3}" begin
            qv = setup_qvector(Vec{3, Float64}, dh, qrc)
            evaluate_quadrature!(qv, qop, u, nothing,
                (ue, qp, cell, element_cache, pe, ctx) -> Vec{3}(x -> Float64(cellid(cell))))
            VTKQuadratureFile(joinpath(tmpdir, "vec_data"), qgrid) do vtk
                write_quadrature_data(vtk, qv, "coords")
            end
            @test isfile(joinpath(tmpdir, "vec_data.vtu"))
        end

        # --- write_quadrature_data accepts a QuadratureDataQuery directly ---
        @testset "write_quadrature_data from QuadratureDataQuery" begin
            query = prepare_quadrature_query(Float64, qop)
            process_query!(query, qop, u, nothing,
                (ue, qp, cell, element_cache, pe, ctx) -> Float64(cellid(cell)))
            VTKQuadratureFile(joinpath(tmpdir, "from_query"), qgrid) do vtk
                write_quadrature_data(vtk, query, "cell_id")
            end
            @test isfile(joinpath(tmpdir, "from_query.vtu"))
        end
    end
end

@testset "QuadratureDataQuery" begin
    import FerriteOperators: QuadratureDataQuery, prepare_quadrature_query, process_query!

    grid       = generate_grid(Hexahedron, (4, 1, 1))
    addcellset!(grid, "left",  x -> x[1] ≤ 0.0)
    addcellset!(grid, "right", x -> x[1] ≥ 0.0)
    dh         = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    qrc        = QuadratureRuleCollection(2)
    integrator = SimpleBilinearDiffusionIntegrator(1.0, qrc, :u)
    strategy   = AssemblyStrategy(SequentialCPUDevice())
    qop        = setup_operator(strategy, integrator, dh)
    u          = zeros(ndofs(dh))
    f_cellid   = (ue, qp, cell, element_cache, pe, ctx) -> Float64(cellid(cell))

    # --- prepare_quadrature_query builds a QVector-backed query ---
    @testset "prepare_quadrature_query" begin
        query = prepare_quadrature_query(Float64, qop)
        @test query.buffer isa QVector{Float64}
        @test length(query.buffer) == getncells(grid) * 8   # 4 cells × 8 QPs each
        @test query.set === nothing                          # no filter by default
    end

    # --- process_query! respects the cell-set filter ---
    @testset "process_query! with cell-set filter" begin
        left_cells = getcellset(grid, "left")
        query = prepare_quadrature_query(Float64, qop; set = left_cells)
        @test query.set === left_cells
        process_query!(query, qop, u, nothing, f_cellid)
        # Cells in the left set must be filled; cells outside must remain zero
        for cellid_val in 1:getncells(grid)
            expected = cellid_val ∈ left_cells ? Float64(cellid_val) : 0.0
            @test all(==(expected), get_range_for_cell(query.buffer, cellid_val))
        end
    end

    # --- prepare_quadrature_query from a prototype reuses the layout ---
    @testset "prepare_quadrature_query from prototype" begin
        proto  = prepare_quadrature_query(Float64, qop)
        query2 = prepare_quadrature_query(Vec{3, Float64}, proto)
        @test query2.buffer isa QVector{Vec{3, Float64}}
        @test length(query2.buffer) == length(proto.buffer)
    end
end
