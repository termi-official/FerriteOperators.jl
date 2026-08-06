using FerriteOperators
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester
using TimerOutputs

@testset "QVector" begin
    import FerriteOperators: QVector, setup_qvector, get_range_for_cell

    # --- Basic construction and AbstractVector interface ---
    @testset "AbstractVector interface" begin
        data    = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        offsets = [1, 3, 5]   # cell 1 starts at 1, cell 2 at 3, cell 3 at 5
        npoints = [2, 2, 2]
        qv = QVector(data, offsets, npoints)
        @test length(qv) == 6
        @test qv[3] == 3.0
        @test eltype(qv) == Float64
        @test collect(qv) == data
    end

    # --- Per-cell view ---
    @testset "get_range_for_cell" begin
        data    = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        offsets = [1, 3, 5]
        npoints = [2, 2, 2]
        qv = QVector(data, offsets, npoints)
        @test get_range_for_cell(qv, 1) == [10.0, 20.0]
        @test get_range_for_cell(qv, 2) == [30.0, 40.0]
        @test get_range_for_cell(qv, 3) == [50.0, 60.0]
        # Mutations through the view affect the underlying data
        get_range_for_cell(qv, 2)[1] = 99.0
        @test qv[3] == 99.0
    end

    # --- setup_qvector from DofHandler + QuadratureRuleCollection ---
    @testset "setup_qvector from DofHandler + QRC" begin
        grid   = generate_grid(Hexahedron, (2, 2, 2))
        dh     = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron, 1}())
        close!(dh)
        qrc    = QuadratureRuleCollection(2)   # 2^3 = 8 QPs per hex cell
        ncells = getncells(grid)               # 8 cells

        qv = setup_qvector(Float64, dh, qrc)
        @test length(qv) == ncells * 8
        @test eltype(qv) == Float64
        for cellid in 1:ncells
            @test length(get_range_for_cell(qv, cellid)) == 8
        end
    end

    # --- setup_qvector from operator ---
    @testset "setup_qvector from operator" begin
        grid       = generate_grid(Hexahedron, (2, 2, 2))
        dh         = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron, 1}())
        close!(dh)
        qrc        = QuadratureRuleCollection(2)
        integrator = FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, qrc, :u)
        strategy   = SequentialAssemblyStrategy(SequentialCPUDevice())
        op         = setup_operator(strategy, integrator, dh)

        qv_op = setup_qvector(Float64, op)
        qv_dh = setup_qvector(Float64, dh, qrc)
        @test length(qv_op) == length(qv_dh)
        @test eltype(qv_op) == Float64
    end

    # --- Partial coverage: cell 1 outside every subdomain (0 QPs) must not throw ---
    @testset "setup_qvector cell 1 uncovered" begin
        grid = generate_grid(Hexahedron, (4, 1, 1))
        addcellset!(grid, "tail", Set([2, 3, 4]))
        dh  = DofHandler(grid)
        sdh = SubDofHandler(dh, getcellset(grid, "tail"))
        add!(sdh, :u, Lagrange{RefHexahedron, 1}())
        close!(dh)
        qrc = QuadratureRuleCollection(2)  # 8 QPs per covered hex

        qv = setup_qvector(Float64, dh, qrc)
        @test length(qv) == 3 * 8
        @test length(get_range_for_cell(qv, 1)) == 0
        for cellid in 2:4
            @test length(get_range_for_cell(qv, cellid)) == 8
        end
    end

    # --- Non-contiguous cell sets: offsets are still correct ---
    @testset "setup_qvector non-contiguous cellsets" begin
        grid = generate_grid(Hexahedron, (4, 1, 1))
        addcellset!(grid, "left",  x -> x[1] ≤ 0.0)  # cells not necessarily 1..2
        addcellset!(grid, "right", x -> x[1] ≥ 0.0)
        dh   = DofHandler(grid)
        sdh1 = SubDofHandler(dh, getcellset(grid, "left"))
        add!(sdh1, :u, Lagrange{RefHexahedron, 1}())
        sdh2 = SubDofHandler(dh, getcellset(grid, "right"))
        add!(sdh2, :u, Lagrange{RefHexahedron, 1}())
        close!(dh)
        qrc = QuadratureRuleCollection(2)  # 8 QPs per hex
        ncells = getncells(grid)

        qv = setup_qvector(Float64, dh, qrc)
        @test length(qv) == ncells * 8
        for cellid in 1:ncells
            @test length(get_range_for_cell(qv, cellid)) == 8
        end
    end
end
