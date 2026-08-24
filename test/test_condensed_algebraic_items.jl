using FerriteOperators
using FerriteOperatorsExampleElements
using Test

# Condensed internal state on ALGEBRAIC items needs a DofHandler that can
# carry dofs outside the mesh (Ferrite's mesh-free algebraic variables), same
# capability gate as `test_algebraic_items.jl`.
if !isdefined(Ferrite, :AlgebraicVariable)
    @info "Skipping the condensed algebraic-item tests: this Ferrite has no `AlgebraicVariable`, " *
          "so a DofHandler cannot carry dofs outside the mesh."
else

    include(joinpath(@__DIR__, "fixture_elements.jl"))

    sequential_strategy() = SequentialAssemblyStrategy(SequentialCPUDevice())

    # A condensed algebraic cache that records the local tolerance its
    # condensation was asked for and reports the convergence verdict a test
    # dictates. Its local problem is a closed-form assignment; only what
    # reaches `condense_algebraic!` and what leaves it in the report matter.
    struct ProbeItemCache
        seen::Base.RefValue{Any}
        converged::Base.RefValue{Bool}
    end
    ProbeItemCache(; converged = true) = ProbeItemCache(Ref{Any}(:unset), Ref(converged))
    FerriteOperators.duplicate_for_device(device, c::ProbeItemCache) = c
    FerriteOperators.has_internal_state(::Type{ProbeItemCache}) = true
    FerriteOperators.internal_state_insensitive(::Type{ProbeItemCache}, kind) = true
    FerriteOperators.get_number_of_internal_dofs_per_algebraic_item(m, ::ProbeItemCache, items) =
        fill(1, length(items))
    function FerriteOperators.assemble_algebraic!(req::ResidualRequest, ::ProbeItemCache, args::AlgebraicArgs)
        req.r[1] += args.states.u[1] - args.states.q[1]
        return nothing
    end
    function FerriteOperators.condense_algebraic!(c::ProbeItemCache, args::AlgebraicArgs, weights::NamedTuple)
        c.seen[] = local_solve_tolerance(args.ctx)
        args.states.q[1] = stage_scaling(args.ctx) * args.states.u[1]
        return CondensationReport(c.converged[], 1, 0, 0, -args.item.index, 0, 0.0, 1.0)
    end

    struct ProbeItemIntegrator{C} <: AbstractNonlinearIntegrator
        cell_integrator::C
        cache::ProbeItemCache
    end
    FerriteOperators.setup_element_cache(m::ProbeItemIntegrator, sdh::SubDofHandler) =
        FerriteOperators.setup_element_cache(m.cell_integrator, sdh)
    FerriteOperators.algebraic_items(::ProbeItemIntegrator, dh::DofHandler) =
        [[only(algebraic_dofs(dh, :p1))]]
    FerriteOperators.setup_algebraic_cache(m::ProbeItemIntegrator, dh::DofHandler) = m.cache

    function probe_item_testbed(strategy, cell_integrator, cache)
        grid = generate_grid(Quadrilateral, (2, 2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        add!(dh, :p1, AlgebraicVariable())
        close!(dh)
        op = setup_operator(strategy, ProbeItemIntegrator(cell_integrator, cache), dh; slots = (:u, :q, :qprev))
        u  = zeros(unknown_size(op))
        view(u, 1:ndofs(dh)) .= 0.4
        return (; op, dh, states = condensed_states(u, zeros(unknown_size(op))))
    end

    @testset "Condensed algebraic item alone" begin
        strategy = sequential_strategy()
        qrc      = QuadratureRuleCollection(2)
        chamber  = ChamberRelaxationParameters(β = 1.3, τ = 0.9)
        tb       = chamber_testbed(strategy, qrc; params = chamber)
        op, dh   = tb.op, tb.dh
        n        = unknown_size(op)
        γ̃        = 0.5
        ctx      = TimeIntegrationContext(0.0, γ̃, γ̃)

        u = zeros(n)
        view(u, 1:ndofs(dh)) .= 0.3 .* sin.(0.6 .* (1:ndofs(dh)))
        u[tb.item_dofs] .= (0.4, -0.2)
        uprev = zeros(n)
        states = condensed_states(u, uprev)

        report = condense_internal!(op, states, nothing, ctx)
        @test report.converged
        @test report.solves == 2   # two items, one closed-form solve each

        r = zeros(residual_size(op))
        evaluate!(op, r, states, nothing, ctx)
        rr = zeros(residual_size(op))
        update_linearization!(op, rr, states, nothing, ctx)
        @test r ≈ rr

        (; β, τ) = chamber
        k = γ̃ * β / τ
        for dof in tb.item_dofs
            p = u[dof]
            q_ref = k * p / (1 + k)   # q₀ = 0, closed-form q = (q₀ + k·p)/(1+k)
            @test r[dof] ≈ β * (p - q_ref) rtol = 1e-12
            @test op.J[dof, dof] ≈ β / (1 + k) rtol = 1e-12
        end
        # The two items are independent (no shared dofs, no coupling declared):
        # each item's row touches only its own diagonal entry.
        @test op.J[tb.item_dofs[1], tb.item_dofs[2]] == 0.0
        @test op.J[tb.item_dofs[2], tb.item_dofs[1]] == 0.0
    end

    @testset "check_derivatives on the condensed-item operator" begin
        strategy = sequential_strategy()
        qrc      = QuadratureRuleCollection(2)
        chamber  = ChamberRelaxationParameters(β = 0.8, τ = 1.4)
        tb       = chamber_testbed(strategy, qrc; params = chamber)
        op, dh   = tb.op, tb.dh
        n        = unknown_size(op)
        ctx      = TimeIntegrationContext(0.0, 0.6, 0.6)

        u = zeros(n)
        view(u, 1:ndofs(dh)) .= 0.2 .* sin.(0.5 .* (1:ndofs(dh)))
        u[tb.item_dofs] .= (0.3, -0.5)
        uprev = zeros(n)
        states = condensed_states(u, uprev)

        # check_derivatives condenses internally at every trial point it
        # probes, so the FD referee is a total — exactly what the analytic
        # `Consistent` kernel (reading the stored `dq/dp`) computes, and what
        # the analytic `ParameterJacobianRequest` kernel (reading `dq/dβ`)
        # computes for ∂F/∂θ.
        res = check_derivatives(op, states, chamber, ctx)
        @test res.passed
        @test res.checks.jacobian.passed
        @test res.checks.jacobian.skipped === nothing
        @test res.checks.parameter_jacobian.passed
        @test res.checks.parameter_jacobian.skipped === nothing
    end

    @testset "Freshness: never-condensed item throws, rollback invalidates, re-condense heals" begin
        strategy = sequential_strategy()
        tb       = chamber_testbed(strategy, QuadratureRuleCollection(2))
        op, dh   = tb.op, tb.dh
        n        = unknown_size(op)

        u = zeros(n)
        view(u, 1:ndofs(dh)) .= 0.1 .* sin.(1:ndofs(dh))
        u[tb.item_dofs] .= (0.2, -0.1)

        # The same four-step contract the condensed CELL family carries, with
        # the rejection naming the item instead of the cell.
        check_freshness_contract(op, condensed_states(u, zeros(n)), u,
                                 TimeIntegrationContext(0.0, 0.4, 0.4); names = "item 1")
    end

    @testset "The layout-collision proof: condensed cell AND condensed item in one operator" begin
        strategy = sequential_strategy()
        qrc      = QuadratureRuleCollection(2)
        mat      = NortonRelaxationParameters()
        chamber  = ChamberRelaxationParameters(β = 1.1, τ = 0.7)
        γ̃        = 0.5
        ctx      = TimeIntegrationContext(0.0, γ̃, γ̃)

        # Three testbeds sharing the same `:u` mesh/interpolation, so `:u`'s
        # dof numbering is identical across all three (algebraic dofs are
        # numbered after every field's cell dofs, and adding `:p1`/`:p2` to a
        # DofHandler does not renumber `:u`).
        tb_cell = relaxation_testbed(strategy, qrc; material = mat)
        tb_item = chamber_testbed(strategy, qrc; params = chamber)
        tb_both = chamber_testbed(strategy, qrc;
                                  cell_integrator = SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q),
                                  params = chamber)
        @test tb_item.item_dofs == tb_both.item_dofs

        n_u = ndofs(tb_cell.dh)
        u_field = 0.3 .* sin.(0.6 .* (1:n_u))
        p_vals  = (0.4, -0.2)

        u_cell = zeros(unknown_size(tb_cell.op)); view(u_cell, 1:n_u) .= u_field
        u_item = zeros(unknown_size(tb_item.op)); view(u_item, 1:n_u) .= u_field
        u_item[tb_item.item_dofs] .= p_vals
        u_both = zeros(unknown_size(tb_both.op)); view(u_both, 1:n_u) .= u_field
        u_both[tb_both.item_dofs] .= p_vals

        uprev_cell = zeros(unknown_size(tb_cell.op))
        uprev_item = zeros(unknown_size(tb_item.op))
        uprev_both = zeros(unknown_size(tb_both.op))

        states_cell = condensed_states(u_cell, uprev_cell)
        states_item = condensed_states(u_item, uprev_item)
        states_both = condensed_states(u_both, uprev_both)

        report_cell = condense_internal!(tb_cell.op, states_cell, nothing, ctx)
        report_item = condense_internal!(tb_item.op, states_item, nothing, ctx)
        report_both = condense_internal!(tb_both.op, states_both, nothing, ctx)
        @test report_both.converged
        @test report_both.solves == report_cell.solves + report_item.solves

        r_cell = zeros(residual_size(tb_cell.op)); evaluate!(tb_cell.op, r_cell, states_cell, nothing, ctx)
        r_item = zeros(residual_size(tb_item.op)); evaluate!(tb_item.op, r_item, states_item, nothing, ctx)
        r_both = zeros(residual_size(tb_both.op)); evaluate!(tb_both.op, r_both, states_both, nothing, ctx)

        rr_cell = zeros(residual_size(tb_cell.op)); update_linearization!(tb_cell.op, rr_cell, states_cell, nothing, ctx)
        rr_item = zeros(residual_size(tb_item.op)); update_linearization!(tb_item.op, rr_item, states_item, nothing, ctx)
        rr_both = zeros(residual_size(tb_both.op)); update_linearization!(tb_both.op, rr_both, states_both, nothing, ctx)

        # Neither block corrupts the other: the combined operator's cell
        # block reproduces the cell-alone reference, and its item block
        # reproduces the item-alone reference — the `[ū | q_cells | q_items]`
        # numbering composes cleanly.
        @test r_both[1:n_u] ≈ r_cell
        @test r_both[tb_both.item_dofs] ≈ r_item[tb_item.item_dofs]
        @test Matrix(tb_both.op.J)[1:n_u, 1:n_u] ≈ Matrix(tb_cell.op.J)
        @test Matrix(tb_both.op.J)[tb_both.item_dofs, tb_both.item_dofs] ≈
              Matrix(tb_item.op.J)[tb_item.item_dofs, tb_item.item_dofs]
        # No coupling was declared between the cell and the item dofs, so the
        # cross block is exactly zero — the layout adds no spurious entries.
        @test all(iszero, Matrix(tb_both.op.J)[1:n_u, tb_both.item_dofs])
        @test all(iszero, Matrix(tb_both.op.J)[tb_both.item_dofs, 1:n_u])
    end

    @testset "∂F/∂q over both families in one rectangular target" begin
        strategy = sequential_strategy()
        qrc      = QuadratureRuleCollection(2)
        mat      = NortonRelaxationParameters()
        chamber  = ChamberRelaxationParameters(β = 1.1, τ = 0.7)
        γ̃        = 0.5
        ctx      = TimeIntegrationContext(0.0, γ̃, γ̃)

        tb = chamber_testbed(strategy, qrc;
                             cell_integrator = SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q),
                             params = chamber)
        op, dh, grid = tb.op, tb.dh, tb.grid
        ivh = op.engine.ivh
        n   = unknown_size(op)
        u   = zeros(n)
        view(u, 1:ndofs(dh)) .= 0.3 .* sin.(0.6 .* (1:ndofs(dh)))
        u[tb.item_dofs] .= (0.4, -0.2)
        uprev  = zeros(n)
        states = condensed_states(u, uprev)
        condense_internal!(op, states, nothing, ctx)

        Kq = allocate_internal_jacobian(op)
        update_internal_jacobian!(Kq, op, states, nothing, ctx)
        @test size(Kq) == (residual_size(op), ndofs(ivh))

        # Both column blocks are populated, and they sit where
        # `[ū | q_cells | q_items]` puts them: the item block after the cell
        # block, each item's column carrying only its own row.
        for (index, dof) in pairs(tb.item_dofs)
            range = internal_variable_range(ivh, FerriteOperators.AlgebraicItem(index, [dof]))
            col = only(range) - residual_size(op)
            @test Kq[dof, col] ≈ -chamber.β rtol = 1e-12
            @test count(!iszero, Kq[:, col]) == 1
        end
        for cellid in 1:getncells(grid)
            range = internal_variable_range(ivh, cellid)
            for col in (range .- residual_size(op))
                @test any(!iszero, Kq[:, col])
                # A cell's column touches its own dofs only — never an item's.
                @test all(iszero, Kq[tb.item_dofs, col])
            end
        end

        # Columns of the assembled block difference the residual w.r.t. the
        # tail, whichever family owns the entry.
        check_internal_jacobian_columns(Kq, op, u, uprev, nothing, ctx)
    end

    @testset "Admissibility: condensed algebraic cache without analytic coverage" begin
        # A cache serving only the mandatory residual, declaring
        # `has_internal_state`: no generic AD `Consistent` bootstrap exists
        # for the algebraic-item family (see `condense_algebraic!`), so
        # `setup_operator` rejects it at setup, naming `assemble_algebraic!`.
        struct NonAnalyticChamberCache
            correctors::ItemStates{Float64}
        end
        FerriteOperators.has_internal_state(::Type{NonAnalyticChamberCache}) = true
        FerriteOperators.assemble_algebraic!(::ResidualRequest, ::NonAnalyticChamberCache, args) = nothing
        FerriteOperators.duplicate_for_device(device, c::NonAnalyticChamberCache) = c
        FerriteOperators.get_number_of_internal_dofs_per_algebraic_item(m, ::NonAnalyticChamberCache, items) =
            fill(1, length(items))
        FerriteOperators.condense_algebraic!(::NonAnalyticChamberCache, args, weights) =
            CondensationReport(true, 1, 0, 0, -args.item.index, 0, 0.0, 1.0)

        struct NonAnalyticChamberIntegrator <: AbstractNonlinearIntegrator end
        FerriteOperators.setup_element_cache(::NonAnalyticChamberIntegrator, sdh::SubDofHandler) =
            FerriteOperators.EmptyVolumetricElementCache()
        FerriteOperators.algebraic_items(::NonAnalyticChamberIntegrator, dh::DofHandler) =
            [[only(algebraic_dofs(dh, :p1))]]
        FerriteOperators.setup_algebraic_cache(::NonAnalyticChamberIntegrator, dh::DofHandler) =
            NonAnalyticChamberCache(ItemStates{Float64}(1))

        grid = generate_grid(Quadrilateral, (1, 1))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        add!(dh, :p1, AlgebraicVariable())
        close!(dh)

        err = @test_throws ArgumentError setup_operator(sequential_strategy(), NonAnalyticChamberIntegrator(), dh)
        @test occursin("assemble_algebraic!", err.value.msg)
        @test occursin("internal_state_insensitive", err.value.msg)
        # The cell-family escape does not apply here.
        @test !occursin("condensed_corrector", err.value.msg)
    end

    @testset "Internal dof counts per item must be uniform" begin
        # A cache whose internal-dof count is whatever the integrator says, so
        # the declaration can be made ragged. `internal_state_insensitive`
        # carries it past the admissibility check; the layout check is what is
        # under test.
        struct RaggedChamberCache
            counts::Vector{Int}
        end
        FerriteOperators.has_internal_state(::Type{RaggedChamberCache}) = true
        FerriteOperators.internal_state_insensitive(::Type{RaggedChamberCache}, kind) = true
        FerriteOperators.assemble_algebraic!(::ResidualRequest, ::RaggedChamberCache, args) = nothing
        FerriteOperators.duplicate_for_device(device, c::RaggedChamberCache) = c
        FerriteOperators.get_number_of_internal_dofs_per_algebraic_item(m, c::RaggedChamberCache, items) = c.counts
        FerriteOperators.condense_algebraic!(::RaggedChamberCache, args, weights) =
            CondensationReport(true, 1, 0, 0, -args.item.index, 0, 0.0, 1.0)

        struct RaggedChamberIntegrator <: AbstractNonlinearIntegrator
            counts::Vector{Int}
        end
        FerriteOperators.setup_element_cache(::RaggedChamberIntegrator, sdh::SubDofHandler) =
            FerriteOperators.EmptyVolumetricElementCache()
        FerriteOperators.algebraic_items(::RaggedChamberIntegrator, dh::DofHandler) =
            [[only(algebraic_dofs(dh, :p1))], [only(algebraic_dofs(dh, :p2))]]
        FerriteOperators.setup_algebraic_cache(m::RaggedChamberIntegrator, dh::DofHandler) =
            RaggedChamberCache(m.counts)

        grid = generate_grid(Quadrilateral, (1, 1))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        add!(dh, :p1, AlgebraicVariable())
        add!(dh, :p2, AlgebraicVariable())
        close!(dh)

        # Two items owning different numbers of internal dofs cannot share one
        # set of fixed-size local buffers.
        err = @test_throws ArgumentError setup_operator(sequential_strategy(), RaggedChamberIntegrator([1, 2]), dh)
        @test occursin("not uniform", err.value.msg)
        # A hook answering for the wrong number of items is its own rejection.
        err = @test_throws ArgumentError setup_operator(sequential_strategy(), RaggedChamberIntegrator([1]), dh)
        @test occursin("get_number_of_internal_dofs_per_algebraic_item", err.value.msg)
        # A uniform declaration passes the same route.
        @test setup_operator(sequential_strategy(), RaggedChamberIntegrator([2, 2]), dh) isa
            FerriteOperators.AbstractNonlinearOperator
    end

    @testset "The sweep's context reaches condense_algebraic! undecorated" begin
        strategy = sequential_strategy()
        qrc      = QuadratureRuleCollection(2)
        base     = TimeIntegrationContext(0.0, 0.5, 0.5)

        cache = ProbeItemCache()
        tb = probe_item_testbed(strategy, PlainPoissonIntegrator(qrc, :u), cache)

        @test condense_internal!(tb.op, tb.states, nothing, base).converged
        @test cache.seen[] === nothing   # a plain context requests no tolerance

        cache.seen[] = :unset
        @test condense_internal!(tb.op, tb.states, nothing, InexactLocalSolveContext(base, 1.0e-3)).converged
        @test cache.seen[] == 1.0e-3
    end

    @testset "CondensationReport.converged merges across the two families" begin
        strategy = sequential_strategy()
        qrc      = QuadratureRuleCollection(2)
        mat      = NortonRelaxationParameters()
        ctx      = TimeIntegrationContext(0.0, 0.5, 0.5)

        converging = SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q)
        # An iteration budget of zero: the local problem is left at its start
        # value and reported as not converged, never thrown.
        stalling = SimpleCondensedPowerLawRelaxation(mat, qrc, :u, :q;
                                                     local_solver = LocalNewtonSettings(max_iterations = 0))

        both_ok = probe_item_testbed(strategy, converging, ProbeItemCache())
        @test condense_internal!(both_ok.op, both_ok.states, mat, ctx).converged

        # The item family alone fails …
        item_fails = probe_item_testbed(strategy, converging, ProbeItemCache(; converged = false))
        @test !condense_internal!(item_fails.op, item_fails.states, mat, ctx).converged

        # … and the cell family alone fails: the merge is not one family's
        # verdict overwriting the other's.
        cell_fails = probe_item_testbed(strategy, stalling, ProbeItemCache())
        @test !condense_internal!(cell_fails.op, cell_fails.states, mat, ctx).converged
    end

end
