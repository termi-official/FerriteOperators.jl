using FerriteOperators
using FerriteOperatorsExampleElements
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester
using TimerOutputs

include(joinpath(@__DIR__, "fixture_elements.jl"))

# A bilinear form with a time-dependent coefficient, A(t) = (1 + t)·M. Time
# reaches a u-independent sweep through the context alone, so this cache
# assembles only where the entry point was handed one.
struct TimedMassIntegrator <: AbstractBilinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct TimedMassCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::TimedMassIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return TimedMassCache(CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::TimedMassCache) =
    TimedMassCache(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.reinit_values!(c::TimedMassCache, cell) = reinit!(c.cv, cell)
FerriteOperators.provides_analytic(::Type{<:TimedMassCache}, ::JacobianKind{:u}) = true

function _timed_mass!(req, c::TimedMassCache, args)
    ρ = 1.0 + evaluation_time(args.ctx)
    for qp in 1:getnquadpoints(c.cv)
        dΩ = getdetJdV(c.cv, qp)
        for i in 1:getnbasefunctions(c.cv)
            Nᵢ = shape_value(c.cv, qp, i) * dΩ
            req isa ResidualRequest && (req.r[i] += ρ * function_value(c.cv, qp, args.states.u) * Nᵢ)
            req isa JacobianRequest && for j in 1:getnbasefunctions(c.cv)
                req.K[i, j] += ρ * Nᵢ * shape_value(c.cv, qp, j)
            end
        end
    end
end
FerriteOperators.assemble_cell!(req::ResidualRequest, c::TimedMassCache, args) = _timed_mass!(req, c, args)
FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, c::TimedMassCache, args) = _timed_mass!(req, c, args)

@testset "Operators" begin
    reset_timer!()
    @testset "Actions" begin
        vin = ones(5)
        vout = ones(5)

        nullop = NullOperator{Float64,5,5}()
        @test eltype(nullop) == Float64
        @test length(vin)  == size(nullop, 1)
        @test size(nullop) == (5, 5)

        mul!(vout, nullop, vin)
        @test vout == zeros(5)

        vout .= ones(5)
        mul!(vout, nullop, vin, 2.0, 1.0)
        @test vout == ones(5)

        @test get_matrix(nullop) ≈ zeros(5,5)

        # Bilinear operators with constant linearization support both update
        # forms, and the fused one fills the residual with the operator's
        # action `F(u) = A·u`.
        update_linearization!(nullop, vin, nothing)
        vres = fill(NaN, 5)
        update_linearization!(nullop, vres, vin, nothing)
        @test vres == zeros(5)


        vin = ones(4)
        vout .= ones(5)
        nullop_rect = NullOperator{Float64,4,5}()

        @test length(vin)  == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)
        @test size(nullop_rect) == (4, 5)

        @test get_matrix(nullop_rect) ≈ zeros(4,5)

        # The rectangular action: `out` is row-shaped, `in` column-shaped.
        out_rect = ones(4)
        in_rect  = ones(5)
        mul!(out_rect, nullop_rect, in_rect)
        @test out_rect == zeros(4)
        out_rect .= ones(4)
        mul!(out_rect, nullop_rect, in_rect, 2.0, 3.0)
        @test out_rect == fill(3.0, 4)
    end

    @testset "Bilinear" begin
        # Setup
        grid = generate_grid(Quadrilateral, (10,9))
        Ferrite.transform_coordinates!(grid, x->Vec{2}(sign.(x.-0.5) .* (x.-0.5).^2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral,1}())
        close!(dh)
        qrc = QuadratureRuleCollection{2}()

        for integrator in [
            SimpleBilinearDiffusionIntegrator(
                1.0,
                QuadratureRuleCollection(2),
                :u
            ),
            SimpleBilinearMassIntegrator(
                1.0,
                QuadratureRuleCollection(1),
                :u
            )
        ]
            bilinop_base = setup_operator(AssemblyStrategy(SequentialCPUDevice()), integrator, dh)
            # Check that assembly works
            @test norm(bilinop_base.A) ≈ 0.0
            update_operator!(bilinop_base, 0.0)
            norm_baseline = norm(bilinop_base.A)
            @test norm_baseline > 0.0
            # Idempotency
            update_operator!(bilinop_base, 0.0)
            @test norm_baseline == norm(bilinop_base.A)

            # The bilinear form induces a linear operator, so the residual
            # entry point must reproduce its action.
            ub = sin.(0.4 .* (1:ndofs(dh)))
            rb = zeros(ndofs(dh))
            evaluate!(bilinop_base, rb, ub, 0.0)
            @test rb ≈ bilinop_base.A * ub rtol = 1e-13

            @testset "Strategy $strategy" for strategy in (
                    AssemblyStrategy(SequentialCPUDevice(); scheduling = ColoredScheduling()),
                    AssemblyStrategy(PolyesterDevice(1); scheduling = ColoredScheduling()),
                    AssemblyStrategy(PolyesterDevice(2); scheduling = ColoredScheduling()),
            )
                bilinop = setup_operator(strategy, integrator, dh)
                # Consistency
                update_operator!(bilinop, 0.0)
                @test bilinop.A ≈ bilinop_base.A
                # Idempotency
                update_operator!(bilinop, 0.0)
                @test bilinop.A ≈ bilinop_base.A
            end
        end
    end

    @testset "u-independent operators see the evaluation context" begin
        grid = generate_grid(Quadrilateral, (3, 2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        qrc = QuadratureRuleCollection(2)
        strategy = AssemblyStrategy(SequentialCPUDevice())
        ctx = TimeIntegrationContext(2.0, 0.1, 0.1)

        Mop = setup_operator(strategy, SimpleBilinearMassIntegrator(1.0, qrc, :u), dh)
        update_operator!(Mop, nothing)
        M0  = copy(Mop.A)
        top = setup_operator(strategy, TimedMassIntegrator(qrc, :u), dh)

        # ρ(t) = 1 + t is read off the context the entry point was handed.
        update_operator!(top, nothing, ctx)
        @test top.A ≈ 3.0 .* Mop.A rtol = 1e-12

        # Without one the element says so instead of freezing the coefficient
        # at t = 0.
        err = @test_throws ArgumentError update_operator!(top, nothing)
        @test occursin("evaluation time", err.value.msg)

        # A ctx-independent cache is unaffected either way, on both families.
        update_operator!(Mop, nothing, ctx)
        @test Mop.A ≈ M0 rtol = 1e-13
        lop = setup_operator(strategy, SimpleLinearIntegrator(1.0, qrc, :u), dh)
        update_operator!(lop, nothing)
        b0 = copy(lop.b)
        update_operator!(lop, nothing, ctx)
        @test lop.b ≈ b0 rtol = 1e-13
    end

    @testset "Nonlinear" begin
        # Setup
        grid = generate_grid(Hexahedron, (3,3,3))
        Ferrite.transform_coordinates!(grid, x->Vec{3}(sign.(x.-0.5) .* (x.-0.5).^2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
        close!(dh)

        residual = zeros(ndofs(dh))
        u = zeros(ndofs(dh))
        apply_analytical!(u, dh, :u, x->0.01x.^2)

        integrator = SimpleHyperelasticityIntegrator(NeoHookean(10.0, 0.3), QuadratureRuleCollection(2), :u)
        nlop_base = setup_operator(AssemblyStrategy(SequentialCPUDevice()), integrator, dh)

        # Check that assembly works
        @test norm(nlop_base.J) ≈ 0.0
        nlop_base.J .= NaN
        update_linearization!(nlop_base, u, 0.0)
        Jnorm_baseline = norm(nlop_base.J)
        @test Jnorm_baseline > 0.0
        yref = zero(u)
        mul!(yref, nlop_base.J, u)

        # Also querying the residual should not change the outcome
        residual .= NaN
        nlop_base.J .= NaN
        update_linearization!(nlop_base, residual, u, 0.0)
        @test Jnorm_baseline ≈ norm(nlop_base.J)
        rnorm_baseline = norm(residual)
        @test rnorm_baseline > 0.0

        # Now just the residual
        residual .= NaN
        evaluate!(nlop_base, residual, u, 0.0)
        @test rnorm_baseline ≈ norm(residual)

        # Idempotency
        update_linearization!(nlop_base, u, 0.0)
        @test Jnorm_baseline ≈ norm(nlop_base.J)
        evaluate!(nlop_base, residual, u, 0.0)
        @test Jnorm_baseline ≈ norm(nlop_base.J)
        @test rnorm_baseline ≈ norm(residual)
        residual_baseline = copy(residual)

        @testset "Full Assembly Strategy $strategy" for strategy in (
            AssemblyStrategy(SequentialCPUDevice()),
            AssemblyStrategy(SequentialCPUDevice(); scheduling = ColoredScheduling()),
            AssemblyStrategy(PolyesterDevice(1); scheduling = ColoredScheduling()),
            AssemblyStrategy(PolyesterDevice(2); scheduling = ColoredScheduling()),
        )
            nlop = setup_operator(strategy, integrator, dh)
            # Consistency: each of the three entry points against the baseline.
            nlop.J .= NaN
            update_linearization!(nlop, u, 0.0)
            @test nlop.J ≈ nlop_base.J

            nlop.J .= NaN
            residual .= NaN
            update_linearization!(nlop, residual, u, 0.0)
            @test nlop.J ≈ nlop_base.J
            @test residual ≈ residual_baseline

            residual .= NaN
            evaluate!(nlop, residual, u, 0.0)
            @test residual ≈ residual_baseline
        end

        # The `ElementAssembly` form accumulates per-element residuals and holds
        # no matrix, so a matrix-target operator is rejected at setup rather
        # than built with a matrix it can never fill.
        @testset "Element Assembly Strategy rejects matrix targets $strategy" for strategy in (
            AssemblyStrategy(SequentialCPUDevice(); form = ElementAssembly()),
            AssemblyStrategy(PolyesterDevice(1); form = ElementAssembly()),
        )
            @test_throws ArgumentError setup_operator(strategy, integrator, dh)
        end
    end

    @testset "Element Assembly Strategy (vector target) $strategy" for strategy in (
        AssemblyStrategy(SequentialCPUDevice(); form = ElementAssembly()),
        AssemblyStrategy(PolyesterDevice(1); form = ElementAssembly()),
        AssemblyStrategy(PolyesterDevice(2); form = ElementAssembly()),
    )
        # The per-element accumulation collapses into exactly the vector full
        # assembly scatters directly.
        grid = generate_grid(Quadrilateral, (5, 4))
        dh   = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)

        m   = SimpleLinearIntegrator(1.0, QuadratureRuleCollection(2), :u)
        ref = setup_operator(AssemblyStrategy(SequentialCPUDevice()), m, dh)
        update_operator!(ref, nothing)

        op = setup_operator(strategy, m, dh)
        @test size(op) == (ndofs(dh),)
        update_operator!(op, nothing)
        @test op.b ≈ ref.b
        # Reassembly zeroes both the per-element buffer and the global vector.
        update_operator!(op, nothing)
        @test op.b ≈ ref.b
    end

    @testset "Condensed Elements" begin
        qrc = QuadratureRuleCollection(2)
        strategy = AssemblyStrategy(SequentialCPUDevice())
        tb = visco_testbed(strategy, qrc, (3, 3, 3);
                           transform = x->Vec{3}(sign.(x.-0.5) .* (x.-0.5).^2))
        nlop, dh, grid = tb.op, tb.dh, tb.grid

        ch = ConstraintHandler(dh);
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> (0,0,0)));
        add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> (0.01,0,0)));
        close!(ch)

        ctx  = TimeIntegrationContext(0.0, π, π)   # backward-Euler local stage: γ̃ = Δt

        residual = zeros(residual_size(nlop))
        u        = zeros(unknown_size(nlop))
        uprev    = zeros(unknown_size(nlop))
        states   = condensed_states(u, uprev)
        apply_analytical!(u, dh, :u, x->0.01x.^2 .+ 0.01)
        @test length(residual) == 3 * (3+1)*(3+1)*(3+1)
        @test length(u)        == 3 * (3+1)*(3+1)*(3+1) + 6 * 8 * 3*3*3 # vdim=3, 4 nodes in each dim, 8 quadrature points, 6 unknowns for the symmetric viscosity tensor, 3*3*3 elements

        apply!(u, ch)
        condense_internal!(nlop, states, nothing, ctx)
        update_linearization!(nlop, residual, states, nothing, ctx)

        apply!(u, ch)
        apply_zero!(nlop.J, residual, ch)
        Δd = nlop.J \ residual
        d = @view u[1:ndofs(dh)]
        d .-= Δd

        condense_internal!(nlop, states, nothing, ctx)
        update_linearization!(nlop, residual, states, nothing, ctx)

        apply_zero!(nlop.J, residual, ch)
        Δd = nlop.J \ residual
        @test norm(Δd)/length(Δd) ≈ 0.0 atol=1e-12

        # Regression pins: recorded from a run of this very setup, with no
        # independent derivation behind either number.
        @test norm(d) ≈ 0.059623465672897884
        @test norm(u[ndofs(dh)+1:end]) ≈ 0.062203435313135984
        # The trial write-back goes into `u` alone: `uprev` is the committed
        # predecessor and no sweep may touch it.
        @test norm(uprev) ≈ 0.0
    end

    @testset "Transfer with different Dof Handlers" begin
        # Multi-cell on purpose: row dofs shared between cells are where a
        # local per-cell projection needs the valence normalization.
        grid = generate_grid(Hexahedron, (2,2,1))
        Ferrite.transform_coordinates!(grid, x->Vec{3}(sign.(x.-0.5) .* (x.-0.5).^2))

        dh2 = DofHandler(grid)
        add!(dh2, :u, Lagrange{RefHexahedron,2}())
        close!(dh2)

        dh1 = DofHandler(grid)
        add!(dh1, :u, Lagrange{RefHexahedron,1}())
        close!(dh1)

        integrator = FerriteOperators.MassProlongatorIntegrator(QuadratureRuleCollection(4), :u)
        strategy   = AssemblyStrategy(SequentialCPUDevice())
        op         = setup_transfer_operator(strategy, integrator, dh2, dh1)
        update_operator!(op, nothing)

        u1 = zeros(ndofs(dh1))
        u2 = zeros(ndofs(dh2))
        apply_analytical!(u1, dh1, :u, x->1.0)
        apply_analytical!(u2, dh2, :u, x->1.0)
        @test u2 ≈ op.P * u1

        # Linear reproduction needs affine geometry (on the curved grid above
        # the P1 interpolant of a linear is not that linear, so the P2
        # interpolant is the wrong reference). A P1 field lies inside the P2
        # row space elementwise, so its interpolant must prolongate exactly.
        grid_affine = generate_grid(Hexahedron, (2,2,1))
        dh2a = DofHandler(grid_affine); add!(dh2a, :u, Lagrange{RefHexahedron,2}()); close!(dh2a)
        dh1a = DofHandler(grid_affine); add!(dh1a, :u, Lagrange{RefHexahedron,1}()); close!(dh1a)
        opa = setup_transfer_operator(strategy, integrator, dh2a, dh1a)
        update_operator!(opa, nothing)
        u1a = zeros(ndofs(dh1a)); u2a = zeros(ndofs(dh2a))
        apply_analytical!(u1a, dh1a, :u, x->x[1] + 2x[2] - x[3])
        apply_analytical!(u2a, dh2a, :u, x->x[1] + 2x[2] - x[3])
        @test u2a ≈ opa.P * u1a
    end

    @testset "Nested transfer (geometric multigrid)" begin
        # Two coarse cells, each uniformly refined 2×2 into 4 conforming fine
        # cells — the minimal nested hierarchy the operator needs. FO ships no
        # refinement utility, so the fine grid, the fine→coarse cell map and
        # each fine cell's node positions in its parent's reference element
        # (`child_ref_coords`) are hand-built from the structured layout
        # `generate_grid` produces.
        coarse_dims = (2, 1)
        refine      = 2
        fine_dims   = (coarse_dims[1] * refine, coarse_dims[2] * refine)

        coarse_grid = generate_grid(Quadrilateral, coarse_dims)
        fine_grid   = generate_grid(Quadrilateral, fine_dims)

        dh_coarse = DofHandler(coarse_grid)
        add!(dh_coarse, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh_coarse)
        dh_fine = DofHandler(fine_grid)
        add!(dh_fine, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh_fine)

        ncol_f, ncol_c   = fine_dims[1], coarse_dims[1]
        fine2coarse      = Vector{Int}(undef, getncells(fine_grid))
        child_ref_coords = Vector{Vector{Vec{2, Float64}}}(undef, getncells(fine_grid))
        for fid in 1:getncells(fine_grid)
            col, row   = mod(fid - 1, ncol_f) + 1, div(fid - 1, ncol_f) + 1
            ccol, crow = div(col - 1, refine) + 1, div(row - 1, refine) + 1
            fine2coarse[fid] = (crow - 1) * ncol_c + ccol

            # This fine cell's quadrant (qx, qy) in its parent's [-1,1]²
            # reference element, its own corners (Ferrite's BL/BR/TR/TL node
            # order) placed at that quadrant's corners.
            qx, qy = mod(col - 1, refine), mod(row - 1, refine)
            dξ     = 2 / refine
            x0, x1 = -1 + qx * dξ, -1 + (qx + 1) * dξ
            y0, y1 = -1 + qy * dξ, -1 + (qy + 1) * dξ
            child_ref_coords[fid] = [Vec((x0, y0)), Vec((x1, y0)), Vec((x1, y1)), Vec((x0, y1))]
        end

        integrator = FerriteOperators.NestedMassProlongatorIntegrator(QuadratureRuleCollection(4), :u)
        strategy   = AssemblyStrategy(SequentialCPUDevice())
        op = setup_nested_transfer_operator(strategy, integrator, dh_fine, dh_coarse, fine2coarse, child_ref_coords)
        update_operator!(op, nothing)

        # Polynomial reproduction: matching P1 spaces means the prolongated
        # coarse nodal values must equal the fine nodal values EXACTLY for a
        # constant and for each linear monomial — the defining property of the
        # projection, and the property a naive per-cell assembly breaks at
        # every fine dof shared between several fine cells.
        for f in (x -> 1.0, x -> x[1], x -> x[2])
            u_coarse = zeros(ndofs(dh_coarse))
            u_fine   = zeros(ndofs(dh_fine))
            apply_analytical!(u_coarse, dh_coarse, :u, f)
            apply_analytical!(u_fine,   dh_fine,   :u, f)
            @test op.P * u_coarse ≈ u_fine atol = 1e-12
        end

        @test_throws ArgumentError setup_nested_transfer_operator(
            AssemblyStrategy(SequentialCPUDevice(); scheduling = ColoredScheduling()), integrator, dh_fine, dh_coarse, fine2coarse, child_ref_coords)
    end

    @testset "Operator sizes" begin
        (; dh, n, strategy) = scalar_quad_testbed((3, 3))

        bilin_op = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u), dh)
        @test size(bilin_op) == (n, n)
        @test size(bilin_op, 1) == n
        @test size(bilin_op, 2) == n

        nl_op = setup_operator(strategy, SimpleHyperelasticityIntegrator(NeoHookean(210e3, 0.3), QuadratureRuleCollection(2), :u), dh)
        @test size(nl_op) == (n, n)
        @test size(nl_op, 1) == n
        @test size(nl_op, 2) == n

        lin_op = setup_operator(strategy, SimpleLinearIntegrator(1.0, QuadratureRuleCollection(2), :u), dh)
        @test size(lin_op) == (n,)
    end

    @testset "Device hooks a device type must implement" begin
        # `AbstractGPUDevice` is the seam a downstream device subtypes; with no
        # method of its own it must reach the loud generic hooks, and the forms
        # whose setup builds storage must refuse it rather than fall through.
        struct _TestGPUDevice{V, I} <: FerriteOperators.AbstractGPUDevice{V, I} end
        device = _TestGPUDevice{Float32, Int32}()

        @test_throws ArgumentError FerriteOperators.setup_device_instances(device, FerriteOperators.EAIndexWorkspace(0), 1)
        @test_throws ArgumentError FerriteOperators.execute_on_device!(nothing, device, nothing, [])
        @test_throws ArgumentError FerriteOperators.setup_operator_strategy_cache(
            AssemblyStrategy(device; form = ElementAssembly()), nothing, nothing)
    end

    @testset "Generic setup_device_instances" begin
        # setup_device_instances should work on any duplicable object, not just AbstractWorkspace
        struct _TestDuplicable
            x::Int
        end
        FerriteOperators.duplicate_for_device(::FerriteOperators.AbstractCPUDevice, d::_TestDuplicable) = _TestDuplicable(d.x)

        seq = SequentialCPUDevice()
        dc_seq = FerriteOperators.setup_device_instances(seq, _TestDuplicable(7), 1)
        @test length(dc_seq) == 1
        @test dc_seq[1].x == 7

        poly = PolyesterDevice()
        dc_poly = FerriteOperators.setup_device_instances(poly, _TestDuplicable(7), 3)
        @test length(dc_poly) == 3
        @test all(d -> d.x == 7, dc_poly)
    end

    @testset "default_strategy resolves to the loaded Polyester ext" begin
        # This file `using Polyester`s above, which activates `FerriteOperatorsPolyesterExt` for the
        # rest of the process — so the parallel branch is the only one this file can observe. The
        # no-Polyester fallback is not exercised here: `ParallelTestRunner` reuses worker processes
        # across test files, so a file that itself avoids `using Polyester` cannot guarantee the
        # extension is still unloaded by the time it runs.
        strategy = default_strategy()
        @test strategy isa AssemblyStrategy
        @test strategy.form isa FullAssembly
        @test strategy.scheduling isa SequentialScheduling
        @test strategy.device isa PolyesterDevice
    end

    @testset "Transfer setup validation" begin
        grid = generate_grid(Hexahedron, (1,1,1))
        dh2 = DofHandler(grid)
        add!(dh2, :u, Lagrange{RefHexahedron,2}())
        close!(dh2)
        dh1 = DofHandler(grid)
        add!(dh1, :u, Lagrange{RefHexahedron,1}())
        close!(dh1)
        integrator = FerriteOperators.MassProlongatorIntegrator(QuadratureRuleCollection(4), :u)
        @test_throws ArgumentError setup_transfer_operator(AssemblyStrategy(SequentialCPUDevice(); scheduling = ColoredScheduling()), integrator, dh2, dh1)
    end

    @testset "Dummy Multi-Physics" begin
        grid = generate_grid(Hexahedron, ntuple(_ -> 4, 3))
        addcellset!(grid, "right_cells", x -> x[1] ≥ 0.0)
        addcellset!(grid, "left_cells", x -> x[1] ≤ 0.0)

        strategy = AssemblyStrategy(SequentialCPUDevice())

        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, getcellset(grid, "right_cells"))
        add!(sdh1, :u, Lagrange{RefHexahedron, 1}())
        sdh2 = SubDofHandler(dh, getcellset(grid, "left_cells"))
        add!(sdh2, :u, Lagrange{RefHexahedron, 1}())
        close!(dh)

        n = 5^3
        qrc = QuadratureRuleCollection(2)

        assemble_linear(subintegrators) = begin
            op = setup_operator(strategy, LinearMultiDomainIntegrator(subintegrators), dh)
            update_operator!(op, nothing)
            op
        end

        # Linear case
        lin_op = assemble_linear(Dict(
            "right_cells" => SimpleLinearIntegrator( 1.0, qrc, :u),
            "left_cells"  => SimpleLinearIntegrator(-1.0, qrc, :u)
        ))
        @test size(lin_op) == (n,)

        # The name→subdomain map is the geometric one: loading only
        # "right_cells" must leave every dof outside that subdomain untouched.
        right_dofs = Set{Int}()
        for cc in CellIterator(sdh1)
            union!(right_dofs, celldofs(cc))
        end
        right_only = assemble_linear(Dict(
            "right_cells" => SimpleLinearIntegrator(1.0, qrc, :u),
            "left_cells"  => SimpleLinearIntegrator(0.0, qrc, :u)
        ))
        @test all(iszero, right_only.b[setdiff(1:n, right_dofs)])
        @test !all(iszero, right_only.b[collect(right_dofs)])

        # Routing is per-domain, so the two assignments of one integrator pair
        # differ, and swapping them sums to the uniform assignment.
        bilin(D_right, D_left) = begin
            op = setup_operator(strategy, BilinearMultiDomainIntegrator(Dict(
                "right_cells" => SimpleBilinearDiffusionIntegrator(D_right, qrc, :u),
                "left_cells"  => SimpleBilinearDiffusionIntegrator(D_left, qrc, :u)
            )), dh)
            update_operator!(op, nothing)
            op.A
        end
        Ka = bilin(1.0, 2.0)
        Kb = bilin(2.0, 1.0)
        @test Ka != Kb
        @test Ka + Kb ≈ bilin(3.0, 3.0)

        # A weak boundary term is resolved through the *volumetric* name of
        # the subdomain carrying it, never through a facetset namespace.
        t̄ = 3.25
        neumann_op = assemble_linear(Dict(
            "right_cells" => LinearNeumannProbe(t̄, :u, Set(getfacetset(grid, "right"))),
            "left_cells"  => SimpleLinearIntegrator(0.0, qrc, :u)
        ))
        @test sum(neumann_op.b) ≈ t̄ * 4.0 rtol = 1e-12

        # Message of the ArgumentError `f` raises, so the cell-exact content of
        # a rejection can be asserted and not just its type.
        function rejection_message(f)
            try
                f()
            catch err
                err isa ArgumentError && return err.msg
                rethrow()
            end
            error("expected an ArgumentError")
        end

        @testset "validation in both modes" begin
            L(v) = SimpleLinearIntegrator(v, qrc, :u)
            addcellset!(grid, "all_cells", x -> true)
            addcellset!(grid, "one_cell", Set([1]))

            # A three-cell grid whose first two cells are the subdomains, so a
            # declared name can be disjoint from every subdomain (cell 3) and an
            # overlap can be placed away from cell 1.
            sub_grid = generate_grid(Hexahedron, (3, 1, 1))
            addcellset!(sub_grid, "a", Set([1]))
            addcellset!(sub_grid, "b", Set([2]))
            addcellset!(sub_grid, "c", Set([3]))
            sub_dh = DofHandler(sub_grid)
            sa = SubDofHandler(sub_dh, getcellset(sub_grid, "a")); add!(sa, :u, Lagrange{RefHexahedron, 1}())
            sb = SubDofHandler(sub_dh, getcellset(sub_grid, "b")); add!(sb, :u, Lagrange{RefHexahedron, 1}())
            close!(sub_dh)

            # These four classes are rejected regardless of resolution mode.
            for mode in (Val(:sample), Val(:full))
                # a declared name that is not a cellset of the grid
                @test_throws ArgumentError FerriteOperators.resolve_subdomain_claims(
                    Dict("right_cells" => L(1.0), "left_cell" => L(1.0)), dh, mode)
                # a subdomain owned by no declared name
                @test_throws ArgumentError FerriteOperators.resolve_subdomain_claims(
                    Dict("right_cells" => L(1.0)), dh, mode)
                # a subdomain claimed by more than one declared name
                @test_throws ArgumentError FerriteOperators.resolve_subdomain_claims(
                    Dict("all_cells" => L(1.0), "right_cells" => L(1.0), "left_cells" => L(1.0)), dh, mode)
                # a declared name claiming no subdomain (disjoint from every subdomain)
                @test FerriteOperators.resolve_subdomain_claims(
                    Dict("a" => L(1.0), "b" => L(1.0)), sub_dh, mode) == ["a", "b"]
                @test_throws ArgumentError FerriteOperators.resolve_subdomain_claims(
                    Dict("a" => L(1.0), "b" => L(1.0), "c" => L(1.0)), sub_dh, mode)
            end

            # Production sampling reads the first cell only, so a subdomain that
            # straddles two declared cellsets resolves by that cell alone; debug
            # mode rejects it, naming the cells at the mismatch.
            addcellset!(grid, "right_head", Set([first(sdh1.cellset)]))
            straddle = Dict("right_head" => L(1.0), "left_cells" => L(1.0))
            @test FerriteOperators.resolve_subdomain_claims(straddle, dh, Val(:sample)) ==
                ["right_head", "left_cells"]
            straddle_msg = rejection_message(
                () -> FerriteOperators.resolve_subdomain_claims(straddle, dh, Val(:full)))
            @test occursin("owned by no declared name", straddle_msg)
            @test occursin("right_head", straddle_msg)

            # Debug mode rejects overlapping DECLARED cellsets at fill time,
            # naming the exact cell, even where no subdomain is ambiguous.
            overlap_msg = rejection_message(() -> FerriteOperators.resolve_subdomain_claims(
                Dict("all_cells" => L(1.0), "one_cell" => L(1.0)), dh, Val(:full)))
            @test occursin("Cell 1 lies in both", overlap_msg)

            # The reported cell is the actual overlap, not the first cell of the
            # grid: "b" and "seam" collide on cell 2 alone.
            addcellset!(sub_grid, "seam", Set([2]))
            seam_msg = rejection_message(() -> FerriteOperators.resolve_subdomain_claims(
                Dict("a" => L(1.0), "b" => L(1.0), "seam" => L(0.0)), sub_dh, Val(:full)))
            @test occursin("Cell 2 lies in both", seam_msg)

            # The operator entry point routes through the compile-time mode.
            @test_throws ArgumentError setup_operator(strategy, LinearMultiDomainIntegrator(
                Dict("right_cells" => L(1.0))), dh)
        end

        # Routing outer, composition inner: a named subdomain whose term is a
        # composite of two bilinear forms.
        composed = setup_operator(strategy, BilinearMultiDomainIntegrator(Dict(
            "right_cells" => BilinearCompositeIntegrator(
                SimpleBilinearDiffusionIntegrator(1.0, qrc, :u),
                SimpleBilinearMassIntegrator(2.0, qrc, :u)),
            "left_cells"  => SimpleBilinearDiffusionIntegrator(1.0, qrc, :u)
        )), dh)
        update_operator!(composed, nothing)
        mass_only = setup_operator(strategy, BilinearMultiDomainIntegrator(Dict(
            "right_cells" => SimpleBilinearMassIntegrator(2.0, qrc, :u),
            "left_cells"  => SimpleBilinearMassIntegrator(0.0, qrc, :u)
        )), dh)
        update_operator!(mass_only, nothing)
        @test composed.A ≈ bilin(1.0, 1.0) + mass_only.A rtol = 1e-12

        # Nonlinear case
        dhv = DofHandler(grid)
        sdhv1 = SubDofHandler(dhv, getcellset(grid, "right_cells"))
        add!(sdhv1, :u, Lagrange{RefHexahedron, 1}()^3)
        sdhv2 = SubDofHandler(dhv, getcellset(grid, "left_cells"))
        add!(sdhv2, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(dhv)
        nl_multi = NonlinearMultiDomainIntegrator(Dict(
            "right_cells" => SimpleHyperelasticityIntegrator(NeoHookean(210e3, 0.30), qrc, :u),
            "left_cells"  => SimpleHyperelasticityIntegrator(NeoHookean(180e3, 0.35), qrc, :u)
        ))
        nl_op = setup_operator(strategy, nl_multi, dhv)
        u = zeros(ndofs(dhv))
        apply_analytical!(u, dhv, :u, x->0.01x.^2)
        update_linearization!(nl_op, u, nothing)
        @test size(nl_op) == (3n, 3n)
        @test size(nl_op, 1) == 3n
        @test size(nl_op, 2) == 3n
    end

    print_timer()
end

# `@allocated` has to be measured from inside a function: at testset scope the
# operator is a captured variable and the boxing of that capture is charged to
# the call being measured rather than to the sweep.
function polyester_sweep_allocations(dims)
    grid = generate_grid(Quadrilateral, dims)
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    op = setup_operator(AssemblyStrategy(PolyesterDevice(1)),
                        SimpleBilinearDiffusionIntegrator(1.0, QuadratureRuleCollection(2), :u), dh)
    for _ in 1:3   # warmup: compilation, and Polyester's own first-batch setup
        update_operator!(op, nothing)
    end
    return @allocated update_operator!(op, nothing)
end

@testset "A threaded sweep allocates per worker, not per item" begin
    # The per-worker workspaces are setup-time state and the task's scatter
    # target is duplicated once per worker, so what one sweep allocates is fixed
    # by the worker count — a 32× larger mesh must not move it. Both meshes
    # carry more cells than there are threads, so both run the same number of
    # workers at `chunksize = 1`.
    nt = Threads.nthreads()
    @test polyester_sweep_allocations((nt, 8)) == polyester_sweep_allocations((nt, 256))
end

# A cache that opts into the request protocol without the mandatory residual
# kernel — what setup-time validation exists to catch.
struct BareEvaluationCache <: AbstractVolumetricElementCache end
struct BareEvaluationIntegrator <: AbstractBilinearIntegrator end
FerriteOperators.setup_element_cache(::BareEvaluationIntegrator, ::SubDofHandler) = BareEvaluationCache()

@testset "Payload-free evaluation operator" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    strategy   = AssemblyStrategy(SequentialCPUDevice())
    qrc        = QuadratureRuleCollection(2)
    integrator = SimpleBilinearDiffusionIntegrator(1.0, qrc, :u)

    op  = setup_operator(strategy, integrator, dh)
    eop = setup_evaluation_operator(strategy, integrator, dh)

    @testset "the engine is the one setup_operator builds" begin
        # Same caches through the same family dispatch, and no payload field.
        @test typeof(eop.engine) === typeof(op.engine)
        @test propertynames(eop) === (:engine, :integrator)
        @test length(eop.engine.subdomain_caches) == length(op.engine.subdomain_caches)
    end

    @testset "the evaluation entry points work off the engine" begin
        u = sin.(0.3 .* (1:ndofs(dh)))
        q = setup_qvector(Float64, dh, qrc)
        evaluate_quadrature!(q, eop, u, nothing, (uₑ, qp, cell, cache, pₑ, ctx) -> Float64(cellid(cell)))
        qref = setup_qvector(Float64, dh, qrc)
        evaluate_quadrature!(qref, op, u, nothing, (uₑ, qp, cell, cache, pₑ, ctx) -> Float64(cellid(cell)))
        @test q.data == qref.data
    end

    @testset "the assembly entry points are a contract error" begin
        u = zeros(ndofs(dh))
        @test_throws ArgumentError update_operator!(eop, nothing)
        @test_throws ArgumentError update_linearization!(eop, u, nothing)
        @test_throws ArgumentError evaluate!(eop, zeros(ndofs(dh)), u, nothing)
        @test_throws ArgumentError evaluate!(eop, zeros(ndofs(dh)), (u = u,), nothing, nothing)
    end

    @testset "setup validation runs as for other operators" begin
        @test_throws ArgumentError setup_evaluation_operator(strategy, BareEvaluationIntegrator(), dh)
    end
end

# A subdomain with no volumetric kernel; its boundary cache stays at the empty
# default, so the pair is what makes an assembly traversal pointless.
struct SilentSubdomainIntegrator <: AbstractBilinearIntegrator end
FerriteOperators.setup_element_cache(::SilentSubdomainIntegrator, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()

# The assembly the engine would run without the skip: every subdomain, in
# subdomain order, which is `execute_on_subdomains!` minus its gate.
function unskipped_assembly!(A, op, kind)
    assembler = start_assemble(op.engine.strategy, A)
    task = FerriteOperators.AssemblyTask(kind, assembler, (;), nothing, nothing)
    for sc in op.engine.subdomain_caches
        FerriteOperators.execute_on_device!(
            task, op.engine.strategy.device, sc.device_cache, sc.partition)
    end
    FerriteOperators.finalize_assembly!(assembler)
    return A
end

@testset "an assembly sweep skips the subdomains that cannot contribute" begin
    grid = generate_grid(Quadrilateral, (4, 4))
    addcellset!(grid, "right_cells", x -> x[1] ≥ 0.0)
    addcellset!(grid, "left_cells",  x -> x[1] ≤ 0.0)
    dh = DofHandler(grid)
    for name in ("right_cells", "left_cells")
        sdh = SubDofHandler(dh, getcellset(grid, name))
        add!(sdh, :u, Lagrange{RefQuadrilateral, 1}())
    end
    close!(dh)
    strategy = AssemblyStrategy(SequentialCPUDevice())
    qrc      = QuadratureRuleCollection(2)

    op = setup_operator(strategy, BilinearMultiDomainIntegrator(Dict(
        "right_cells" => SimpleBilinearDiffusionIntegrator(1.0, qrc, :u),
        "left_cells"  => SilentSubdomainIntegrator(),
    )), dh)
    live, silent = op.engine.subdomain_caches

    # The verdict is structural and taken once, at setup.
    @test live.contributes
    @test !silent.contributes

    silent_ws = first(silent.device_cache)
    live_ws   = first(live.device_cache)
    parked    = cellid(silent_ws.cell)

    update_operator!(op, nothing)
    A_skipped = copy(op.A)

    # The declining subdomain was never positioned on a cell, while the
    # contributing one was.
    @test cellid(silent_ws.cell) == parked
    @test cellid(live_ws.cell) != parked

    # Bit-identical to the traversal that visits it: an empty cache's kernel
    # writes nothing, so its scatter only ever added zeros.
    A_reference = unskipped_assembly!(similar(op.A), op, FerriteOperators.BilinearKind())
    @test A_skipped == A_reference
    @test !iszero(A_skipped)

    # An empty ELEMENT cache alone is not the verdict: the fused boundary route
    # rides the cell sweep, so a subdomain carrying a surface cache is traversed.
    t̄  = 3.25
    nop = setup_operator(strategy, LinearMultiDomainIntegrator(Dict(
        "right_cells" => LinearNeumannProbe(t̄, :u, Set(getfacetset(grid, "right"))),
        "left_cells"  => SimpleLinearIntegrator(0.0, qrc, :u),
    )), dh)
    @test first(nop.engine.subdomain_caches).domain.element isa FerriteOperators.EmptyVolumetricElementCache
    @test all(sc -> sc.contributes, nop.engine.subdomain_caches)
    update_operator!(nop, nothing)
    @test sum(nop.b) ≈ t̄ * 2.0 rtol = 1e-12       # |Γ_right| = 2 on the [-1, 1]² grid
end
