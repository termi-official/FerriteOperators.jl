# The rate form `M⁻¹ · rhs` and the element-matrix STRUCTURE election it reads.
#
# Every arm is checked against the same independent reference: the mass and the
# diffusion matrix assembled on their own, and `M \ (K·u)` formed with
# LinearAlgebra. What the realizations may differ in is WHERE `M⁻¹` is applied,
# never in the operator they represent.

include(joinpath(@__DIR__, "fixture_elements.jl"))

using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using LinearAlgebra
using SparseArrays

const RF_D = 1.3
const RF_ρ = 2.1

rf_diffusion(order = 2) = SimpleBilinearDiffusionIntegrator(RF_D, QuadratureRuleCollection(order), :u)
rf_mass(order = 2) = SimpleBilinearMassIntegrator(RF_ρ, QuadratureRuleCollection(order), :u)
rf_probe(dh) = Float64[sin(0.9i) + 0.3cos(1.7i) for i in 1:ndofs(dh)]
rf_seq() = AssemblyStrategy(SequentialCPUDevice())
rf_element_arm() = AssemblyStrategy(MatrixFreeAction(; storage = ElementAssembly()),
                                    SequentialScheduling(), SequentialCPUDevice())

function rf_assembled(strategy, integrator, dh)
    op = setup_operator(strategy, integrator, dh)
    update_operator!(op, nothing)
    return op
end

rf_action(op, u) = (y = zeros(length(u)); mul!(y, op, u); y)

# A DISCONTINUOUS handler, where a dense (unlumped) mass is invertible cell by
# cell and the rate form's block route applies.
function rf_dg_handler(dims = (3, 2); order = 1)
    grid = generate_grid(Quadrilateral, dims)
    dh   = DofHandler(grid)
    add!(dh, :u, DiscontinuousLagrange{RefQuadrilateral, order}())
    close!(dh)
    return dh
end

####################################
## Doubles
####################################

# A mass whose kernel reads the CONTEXT: `M(t) = t·M(1)`, so a sweep carrying no
# context fails in `evaluation_time` and the weighted action scales with `1/t`.
struct TimeScaledMassIntegrator <: AbstractBilinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct TimeScaledMassCache{CV} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::TimeScaledMassIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return TimeScaledMassCache(CellValues(qr, ip, ip_geo))
end
Ferrite.getnquadpoints(c::TimeScaledMassCache) = getnquadpoints(c.cv)
FerriteOperators.reinit_values!(c::TimeScaledMassCache, cell) = reinit!(c.cv, cell)
FerriteOperators.duplicate_for_device(device, c::TimeScaledMassCache) =
    TimeScaledMassCache(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.provides_analytic(::Type{<:TimeScaledMassCache}, ::JacobianKind{:u}) = true
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, c::TimeScaledMassCache, args::CellArgs)
    t = evaluation_time(args.ctx)
    for qp in 1:getnquadpoints(c.cv)
        dΩ = getdetJdV(c.cv, qp)
        for i in 1:getnbasefunctions(c.cv), j in 1:getnbasefunctions(c.cv)
            req.K[i, j] += t * shape_value(c.cv, qp, i) * shape_value(c.cv, qp, j) * dΩ
        end
    end
end
function FerriteOperators.assemble_cell!(req::ResidualRequest, c::TimeScaledMassCache, args::CellArgs)
    t  = evaluation_time(args.ctx)
    uₑ = args.states.u
    for qp in 1:getnquadpoints(c.cv)
        dΩ   = getdetJdV(c.cv, qp)
        uval = function_value(c.cv, qp, uₑ)
        for i in 1:getnbasefunctions(c.cv)
            req.r[i] += t * uval * shape_value(c.cv, qp, i) * dΩ
        end
    end
end
rf_time_mass() = RowSumLumped(TimeScaledMassIntegrator(QuadratureRuleCollection(2), :u))

# A bilinear integrator serving the RESIDUAL alone — what `RowSumLumped` and the
# rate form both refuse, there being no element matrix to read.
struct NoMatrixKernelIntegrator <: AbstractBilinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct NoMatrixKernelCache{CV} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::NoMatrixKernelIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return NoMatrixKernelCache(CellValues(qr, ip, ip_geo))
end
Ferrite.getnquadpoints(c::NoMatrixKernelCache) = getnquadpoints(c.cv)
FerriteOperators.reinit_values!(c::NoMatrixKernelCache, cell) = reinit!(c.cv, cell)
FerriteOperators.assemble_cell!(req::ResidualRequest, c::NoMatrixKernelCache, args::CellArgs) = nothing

# A nonlinear integrator, to be refused by the constructor before any setup.
struct RFNonlinearIntegrator <: AbstractNonlinearIntegrator end

# A mass declaring a local-system tail the rate form's own cell sweep would
# never visit.
struct GlobalDofMassIntegrator <: AbstractBilinearIntegrator end
FerriteOperators.global_dofs(::GlobalDofMassIntegrator, sdh::SubDofHandler) = (1,)

# A decorator that declares nothing of its own: what reaches it must have been
# FORWARDED off the wrapped cache.
struct RFPassthrough{I} <: FerriteOperators.AbstractElementCacheDecorator{I}
    inner::I
end

####################################
## The tests
####################################

@testset "The element-matrix structure election" begin
    (; dh) = scalar_quad_testbed((3, 2))
    sdh = dh.subdofhandlers[1]
    nb  = ndofs_per_cell(sdh)

    consistent = FerriteOperators.setup_element_cache(rf_mass(), sdh)
    lumped     = FerriteOperators.setup_element_cache(RowSumLumped(rf_mass()), sdh)

    @testset "the default is dense and the buffer is square" begin
        @test element_matrix_structure(consistent) isa DenseElementMatrix
        @test size(FerriteOperators.allocate_element_matrix(consistent, sdh)) == (nb, nb)
    end

    @testset "a diagonal election allocates the diagonal VECTOR" begin
        @test element_matrix_structure(lumped) isa DiagonalElementMatrix
        Mₑ = FerriteOperators.allocate_element_matrix(lumped, sdh)
        @test Mₑ isa AbstractVector
        @test length(Mₑ) == nb
    end

    # A cache-keyed declaration that a decorator does not forward is silently
    # lost; this is the new election's own forwarding gate.
    @testset "a decorator forwards the election and the buffer it sizes" begin
        @test element_matrix_structure(RFPassthrough(lumped)) isa DiagonalElementMatrix
        @test FerriteOperators.allocate_element_matrix(RFPassthrough(lumped), sdh) isa AbstractVector
        @test element_matrix_structure(RFPassthrough(consistent)) isa DenseElementMatrix
        @test size(FerriteOperators.allocate_element_matrix(RFPassthrough(consistent), sdh)) == (nb, nb)
    end

    @testset "the ELEMENT storage levels refuse a diagonal cache by name" begin
        for (election, storage) in ("ElementAssembly" => ElementAssembly(),
                                    "BlockRowAssembly" => BlockRowAssembly())
            err = @test_throws ArgumentError with_action_storage(lumped, storage, sdh)
            @test occursin("DiagonalElementMatrix", err.value.msg)
            @test occursin(election, err.value.msg)
            @test occursin("FullAssembly", err.value.msg)
        end
    end
end

@testset "RowSumLumped" begin
    (; dh) = scalar_quad_testbed((4, 3))
    M = get_matrix(rf_assembled(rf_seq(), rf_mass(), dh))

    @testset "the operator's matrix IS its diagonal" begin
        op = rf_assembled(rf_seq(), RowSumLumped(rf_mass()), dh)
        D  = get_matrix(op)
        @test D isa Diagonal
        # Element-level row-sum equals the GLOBAL row sum: the scatter is
        # linear, so lumping per cell and assembling cannot disagree with
        # assembling and then summing the rows.
        @test diag(D) ≈ vec(sum(M; dims = 2)) rtol = 1.0e-12
        # A lumped mass conserves the total mass.
        @test sum(diag(D)) ≈ sum(M) rtol = 1.0e-12
    end

    @testset "its action is the diagonal's" begin
        op = rf_assembled(rf_seq(), RowSumLumped(rf_mass()), dh)
        u  = rf_probe(dh)
        r  = zeros(ndofs(dh))
        evaluate!(op, r, u, nothing)
        @test r ≈ diag(get_matrix(op)) .* u rtol = 1.0e-12
    end

    @testset "an inner without an element-matrix kernel is refused" begin
        err = @test_throws ArgumentError FerriteOperators.setup_element_cache(
            RowSumLumped(NoMatrixKernelIntegrator(QuadratureRuleCollection(2), :u)),
            dh.subdofhandlers[1])
        @test occursin("JacobianKind{:u}", err.value.msg)
    end
end

@testset "The rate form over a lumped mass" begin
    (; dh) = scalar_quad_testbed((4, 3))
    u = rf_probe(dh)
    K = get_matrix(rf_assembled(rf_seq(), rf_diffusion(), dh))
    D = get_matrix(rf_assembled(rf_seq(), RowSumLumped(rf_mass()), dh))
    reference = D \ (K * u)
    rate = RateFormIntegrator(rf_diffusion(), RowSumLumped(rf_mass()))

    @testset "FullAssembly scales the rows of the assembled matrix" begin
        op = rf_assembled(rf_seq(), rate, dh)
        @test op isa RateFormFerriteOperator
        @test rf_action(op, u) ≈ reference rtol = 1.0e-12
        # The weighting is not a no-op the tolerance would hide.
        @test !isapprox(rf_action(op, u), K * u; rtol = 1.0e-3)
        # Same sparsity: the fusion is a row scaling, not a second matrix.
        @test nnz(get_matrix(op)) == nnz(K)
        @test get_matrix(op) ≈ D \ Matrix(K) rtol = 1.0e-12
    end

    @testset "a refill re-weights rather than weighting twice" begin
        op = rf_assembled(rf_seq(), rate, dh)
        update_operator!(op, nothing)
        update_operator!(op, nothing)
        @test rf_action(op, u) ≈ reference rtol = 1.0e-12
    end

    @testset "the residual route is weighted too" begin
        op = rf_assembled(rf_seq(), rate, dh)
        r  = zeros(ndofs(dh))
        evaluate!(op, r, u, nothing)
        @test r ≈ reference rtol = 1.0e-12
    end

    @testset "the matrix-free ELEMENT level scales the action" begin
        op = setup_operator(rf_element_arm(), rate, dh)
        @test rf_action(op, u) ≈ reference rtol = 1.0e-12
        update_operator!(op, nothing)
        @test rf_action(op, u) ≈ reference rtol = 1.0e-12
    end

    @testset "the five-argument mul! scales and accumulates — $label" for
            (label, strategy) in ("FullAssembly" => rf_seq(),
                                  "ElementAssembly" => rf_element_arm())
        op = rf_assembled(strategy, rate, dh)
        y = fill(2.0, ndofs(dh))
        mul!(y, op, u, 3.0, 0.5)
        @test y ≈ 3.0 .* reference .+ 1.0 rtol = 1.0e-12
        # `β = 0` ASSIGNS: a NaN already in `y` must not propagate.
        z = fill(NaN, ndofs(dh))
        mul!(z, op, u, 1.0, 0.0)
        @test z ≈ reference rtol = 1.0e-12
    end
end

@testset "The rate form over a dense mass" begin
    @testset "a continuous space is refused, and the escapes are named" begin
        (; dh) = scalar_quad_testbed((3, 2))
        err = @test_throws ArgumentError setup_operator(
            rf_seq(), RateFormIntegrator(rf_diffusion(), rf_mass()), dh)
        msg = err.value.msg
        @test occursin("RowSumLumped", msg)
        @test occursin("DiagonalElementMatrix", msg)
        @test occursin("CONTINUOUS", msg)
    end

    @testset "a discontinuous space is block-solved per cell" begin
        dh = rf_dg_handler((3, 2))
        u  = rf_probe(dh)
        K  = get_matrix(rf_assembled(rf_seq(), rf_diffusion(), dh))
        M  = get_matrix(rf_assembled(rf_seq(), rf_mass(), dh))
        reference = M \ (K * u)

        op = rf_assembled(rf_seq(), RateFormIntegrator(rf_diffusion(), rf_mass()), dh)
        @test rf_action(op, u) ≈ reference rtol = 1.0e-10
        @test nnz(get_matrix(op)) == nnz(K)
        # A refill re-derives the mass and re-fuses.
        update_operator!(op, nothing)
        @test rf_action(op, u) ≈ reference rtol = 1.0e-10
    end

    @testset "a matrix-free level with no row to fuse into is refused" begin
        dh  = rf_dg_handler((3, 2))
        err = @test_throws ArgumentError setup_operator(
            rf_element_arm(), RateFormIntegrator(rf_diffusion(), rf_mass()), dh)
        msg = err.value.msg
        @test occursin("BlockRowAssembly", msg)
        @test occursin("RowSumLumped", msg)
    end
end

@testset "A linear rate form carries M⁻¹b" begin
    (; dh) = scalar_quad_testbed((3, 3))
    source = SimpleLinearIntegrator(1.7, QuadratureRuleCollection(2), :u)
    b = let op = setup_operator(rf_seq(), source, dh)
        update_operator!(op, nothing)
        copy(op.b)
    end
    D = get_matrix(rf_assembled(rf_seq(), RowSumLumped(rf_mass()), dh))

    op = setup_operator(rf_seq(), RateFormIntegrator(source, RowSumLumped(rf_mass())), dh)
    @test op isa LinearRateFormFerriteOperator
    update_operator!(op, nothing)
    @test op.b ≈ D \ b rtol = 1.0e-12
    # A refill weights the freshly assembled vector once, never the weighted one
    # again.
    update_operator!(op, nothing)
    @test op.b ≈ D \ b rtol = 1.0e-12
    # The `AbstractLinearOperator` surface reads the weighted vector.
    acc = zeros(ndofs(dh))
    Ferrite.add!(acc, op)
    @test acc ≈ D \ b rtol = 1.0e-12
end

@testset "What a rate form refuses to be" begin
    (; dh) = scalar_quad_testbed((2, 2))

    @testset "a nonlinear right-hand side" begin
        err = @test_throws ArgumentError RateFormIntegrator(RFNonlinearIntegrator(), rf_mass())
        @test occursin("BILINEAR", err.value.msg)
    end

    # `M⁻¹` does not distribute over a sum of element contributions, so the pair
    # is an operator-level term and composing it into one local system is an
    # error rather than a silently different operator.
    @testset "a term inside someone else's element cache" begin
        err = @test_throws ArgumentError FerriteOperators.setup_element_cache(
            RateFormIntegrator(rf_diffusion(), RowSumLumped(rf_mass())), dh.subdofhandlers[1])
        @test occursin("OPERATOR-level", err.value.msg)
    end

    @testset "a mass whose support is not the cells" begin
        err = @test_throws ArgumentError setup_operator(
            rf_seq(), RateFormIntegrator(rf_diffusion(), GlobalDofMassIntegrator()), dh)
        @test occursin("global_dofs", err.value.msg)
    end

    # A load vector has no action, so the refusal names the PAIR rather than
    # letting the wrapped linear integrator's own rejection speak for it.
    @testset "a linear right-hand side under a matrix-free form" begin
        source = SimpleLinearIntegrator(1.0, QuadratureRuleCollection(2), :u)
        err = @test_throws ArgumentError setup_operator(
            rf_element_arm(), RateFormIntegrator(source, RowSumLumped(rf_mass())), dh)
        @test occursin("rate form", err.value.msg)
        @test occursin("FullAssembly", err.value.msg)
    end
end

@testset "The mass is evaluated at the fill's (p, ctx)" begin
    (; dh) = scalar_quad_testbed((3, 2))
    u    = rf_probe(dh)
    K    = get_matrix(rf_assembled(rf_seq(), rf_diffusion(), dh))
    rate = RateFormIntegrator(rf_diffusion(), rf_time_mass())

    # `M(t) = t·M(1)`, so the weighted action scales with `1/t`.
    unit = let op = setup_operator(rf_seq(), rf_time_mass(), dh)
        update_operator!(op, nothing, TimeIntegrationContext(1.0, 1.0, 1.0))
        get_matrix(op)
    end
    at(t) = (t * unit) \ (K * u)

    @testset "a context-reading mass needs the initial pair at setup" begin
        err = @test_throws ArgumentError setup_operator(rf_seq(), rate, dh)
        @test occursin("carries no context", err.value.msg)
    end

    @testset "the initial pair reaches the mass, and a refill re-derives it" begin
        op = setup_operator(rf_seq(), rate, dh;
                            initial_context = TimeIntegrationContext(2.0, 1.0, 1.0))
        update_operator!(op, nothing, TimeIntegrationContext(2.0, 1.0, 1.0))
        @test rf_action(op, u) ≈ at(2.0) rtol = 1.0e-12
        update_operator!(op, nothing, TimeIntegrationContext(5.0, 1.0, 1.0))
        @test rf_action(op, u) ≈ at(5.0) rtol = 1.0e-12
        @test !isapprox(rf_action(op, u), at(2.0); rtol = 1.0e-3)
    end

    @testset "the matrix-free setup fills its store with the initial pair" begin
        op = setup_operator(rf_element_arm(), rate, dh;
                            initial_context = TimeIntegrationContext(2.0, 1.0, 1.0))
        @test rf_action(op, u) ≈ at(2.0) rtol = 1.0e-12
    end
end
