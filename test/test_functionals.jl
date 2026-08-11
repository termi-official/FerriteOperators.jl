using FerriteOperators
using Test
using Polyester
import LinearAlgebra: dot

# Diffusion-type element contributing two functionals: the Dirichlet energy
# ½∫∇u⋅∇u dΩ (scalar) and ∫∇u dΩ (a Vec).
struct FunctionalTestIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
struct FunctionalTestCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end
function FerriteOperators.setup_element_cache(m::FunctionalTestIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return FunctionalTestCache(CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::FunctionalTestCache) =
    FunctionalTestCache(FerriteOperators.duplicate_for_device(device, c.cv))
FerriteOperators.reinit_values!(c::FunctionalTestCache, cell) = reinit!(c.cv, cell)
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::FunctionalTestCache, args)
    (; cv) = cache
    uₑ = args.states.u
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u) * dΩ
        end
    end
end
function FerriteOperators.evaluate_cell_functional(::FunctionalKind{:energy}, cache::FunctionalTestCache, args)
    (; cv) = cache
    uₑ = args.states.u
    Φ = 0.0
    for qp in 1:getnquadpoints(cv)
        ∇u = function_gradient(cv, qp, uₑ)
        Φ += (∇u ⋅ ∇u) / 2 * getdetJdV(cv, qp)
    end
    return Φ
end
function FerriteOperators.evaluate_cell_functional(::FunctionalKind{:gradient_volume}, cache::FunctionalTestCache, args)
    (; cv) = cache
    uₑ = args.states.u
    g = zero(Vec{2, Float64})
    for qp in 1:getnquadpoints(cv)
        g += function_gradient(cv, qp, uₑ) * getdetJdV(cv, qp)
    end
    return g
end

@testset "Functional value requests" begin
    grid = generate_grid(Quadrilateral, (4, 3))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(2)
    n   = ndofs(dh)
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    op = setup_operator(strategy, FunctionalTestIntegrator(qrc, :u), dh)
    u = sin.(0.3 .* (1:n))

    @testset "scalar energy matches ½u'Ku" begin
        Kop = setup_operator(strategy, FerriteOperators.SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
        update_operator!(Kop, nothing)
        Φ = evaluate_functional(op, FunctionalKind(:energy), u, nothing)
        @test Φ ≈ dot(u, Kop.A * u) / 2 rtol = 1e-13
        @test Φ isa Float64
    end

    @testset "tensor-valued functional matches a hand-rolled reference" begin
        g = evaluate_functional(op, FunctionalKind(:gradient_volume), u, nothing)
        @test g isa Vec{2, Float64}

        ip = Lagrange{RefQuadrilateral, 1}()
        cv = CellValues(getquadraturerule(qrc, dh.subdofhandlers[1]), ip, ip)
        gref = zero(Vec{2, Float64})
        uₑ = zeros(4)
        for cell in CellIterator(dh)
            reinit!(cv, cell)
            uₑ .= u[celldofs(cell)]
            for qp in 1:getnquadpoints(cv)
                gref += function_gradient(cv, qp, uₑ) * getdetJdV(cv, qp)
            end
        end
        @test g ≈ gref rtol = 1e-14
    end

    @testset "parallel workers reduce to the same value" begin
        pop = setup_operator(PerColorAssemblyStrategy(PolyesterDevice(2)), FunctionalTestIntegrator(qrc, :u), dh)
        Φs = evaluate_functional(op, FunctionalKind(:energy), u, nothing)
        Φp = evaluate_functional(pop, FunctionalKind(:energy), u, nothing)
        @test Φs ≈ Φp rtol = 1e-12
        gp = evaluate_functional(pop, FunctionalKind(:gradient_volume), u, nothing)
        gs = evaluate_functional(op, FunctionalKind(:gradient_volume), u, nothing)
        @test gs ≈ gp rtol = 1e-12
    end

    @testset "repeated evaluation resets the accumulators" begin
        Φ1 = evaluate_functional(op, FunctionalKind(:energy), u, nothing)
        Φ2 = evaluate_functional(op, FunctionalKind(:energy), u, nothing)
        @test Φ1 == Φ2
    end

    @testset "missing kernel and undeclared slots fail loudly" begin
        @test_throws MethodError evaluate_functional(op, FunctionalKind(:enthalpy), u, nothing)
        @test_throws ArgumentError evaluate_functional(op, FunctionalKind(:energy), (u = u, uprev = copy(u)), nothing, nothing)
    end
end
