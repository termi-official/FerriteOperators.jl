using FerriteOperators
using FerriteOperatorsExampleElements
using Test
using Polyester
import LinearAlgebra: dot

include(joinpath(@__DIR__, "fixture_elements.jl"))

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

# Declared reduction value types: required under a parallel device, and what
# makes the fold typed from `zero(T)` on either device.
FerriteOperators.functional_value_type(::FunctionalKind{:energy}) = Float64
FerriteOperators.functional_value_type(::FunctionalKind{:gradient_volume}) = Vec{2, Float64}

# The same two integrands under undeclared tags, so the scan path stays covered
# and can be compared against the typed one.
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:energy_undeclared}, cache::FunctionalTestCache, args) =
    FerriteOperators.evaluate_cell_functional(FunctionalKind(:energy), cache, args)
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:gradient_undeclared}, cache::FunctionalTestCache, args) =
    FerriteOperators.evaluate_cell_functional(FunctionalKind(:gradient_volume), cache, args)

# A declaration disagreeing with its kernel.
FerriteOperators.functional_value_type(::FunctionalKind{:mistyped}) = Float64
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:mistyped}, cache::FunctionalTestCache, args) =
    zero(Vec{2, Float64})

# Kernels that run on every cell of a non-empty domain and contribute nothing:
# an empty sum, which only the declared tag can answer.
FerriteOperators.functional_value_type(::FunctionalKind{:quiet}) = Float64
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:quiet}, cache::FunctionalTestCache, args) = nothing
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:quiet_undeclared}, cache::FunctionalTestCache, args) = nothing

# An integrator whose subdomains cannot contribute at all.
struct EmptyCacheIntegrator <: AbstractNonlinearIntegrator end
FerriteOperators.setup_element_cache(::EmptyCacheIntegrator, sdh::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()

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
        Kop = setup_operator(strategy, SimpleBilinearDiffusionIntegrator(1.0, qrc, :u), dh)
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

    @testset "repeated evaluation carries nothing over" begin
        Φ1 = evaluate_functional(op, FunctionalKind(:energy), u, nothing)
        Φ2 = evaluate_functional(op, FunctionalKind(:energy), u, nothing)
        @test Φ1 == Φ2
    end

    @testset "declared value type reproduces the scan path" begin
        @test evaluate_functional(op, FunctionalKind(:energy), u, nothing) ===
              evaluate_functional(op, FunctionalKind(:energy_undeclared), u, nothing)
        @test evaluate_functional(op, FunctionalKind(:gradient_volume), u, nothing) ===
              evaluate_functional(op, FunctionalKind(:gradient_undeclared), u, nothing)
    end

    @testset "the per-worker fold is type stable" begin
        ws     = first_workspace(op)
        task   = FerriteOperators.AssemblyTask(FunctionalKind(:energy), nothing, (u = u,), nothing, nothing)
        untask = FerriteOperators.AssemblyTask(FunctionalKind(:energy_undeclared), nothing, (u = u,), nothing, nothing)
        # The driver body returns the cell value rather than parking it.
        @test last(only(code_typed(FerriteOperators.functional_cell_sweep,
                                   Tuple{FunctionalKind{:energy}, typeof(task), typeof(ws)}))) === Float64
        # Declared: the fold is Float64-typed from the seed, no `Nothing` in the
        # return, and the seed itself is the concretely typed additive identity.
        @test last(only(code_typed(FerriteOperators.fold_items,
                                   Tuple{typeof(task), typeof(ws), Vector{Int}}))) === Float64
        @test FerriteOperators.initial_partial(FunctionalKind(:energy)) === 0.0
        @test FerriteOperators.initial_partial(FunctionalKind(:gradient_volume)) === zero(Vec{2, Float64})
        # …so the parallel route's partials array is concretely typed.
        @test typeof(zeros(FerriteOperators.functional_value_type(FunctionalKind(:gradient_volume)), 3)) ===
              Vector{Vec{2, Float64}}
        # Undeclared: the first value fixes the accumulator, so the loop doing
        # the work still carries a concrete Float64 and dispatches nothing per cell.
        @test last(only(code_typed(FerriteOperators.fold_items,
                                   Tuple{typeof(untask), typeof(ws), Vector{Int}}))) === Union{Nothing, Float64}
        @test last(only(code_typed(FerriteOperators._fold_items_from,
                                   Tuple{typeof(untask), typeof(ws), Vector{Int}, Int, Float64, Type{Nothing}}))) === Float64
    end

    @testset "the value-type declaration is a loud contract" begin
        # A kernel disagreeing with the declaration fails by name.
        @test_throws ArgumentError evaluate_functional(op, FunctionalKind(:mistyped), u, nothing)
        # An undeclared kind cannot run in parallel: the partials are allocated
        # before the batch, so the type must be known up front.
        pop = setup_operator(PerColorAssemblyStrategy(PolyesterDevice(2)), FunctionalTestIntegrator(qrc, :u), dh)
        @test_throws ArgumentError evaluate_functional(pop, FunctionalKind(:energy_undeclared), u, nothing)
        # …while the declared one does.
        @test evaluate_functional(pop, FunctionalKind(:energy), u, nothing) isa Float64
    end

    @testset "structural emptiness fails before any cell runs" begin
        # No items to reduce over — the same error whether or not the kind
        # declares a value type, because neither can integrate over nothing.
        sc = first(op.engine.subdomain_caches)
        empty_partition = (engine = FerriteOperators.AssemblyEngine(
            op.engine.strategy,
            [FerriteOperators.SubdomainCache(sc.domain, sc.device_cache, (Int[],))],
            op.engine.dh, op.engine.ivh, op.engine.protocol),)
        @test_throws ArgumentError evaluate_functional(empty_partition, FunctionalKind(:energy), u, nothing)
        @test_throws ArgumentError evaluate_functional(empty_partition, FunctionalKind(:energy_undeclared), u, nothing)

        # Items exist but no subdomain can contribute: a type-level verdict.
        eop = setup_operator(strategy, EmptyCacheIntegrator(), dh)
        @test_throws ArgumentError evaluate_functional(eop, FunctionalKind(:energy), u, nothing)
        @test_throws ArgumentError evaluate_functional(eop, FunctionalKind(:energy_undeclared), u, nothing)
    end

    @testset "an all-quiet sweep is an empty sum" begin
        # The domain and the caches are fine; the kernels simply contribute
        # nothing. Declared, that is the additive identity…
        @test evaluate_functional(op, FunctionalKind(:quiet), u, nothing) === 0.0
        # …undeclared, there is no type to take the identity of.
        @test_throws ArgumentError evaluate_functional(op, FunctionalKind(:quiet_undeclared), u, nothing)
    end

    @testset "missing kernel and undeclared slots fail loudly" begin
        @test_throws MethodError evaluate_functional(op, FunctionalKind(:enthalpy), u, nothing)
        @test_throws ArgumentError evaluate_functional(op, FunctionalKind(:energy), (u = u, uprev = copy(u)), nothing, nothing)
    end
end
