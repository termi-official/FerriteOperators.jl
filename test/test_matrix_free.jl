using FerriteOperators
using FerriteOperatorsExampleElements
using FerriteOperatorsTensorProduct
using Test
using LinearAlgebra
using Polyester
# FerriteKAExt — which supplies the device handler and every Adapt rule the
# device kernels build on — is triggered by these four together.
import Adapt, GPUArrays, GPUArraysCore
import KernelAbstractions as KA

# Deterministic, dependency-free probes.
probe(::Type{T}, n, k) where {T} = T[sin(T(0.7) * k * i + T(0.3) * k) for i in 1:n]
wobble(::Type{T}, node, d) where {T} = T(sin(2.7 * node + 1.3 * d))

# Perturbed interior nodes: the element must not assume an affine map.
function distorted_testbed(cellT, interpolation, ::Type{T}, dims, p; distortion = 0.15) where {T}
    dim = length(dims)
    grid = generate_grid(cellT, dims, Vec{dim}(ntuple(_ -> -one(T), dim)),
                         Vec{dim}(ntuple(_ -> one(T), dim)))
    h = T(2) / maximum(dims)
    nodes = [Ferrite.Node(Vec{dim, T}(ntuple(d -> node.x[d] +
                (all(abs.(node.x) .< 1 - 1.0f-4) ? T(distortion) * h * wobble(T, i, d) : zero(T)), dim)))
             for (i, node) in enumerate(Ferrite.getnodes(grid))]
    dh = DofHandler(Grid(Ferrite.getcells(grid), nodes))
    add!(dh, :u, interpolation(p))
    close!(dh)
    return dh
end

matrix_free_ka(::Type{T}, mapping; scheduling = SequentialScheduling(), storage = Stored()) where {T} = AssemblyStrategy(
    MatrixFreeAction(; element_mapping = mapping, storage), scheduling,
    KernelAbstractionsDevice(KA.CPU(); value_type = T, index_type = Int,
                             items_per_worker = 2, max_workgroup_size = 8))

####################################
## A MatrixKernelFill + SymmetricElementMatrix() fixture
####################################
#
# `SimpleBilinearDiffusionIntegrator`'s cache verbatim, plus the symmetry
# declaration. The sum-factorized caches below have no analytic Jacobian kernel,
# so they never exercise the MatrixKernelFill route under ElementAssembly();
# this fixture covers the packed election's per-worker ndofs² scratch fill path.
struct SymmetricAnalyticDiffusionCache{CV} <: FerriteOperators.AbstractVolumetricElementCache
    D::Float64
    cellvalues::CV
end
Ferrite.getnquadpoints(e::SymmetricAnalyticDiffusionCache) = getnquadpoints(e.cellvalues)
FerriteOperators.reinit_values!(e::SymmetricAnalyticDiffusionCache, cell) = Ferrite.reinit!(e.cellvalues, cell)
FerriteOperators.element_value_type(c::SymmetricAnalyticDiffusionCache) = element_value_type(c.cellvalues)
FerriteOperators.provides_analytic(::Type{<:SymmetricAnalyticDiffusionCache}, ::JacobianKind{:u}) = true
FerriteOperators.element_matrix_symmetry(::SymmetricAnalyticDiffusionCache) = SymmetricElementMatrix()

function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, cache::SymmetricAnalyticDiffusionCache, args::CellArgs)
    Kₑ = req.K
    (; cellvalues, D) = cache
    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:getnbasefunctions(cellvalues)
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            for j in 1:getnbasefunctions(cellvalues)
                ∇Nⱼ = shape_gradient(cellvalues, qp, j)
                Kₑ[i, j] += D * ∇Nⱼ ⋅ ∇Nᵢ * dΩ
            end
        end
    end
end
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::SymmetricAnalyticDiffusionCache, args::CellArgs)
    (; cellvalues, D) = cache
    uₑ = args.states.u
    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        ∇u = function_gradient(cellvalues, qp, uₑ)
        for i in 1:getnbasefunctions(cellvalues)
            req.r[i] += D * (∇u ⋅ shape_gradient(cellvalues, qp, i)) * dΩ
        end
    end
end

struct SymmetricAnalyticDiffusionIntegrator <: AbstractBilinearIntegrator
    D::Float64
    qrc::QuadratureRuleCollection
    field_name::Symbol
end
function FerriteOperators.setup_element_cache(m::SymmetricAnalyticDiffusionIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(m.qrc, sdh)
    T      = element_value_type(m.qrc)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return SymmetricAnalyticDiffusionCache(m.D, CellValues(T, qr, ip, ip_geo))
end

####################################
## A downstream decorator over a matrix-free element
####################################
#
# Written entirely outside `src/`: it forwards what a decorator OWNS (the
# kernels it serves, and its claims about them) and declares nothing about the
# wrapped ELEMENT — the blanket-forward half of
# `AbstractElementCacheDecorator`, storage election and store fill included.
struct PassthroughDecorator{I} <: FerriteOperators.AbstractElementCacheDecorator{I}
    inner::I
end
FerriteOperators.assemble_cell!(req, d::PassthroughDecorator, args) =
    FerriteOperators.assemble_cell!(req, d.inner, args)
FerriteOperators.provides_analytic(::Type{<:PassthroughDecorator{I}}, kind) where {I} =
    FerriteOperators.provides_analytic(I, kind)
FerriteOperators.serves_kind(::Type{<:PassthroughDecorator{I}}, kind) where {I} =
    FerriteOperators.serves_kind(I, kind)
FerriteOperators.apply_element_action!(y, d::PassthroughDecorator, u, args::CellArgs) =
    FerriteOperators.apply_element_action!(y, d.inner, u, args)

struct DecoratedIntegrator{I} <: AbstractBilinearIntegrator
    inner::I
end
FerriteOperators.setup_element_cache(m::DecoratedIntegrator, sdh::SubDofHandler) =
    PassthroughDecorator(FerriteOperators.setup_element_cache(m.inner, sdh))

####################################
## An element whose store is allocated EAGERLY and filled only by the fill sweep
####################################
#
# The store exists whatever the election says, starts at zero, and the action
# reads it, so `with_action_storage` cannot mask an unforwarded
# `fill_quadrature_data!`. A fill that does not reach the element gives `y == 0`
# — no error and no `MethodError`, which is why this is a test and not a
# capability wall. The action is `Kₑ = f·I`, so the reference is closed form.
struct EagerStoreCache{S} <: FerriteOperators.AbstractVolumetricElementCache
    factor::Float64
    store::S          # one scalar per cell, written by `fill_quadrature_data!`
end
FerriteOperators.reinit_values!(::EagerStoreCache, cell) = nothing
FerriteOperators.assemble_cell!(::ResidualRequest, ::EagerStoreCache, ::CellArgs) = nothing
FerriteOperators.fill_quadrature_data!(c::EagerStoreCache, args::CellArgs) =
    (c.store[cellid(args.cell)] = c.factor; nothing)
function FerriteOperators.apply_element_action!(yₑ, c::EagerStoreCache, uₑ, args::CellArgs)
    f = @inbounds c.store[cellid(args.cell)]
    for i in eachindex(yₑ)
        yₑ[i] += f * uₑ[i]
    end
    return nothing
end

struct EagerStoreIntegrator <: AbstractBilinearIntegrator
    factor::Float64
end
FerriteOperators.setup_element_cache(m::EagerStoreIntegrator, sdh::SubDofHandler) =
    EagerStoreCache(m.factor, zeros(getncells(Ferrite.get_grid(sdh.dh))))

# From INSIDE a function: at testset scope the block's own boxing would count.
function action_allocations(op, y, u)
    mul!(y, op, u)
    mul!(y, op, u)
    return @allocated mul!(y, op, u)
end

@testset "MatrixFreeAction" begin
    # ONE element definition under every execution mapping the strategy axis
    # offers, against the assembled matrix of the same form.
    @testset "action matches the assembled operator ($label, $T, p = $p)" for
            (label, cellT, interpolation, dims) in (
                ("quad", Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), (4, 3)),
                ("hex",  Hexahedron,    o -> Lagrange{RefHexahedron, o}(),    (3, 2, 2))),
            T in (Float64, Float32), p in 1:3

        dh   = distorted_testbed(cellT, interpolation, T, dims, p)
        qrc  = QuadratureRuleCollection(T, p + 1)
        rtol = T === Float32 ? 1.0f-3 : 1.0e-11

        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice{T, Int}()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(T, ndofs(dh), 7)
        reference = assembled.A * u

        integrator = SumFactorizedDiffusionIntegrator(T(2.5), qrc, :u)
        @testset "$arm" for (arm, strategy) in (
                ("sequential", AssemblyStrategy(SequentialCPUDevice{T, Int}(); form = MatrixFreeAction())),
                ("polyester",  AssemblyStrategy(PolyesterDevice{T, Int}(4); form = MatrixFreeAction(),
                                                scheduling = ColoredScheduling())),
                ("KA worker-per-element", matrix_free_ka(T, WorkerPerElement())),
                ("KA cooperative",        matrix_free_ka(T, CooperativeElement())))
            op = setup_operator(strategy, integrator, dh)
            @test size(op) == (ndofs(dh), ndofs(dh))
            @test size(op, 1) == ndofs(dh)
            @test eltype(op) === T
            y = zeros(T, ndofs(dh))
            mul!(y, op, u)
            @test y ≈ reference rtol = rtol
        end
    end

    # The second consumer of the sum-factorization core: a different pointwise
    # map (a scalar on the interpolated value rather than a tensor on the
    # reference gradient) over the same contractions, mappings and elections.
    @testset "the mass action matches the assembled mass operator ($label, $T, p = $p)" for
            (label, cellT, interpolation, dims) in (
                ("quad", Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), (4, 3)),
                ("hex",  Hexahedron,    o -> Lagrange{RefHexahedron, o}(),    (3, 2, 2))),
            T in (Float64, Float32), p in 1:3

        dh   = distorted_testbed(cellT, interpolation, T, dims, p)
        qrc  = QuadratureRuleCollection(T, p + 1)
        rtol = T === Float32 ? 1.0f-3 : 1.0e-11

        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice{T, Int}()),
                                   SimpleBilinearMassIntegrator(1.7, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(T, ndofs(dh), 7)
        reference = assembled.A * u

        integrator = SumFactorizedMassIntegrator(1.7, qrc, :u)
        @testset "$arm" for (arm, strategy) in (
                ("sequential", AssemblyStrategy(SequentialCPUDevice{T, Int}(); form = MatrixFreeAction())),
                ("polyester",  AssemblyStrategy(PolyesterDevice{T, Int}(4); form = MatrixFreeAction(),
                                                scheduling = ColoredScheduling())),
                ("KA worker-per-element", matrix_free_ka(T, WorkerPerElement())),
                ("KA cooperative",        matrix_free_ka(T, CooperativeElement())))
            op = setup_operator(strategy, integrator, dh)
            y = zeros(T, ndofs(dh))
            mul!(y, op, u)
            @test y ≈ reference rtol = rtol
        end
    end

    # The three storage levels are three ways to keep the same operator.
    # `Stored()` reads the pointwise factor the fill sweep formed and
    # `Recompute()` forms it at the point of use — the SAME expression, so the
    # two reproduce bit for bit on the sequential arm. `ElementAssembly()`
    # contracts differently (a dense product over columns the fill built from
    # the action), so it agrees to a tolerance and not bitwise.
    @testset "the storage levels agree ($(nameof(typeof(integrator))), p = $p)" for
            integrator in (SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u),
                           SumFactorizedMassIntegrator(1.7, QuadratureRuleCollection(3), :u)),
            p in 1:3

        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), p)
        u = probe(Float64, ndofs(dh), 5)
        actions = map((Stored(), Recompute(), ElementAssembly())) do storage
            op = setup_operator(AssemblyStrategy(SequentialCPUDevice();
                                                 form = MatrixFreeAction(; storage)), integrator, dh)
            y = zeros(ndofs(dh))
            mul!(y, op, u)
            y
        end
        @test actions[1] == actions[2]
        @test actions[3] ≈ actions[1] rtol = 1.0e-11
    end

    @testset "the storage levels agree on the device backend" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        integrator = SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u)
        u = probe(Float64, ndofs(dh), 5)
        action(strategy) = (y = zeros(ndofs(dh)); mul!(y, setup_operator(strategy, integrator, dh), u); y)
        @testset "$(nameof(typeof(mapping)))" for mapping in (WorkerPerElement(), CooperativeElement())
            @test action(matrix_free_ka(Float64, mapping; storage = Stored())) ≈
                  action(matrix_free_ka(Float64, mapping; storage = Recompute())) rtol = 1.0e-11
        end
        # The ELEMENT level is worker-per-element only.
        @test action(matrix_free_ka(Float64, WorkerPerElement(); storage = ElementAssembly())) ≈
              action(matrix_free_ka(Float64, WorkerPerElement(); storage = Stored())) rtol = 1.0e-11
    end

    # Two device-cursor shortcuts are elected from a subdomain's shape alone: a
    # sequential partition's chunk is a `UnitRange` where the cellset is
    # contiguous, and a cursor's dof-window offset is arithmetic where
    # `cell_dofs_offset` is affine in the cell id. Every test above runs one
    # contiguous, single-order subdomain and takes both; these two put the
    # subdomain out of that shape and check each fallback.
    @testset "the action matches the assembled operator over a non-contiguous cellset" begin
        grid = generate_grid(Hexahedron, (3, 2, 2), Vec{3}((-1.0, -1.0, -1.0)), Vec{3}((1.0, 1.0, 1.0)))
        ncells = getncells(grid)
        addcellset!(grid, "odd", Set(1:2:ncells))
        addcellset!(grid, "even", Set(2:2:ncells))

        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, getcellset(grid, "odd"));  add!(sdh1, :u, Lagrange{RefHexahedron, 1}())
        sdh2 = SubDofHandler(dh, getcellset(grid, "even")); add!(sdh2, :u, Lagrange{RefHexahedron, 1}())
        close!(dh)

        qrc = QuadratureRuleCollection(2)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 5)
        reference = assembled.A * u

        integrator = SumFactorizedDiffusionIntegrator(2.5, qrc, :u)
        op = setup_operator(matrix_free_ka(Float64, WorkerPerElement(); storage = ElementAssembly()), integrator, dh)
        y = zeros(ndofs(dh))
        mul!(y, op, u)
        @test y ≈ reference rtol = 1.0e-11

        # Neither subdomain's chunk is contiguous, so the `Vector` fallback stands.
        @test all(sc -> all(chunk -> chunk isa Vector{Int}, sc.partition), get_subdomain_caches(op))

        lanes = setup_operator(matrix_free_ka(Float64, LanesPerElement(); storage = ElementAssembly()),
                               integrator, dh)
        fill!(y, 0.0)
        mul!(y, lanes, u)
        @test y ≈ reference rtol = 1.0e-11
    end

    @testset "the action matches the assembled operator over a mixed-order subdomain" begin
        grid = generate_grid(Hexahedron, (4, 2, 2), Vec{3}((-1.0, -1.0, -1.0)), Vec{3}((1.0, 1.0, 1.0)))
        ncells = getncells(grid)
        k = ncells ÷ 2
        addcellset!(grid, "p1", Set(1:k))
        addcellset!(grid, "p2", Set((k + 1):ncells))

        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, getcellset(grid, "p1")); add!(sdh1, :u, Lagrange{RefHexahedron, 1}())
        sdh2 = SubDofHandler(dh, getcellset(grid, "p2")); add!(sdh2, :u, Lagrange{RefHexahedron, 2}())
        close!(dh)

        qrc = QuadratureRuleCollection(3)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 5)
        reference = assembled.A * u

        integrator = SumFactorizedDiffusionIntegrator(2.5, qrc, :u)
        op = setup_operator(matrix_free_ka(Float64, WorkerPerElement(); storage = ElementAssembly()), integrator, dh)
        y = zeros(ndofs(dh))
        mul!(y, op, u)
        @test y ≈ reference rtol = 1.0e-11

        # Different extents, so the lane mapping launches a different count for each.
        lanes = setup_operator(matrix_free_ka(Float64, LanesPerElement(); storage = ElementAssembly()),
                               integrator, dh)
        fill!(y, 0.0)
        mul!(y, lanes, u)
        @test y ≈ reference rtol = 1.0e-11
    end

    # The device sweeps position their items on ONE iterator but refresh what
    # each KIND reads: the stored levels' action reads a cell id and a dof
    # range, while the quadrature-data fill forms the geometry. Interleaving the
    # two catches an action leaving the iterator in a state the next fill reads,
    # or a fill's staging being skipped.
    @testset "the action and the fill share one device iterator ($(nameof(typeof(storage))))" for
            storage in (Stored(), Recompute(), ElementAssembly())

        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 5)
        reference = assembled.A * u

        op = setup_operator(matrix_free_ka(Float64, WorkerPerElement(); storage),
                            SumFactorizedDiffusionIntegrator(2.5, qrc, :u), dh)
        y = zeros(ndofs(dh))
        for _ in 1:2
            mul!(y, op, u)
            @test y ≈ reference rtol = 1.0e-11
            update_operator!(op, nothing)
        end
        mul!(y, op, u)
        @test y ≈ reference rtol = 1.0e-11
    end

    # The fill sweep carries no assembler, so its task can have every field a
    # singleton, and a parallel device must not assume per-worker task state it
    # can index. `min_items_per_worker = 1` puts more than one worker on this
    # mesh's largest colour, where that assumption would show.
    @testset "the fill sweep runs on several workers ($(nameof(typeof(storage))))" for
            storage in (Stored(), ElementAssembly())

        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 5)
        op = setup_operator(
            AssemblyStrategy(MatrixFreeAction(; storage), ColoredScheduling(), PolyesterDevice{Float64, Int}(1)),
            SumFactorizedDiffusionIntegrator(2.5, qrc, :u), dh)
        y = zeros(ndofs(dh))
        mul!(y, op, u)
        @test y ≈ assembled.A * u rtol = 1.0e-11
        update_operator!(op, nothing)
        mul!(y, op, u)
        @test y ≈ assembled.A * u rtol = 1.0e-11
    end

    # The ELEMENT level is element-agnostic: it keeps the dense matrices the
    # element's own kernels produce, so a cache with an element-matrix kernel and
    # NO matrix-free kernel serves it too.
    @testset "the ELEMENT level serves a cache with no matrix-free kernel" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        u = probe(Float64, ndofs(dh), 5)
        @testset "$(nameof(typeof(integrator)))" for integrator in (
                SimpleBilinearDiffusionIntegrator(2.5, qrc, :u),
                SimpleBilinearMassIntegrator(1.7, qrc, :u))
            assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()), integrator, dh)
            update_operator!(assembled, nothing)
            op = setup_operator(AssemblyStrategy(SequentialCPUDevice();
                                                 form = MatrixFreeAction(; storage = ElementAssembly())),
                                integrator, dh)
            y = zeros(ndofs(dh))
            mul!(y, op, u)
            @test y ≈ assembled.A * u rtol = 1.0e-12
            # Neither cache declares `element_matrix_symmetry`, so `K` stays dense.
            cache = get_subdomain_caches(op)[1].domain.element
            @test cache.symmetry isa GeneralElementMatrix
            @test ndims(cache.K) == 3
        end
    end

    # The ELEMENT level over the sum-factorized cache, whose matrices the fill
    # builds from the action itself.
    @testset "the ELEMENT level matches the assembled operator ($label, $T, p = $p)" for
            (label, cellT, interpolation, dims) in (
                ("quad", Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), (4, 3)),
                ("hex",  Hexahedron,    o -> Lagrange{RefHexahedron, o}(),    (3, 2, 2))),
            T in (Float64, Float32), p in 1:3

        dh   = distorted_testbed(cellT, interpolation, T, dims, p)
        qrc  = QuadratureRuleCollection(T, p + 1)
        rtol = T === Float32 ? 1.0f-3 : 1.0e-11
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice{T, Int}()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(T, ndofs(dh), 7)
        reference = assembled.A * u

        integrator = SumFactorizedDiffusionIntegrator(T(2.5), qrc, :u)
        ea = MatrixFreeAction(; storage = ElementAssembly())
        @testset "$arm" for (arm, strategy) in (
                ("sequential", AssemblyStrategy(SequentialCPUDevice{T, Int}(); form = ea)),
                ("polyester",  AssemblyStrategy(PolyesterDevice{T, Int}(4); form = ea,
                                                scheduling = ColoredScheduling())),
                ("KA worker-per-element", matrix_free_ka(T, WorkerPerElement(); storage = ElementAssembly())),
                ("KA lanes-per-element",  matrix_free_ka(T, LanesPerElement(); storage = ElementAssembly())))
            op = setup_operator(strategy, integrator, dh)
            y = zeros(T, ndofs(dh))
            mul!(y, op, u)
            @test y ≈ reference rtol = rtol
            # Isotropic D is symmetric, so the tensor-product cache declares
            # SymmetricElementMatrix() and this arm runs the PACKED layout.
            cache = get_subdomain_caches(op)[1].domain.element
            @test cache.symmetry isa SymmetricElementMatrix
            @test ndims(cache.K) == 2
            @test size(cache.K, 2) == (ndofs_per_cell(dh.subdofhandlers[1]) * (ndofs_per_cell(dh.subdofhandlers[1]) + 1)) ÷ 2
        end
    end

    # The MatrixKernelFill route (an analytic element-matrix kernel, unlike the
    # sum-factorized caches above, which only fill via the action) under a
    # symmetric election — the per-worker ndofs² scratch fill path. Same physics
    # as `SimpleBilinearDiffusionIntegrator`, so the packed action and the dense
    # one it is compared against are the SAME `Kₑ` through the SAME kernel,
    # differing only in the symmetry declaration.
    @testset "ElementAssembly, MatrixKernelFill route, packed matches dense ($T, p = $p)" for
            T in (Float64, Float32), p in 1:3

        dh   = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), T, (3, 2, 2), p)
        qrc  = QuadratureRuleCollection(T, p + 1)
        rtol = T === Float32 ? 1.0f-3 : 1.0e-11
        u = probe(T, ndofs(dh), 7)

        dense_op = setup_operator(AssemblyStrategy(SequentialCPUDevice{T, Int}();
                                                    form = MatrixFreeAction(; storage = ElementAssembly())),
                                  SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        y_dense = zeros(T, ndofs(dh))
        mul!(y_dense, dense_op, u)
        dense_cache = get_subdomain_caches(dense_op)[1].domain.element
        @test dense_cache.symmetry isa GeneralElementMatrix
        @test dense_cache.route isa FerriteOperators.MatrixKernelFill

        packed_op = setup_operator(AssemblyStrategy(SequentialCPUDevice{T, Int}();
                                                     form = MatrixFreeAction(; storage = ElementAssembly())),
                                   SymmetricAnalyticDiffusionIntegrator(2.5, qrc, :u), dh)
        y_packed = zeros(T, ndofs(dh))
        mul!(y_packed, packed_op, u)
        packed_cache = get_subdomain_caches(packed_op)[1].domain.element
        @test packed_cache.symmetry isa SymmetricElementMatrix
        @test packed_cache.route isa FerriteOperators.MatrixKernelFill

        @test y_packed ≈ y_dense rtol = rtol
    end

    # The lane mapping reads a ROW of the stored matrix, and the two layouts put
    # a row in different places: dense walks `K[slot, i, :]`, packed walks the
    # triangle through `_packed_index`. Both are checked against the assembled
    # matrix AND against `WorkerPerElement` on the same store, so a wrongly-read
    # layout cannot hide behind a matching reference.
    @testset "the lane mapping serves both element-matrix layouts ($T, p = $p)" for
            T in (Float64, Float32), p in 1:3

        dh   = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), T, (3, 2, 2), p)
        qrc  = QuadratureRuleCollection(T, p + 1)
        rtol = T === Float32 ? 1.0f-3 : 1.0e-11
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice{T, Int}()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(T, ndofs(dh), 7)
        reference = assembled.A * u

        @testset "$layout" for (layout, integrator, symmetry) in (
                ("packed", SumFactorizedDiffusionIntegrator(T(2.5), qrc, :u), SymmetricElementMatrix),
                ("dense",  SimpleBilinearDiffusionIntegrator(2.5, qrc, :u),   GeneralElementMatrix))

            worker = setup_operator(matrix_free_ka(T, WorkerPerElement(); storage = ElementAssembly()),
                                    integrator, dh)
            y_worker = zeros(T, ndofs(dh))
            mul!(y_worker, worker, u)
            @test get_subdomain_caches(worker)[1].domain.element.symmetry isa symmetry

            lanes = setup_operator(matrix_free_ka(T, LanesPerElement(); storage = ElementAssembly()),
                                   integrator, dh)
            y_lanes = zeros(T, ndofs(dh))
            mul!(y_lanes, lanes, u)
            @test y_lanes ≈ reference rtol = rtol
            @test y_lanes ≈ y_worker rtol = rtol
        end
    end

    # `nlanes` is a launch policy, not element math: every count gives the same
    # action, whether it matches the element's extent, divides it, or exceeds it
    # (leaving lanes with no row). The wide group additionally puts SEVERAL
    # elements in one workgroup — the geometry a small element runs — where the
    # narrow one leaves a single block per group.
    @testset "the lane count is a launch policy alone (lanes = $lanes, group = $max_group)" for
            (lanes, max_group) in ((nothing, 8), (nothing, 64), (1, 8), (5, 8),
                                   (5, 64), (27, 64), (64, 64))

        dh  = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 7)

        device = KernelAbstractionsDevice(KA.CPU(); value_type = Float64, index_type = Int,
                                          items_per_worker = 2, max_workgroup_size = max_group)
        op = setup_operator(AssemblyStrategy(
                MatrixFreeAction(; element_mapping = LanesPerElement(; lanes), storage = ElementAssembly()),
                SequentialScheduling(), device),
            SumFactorizedDiffusionIntegrator(2.5, qrc, :u), dh)
        y = zeros(ndofs(dh))
        mul!(y, op, u)
        @test y ≈ assembled.A * u rtol = 1.0e-11
    end

    # The hazard `element_matrix_symmetry`'s docstring warns about: a
    # declared-symmetric element whose assembled `Kₑ` is not symmetric would
    # silently symmetrize the operator. Checked on a genuinely off-diagonally
    # coupled `D`, a diagonal coefficient never exercising `_packed_index`'s
    # off-diagonal arithmetic at all.
    @testset "a declared-symmetric element's ElementAssembly operator stays symmetric (anisotropic D)" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        D = SymmetricTensor{2, 3}((2.0, 0.3, -0.2, 1.4, 0.1, 3.1))
        integrator = SumFactorizedDiffusionIntegrator(D, qrc, :u)

        op = setup_operator(AssemblyStrategy(SequentialCPUDevice();
                                             form = MatrixFreeAction(; storage = ElementAssembly())),
                            integrator, dh)
        cache = get_subdomain_caches(op)[1].domain.element
        @test cache.symmetry isa SymmetricElementMatrix
        @test ndims(cache.K) == 2

        u = probe(Float64, ndofs(dh), 3)
        v = probe(Float64, ndofs(dh), 4)
        Au, Av = zeros(ndofs(dh)), zeros(ndofs(dh))
        mul!(Au, op, u)
        mul!(Av, op, v)
        @test dot(v, Au) ≈ dot(u, Av) rtol = 1.0e-12

        # Independent reference: the SAME anisotropic form under `Stored()`, a
        # different code path — sum-factorized contraction, no dense `Kₑ`.
        stored = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()), integrator, dh)
        y_stored = zeros(ndofs(dh))
        mul!(y_stored, stored, u)
        @test Au ≈ y_stored rtol = 1.0e-11
    end

    # The same freshness contract an assembled operator has.
    @testset "update_operator! refills the quadrature-data store" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        integrator = SumFactorizedDiffusionIntegrator(2.5, qrc, :u)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 5)
        reference = assembled.A * u

        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()), integrator, dh)
        y = zeros(ndofs(dh))
        mul!(y, op, u)
        @test y ≈ reference rtol = 1.0e-11

        store = get_subdomain_caches(op)[1].domain.element.qdata
        fill!(store.data, zero(eltype(store.data)))
        mul!(y, op, u)
        @test iszero(y)

        update_operator!(op, nothing)
        mul!(y, op, u)
        @test y ≈ reference rtol = 1.0e-11

        recomputing = setup_operator(
            AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction(; storage = Recompute())),
            integrator, dh)
        @test update_operator!(recomputing, nothing) === nothing
        @test get_subdomain_caches(recomputing)[1].domain.element.qdata === nothing
    end

    @testset "colored scatter on the device backend" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 11)
        integrator = SumFactorizedDiffusionIntegrator(2.5, qrc, :u)
        # The vector scatter is atomic under `SequentialScheduling` and plain
        # under `ColoredScheduling`; both are race-free and must agree. The lane
        # mapping adds one reason: the lanes of ONE element scatter to different
        # rows of the local system, which are different dofs.
        for (mapping, storage) in ((WorkerPerElement(), Stored()),
                                   (CooperativeElement(), Stored()),
                                   (LanesPerElement(), ElementAssembly()))
            op = setup_operator(matrix_free_ka(Float64, mapping; storage,
                                               scheduling = ColoredScheduling()), integrator, dh)
            y = zeros(ndofs(dh))
            mul!(y, op, u)
            @test y ≈ assembled.A * u rtol = 1.0e-11
        end
    end

    @testset "the sequential arm is deterministic and allocation-free" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (4, 4, 4), 2)
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u), dh)
        u = probe(Float64, ndofs(dh), 13)
        y, z = zeros(ndofs(dh)), zeros(ndofs(dh))
        mul!(y, op, u)
        mul!(z, op, u)
        # No atomics and a fixed item order, so the two sweeps agree bit for bit.
        @test y == z
        @test (@allocated mul!(y, op, u)) == 0
    end

    @testset "every ELEMENT-level arm's per-mul! host allocations stay O(1)" begin
        # The gate the `Stored()` arm above carries, extended to the ELEMENT
        # level's arms. It protects an INVARIANT rather than a byte budget: an
        # action allocates per CALL and never per item, so the count must not
        # follow the cell count.
        packed(o) = SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(o + 1), :u)
        dense(o) = SimpleBilinearDiffusionIntegrator(2.5, QuadratureRuleCollection(o + 1), :u)
        ea = MatrixFreeAction(; storage = ElementAssembly())

        for (layout, build) in ("packed" => packed, "dense" => dense)
            @testset "sequential, $layout" begin
                dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (4, 4, 4), 2)
                op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = ea), build(2), dh)
                @test action_allocations(op, zeros(ndofs(dh)), probe(Float64, ndofs(dh), 23)) == 0
            end
        end

        # The device arms track one workgroup object per launch, a per-CALL
        # constant. Gated for the PACKED layout on both mappings, where that
        # holds on every supported Julia. The DENSE arms are NOT gated: on
        # Julia 1.10 their per-worker views of the analytic cache allocate —
        # between a 27-cell and a 216-cell mesh the count grows by 96 kB
        # (worker) and 774 kB (lanes) — while Julia 1.12 keeps both flat at
        # ~3 kB, and no single bound describes both.
        for mapping in (WorkerPerElement(), LanesPerElement())
            @testset "KA $(nameof(typeof(mapping))), packed" begin
                counts = map(((3, 3, 3), (6, 6, 6))) do dims
                    dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, dims, 2)
                    op = setup_operator(matrix_free_ka(Float64, mapping; storage = ElementAssembly()),
                                        packed(2), dh)
                    action_allocations(op, zeros(ndofs(dh)), probe(Float64, ndofs(dh), 23))
                end
                @test all(<(16_384), counts)
                # 8x the cells; the count must not follow.
                @test counts[2] - counts[1] < 1024
            end
        end
    end

    @testset "a downstream decorator inherits the wrapped element's storage election" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        qrc = QuadratureRuleCollection(3)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, qrc, :u), dh)
        update_operator!(assembled, nothing)
        u = probe(Float64, ndofs(dh), 17)
        inner = SumFactorizedDiffusionIntegrator(2.5, qrc, :u)

        for storage in (Stored(), Recompute(), ElementAssembly())
            op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction(; storage)),
                                DecoratedIntegrator(inner), dh)
            y = zeros(ndofs(dh))
            mul!(y, op, u)
            @test y ≈ assembled.A * u rtol = 1.0e-11
        end

        # `Stored()` is the PARTIAL level, and the store it elects is the WRAPPED
        # element's. An unforwarded election is not an error: the cache comes
        # back untouched, nothing is allocated and the action silently runs the
        # `Recompute()` path — the same number for THIS element, a different one
        # for an element whose factors are not re-derivable.
        stored = MatrixFreeAction(; storage = Stored())
        bare = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = stored), inner, dh)
        decorated = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = stored),
                                   DecoratedIntegrator(inner), dh)
        element = first(get_subdomain_caches(decorated)).domain.element
        @test element isa PassthroughDecorator
        @test element.inner.qdata !== nothing
        @test typeof(element.inner.qdata) ===
            typeof(first(get_subdomain_caches(bare)).domain.element.qdata)
    end

    @testset "a downstream decorator forwards the quadrature-data fill" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 1)
        u = probe(Float64, ndofs(dh), 19)
        integrator = EagerStoreIntegrator(1.5)
        # `Kₑ = 1.5·I` over each cell's dofs, so a dof carries 1.5 per cell it
        # belongs to — from the DofHandler, sharing no code with the engine.
        multiplicity = zeros(ndofs(dh))
        for cell in 1:getncells(Ferrite.get_grid(dh)), d in celldofs(dh, cell)
            multiplicity[d] += 1.5
        end
        expected = multiplicity .* u

        for (label, integ) in ("bare" => integrator, "decorated" => DecoratedIntegrator(integrator))
            op = setup_operator(
                AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction(; storage = Stored())),
                integ, dh)
            y = zeros(ndofs(dh))
            mul!(y, op, u)
            @test y ≈ expected rtol = 1.0e-12
            @test !iszero(y)                      # the fill reached the element
        end
    end

    @testset "an anisotropic tensor gives a symmetric operator" begin
        dh = distorted_testbed(Hexahedron, o -> Lagrange{RefHexahedron, o}(), Float64, (3, 2, 2), 2)
        D = SymmetricTensor{2, 3}((2.0, 0.3, -0.2, 1.4, 0.1, 3.1))
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(D, QuadratureRuleCollection(3), :u), dh)
        u = probe(Float64, ndofs(dh), 3)
        v = probe(Float64, ndofs(dh), 4)
        Au, Av = zeros(ndofs(dh)), zeros(ndofs(dh))
        mul!(Au, op, u)
        mul!(Av, op, v)
        @test dot(v, Au) ≈ dot(u, Av) rtol = 1.0e-12
        isotropic = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                                   SumFactorizedDiffusionIntegrator(2.5 * one(SymmetricTensor{2, 3}),
                                                                    QuadratureRuleCollection(3), :u), dh)
        assembled = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                                   SimpleBilinearDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u), dh)
        update_operator!(assembled, nothing)
        y = zeros(ndofs(dh))
        mul!(y, isotropic, u)
        @test y ≈ assembled.A * u rtol = 1.0e-11
    end

    @testset "the five-argument mul! scales as LinearAlgebra promises" begin
        dh = distorted_testbed(Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), Float64, (4, 4), 2)
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u), dh)
        u = probe(Float64, ndofs(dh), 17)
        base = probe(Float64, ndofs(dh), 19)
        action = zeros(ndofs(dh))
        mul!(action, op, u)
        for (α, β) in ((1.0, 0.0), (-1.0, 1.0), (2.0, 0.5), (0.0, 3.0))
            y = copy(base)
            mul!(y, op, u, α, β)
            @test y ≈ α .* action .+ β .* base rtol = 1.0e-10
        end
        @test op * u ≈ action rtol = 1.0e-12
    end

    @testset "β = 0 ASSIGNS rather than scales, even against a NaN-filled y" begin
        # `rmul!(y, 0)` propagates a NaN/Inf already sitting in `y`; the LinearAlgebra convention
        # for the 5-arg `mul!` is that `β = 0` overwrites `y` instead, regardless of its contents.
        dh = distorted_testbed(Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), Float64, (4, 4), 2)
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(3), :u), dh)
        u = probe(Float64, ndofs(dh), 17)
        action = zeros(ndofs(dh))
        mul!(action, op, u)

        y = fill(NaN, ndofs(dh))
        mul!(y, op, u, 1.0, 0.0)
        @test y ≈ action rtol = 1.0e-11

        y = fill(NaN, ndofs(dh))
        mul!(y, op, u, 0.0, 0.0)
        @test all(iszero, y)
    end

    @testset "evaluate! is the action with parameters and a context" begin
        dh = distorted_testbed(Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), Float64, (3, 3), 1)
        op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                            SumFactorizedDiffusionIntegrator(2.5, QuadratureRuleCollection(2), :u), dh)
        u = probe(Float64, ndofs(dh), 23)
        y, z = zeros(ndofs(dh)), zeros(ndofs(dh))
        mul!(y, op, u)
        evaluate!(op, z, u, nothing)
        @test y == z
        residual = zeros(ndofs(dh))
        update_linearization!(op, residual, u, nothing)
        @test residual == y
        @test update_operator!(op, nothing) === nothing
    end
end

####################################
## Capability walls
####################################

# Neither fill route.
struct NoRouteCache <: FerriteOperators.AbstractVolumetricElementCache end

@testset "MatrixFreeAction capability walls" begin
    dh = distorted_testbed(Quadrilateral, o -> Lagrange{RefQuadrilateral, o}(), Float64, (3, 3), 1)
    qrc = QuadratureRuleCollection(2)
    sum_factorized = SumFactorizedDiffusionIntegrator(2.5, qrc, :u)
    assembled_form = SimpleBilinearDiffusionIntegrator(2.5, qrc, :u)

    @testset "the cooperative mapping needs a KernelAbstractions device" begin
        for device in (SequentialCPUDevice(), PolyesterDevice())
            strategy = AssemblyStrategy(device;
                form = MatrixFreeAction(; element_mapping = CooperativeElement()),
                scheduling = ColoredScheduling())
            err = @test_throws ArgumentError setup_operator(strategy, sum_factorized, dh)
            @test occursin("CooperativeElement", err.value.msg)
            @test occursin("KernelAbstractionsDevice", err.value.msg)
        end
    end

    @testset "a cache without the action entry is refused" begin
        strategy = AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction())
        err = @test_throws ArgumentError setup_operator(strategy, assembled_form, dh)
        @test occursin("apply_element_action!", err.value.msg)
    end

    @testset "a cache without the cooperative entries is refused" begin
        err = @test_throws ArgumentError setup_operator(
            matrix_free_ka(Float64, CooperativeElement()), assembled_form, dh)
        # `setup_operator` reports the action entry first; the direct call below
        # exercises the cooperative half.
        @test occursin("apply_element_action!", err.value.msg)
        cache = setup_element_cache(assembled_form, dh.subdofhandlers[1])
        err = @test_throws ArgumentError FerriteOperators._assert_mapping_capability(
            CooperativeElement(), Stored(), cache)
        @test occursin("cooperative_lattice_dim", err.value.msg)
        @test occursin("WorkerPerElement", err.value.msg)
    end

    @testset "the lane mapping needs a KernelAbstractions device" begin
        for device in (SequentialCPUDevice(), PolyesterDevice())
            strategy = AssemblyStrategy(device;
                form = MatrixFreeAction(; element_mapping = LanesPerElement(),
                                        storage = ElementAssembly()),
                scheduling = ColoredScheduling())
            err = @test_throws ArgumentError setup_operator(strategy, sum_factorized, dh)
            @test occursin("LanesPerElement", err.value.msg)
            @test occursin("KernelAbstractionsDevice", err.value.msg)
        end
    end

    @testset "the lane mapping is the ELEMENT level's only ($(nameof(typeof(storage))))" for
            storage in (Stored(), Recompute())

        err = @test_throws ArgumentError setup_operator(
            matrix_free_ka(Float64, LanesPerElement(); storage), sum_factorized, dh)
        @test occursin("LanesPerElement", err.value.msg)
        @test occursin("ElementAssembly", err.value.msg)
    end

    # The decorator implements the row entry, so the refusal is exercised
    # directly on the cache it wraps.
    @testset "a cache without the row entry is refused" begin
        cache = setup_element_cache(assembled_form, dh.subdofhandlers[1])
        err = @test_throws ArgumentError FerriteOperators._assert_mapping_capability(
            LanesPerElement(), ElementAssembly(), cache)
        @test occursin("element_action_row", err.value.msg)
        @test occursin("WorkerPerElement", err.value.msg)
    end

    @testset "a lane count wider than the workgroup is refused" begin
        err = @test_throws ArgumentError FerriteOperators.with_element_mapping(
            KernelAbstractionsDevice(KA.CPU(); max_workgroup_size = 16), LanesPerElement(; lanes = 64))
        @test occursin("max_workgroup_size", err.value.msg)
    end

    @testset "the lane kernel serves the action kind only" begin
        device = FerriteOperators.with_element_mapping(
            KernelAbstractionsDevice(KA.CPU()), LanesPerElement())
        task = FerriteOperators.AssemblyTask(FerriteOperators.BilinearKind(), nothing, (;), nothing, nothing)
        err = @test_throws ArgumentError FerriteOperators.execute_on_device!(
            task, device, nothing, ())
        @test occursin("BilinearKind", err.value.msg)
    end

    @testset "an assembling form on a LanesPerElement device is refused at setup" begin
        device = FerriteOperators.with_element_mapping(
            KernelAbstractionsDevice(KA.CPU()), LanesPerElement())
        strategy = AssemblyStrategy(device; scheduling = ColoredScheduling())
        err = @test_throws ArgumentError setup_operator(strategy, assembled_form, dh)
        @test occursin("LanesPerElement", err.value.msg)
        @test occursin("WorkerPerElement", err.value.msg)
    end

    @testset "only the bilinear family takes the matrix-free form" begin
        strategy = AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction())
        err = @test_throws ArgumentError setup_operator(
            strategy, SimpleLinearIntegrator(3.1, qrc, :u), dh)
        @test occursin("no argument to act on", err.value.msg)
    end

    @testset "the action needs the :u slot" begin
        strategy = AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction())
        err = @test_throws ArgumentError setup_operator(strategy, sum_factorized, dh; slots = (:v,))
        @test occursin(":u", err.value.msg)
    end

    @testset "the matrix-free element has no element matrix" begin
        for integrator in (sum_factorized, SumFactorizedMassIntegrator(1.7, qrc, :u))
            op = setup_operator(AssemblyStrategy(SequentialCPUDevice()), integrator, dh)
            err = @test_throws ArgumentError update_operator!(op, nothing)
            @test occursin("forms no element matrix", err.value.msg)
            @test occursin("MatrixFreeAction", err.value.msg)
        end
    end

    @testset "the storage election names the three levels" begin
        err = @test_throws ArgumentError MatrixFreeAction(; storage = :stored)
        @test occursin("Stored()", err.value.msg)
        @test occursin("ElementAssembly()", err.value.msg)
    end

    @testset "the ELEMENT level refuses the cooperative mapping" begin
        err = @test_throws ArgumentError setup_operator(
            matrix_free_ka(Float64, CooperativeElement(); storage = ElementAssembly()), sum_factorized, dh)
        @test occursin("ElementAssembly", err.value.msg)
        @test occursin("WorkerPerElement", err.value.msg)
        @test occursin("LanesPerElement", err.value.msg)
    end

    @testset "a cache serving neither fill route is refused" begin
        err = @test_throws ArgumentError FerriteOperators.element_matrix_fill_route(NoRouteCache)
        @test occursin("provides_analytic", err.value.msg)
        @test occursin("apply_element_action!", err.value.msg)
    end

    @testset "the cooperative kernel serves the action kind only" begin
        device = FerriteOperators.with_element_mapping(
            KernelAbstractionsDevice(KA.CPU()), CooperativeElement())
        task = FerriteOperators.AssemblyTask(FerriteOperators.BilinearKind(), nothing, (;), nothing, nothing)
        err = @test_throws ArgumentError FerriteOperators.execute_on_device!(
            task, device, nothing, ())
        @test occursin("BilinearKind", err.value.msg)
    end

    # Complements the call-time check above: an assembling form on a
    # `CooperativeElement` device is refused at `setup_operator`, before any
    # cache or sweep exists.
    @testset "an assembling form on a CooperativeElement device is refused at setup" begin
        device = FerriteOperators.with_element_mapping(
            KernelAbstractionsDevice(KA.CPU()), CooperativeElement())
        strategy = AssemblyStrategy(device; scheduling = ColoredScheduling())
        err = @test_throws ArgumentError setup_operator(strategy, assembled_form, dh)
        @test occursin("CooperativeElement", err.value.msg)
        @test occursin("WorkerPerElement", err.value.msg)
    end
end
