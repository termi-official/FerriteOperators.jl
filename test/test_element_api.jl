using FerriteOperators
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester
using TimerOutputs

# A real facet kernel exercising the framework-owned boundary driver: a
# constant Neumann load t̄ on a facet set, with the analytic reference
# sum(b) = t̄ · |Γ|.
struct NeumannTestIntegrator <: AbstractLinearIntegrator
    t̄::Float64
    qrc::QuadratureRuleCollection
    field_name::Symbol
    facetset::Set{FacetIndex}
end
struct NeumannTestCache{FV <: FacetValues} <: FerriteOperators.AbstractSurfaceElementCache
    t̄::Float64
    fv::FV
    facetset::Set{FacetIndex}
end
function FerriteOperators.setup_element_cache(m::NeumannTestIntegrator, sdh::SubDofHandler)
    return FerriteOperators.EmptyVolumetricElementCache()
end
function FerriteOperators.setup_boundary_cache(m::NeumannTestIntegrator, sdh::SubDofHandler)
    fqr = FacetQuadratureRule{RefHexahedron}(2)
    ip  = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return NeumannTestCache(m.t̄, FacetValues(fqr, ip, ip_geo), m.facetset)
end
FerriteOperators.duplicate_for_device(device, c::NeumannTestCache) =
    NeumannTestCache(c.t̄, FerriteOperators.duplicate_for_device(device, c.fv), c.facetset)
FerriteOperators.is_facet_in_cache(idx::FacetIndex, cell, c::NeumannTestCache) = idx ∈ c.facetset
function FerriteOperators.assemble_facet!(req::ResidualRequest, c::NeumannTestCache, args::KernelArgs, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    for qp in 1:getnquadpoints(c.fv)
        dΓ = getdetJdV(c.fv, qp)
        for i in 1:getnbasefunctions(c.fv)
            req.r[i] += c.t̄ * shape_value(c.fv, qp, i) * dΓ
        end
    end
end

@testset "Facet driver with a real Neumann kernel" begin
    grid = generate_grid(Hexahedron, (2, 2, 2))   # unit cube [-1,1]³ → right face area 4.0
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}())
    close!(dh)
    t̄ = 3.25
    right = Set(getfacetset(grid, "right"))
    integrator = NeumannTestIntegrator(t̄, QuadratureRuleCollection(2), :u, right)

    op = setup_operator(SequentialAssemblyStrategy(SequentialCPUDevice()), integrator, dh)
    update_operator!(op, nothing)
    area = 4.0
    @test sum(op.b) ≈ t̄ * area rtol = 1e-12
    # only dofs on the loaded face carry entries
    @test count(!iszero, op.b) == 9

    # parallel path through the same driver
    opp = setup_operator(PerColorAssemblyStrategy(PolyesterDevice(2)), integrator, dh)
    update_operator!(opp, nothing)
    @test opp.b ≈ op.b rtol = 1e-13

    # composite surface fan-out re-gates per inner cache: two identical inner
    # caches double the load
    sdh = first(dh.subdofhandlers)
    inner = FerriteOperators.setup_boundary_cache(integrator, sdh)
    comp  = FerriteOperators.CompositeSurfaceElementCache((inner, inner))
    b2 = zeros(ndofs(dh))
    cc = Ferrite.CellCache(sdh)
    for cellid in 1:getncells(grid)
        reinit!(cc, cellid)
        rₑ = zeros(ndofs_per_cell(sdh))
        args = KernelArgs((;), cc, nothing, nothing, nothing)
        for lfi in 1:nfacets(cc)
            if FerriteOperators.is_facet_in_cache(FacetIndex(cellid, lfi), cc, comp)
                FerriteOperators.assemble_facet!(ResidualRequest(rₑ), comp, args, lfi)
            end
        end
        b2[celldofs(cc)] .+= rₑ
    end
    @test sum(b2) ≈ 2 * t̄ * area rtol = 1e-12
end

@testset "Element API" begin
    import FerriteOperators: assemble_cell!, assemble_facet!
    import FerriteOperators: setup_element_cache, setup_boundary_cache
    import FerriteOperators: SimpleBilinearMassIntegrator, SimpleBilinearDiffusionIntegrator, SimpleLinearIntegrator
    import FerriteOperators

    setup_test_cache(kwargs...) =
        FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
    function setup_test_composite_volume_cache(kwargs...)
        element_cache =
            FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_element_cache(kwargs...))
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeVolumetricElementCache((element_cache, element_cache)),
        )
    end
    function setup_test_composite_surface_cache(kwargs...)
        element_cache =
            FerriteOperators.duplicate_for_device(PolyesterDevice(), setup_boundary_cache(kwargs...))
        return FerriteOperators.duplicate_for_device(
            PolyesterDevice(),
            FerriteOperators.CompositeSurfaceElementCache((element_cache, element_cache)),
        )
    end

    grid = generate_grid(Hexahedron, (1, 1, 1))
    qrc  = QuadratureRuleCollection(3)
    qr   = QuadratureRule{RefHexahedron}(3)
    qrcf = QuadratureRuleCollection(3)
    qrf  = FacetQuadratureRule{RefHexahedron}(3)
    ip   = Lagrange{RefHexahedron, 1}()

    dhs = DofHandler(grid)
    add!(dhs, :u, ip)
    close!(dhs)
    sdhs = first(dhs.subdofhandlers)
    cell_cache_s = Ferrite.CellCache(sdhs)
    Ferrite.reinit!(cell_cache_s, 1)
    uₑs = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0] .* 1e-4

    ipv = ip^3
    dhv = DofHandler(grid)
    add!(dhv, :u, ipv)
    close!(dhv)
    sdhv = first(dhv.subdofhandlers)
    cell_cache_v = Ferrite.CellCache(sdhv)
    Ferrite.reinit!(cell_cache_v, 1)
    uₑv =
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ] .* 1e-4

    # We check for pairwise consistency of the assembly operations
    # First we check if the empty caches work correctly
    @testset "Empty caches" begin
        rₑ¹ = zeros(ndofs(dhs))
        rₑ² = zeros(ndofs(dhs))
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        args = KernelArgs((u = uₑs,), cell_cache_s, 0.0, nothing, nothing)

        # Volume
        assemble_cell!(JacobianResidualRequest(Kₑ¹, rₑ¹), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(Kₑ¹)
        @test iszero(rₑ¹)

        assemble_cell!(ResidualRequest(rₑ²), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(rₑ²)

        assemble_cell!(JacobianRequest{:u}(Kₑ²), FerriteOperators.EmptyVolumetricElementCache(), args)
        @test iszero(Kₑ²)

        # Surface: the empty cache never claims a facet, and its kernels are no-ops
        for local_facet_index = 1:nfacets(cell_cache_s)
            @test !FerriteOperators.is_facet_in_cache(FacetIndex(1, local_facet_index), cell_cache_s, FerriteOperators.EmptySurfaceElementCache())
            assemble_facet!(JacobianResidualRequest(Kₑ¹, rₑ¹), FerriteOperators.EmptySurfaceElementCache(), args, local_facet_index)
            @test iszero(Kₑ¹)
            @test iszero(rₑ¹)

            assemble_facet!(ResidualRequest(rₑ²), FerriteOperators.EmptySurfaceElementCache(), args, local_facet_index)
            @test iszero(rₑ²)

            assemble_facet!(JacobianRequest{:u}(Kₑ²), FerriteOperators.EmptySurfaceElementCache(), args, local_facet_index)
            @test iszero(Kₑ²)
        end
    end

    @testset "Scalar volumetric bilinear composite elements: $model" for model in (
        SimpleBilinearMassIntegrator(1.0, qrc, :u),
        SimpleBilinearDiffusionIntegrator(1.0, qrc, :u),
    )
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        args = KernelArgs((;), cell_cache_s, 0.0, nothing, nothing)
        assemble_cell!(JacobianRequest{:u}(Kₑ¹), element_cache, args)
        @test !iszero(Kₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        assemble_cell!(JacobianRequest{:u}(Kₑ²), composite_element_cache, args)
        @test 2Kₑ¹ ≈ Kₑ²
    end

    @testset "Scalar linear composite elements: $model" for model in (
        SimpleLinearIntegrator(1.0, qrc, :u),
    )
        bₑ¹ = zeros(ndofs(dhs))
        bₑ² = zeros(ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        args = KernelArgs((;), cell_cache_s, 0.0, nothing, nothing)
        assemble_cell!(ResidualRequest(bₑ¹), element_cache, args)
        @test !iszero(bₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        assemble_cell!(ResidualRequest(bₑ²), composite_element_cache, args)
        @test 2bₑ¹ ≈ bₑ²
    end
end
