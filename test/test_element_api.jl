using FerriteOperators
import FerriteOperators: get_matrix
using Test
import LinearAlgebra: mul!
using SparseArrays
using Polyester
using TimerOutputs

@testset "Element API" begin
    import FerriteOperators: assemble_element!, assemble_facet!
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

        # Volume
        assemble_element!(
            Kₑ¹,
            rₑ¹,
            uₑs,
            cell_cache_s,
            FerriteOperators.EmptyVolumetricElementCache(),
            0.0,
        )
        @test iszero(Kₑ¹)
        @test iszero(rₑ¹)

        assemble_element!(rₑ², uₑs, cell_cache_s, FerriteOperators.EmptyVolumetricElementCache(), 0.0)
        @test iszero(rₑ²)

        assemble_element!(Kₑ², uₑs, cell_cache_s, FerriteOperators.EmptyVolumetricElementCache(), 0.0)
        @test iszero(Kₑ²)

        # Surface
        for local_facet_index = 1:nfacets(cell_cache_s)
            assemble_facet!(
                Kₑ¹,
                rₑ¹,
                uₑs,
                cell_cache_s,
                local_facet_index,
                FerriteOperators.EmptySurfaceElementCache(),
                0.0,
            )
            @test iszero(Kₑ¹)
            @test iszero(rₑ¹)

            assemble_facet!(
                rₑ²,
                uₑs,
                cell_cache_s,
                local_facet_index,
                FerriteOperators.EmptySurfaceElementCache(),
                0.0,
            )
            @test iszero(rₑ²)

            assemble_facet!(
                Kₑ²,
                uₑs,
                cell_cache_s,
                local_facet_index,
                FerriteOperators.EmptySurfaceElementCache(),
                0.0,
            )
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

        assemble_element!(Kₑ¹, cell_cache_s, element_cache, 0.0)
        @test !iszero(Kₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        assemble_element!(Kₑ², cell_cache_s, composite_element_cache, 0.0)
        @test 2Kₑ¹ ≈ Kₑ²
    end

    @testset "Scalar linear composite elements: $model" for model in (
        SimpleLinearIntegrator(1.0, qrc, :u),
    )
        bₑ¹ = zeros(ndofs(dhs))
        bₑ² = zeros(ndofs(dhs))

        element_cache = setup_test_cache(model, sdhs)

        assemble_element!(bₑ¹, cell_cache_s, element_cache, 0.0)
        @test !iszero(bₑ¹)

        composite_element_cache = setup_test_composite_volume_cache(model, sdhs)

        assemble_element!(bₑ², cell_cache_s, composite_element_cache, 0.0)
        @test 2bₑ¹ ≈ bₑ²
    end
end
