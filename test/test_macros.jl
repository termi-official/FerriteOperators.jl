using Test
using FerriteOperators: @device_element, @per_thread_structure, per_thread, duplicate_for_device, AbstractVolumetricElementCache
using Adapt

@testset "@device_element macro" begin
    # Test 1: typed fields get parametrized
    @device_element struct _TestTypedCache <: AbstractVolumetricElementCache
        D::Float64
        cellvalues::CellValues
    end

    @test fieldcount(_TestTypedCache) == 2
    @test _TestTypedCache <: AbstractVolumetricElementCache
    # Constructor works with different types for each field
    cache = _TestTypedCache(1.0, :placeholder)
    @test cache.D == 1.0
    @test cache.cellvalues == :placeholder
    # Type params are inferred, not fixed
    @test typeof(cache) !== typeof(_TestTypedCache(1, "other"))

    # Test 2: untyped fields also get parametrized
    @device_element struct _TestUntypedCache <: AbstractVolumetricElementCache
        D
        cellvalues
    end

    cache2 = _TestUntypedCache(1.0, :placeholder)
    @test cache2.D == 1.0
    @test typeof(cache2) !== typeof(_TestUntypedCache("a", "b"))

    # Test 3: adapt_structure is generated
    @test hasmethod(Adapt.adapt_structure, Tuple{Any, _TestTypedCache})

    # Test 4: getindex (per_thread) is generated
    @test hasmethod(Base.getindex, Tuple{_TestTypedCache, Int})

    # Test 5: per_thread identity for non-containers
    cache3 = _TestTypedCache(2.5, :foo)
    extracted = cache3[1]
    @test extracted.D == 2.5
    @test extracted.cellvalues == :foo

    # Test 6: adapt_structure preserves values for isbits
    adapted = Adapt.adapt_structure(nothing, cache3)
    @test adapted.D == 2.5
    @test adapted.cellvalues == :foo

    # Test 7: no supertype works
    @device_element struct _TestNoSuper
        x::Int
    end
    @test _TestNoSuper(1).x == 1

    # Test 8: duplicate_for_device recursive fallback
    cache4 = _TestTypedCache(3.0, 42)
    dup = duplicate_for_device(SequentialCPUDevice(), cache4)
    @test dup.D == 3.0
    @test dup.cellvalues == 42

    # Test 9: user-specified type params are preserved
    @device_element struct _TestUserParams{EnergyType} <: AbstractVolumetricElementCache
        ψ::EnergyType
        cv::CellValues
    end

    @test fieldcount(_TestUserParams) == 2
    # EnergyType is user param, cv gets auto-param _T1
    cache5 = _TestUserParams(sin, :cv_placeholder)
    @test cache5.ψ === sin
    @test cache5.cv == :cv_placeholder
    # Different energy types produce different concrete types
    @test typeof(_TestUserParams(sin, :a)) !== typeof(_TestUserParams(cos, :a))

    # Test 10: user params with bounds — bounds stripped for GPU compatibility
    @device_element struct _TestBoundedParams{T <: Number} <: AbstractVolumetricElementCache
        val::T
        cv::CellValues
    end

    cache6 = _TestBoundedParams(1.0, :cv)
    @test cache6.val == 1.0
    # Bound is stripped — accepts any type (GPU-safe)
    cache6b = _TestBoundedParams("not a number", :cv)
    @test cache6b.val == "not a number"

    # Test 11: mixed user + untyped fields
    @device_element struct _TestMixed{E}
        energy::E
        data
    end

    cache7 = _TestMixed(:energy, [1,2,3])
    @test cache7.energy == :energy
    @test cache7.data == [1,2,3]
end
