## GPU Element Tests: duplicate_for_device + per-thread extraction ##

function test_element_gpu_lifecycle(device, element_cache, isbits_fields::Dict)
    nt = FerriteOperators.total_nthreads(device)

    # Step 1: duplicate_for_device (CPU → GPU)
    gpu_cache = duplicate_for_device(device, element_cache)

    @testset "duplicate_for_device" begin
        # CellValues should become CellValuesContainer
        has_container = any(getfield(gpu_cache, i) isa Ferrite.CellValuesContainer for i in 1:fieldcount(typeof(gpu_cache)))
        @test has_container

        # isbits fields should be unchanged
        for (name, expected) in isbits_fields
            @test getfield(gpu_cache, name) == expected
        end
    end

    # Step 2: per-thread extraction (container[tid] → per-thread cache)
    @testset "per-thread extraction" begin
        for tid in (1, nt)
            thread_cache = gpu_cache[tid]

            # Result type should match original (unwrapped)
            @test typeof(thread_cache).name.wrapper === typeof(element_cache).name.wrapper

            # CellValues field should be CellValues (not CellValuesContainer)
            has_cellvalues = any(getfield(thread_cache, i) isa CellValues for i in 1:fieldcount(typeof(thread_cache)))
            @test has_cellvalues

            # isbits fields preserved
            for (name, expected) in isbits_fields
                @test getfield(thread_cache, name) == expected
            end
        end
    end
end

function test_composite_gpu_lifecycle(device, composite_cache, inner_isbits::Vector)
    nt = FerriteOperators.total_nthreads(device)

    @testset "duplicate_for_device" begin
        gpu_composite = duplicate_for_device(device, composite_cache)
        inner = gpu_composite.inner_caches
        for (idx, isbits_fields) in enumerate(inner_isbits)
            # Each inner cache should have CellValuesContainer
            has_container = any(getfield(inner[idx], i) isa Ferrite.CellValuesContainer for i in 1:fieldcount(typeof(inner[idx])))
            @test has_container
            for (name, expected) in isbits_fields
                @test getfield(inner[idx], name) == expected
            end
        end
    end

    @testset "per-thread extraction" begin
        gpu_composite = duplicate_for_device(device, composite_cache)
        for tid in (1, nt)
            thread_composite = gpu_composite[tid]
            inner = thread_composite.inner_caches
            for (idx, isbits_fields) in enumerate(inner_isbits)
                # Each inner cache should have CellValues (not Container)
                has_cellvalues = any(getfield(inner[idx], i) isa CellValues for i in 1:fieldcount(typeof(inner[idx])))
                @test has_cellvalues
                for (name, expected) in isbits_fields
                    @test getfield(inner[idx], name) == expected
                end
            end
        end
    end
end

# NeoHookean for hyperelasticity test (defined outside testset to avoid struct-in-local-scope issues)
struct _TestNeoHookean
    E::Float64
    ν::Float64
end
function (p::_TestNeoHookean)(F)
    μ = p.E / (2(1 + p.ν))
    λ = (p.E * p.ν) / ((1 + p.ν) * (1 - 2p.ν))
    C = tdot(F); Ic = tr(C); J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

@testset "Element GPU Lifecycle" begin
    run_on_backends(cuda_f32=false) do device
        # --- SimpleBilinearDiffusionElementCache ---
        @testset "SimpleBilinearDiffusion" begin
            grid = generate_grid(Quadrilateral, (4, 4))
            dh = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
            sdh = first(dh.subdofhandlers)
            integrator = FerriteOperators.SimpleBilinearDiffusionIntegrator(2.5, QuadratureRuleCollection(2), :u)
            cache = FerriteOperators.setup_element_cache(integrator, sdh)
            test_element_gpu_lifecycle(device, cache, Dict(:D => 2.5))
        end

        # --- SimpleBilinearMassElementCache ---
        @testset "SimpleBilinearMass" begin
            grid = generate_grid(Quadrilateral, (4, 4))
            dh = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
            sdh = first(dh.subdofhandlers)
            integrator = FerriteOperators.SimpleBilinearMassIntegrator(1.5, QuadratureRuleCollection(1), :u)
            cache = FerriteOperators.setup_element_cache(integrator, sdh)
            test_element_gpu_lifecycle(device, cache, Dict(:ρ => 1.5))
        end

        # --- SimpleHyperelasticityElementCache ---
        @testset "SimpleHyperelasticity" begin
            grid = generate_grid(Hexahedron, (2, 2, 2))
            dh = DofHandler(grid); add!(dh, :u, Lagrange{RefHexahedron, 1}()^3); close!(dh)
            sdh = first(dh.subdofhandlers)
            ψ = _TestNeoHookean(10.0, 0.3)
            integrator = FerriteOperators.SimpleHyperelasticityIntegrator(ψ, QuadratureRuleCollection(2), :u)
            cache = FerriteOperators.setup_element_cache(integrator, sdh)
            test_element_gpu_lifecycle(device, cache, Dict(:ψ => ψ))
        end

        # --- SimpleCondensedLinearViscoelasticityCache ---
        @testset "SimpleCondensedLinearViscoelasticity" begin
            grid = generate_grid(Hexahedron, (2, 2, 2))
            dh = DofHandler(grid); add!(dh, :u, Lagrange{RefHexahedron, 1}()^3); close!(dh)
            sdh = first(dh.subdofhandlers)
            mp = FerriteOperators.MaxwellParameters()
            integrator = FerriteOperators.SimpleCondensedLinearViscoelasticity(mp, QuadratureRuleCollection(2), :u, :εᵛ)
            cache = FerriteOperators.setup_element_cache(integrator, sdh)
            test_element_gpu_lifecycle(device, cache, Dict(
                :material_parameters => mp,
                :displacement_range => cache.displacement_range,
                :viscosity_range => cache.viscosity_range,
            ))
        end

        # --- CompositeVolumetricElementCache ---
        @testset "CompositeVolumetricElement" begin
            grid = generate_grid(Quadrilateral, (4, 4))
            dh = DofHandler(grid); add!(dh, :u, Lagrange{RefQuadrilateral, 1}()); close!(dh)
            sdh = first(dh.subdofhandlers)

            diff_cache = FerriteOperators.setup_element_cache(
                FerriteOperators.SimpleBilinearDiffusionIntegrator(2.5, QuadratureRuleCollection(2), :u), sdh)
            mass_cache = FerriteOperators.setup_element_cache(
                FerriteOperators.SimpleBilinearMassIntegrator(1.0, QuadratureRuleCollection(1), :u), sdh)
            composite_cache = FerriteOperators.CompositeVolumetricElementCache((diff_cache, mass_cache))

            test_composite_gpu_lifecycle(device, composite_cache, [
                Dict(:D => 2.5),
                Dict(:ρ => 1.0),
            ])
        end
    end
end
