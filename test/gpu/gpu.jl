# Backend-agnostic GPU test suite for FerriteOperators.
#
# This file defines the device-parameterised test functions and the `run_gpu_tests`
# entry point. It does NOT load any GPU backend package. Drive it from a per-backend
# file (`cuda.jl` / `amd.jl`) that loads its backend and calls `run_gpu_tests(device)`.

using Test
using FerriteOperators
using FerriteOperators: duplicate_for_device
using Adapt
using SparseArrays
import KernelAbstractions as KA
import LinearAlgebra: mul!, norm

#####################
## Shared setup    ##
#####################

# NeoHookean energy (defined at top level to avoid struct-in-local-scope issues)
struct GPUTestNeoHookean
    E::Float64
    ν::Float64
end
function (p::GPUTestNeoHookean)(F)
    (; E, ν) = p
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    C = tdot(F); Ic = tr(C); J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

function setup_diffusion_problem()
    grid = generate_grid(Quadrilateral, (4, 4))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    D = 2.5
    integrator = FerriteOperators.SimpleBilinearDiffusionIntegrator(
        D,
        QuadratureRuleCollection(2),
        :u,
    )
    return dh, integrator
end

function setup_mass_problem()
    grid = generate_grid(Quadrilateral, (4, 4))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    integrator = FerriteOperators.SimpleBilinearMassIntegrator(
        1.5,
        QuadratureRuleCollection(1),
        :u,
    )
    return dh, integrator
end

function setup_hyperelasticity_problem(device=nothing)
    grid = generate_grid(Hexahedron, (3, 3, 3))
    Ferrite.transform_coordinates!(grid, x -> Vec{3}(sign.(x .- 0.5) .* (x .- 0.5) .^ 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(dh)

    integrator = FerriteOperators.SimpleHyperelasticityIntegrator(
        GPUTestNeoHookean(10.0, 0.3),
        QuadratureRuleCollection(2),
        :u,
    )

    u_cpu = zeros(ndofs(dh))
    apply_analytical!(u_cpu, dh, :u, x -> 0.01x .^ 2)

    if device === nothing
        return dh, integrator, u_cpu
    else
        return dh, integrator, Adapt.adapt(KA.backend(device), u_cpu)
    end
end

#####################
## Test functions  ##
#####################
# Each file defines its `test_*` helpers and a `run_*_tests(device)` driver.
include("test_elements.jl")
include("test_element_assembly.jl")
include("test_sequential_assembly.jl")
include("test_percolor_assembly.jl")
include("test_sizes.jl")

#####################
## Entry point     ##
#####################
"""
    run_gpu_tests(device; testset_name="FerriteOperators GPU")

Run the full GPU test suite for the given `device` (e.g. `CudaDevice`, `RocDevice`).
Backend-agnostic: the caller (`cuda.jl` / `amd.jl`) loads the backend package first.
"""
function run_gpu_tests(device; testset_name="FerriteOperators GPU")
    @testset "$testset_name" begin
        run_lifecycle_tests(device)
        run_ea_tests(device)
        run_sequential_tests(device)
        run_percolor_tests(device)
        run_sizes_tests(device)
    end
end
