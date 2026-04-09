using Adapt
using SparseArrays
import KernelAbstractions as KA
import LinearAlgebra: mul!, norm

## Detect backends ##
roc_available = try
    using AMDGPU
    include(joinpath(@__DIR__, "..", "..", "ext", "FerriteOperatorsAMDGPUExt.jl"))
    AMDGPU.functional()
catch; false end

cuda_available = try
    using CUDA
    include(joinpath(@__DIR__, "..", "..", "ext", "FerriteOperatorsCUDAExt.jl"))
    CUDA.functional()
catch; false end

## Run tests per available backend ##
function run_on_backends(f; cuda_f32=true, cuda_f64=true, roc=true)
    if cuda_available
        cuda_f32 && @testset "CUDA (Float32)" begin
            f(CudaDevice())
        end
        cuda_f64 && @testset "CUDA (Float64)" begin
            f(CudaDevice{Float64, Int32}(Int32(64), Int32(4)))
        end
    else
        @info "CUDA not available, skipping"
    end
    if roc_available
        roc && @testset "ROC" begin
            f(RocDevice())
        end
    else
        @info "AMDGPU not available, skipping ROC"
    end
end

## Shared test setup ##

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
