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

## Shared test setup ##
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
