module FerriteOperators

using Reexport
@reexport using Ferrite
using TimerOutputs
using Adapt
using Unrolled
using SparseArrays, StaticArrays

using ConcreteStructs

import LinearAlgebra: mul!

import Base: *, +, -, @kwdef

using Polyester # TODO extension

import Atomix

import Ferrite: AbstractDofHandler, AbstractGrid, AbstractRefShape, AbstractCell, get_grid, get_coordinate_eltype
import Ferrite: vertices, edges, faces, sortedge, sortface
import Ferrite: get_coordinate_type, getspatialdim
import Ferrite: reference_shape_value

include("core/device.jl")
include("core/strategy.jl")
include("core/element_interface.jl")
include("core/utils.jl")

include("core/ferrite-addons/collections.jl")
include("core/ferrite-addons/assembly.jl")
include("core/ferrite-addons/parallel_duplication_api.jl")

abstract type AbstractBilinearIntegrator end
abstract type AbstractNonlinearIntegrator end
abstract type AbstractLinearIntegrator end

include("elements/composite_elements.jl")
include("elements/simple_diffusion.jl")
include("elements/simple_mass.jl")
include("elements/simple_hyperelasticity.jl")

include("operators/general.jl")
include("operators/matrix-free.jl")
include("operators/ferrite.jl")
include("operators/setup.jl")

export QuadratureRuleCollection

export setup_operator, update_operator!, update_linearization!

export NullOperator, DiagonalOperator

export SequentialCPUDevice, PolyesterDevice, CudaDevice
export SequentialAssemblyStrategy, ElementAssemblyStrategy, PerColorAssemblyStrategy

end
