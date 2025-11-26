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

include("core/device.jl")    # Utilities to manage devices (e.g. CPU threads or GPUs)
include("core/strategy.jl")  # Utilities to control the assembly strategy
include("core/tasks.jl")     # Contains the basic task system

include("core/element_interface.jl") # This is the basic element interface used for the operators
include("core/utils.jl")             # Internal helpers

# These are
#   1. addons to make life with Ferrite easier
#   2. potentially missing dispatches which will be temporarily pirated before upstreamed into Ferrite.jl
include("core/ferrite-addons/collections.jl")
include("core/ferrite-addons/mappings.jl")
include("core/ferrite-addons/assembly.jl")
include("core/ferrite-addons/parallel_duplication_api.jl")

# Some generic integrators
abstract type AbstractBilinearIntegrator end
abstract type AbstractNonlinearIntegrator end
abstract type AbstractLinearIntegrator end

include("elements/composite_elements.jl")     # This is the key component to allow high level composition of operators
include("elements/simple_diffusion.jl")       # Example element for diffusion
include("elements/simple_mass.jl")            # Example element for mass matrices
include("elements/simple_hyperelasticity.jl") # Example element for hyperelasticity

include("operators/general.jl")         # Some general operators which might be handy
include("operators/matrix_free.jl")     # Everything related to the fundamental decomposition
include("operators/tasks.jl")           # Here are all the tasks to handle the assembly and action of operators
include("operators/setup.jl")           # Nitty gritty helpers to handle the setup of operators without poking into internals

export QuadratureRuleCollection

export setup_operator, update_operator!, update_linearization!

export NullOperator, DiagonalOperator

export SequentialCPUDevice, PolyesterDevice, CudaDevice
export SequentialAssemblyStrategy, ElementAssemblyStrategy, PerColorAssemblyStrategy

end
