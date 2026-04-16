module FerriteOperators

using Reexport
@reexport using Ferrite
using TimerOutputs
using Adapt
using Unrolled
using SparseArrays, StaticArrays
import SparseArrays: AbstractSparseMatrixCSC

using ConcreteStructs

import LinearAlgebra: mul!, ldiv!, qr, cholesky!, Symmetric

import Base: *, +, -, @kwdef, @propagate_inbounds

import Atomix

import Ferrite: AbstractDofHandler, AbstractGrid, AbstractRefShape, AbstractCell, get_grid, get_coordinate_eltype
import Ferrite: SparsityPattern, allocate_matrix
import Ferrite: AbstractCSCAssembler, AbstractCSRAssembler, matrix_handle, fillzero!
import Ferrite: vertices, edges, faces, sortedge, sortface
import Ferrite: get_coordinate_type, getspatialdim
import Ferrite: reference_shape_value
import Ferrite: IntegerCollection
import Ferrite: nnodes_per_cell, cellnodes!, getcoordinates!

include("core/device.jl")    # Utilities to manage devices (e.g. CPU threads or GPUs)
include("core/strategy.jl")  # Utilities to control the assembly strategy
include("core/tasks.jl")     # Contains the basic task system
include("core/iterators.jl") # Transfer cell iterators for two-DofHandler assembly

include("core/element_interface.jl") # This is the basic element interface used for the operators
include("core/utils.jl")             # Internal helpers

# These are
#   1. addons to make life with Ferrite easier
#   2. potentially missing dispatches which will be temporarily pirated before upstreamed into Ferrite.jl
include("core/ferrite-addons/collections.jl")
include("core/ferrite-addons/mappings.jl")
include("core/ferrite-addons/assembly.jl")
include("core/ferrite-addons/parallel_duplication_api.jl")
include("core/ferrite-addons/internal_variable_handler.jl")

# Some generic integrator types
abstract type AbstractBilinearIntegrator end
abstract type AbstractNonlinearIntegrator end
abstract type AbstractCondensedNonlinearIntegrator <: AbstractNonlinearIntegrator end
# Simple means that it has a constant number of dofs per quadrature point
abstract type AbstractSimpleCondensedNonlinearIntegrator <: AbstractNonlinearIntegrator end
abstract type AbstractLinearIntegrator end

include("elements/composite_elements.jl")     # This is the key component to allow high level composition of operators
include("elements/generic_first_order_time_element.jl")

include("operators/general.jl")         # Some general operators which might be handy
include("operators/matrix_free.jl")     # Everything related to the fundamental decomposition
include("operators/nonlinear.jl")       # Here are all the tasks to handle the assembly and action of operators
include("operators/bilinear.jl")
include("operators/linear.jl")
include("operators/transfer.jl")        # Transfer (prolongation/restriction) operators
include("operators/setup.jl")           # Nitty gritty helpers to handle the setup of operators without poking into internals

include("elements/simple_diffusion.jl")       # Example element for diffusion
include("elements/simple_mass.jl")            # Example element for mass matrices
include("elements/simple_hyperelasticity.jl") # Example element for hyperelasticity
include("elements/simple_linear_viscoelasticity.jl")

export QuadratureRuleCollection, QuadratureInterpolation, InternalVariableHandler
export getquadraturerule
export AbstractBilinearIntegrator, AbstractNonlinearIntegrator, AbstractLinearIntegrator

export setup_operator, update_operator!, update_linearization!

export residual_size, unknown_size

export NullOperator, DiagonalOperator

export SequentialCPUDevice, PolyesterDevice, CudaDevice
export SequentialAssemblyStrategy, ElementAssemblyStrategy, PerColorAssemblyStrategy

# Transfer operator infrastructure
export SameGridCellCache, SameGridCellIterator
export NestedGridCellCache, NestedGridCellIterator
export getrowdofs, getcolumndofs
export get_fine_coordinates, get_coarse_coordinates, get_child_ref_coords
export AbstractTransferIntegrator, AbstractTransferElementCache
export AbstractVolumetricElementCache
export MassProlongatorIntegrator
export NestedMassProlongatorIntegrator
export setup_element_cache, assemble_element!
export TransferFerriteOperator, setup_transfer_operator, init_transfer_sparsity_pattern
export NestedTransferFerriteOperator, setup_nested_transfer_operator, init_nested_transfer_sparsity_pattern

end
