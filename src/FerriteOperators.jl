module FerriteOperators

using Reexport
@reexport using Ferrite
using TimerOutputs
using Adapt
using Unrolled
using SparseArrays, StaticArrays
import SparseArrays: AbstractSparseMatrixCSC, getcolptr

using ConcreteStructs

import LinearAlgebra: mul!, ldiv!, qr, cholesky!, Symmetric, dot, norm

import ForwardDiff

import Base: *, +, -, @kwdef, @propagate_inbounds

import Atomix
import Preferences

import VTKBase
import WriteVTK

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
include("core/requests.jl")           # Assembly requests: the element interface v2 contract
include("core/element_interface.jl")  # Cache supertypes + the empty caches
include("core/ad.jl")                 # ForwardDiff fallbacks deriving requests from residual kernels
include("core/tasks.jl")              # Contains the basic task system
include("core/iterators.jl")          # Transfer cell iterators for two-DofHandler assembly

include("core/utils.jl")             # Internal helpers
include("core/qvector.jl")           # Flat per-cell quadrature data storage

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
abstract type AbstractLinearIntegrator end

include("elements/composite_elements.jl")     # This is the key component to allow high level composition of operators

include("operators/general.jl")         # Some general operators which might be handy
include("operators/matrix_free.jl")     # Everything related to the fundamental decomposition
include("operators/nonlinear.jl")       # Here are all the tasks to handle the assembly and action of operators
include("operators/bilinear.jl")
include("operators/linear.jl")
include("operators/transfer.jl")        # Transfer (prolongation/restriction) operators
include("elements/prolongators.jl")     # Transfer integrators assembling mass-based prolongators
include("operators/setup.jl")           # Nitty gritty helpers to handle the setup of operators without poking into internals
include("elements/domain_elements.jl")  # Subdomain routing; specializes setup.jl's per-DofHandler cache setup seam
include("operators/components.jl")      # Component bags over a shared sparsity pattern + combine!
include("operators/stage_block.jl")     # Stage-block operator for fully implicit Runge-Kutta schemes
include("operators/verification.jl")    # check_derivatives: FD referee for analytic kernels and AD paths

include("core/quadrature-task.jl")      # Task + operator for evaluating functions at quadrature points
include("core/patch-task.jl")           # Patch items: multi-cell work items with patch-local scatter (experimental)

include("postprocessing/quadrature-grid.jl")  # VTKQuadratureGrid — QP positions as a VTK mesh
include("postprocessing/quadrature-query.jl") # VTKQuadratureFile + write_quadrature_data

export QuadratureRuleCollection, InternalVariableHandler
export internal_variable_offset, internal_variable_range
export getquadraturerule
export AbstractBilinearIntegrator, AbstractNonlinearIntegrator, AbstractLinearIntegrator

export QVector, setup_qvector, get_range_for_cell
export evaluate_quadrature!
export query_element_quadrature_data, store_quadrature_data!
export VTKQuadratureGrid, VTKQuadratureFile, write_quadrature_data
export QuadratureDataQuery, QuadratureDataMultiQuery, prepare_quadrature_query, process_query!

export setup_operator, update_operator!, update_linearization!, evaluate!
export AbstractSchemeProtocol, DefaultProtocol
export declared_slots, declared_kinds, declared_scratch, declared_args_type, mandatory_kinds
export assemble_slot_jacobian!, assemble_weighted_jacobian!
export allocate_components, share_pattern, combine!
export StageBlockOperator, assemble_stages!
export update_parameter_jacobian!, parameter_vjp!, time_sensitivity!
export ADSensitivity, FiniteDifferenceSensitivity, has_internal_state, internal_state_insensitive
export state_jvp!, state_vjp!, StateJVPRequest, StateVJPRequest
export check_derivatives
export parameter_vector, rebuild_parameters
export TimeIntegrationContext, evaluation_time, KernelArgs, assemble_cell!
export AffineRate
export AbstractAssemblyRequest, ResidualRequest, JacobianRequest, JacobianResidualRequest
export WeightedJacobianRequest
export ParameterJacobianRequest, ParameterVJPRequest, TimeSensitivityRequest
export ResidualKind, JacobianKind, JacobianResidualKind, WeightedJacobianKind
export ParameterJacobianKind, ParameterVJPKind, TimeSensitivityKind, StateJVPKind, StateVJPKind
export FunctionalKind, evaluate_functional, evaluate_cell_functional
export PatchItems, npatches, patch_dofs, patch_ndofs, patch_cells, patch_cell_groups, assemble_patch_matrices!
export patch_free_dofs, patch_prescribed_dofs, augment_prescribed_dofs!, patch_vertices, patch_vertex_dofs
export WholePatch, CellGroup, PatchTerm, patch_term_active, any_patch_term_active
export whole_patch_terms, assemble_patch_cell!
export PatchMatrixKind, PatchVectorKind, PatchCallbackKind, assemble_patches!, foreach_patch
export PatchAssemblyWorkspace, patch_workspace, current_patch, patch_provider
export assemble_patch_target!, patch_chunks
export AbstractPatchSink, patch_target, patch_scatter, patch_emit!
export PatchLocalSink, PatchAssemblerSink, PatchGlobalVectorSink, PatchTripletSink, emit_patch_column!
export PatchItemStates, item_state, set_item_state!, has_item_state, invalidate_item_state!, invalidate_item_states!
export provides_analytic
export query_cell_parameters, query_facet_parameters, unwrap_parameters, assemble_facet!, is_facet_in_cache
export reinit_values!

export residual_size, unknown_size

export NullOperator, DiagonalOperator

export SequentialCPUDevice, PolyesterDevice, CudaDevice
export SequentialAssemblyStrategy, ElementAssemblyStrategy, PerColorAssemblyStrategy
export AssemblyStrategy, AbstractAssemblyForm, FullAssembly, ElementAssembly
export AbstractSchedulingPolicy, SequentialScheduling, ColoredScheduling

# Transfer operator infrastructure
export SameGridCellCache, SameGridCellIterator
export NestedGridCellCache, NestedGridCellIterator
export getrowdofs, getcolumndofs
export get_fine_coordinates, get_coarse_coordinates, get_child_ref_coords
export AbstractTransferIntegrator, AbstractTransferElementCache
export AbstractVolumetricElementCache
export MassProlongatorIntegrator
export NestedMassProlongatorIntegrator
export setup_element_cache, setup_boundary_cache
export TransferFerriteOperator, setup_transfer_operator, init_transfer_sparsity_pattern
export NestedTransferFerriteOperator, setup_nested_transfer_operator, init_nested_transfer_sparsity_pattern

export NonlinearMultiDomainIntegrator, BilinearMultiDomainIntegrator, LinearMultiDomainIntegrator
export NonlinearCompositeIntegrator, BilinearCompositeIntegrator, LinearCompositeIntegrator

end
