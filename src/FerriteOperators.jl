module FerriteOperators

using Reexport
@reexport using Ferrite
using TimerOutputs
using Adapt
using Unrolled
using SparseArrays, StaticArrays
import SparseArrays: AbstractSparseMatrixCSC, getcolptr

using ConcreteStructs

import LinearAlgebra: mul!, ldiv!, qr, lu!, cholesky!, Symmetric, dot, norm

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

include("core/device.jl")    # Device abstraction: CPU threads, GPUs
include("core/strategy.jl")  # Assembly strategies
include("core/requests.jl")           # Assembly requests: the element kernel contract
include("core/element_interface.jl")  # Cache supertypes + the empty caches
include("core/tasks.jl")              # Assembly kinds and the task system
include("core/iterators.jl")          # Two-DofHandler cell iterators

include("core/utils.jl")
include("core/qvector.jl")           # Flat per-cell quadrature data storage

# Conveniences for working with Ferrite, plus dispatches missing from
# Ferrite.jl and pirated here.
include("core/ferrite-addons/collections.jl")
include("core/ferrite-addons/mappings.jl")
include("core/ferrite-addons/assembly.jl")
include("core/ferrite-addons/parallel_duplication_api.jl")
include("core/ferrite-addons/internal_variable_handler.jl")

"""
    AbstractBilinearIntegrator

Supertype of integrators describing a **bilinear form** `a(u, v)`: the element
kernel fills a state-independent element matrix. The operator that form induces
is linear, so its assembled matrix IS that operator's Jacobian and its residual
is the action `F(u) = A·u`.

A structural declaration, not a performance hint: such an operator issues no
sensitivity kind, so it carries neither the [`ADElementCache`](@ref) decoration
nor the per-worker [`SensitivityBuffers`](@ref), whatever its element caches
implement analytically ([`needs_ad_decoration`](@ref)).
"""
abstract type AbstractBilinearIntegrator end

"""
    AbstractNonlinearIntegrator

Supertype of integrators describing a state-dependent residual form — the
general case, and the only family whose operators carry differentiation
machinery. The residual kernel is mandatory; every other request is served
analytically where [`provides_analytic`](@ref) declares it, otherwise by
differentiating that kernel — what the [`ADElementCache`](@ref) decoration and
the per-worker [`SensitivityBuffers`](@ref) exist for.

A bilinear sub-integrator composes into this family (the operator its form
induces scatters into the same matrix and residual); a linear one does not, its
sink being a vector alone.
"""
abstract type AbstractNonlinearIntegrator end

abstract type AbstractCondensedNonlinearIntegrator <: AbstractNonlinearIntegrator end

"""
    AbstractLinearIntegrator

Supertype of integrators describing a **linear form** `l(v)`: the element kernel
fills the form's load vector, and the operator holds that vector and no matrix.
The form has no state to depend on, so this family too carries no AD or
sensitivity machinery, and a [`BlockedOperatorSpecification`](@ref) on such an
operator is rejected at `setup_operator` — there is no matrix to lay out.
"""
abstract type AbstractLinearIntegrator end

include("elements/composite_elements.jl")     # High-level composition of operators
include("elements/ad_element.jl")             # ADElementCache: AD as an element cache decorator

include("operators/general.jl")         # Domain descriptors, NullOperator, DiagonalOperator
include("operators/matrix_free.jl")     # Element-assembly storage and matrix-free products
include("operators/nonlinear.jl")       # Assembly and action tasks
include("operators/bilinear.jl")
include("operators/linear.jl")
include("operators/transfer.jl")        # Prolongation/restriction operators
include("elements/prolongators.jl")     # Mass-based prolongator integrators
include("operators/ad_decoration.jl")   # Construction-time ADElementCache/FusedFromSplit wrapping policy
include("operators/setup.jl")           # Operator setup, so callers need not poke into internals
include("elements/domain_elements.jl")  # Subdomain routing; specializes setup.jl's per-DofHandler cache seam
include("operators/components.jl")      # Component bags over a shared sparsity pattern + combine!
include("operators/stage_block.jl")     # Fully implicit Runge-Kutta stage blocks
include("operators/verification.jl")    # check_derivatives: FD referee for analytic kernels and AD paths
include("operators/condensation.jl")    # condense_internal!: element-local solves up front, pure evaluation after

include("core/quadrature-task.jl")      # Evaluating functions at quadrature points
include("core/item_states.jl")          # ItemStates: provider-agnostic per-item persistent storage
include("core/patch-task.jl")           # Patch items: multi-cell work items with patch-local scatter (experimental)
include("core/facet-task.jl")           # Facet items: facet-set-restricted boundary terms as their own traversal
include("core/algebraic-task.jl")       # Algebraic items: work items that are a dof set and nothing else

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
export get_declared_slots, get_declared_kinds
export assemble_slot_jacobian!, assemble_weighted_jacobian!
export allocate_components, share_pattern, combine!
export StageBlockOperator, assemble_stages!
export update_parameter_jacobian!, parameter_vjp!, time_sensitivity!
export ADSensitivity, FiniteDifferenceSensitivity, has_internal_state, internal_state_insensitive
export state_jvp!, state_vjp!, StateJVPRequest, StateVJPRequest
export check_derivatives
export parameter_vector, rebuild_parameters
export TimeIntegrationContext, evaluation_time, with_time, stage_scaling, CellArgs, FacetArgs, assemble_cell!
export AffineRate, InternalSource
export CorrectionMode, Consistent, FrozenQ
export CondensationReport, condense_internal!, condense_cell!, condense_algebraic!, CondensationKind
export local_conditions!
export condensed_update_linearization!, rollback_state!, commit_state!, invalidate_correctors!
export allocate_internal_jacobian, update_internal_jacobian!
export CorrectorElection, Stored, Recompute, corrector_election, corrector_election_error
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
export ItemStates, item_state, set_item_state!, has_item_state, invalidate_item_state!, invalidate_item_states!
export provides_analytic
export query_cell_parameters, query_facet_parameters, unwrap_parameters, assemble_facet!, is_facet_in_cache
export reinit_values!
export ADElementCache, AbstractElementCacheDecorator, unwrap, ForwardDiffAD, FusedFromSplit, condensed_corrector
export decorate_element_cache, needs_ad_decoration, fully_analytic

export residual_size, unknown_size

export NullOperator, DiagonalOperator

export SequentialCPUDevice, PolyesterDevice
export SequentialAssemblyStrategy, ElementAssemblyStrategy, PerColorAssemblyStrategy
export AssemblyStrategy, AbstractAssemblyForm, FullAssembly, ElementAssembly
export AbstractSchedulingPolicy, SequentialScheduling, ColoredScheduling
export StandardOperatorSpecification, BlockedOperatorSpecification

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
export global_dofs, global_dof_range
export facet_items, setup_facet_item_cache, facet_item_global_dofs, facet_item_global_dof_range
export algebraic_items, setup_algebraic_cache, assemble_algebraic!, evaluate_algebraic_functional
export AlgebraicItem, AlgebraicArgs
export TransferFerriteOperator, setup_transfer_operator, init_transfer_sparsity_pattern
export NestedTransferFerriteOperator, setup_nested_transfer_operator, init_nested_transfer_sparsity_pattern

export NonlinearMultiDomainIntegrator, BilinearMultiDomainIntegrator, LinearMultiDomainIntegrator
export NonlinearCompositeIntegrator, BilinearCompositeIntegrator, LinearCompositeIntegrator

end
