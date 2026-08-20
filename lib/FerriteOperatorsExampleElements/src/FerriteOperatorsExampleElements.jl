"""
Example elements for [FerriteOperators](https://github.com/termi-official/FerriteOperators.jl).

The elements here are minimal, readable implementations of the element
contract — one per feature the contract exposes: a bilinear form
([`SimpleBilinearDiffusionIntegrator`](@ref), [`SimpleBilinearMassIntegrator`](@ref)),
a linear form ([`SimpleLinearIntegrator`](@ref)), a nonlinear element with
analytic tangent ([`SimpleHyperelasticityIntegrator`](@ref)), a condensed
element with per-quadrature-point internal state
([`SimpleCondensedLinearViscoelasticity`](@ref)) and a condensed element whose
local problem is nonlinear and communicates with the outer solver
([`SimpleCondensedPowerLawRelaxation`](@ref)).

They are meant to be read, copied and used as test fixtures. They are not
tuned for production use and carry no stability guarantee beyond the element
contract they demonstrate.
"""
module FerriteOperatorsExampleElements

using FerriteOperators
using Ferrite
using Tensors
using StaticArrays

import Ferrite: getnquadpoints

import FerriteOperators: AbstractBilinearIntegrator, AbstractLinearIntegrator,
    AbstractCondensedNonlinearIntegrator, AbstractNonlinearIntegrator,
    AbstractVolumetricElementCache
import FerriteOperators: assemble_cell!, setup_element_cache, reinit_values!,
    provides_analytic, has_internal_state, duplicate_for_device,
    geometric_subdomain_interpolation, get_number_of_internal_dofs_per_element,
    internal_variable_offset, internal_variable_range,
    evaluation_time, with_time, stage_scaling, CellArgs,
    CorrectionMode, Consistent, FrozenQ, InternalSource,
    JacobianKind, JacobianResidualKind, JacobianRequest, JacobianResidualRequest,
    ParameterJacobianKind, ParameterJacobianRequest,
    CondensationReport, condense_cell!,
    CorrectorElection, Stored, Recompute, corrector_election, corrector_election_error,
    ItemStates, item_state, set_item_state!, has_item_state, invalidate_item_states!

include("simple_diffusion.jl")             # Bilinear form + its induced residual
include("simple_mass.jl")                  # Linear form and mass bilinear form
include("simple_hyperelasticity.jl")       # Nonlinear element with analytic tangent
include("simple_linear_viscoelasticity.jl") # Condensed element with internal state
include("simple_power_law_relaxation.jl")   # Condensed element with a nonlinear local solve

# The integrators are the public handle; the caches they set up are internal,
# reachable as `FerriteOperatorsExampleElements.Simple…ElementCache`.
export SimpleBilinearDiffusionIntegrator
export SimpleLinearIntegrator
export SimpleBilinearMassIntegrator
export SimpleHyperelasticityIntegrator
export SimpleCondensedLinearViscoelasticity
export MaxwellParameters
export SimpleCondensedPowerLawRelaxation
export NortonRelaxationParameters, LocalNewtonSettings
export InexactLocalSolveContext, local_solve_tolerance

end
