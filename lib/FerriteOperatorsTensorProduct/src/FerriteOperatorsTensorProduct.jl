"""
The tensor-product sum-factorization core for
[FerriteOperators](https://github.com/termi-official/FerriteOperators.jl)'
matrix-free elements: the 1D reference operators, the lattice permutations,
the contraction pipeline, and the two reference elements written against it —
[`SumFactorizedDiffusionIntegrator`](@ref) and
[`SumFactorizedMassIntegrator`](@ref).

An element in this family is [`AbstractTensorProductElementCache`](@ref) plus
a POINTWISE MAP ([`tensor_product_pointwise`](@ref)); everything else — the
`B`/`G` blocks of the MFEM/libCEED decomposition and both execution mappings
([`WorkerPerElement`](@ref FerriteOperators.WorkerPerElement)/[`CooperativeElement`](@ref)) — is this package's.

Scope: tensor-product Lagrange on `RefQuadrilateral`/`RefHexahedron`, the
shapes Ferrite ships such a basis for.

!!! warning "Experimental surface"
    This package's names and the element entry points they call may change in
    a minor release.
"""
module FerriteOperatorsTensorProduct

using Ferrite
using StaticArrays
using Tensors

using LinearAlgebra: norm, det, inv

import FerriteOperators: element_value_type, get_first_cell, AbstractVolumetricElementCache,
    item_update_flags, MatrixFreeActionKind, allocate_element_matrix, reinit_values!,
    assemble_cell!, ResidualRequest, JacobianRequest, JacobianResidualRequest, CellArgs,
    apply_element_action!,
    cooperative_lattice_dim, cooperative_group_size, cooperative_scratch_shape,
    cooperative_load!, cooperative_stage!, cooperative_store!,
    element_mapping, CooperativeElement,
    QuadratureRuleCollection,
    AbstractBilinearIntegrator, AbstractGPUDevice,
    duplicate_for_device, setup_device_instances, device_worker_view, adapt_shared,
    setup_element_cache, with_action_storage, fill_quadrature_data!, Stored,
    setup_qvector, get_range_for_cell

include("tensor_product.jl")             # 1D operators, lattice, contractions, both mapping pipelines
include("sum_factorized_diffusion.jl")   # Reference consumer: diffusion bilinear form
include("sum_factorized_mass.jl")        # Reference consumer: mass bilinear form

export TensorProductValues, AbstractTensorProductElementCache
export AbstractQuadratureQuantity, QuadratureValue, QuadratureGradient
export tensor_product_values, tensor_product_scratch, tensor_product_quantity, tensor_product_pointwise
export tensor_product_jacobian, tensor_product_weight, tensor_product_contract!
export allocate_tensor_product_scratch, tensor_product_scratch_prototype, quadrature_lattice_index

export SumFactorizedDiffusionIntegrator
export SumFactorizedMassIntegrator

end
