@doc raw"""
    SumFactorizedMassIntegrator(ρ, qrc, field_name)

The same bilinear form as [`SimpleBilinearMassIntegrator`](@ref FerriteOperatorsExampleElements.SimpleBilinearMassIntegrator),
``a(u,v) = \int v \, \rho \, u \,dx``, written for the MATRIX-FREE level: its
cache evaluates the ACTION `yₑ = Mₑ·uₑ` by sum factorization, reusing
[`AbstractTensorProductElementCache`](@ref) whole. `ρ` is a constant scalar.

The whole element is its pointwise map ``\rho \, w_q \det(J_q)`` — a SCALAR
factor on the interpolated value, where
[`SumFactorizedDiffusionIntegrator`](@ref) has a tensor on the reference
gradient. Everything else is [`AbstractTensorProductElementCache`](@ref)'s.

Set the operator up with `form = MatrixFreeAction()`. The cache has no
element-matrix kernel; assemble [`SimpleBilinearMassIntegrator`](@ref FerriteOperatorsExampleElements.SimpleBilinearMassIntegrator) for a
matrix of the same form.
"""
struct SumFactorizedMassIntegrator{QRC} <: AbstractBilinearIntegrator
    ρ::Float64
    qrc::QRC
    field_name::Symbol
end

"""
    SumFactorizedMassElementCache

The cache [`SumFactorizedMassIntegrator`](@ref) sets up: the density, one
`TensorProductValues`, the per-quadrature-point factor store the form's `storage`
election allocated (`nothing` under `Recompute()`), and the contraction scratch.
"""
struct SumFactorizedMassElementCache{T, VT, QT, ST} <: AbstractTensorProductElementCache
    ρ::T
    values::VT
    qdata::QT
    scratch::ST
end

tensor_product_quantity(::SumFactorizedMassElementCache) = QuadratureValue()

duplicate_for_device(device, c::SumFactorizedMassElementCache) =
    SumFactorizedMassElementCache(c.ρ, c.values, c.qdata, copy(c.scratch))
setup_device_instances(device::AbstractGPUDevice, c::SumFactorizedMassElementCache, n) =
    SumFactorizedMassElementCache(c.ρ, c.values, adapt_shared(device, c.qdata),
        setup_device_instances(device, tensor_product_scratch_prototype(device, c.scratch), n))
device_worker_view(c::SumFactorizedMassElementCache, worker) =
    SumFactorizedMassElementCache(c.ρ, c.values, c.qdata, device_worker_view(c.scratch, worker))

# As for the diffusion cache: `Recompute()` re-derives `w_q det(J_q)` from the
# cell's coordinates, the stored level reads it by cell id.
item_update_flags(::MatrixFreeActionKind, c::SumFactorizedMassElementCache) =
    Ferrite.UpdateFlags(nodes = false, coords = c.qdata === nothing, dofs = true)

# `a(u, v) = ∫ v ρ u` is symmetric for every scalar `ρ` — `u` and `v` enter
# through the same scalar multiplication ([`element_matrix_symmetry`](@ref)).
# The packed layout's action-time trade-off is the diffusion cache's, being a
# property of the packed index map rather than of the physics.
element_matrix_symmetry(::SumFactorizedMassElementCache) = SymmetricElementMatrix()

function setup_element_cache(model::SumFactorizedMassIntegrator, sdh::SubDofHandler)
    values = TensorProductValues(sdh, model.field_name, model.qrc)
    return SumFactorizedMassElementCache(element_value_type(values)(model.ρ), values,
        nothing, allocate_tensor_product_scratch(values, QuadratureValue()))
end

with_action_storage(cache::SumFactorizedMassElementCache, storage::Stored, sdh) =
    SumFactorizedMassElementCache(cache.ρ, cache.values,
        _mass_qdata(cache.values, sdh), cache.scratch)

_mass_qdata(values, sdh) =
    setup_qvector(element_value_type(values), sdh, getnquadpoints(values))

####################################
## The pointwise map
####################################

@inline function _form_mass_factor(cache::SumFactorizedMassElementCache, coordinates, q)
    values = tensor_product_values(cache)
    return tensor_product_weight(values, q) * det(tensor_product_jacobian(values, coordinates, q))
end

@inline tensor_product_pointwise(cache::SumFactorizedMassElementCache, args, q, qp::Int, u) =
    (cache.ρ * _mass_factor(cache.qdata, cache, args, q, qp)) * u

@inline _mass_factor(::Nothing, cache, args, q, qp::Int) = _form_mass_factor(cache, getcoordinates(args.cell), q)
@inline _mass_factor(qdata, cache, args, q, qp::Int) =
    @inbounds get_range_for_cell(qdata, cellid(args.cell))[qp]

function fill_quadrature_data!(cache::SumFactorizedMassElementCache, args::CellArgs)
    cache.qdata === nothing && return nothing
    values = tensor_product_values(cache)
    coordinates = getcoordinates(args.cell)
    w = get_range_for_cell(cache.qdata, cellid(args.cell))
    for qp in 1:length(w)
        @inbounds w[qp] = _form_mass_factor(cache, coordinates, quadrature_lattice_index(values, qp))
    end
    return nothing
end
