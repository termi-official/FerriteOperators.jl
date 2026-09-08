@doc raw"""
    SumFactorizedDiffusionIntegrator(D, qrc, field_name)

The same bilinear form as [`SimpleBilinearDiffusionIntegrator`](@ref),
``a(u,v) = \int \nabla v \cdot D \nabla u \,dx``, written for the MATRIX-FREE
level: its cache evaluates the ACTION `yₑ = Kₑ·uₑ` by sum factorization
(Deville–Fischer–Mund) in `O(p^{d+1})` per cell instead of forming `Kₑ`.

`D` is a constant scalar (isotropic) or a `SymmetricTensor{2, dim}`. Set the
operator up with `form = MatrixFreeAction()`; the cache serves both element
mappings and both `qdata` elections, so one definition of the element math runs
one worker per element and one workgroup per element, with the geometric factors
stored or re-derived.

The element math is the pointwise map ``W_q = w_q \det(J_q) J_q^{-1} D
J_q^{-T}``; everything around it — the 1D operators, the lattice permutations
and the contractions — is [`AbstractTensorProductElementCache`](@ref)'s, whose
scope (tensor-product Lagrange on `RefQuadrilateral`/`RefHexahedron`, distorted
cells included) is this integrator's.

The cache has no element-matrix kernel: `FullAssembly` over this integrator is
refused where the matrix would be formed. Assemble
[`SimpleBilinearDiffusionIntegrator`](@ref) for a matrix of the same form.
"""
struct SumFactorizedDiffusionIntegrator{DT, QRC} <: AbstractBilinearIntegrator
    D::DT
    qrc::QRC
    field_name::Symbol
end

"""
    SumFactorizedDiffusionElementCache

The cache [`SumFactorizedDiffusionIntegrator`](@ref) sets up: the diffusion
tensor in the reference frame, one `TensorProductValues`, the per-quadrature-point
factor store the form's `qdata` election allocated (`nothing` under
`Recompute()`), and the contraction scratch.
"""
struct SumFactorizedDiffusionElementCache{DT, VT, QT, ST} <: AbstractTensorProductElementCache
    D::DT
    values::VT
    qdata::QT
    scratch::ST
end

tensor_product_quantity(::SumFactorizedDiffusionElementCache) = QuadratureGradient()

# The reference operators and the factor store are `isbits` or shared read-only;
# the scratch is the one per-worker field, so the device pair batches and slices
# exactly that.
duplicate_for_device(device, c::SumFactorizedDiffusionElementCache) =
    SumFactorizedDiffusionElementCache(c.D, c.values, c.qdata, copy(c.scratch))
setup_device_instances(device::AbstractGPUDevice, c::SumFactorizedDiffusionElementCache, n) =
    SumFactorizedDiffusionElementCache(c.D, c.values, adapt_shared(device, c.qdata),
        setup_device_instances(device, tensor_product_scratch_prototype(device, c.scratch), n))
device_worker_view(c::SumFactorizedDiffusionElementCache, worker) =
    SumFactorizedDiffusionElementCache(c.D, c.values, c.qdata, device_worker_view(c.scratch, worker))

####################################
## Setup
####################################

_diffusion_matrix(D, ::TensorProductValues{dim, Nb, Nq, Nn, T}) where {dim, Nb, Nq, Nn, T} =
    _diffusion_matrix(D, Val(dim), T)
_diffusion_matrix(D::Number, ::Val{dim}, ::Type{T}) where {dim, T} = SMatrix{dim, dim, T}(
    ntuple(k -> (k - 1) % dim == (k - 1) ÷ dim ? T(D) : zero(T), dim * dim))
_diffusion_matrix(D::SymmetricTensor{2, dim}, ::Val{dim}, ::Type{T}) where {dim, T} = SMatrix{dim, dim, T}(
    ntuple(k -> T(D[(k - 1) % dim + 1, (k - 1) ÷ dim + 1]), dim * dim))

function setup_element_cache(model::SumFactorizedDiffusionIntegrator, sdh::SubDofHandler)
    values = TensorProductValues(sdh, model.field_name, model.qrc)
    return SumFactorizedDiffusionElementCache(
        _diffusion_matrix(model.D, values), values,
        nothing, allocate_tensor_product_scratch(values, QuadratureGradient()))
end

_factor_type(::TensorProductValues{dim, Nb, Nq, Nn, T}) where {dim, Nb, Nq, Nn, T} =
    SMatrix{dim, dim, T, dim * dim}

# The form's `qdata` election: one `W_q` per quadrature point of every cell of
# the subdomain, or nothing to store.
with_action_storage(cache::SumFactorizedDiffusionElementCache, storage::Stored, sdh) =
    SumFactorizedDiffusionElementCache(cache.D, cache.values,
        _diffusion_qdata(cache.values, sdh), cache.scratch)

_diffusion_qdata(values, sdh) =
    setup_qvector(_factor_type(values), sdh, getnquadpoints(values))

####################################
## The pointwise map
####################################

@inline function _form_diffusion_factor(cache::SumFactorizedDiffusionElementCache, coordinates, q)
    values = tensor_product_values(cache)
    J = tensor_product_jacobian(values, coordinates, q)
    Jinv = inv(J)
    return (tensor_product_weight(values, q) * det(J)) * (Jinv * cache.D * Jinv')
end

@inline tensor_product_pointwise(cache::SumFactorizedDiffusionElementCache, args, q, qp::Int, g) =
    _diffusion_factor(cache.qdata, cache, args, q, qp) * g

# `Recompute()`: the factor is formed at the quadrature point that consumes it,
# by the lane that produced it, so the geometry needs neither storage nor a
# barrier. `Stored()`: the same expression, evaluated once by the fill sweep
# below and read back here.
@inline _diffusion_factor(::Nothing, cache, args, q, qp::Int) =
    _form_diffusion_factor(cache, getcoordinates(args.cell), q)
@inline _diffusion_factor(qdata, cache, args, q, qp::Int) =
    @inbounds get_range_for_cell(qdata, cellid(args.cell))[qp]

function fill_quadrature_data!(cache::SumFactorizedDiffusionElementCache, args::CellArgs)
    cache.qdata === nothing && return nothing
    values = tensor_product_values(cache)
    coordinates = getcoordinates(args.cell)
    W = get_range_for_cell(cache.qdata, cellid(args.cell))
    for qp in 1:length(W)
        @inbounds W[qp] = _form_diffusion_factor(cache, coordinates, quadrature_lattice_index(values, qp))
    end
    return nothing
end
