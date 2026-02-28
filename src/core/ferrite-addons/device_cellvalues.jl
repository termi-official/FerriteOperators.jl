## DeviceCellValuesFactory ##
# Pre-allocates pools for mutable CellValues data (detJdV, dNdx) on GPU.
# Immutable reference data (dNdξ, dMdξ, Nξ, weights) is shared across threads.
# Same Factory → materialize pattern as DeviceCellCacheFactory.

struct DeviceCellValuesFactory{DetPool, DNdxPool, DNdξT, DMdξT, NξT, WT, Ti <: Integer}
    # Per-thread mutable pools
    detJdV_pool::DetPool     # GPU array (nqp, total_nthreads)
    dNdx_pool::DNdxPool      # GPU array (nbf, nqp, total_nthreads)
    # Shared immutable data (on GPU)
    dNdξ::DNdξT              # GPU array (nbf, nqp) — reference gradients
    dMdξ::DMdξT              # GPU array (ngeo, nqp) — geometric ref gradients
    Nξ::NξT                  # GPU array (nbf, nqp) — reference shape values
    weights::WT              # GPU array (nqp,)      — quadrature weights
    nqp::Ti
    nbf::Ti
    ngeo::Ti
end

"""
    DeviceCellValuesFactory(cv::CellValues, device::AbstractGPUDevice; total_nthreads)

Build a `DeviceCellValuesFactory` from a CPU `CellValues`.
Extracts immutable reference data to GPU and allocates per-thread pools.
"""
function DeviceCellValuesFactory(cv::CellValues, device::AbstractGPUDevice; total_nthreads)
    Ti = index_type(device)
    backend = default_backend(device)

    # Dimensions
    nqp  = getnquadpoints(cv)
    nbf  = getnbasefunctions(cv)
    ngeo = Ferrite.getngeobasefunctions(cv.geo_mapping)

    # Shared immutable data → GPU
    dNdξ_gpu    = Adapt.adapt(backend, cv.fun_values.dNdξ)
    dMdξ_gpu    = Adapt.adapt(backend, cv.geo_mapping.dMdξ)
    Nξ_gpu      = Adapt.adapt(backend, cv.fun_values.Nξ)
    weights_gpu = Adapt.adapt(backend, cv.qr.weights)

    # Per-thread mutable pools
    DetT  = eltype(cv.detJdV)                # e.g. Float64
    GradT = eltype(cv.fun_values.dNdx)       # e.g. Vec{2, Float64}
    detJdV_pool = _gpu_zeros(device, DetT, nqp, total_nthreads)
    dNdx_pool   = _gpu_zeros(device, GradT, nbf, nqp, total_nthreads)

    return DeviceCellValuesFactory(
        detJdV_pool, dNdx_pool,
        dNdξ_gpu, dMdξ_gpu, Nξ_gpu, weights_gpu,
        Ti(nqp), Ti(nbf), Ti(ngeo),
    )
end


## DeviceCellValues ##
# Immutable struct — mutable fields are views into pools.
# Created once per thread via materialize(), updated per cell via reinit!().
# Implements the Ferrite CellValues API (identity mapping only).

struct DeviceCellValues{DetView, DNdxView, DNdξT, DMdξT, NξT, WT, Ti <: Integer} <: Ferrite.AbstractCellValues
    # Per-thread mutable views
    detJdV::DetView          # view into detJdV_pool[:, tid]
    dNdx::DNdxView           # view into dNdx_pool[:, :, tid]
    # Shared immutable data (references, not copies)
    dNdξ::DNdξT
    dMdξ::DMdξT
    Nξ::NξT
    weights::WT
    nqp::Ti
    nbf::Ti
    ngeo::Ti
end

@inline function materialize(factory::DeviceCellValuesFactory, tid)
    DeviceCellValues(
        view(factory.detJdV_pool, :, tid),
        view(factory.dNdx_pool, :, :, tid),
        factory.dNdξ,
        factory.dMdξ,
        factory.Nξ,
        factory.weights,
        factory.nqp,
        factory.nbf,
        factory.ngeo,
    )
end


## Ferrite-compatible accessors ##
@inline Ferrite.getnquadpoints(cv::DeviceCellValues) = cv.nqp
@inline Ferrite.getnbasefunctions(cv::DeviceCellValues) = cv.nbf
@inline Ferrite.getdetJdV(cv::DeviceCellValues, qp::Integer) = cv.detJdV[qp]
@inline Ferrite.shape_gradient(cv::DeviceCellValues, qp::Integer, i::Integer) = cv.dNdx[i, qp]
@inline Ferrite.shape_value(cv::DeviceCellValues, qp::Integer, i::Integer) = cv.Nξ[i, qp]


## reinit! — computes Jacobian, detJdV, and physical-space gradients ##

# Dispatch: DeviceCellValues + DeviceCellCache → extract coords and forward
@inline function Ferrite.reinit!(cv::DeviceCellValues, cc::DeviceCellCache)
    _reinit_device_cv!(cv, cc.coords)
    return nothing
end

# Dispatch: DeviceCellValues + coords vector
@inline function Ferrite.reinit!(cv::DeviceCellValues, x::AbstractVector)
    _reinit_device_cv!(cv, x)
    return nothing
end

# Actual implementation — separate function avoids Ferrite dispatch ambiguity
@inline function _reinit_device_cv!(cv::DeviceCellValues, x)
    @inbounds for q in 1:cv.nqp
        # Compute Jacobian: J = Σⱼ x[j] ⊗ dMdξ[j, q]
        J = _compute_jacobian(cv.dMdξ, x, cv.ngeo, q)
        # det(J) × quadrature weight
        cv.detJdV[q] = det(J) * cv.weights[q]
        # Physical-space gradients: dNdx = dNdξ · J⁻¹
        Jinv = inv(J)
        for i in 1:cv.nbf
            cv.dNdx[i, q] = cv.dNdξ[i, q] ⋅ Jinv
        end
    end
    return nothing
end

# Compute Jacobian from geometric mapping gradients and coordinates
@inline function _compute_jacobian(dMdξ, coords, ngeo, q)
    @inbounds begin
        J = coords[1] ⊗ dMdξ[1, q]
        for j in 2:ngeo
            J += coords[j] ⊗ dMdξ[j, q]
        end
    end
    return J
end


## GPU duplicate_for_device: CellValues → DeviceCellValuesFactory ##
# Overrides the CPU deep-copy path in parallel_duplication_api.jl.
# Uses total_nthreads(device) — device must be resolved via resolve_launch_config first.
function duplicate_for_device(device::AbstractGPUDevice, cv::CellValues)
    return DeviceCellValuesFactory(cv, device; total_nthreads=total_nthreads(device))
end
