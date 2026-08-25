# Adaption of the API presented in Ferrite.jl#1070 to general devices; essentially an Adapt.jl wrapper.
# The assemblers reconstruct through `typeof(asm)`: the concrete type carries its own parameters, so
# the duplication survives Ferrite adding or reordering type parameters as long as the field set
# (shared matrix and vector, four permutation scratches) is stable.
duplicate_for_device(device, ::Nothing) = nothing
for Assembler in (:CSCAssembler, :SymmetricCSCAssembler, :CSRAssembler)
    @eval function duplicate_for_device(device, asm::Ferrite.$Assembler)
        return typeof(asm)(
            asm.K,
            asm.f,
            duplicate_for_device(device, asm.rowpermutation),
            duplicate_for_device(device, asm.colpermutation),
            duplicate_for_device(device, asm.sortedrowdofs),
            duplicate_for_device(device, asm.sortedcoldofs),
        )
    end
end

function duplicate_for_device(device, fv::FacetValues)
    return FacetValues(
        duplicate_for_device(device, fv.fun_values),
        duplicate_for_device(device, fv.geo_mapping),
        duplicate_for_device(device, fv.fqr),
        duplicate_for_device(device, fv.detJdV),
        duplicate_for_device(device, fv.normals),
        duplicate_for_device(device, fv.current_facet),
    )
end

function duplicate_for_device(device, cv::CellValues)
    return CellValues(
        duplicate_for_device(device, cv.fun_values),
        duplicate_for_device(device, cv.geo_mapping),
        duplicate_for_device(device, cv.qr),
        duplicate_for_device(device, cv.detJdV),
    )
end

function duplicate_for_device(device, v::Ferrite.FunctionValues)
    Nξ = v.Nξ
    Nx = v.Nξ === v.Nx ? Nξ : duplicate_for_device(device, v.Nx) # Preserve aliasing
    return Ferrite.FunctionValues(
        duplicate_for_device(device, v.ip),
        duplicate_for_device(device, v.Nx),
        Nξ,
        duplicate_for_device(device, v.dNdx),
        v.dNdξ,
        duplicate_for_device(device, v.d2Ndx2),
        v.d2Ndξ2,
    )
end

function duplicate_for_device(device, v::Ferrite.GeometryMapping)
    return Ferrite.GeometryMapping(
        duplicate_for_device(device, v.ip),
        v.M,
        v.dMdξ,
        v.d2Mdξ2
    )
end

function duplicate_for_device(device, qr::QR) where {refshape, QR <: QuadratureRule{refshape}}
    return QuadratureRule{refshape}(duplicate_for_device(device, qr.weights), duplicate_for_device(device, qr.points))::QR
end

function duplicate_for_device(device, qr::QR) where {refshape, QR <: FacetQuadratureRule{refshape}}
    return FacetQuadratureRule{refshape}(duplicate_for_device(device, qr.facet_rules))::QR
end

duplicate_for_device(device, ip::Ferrite.Interpolation) = ip

function duplicate_for_device(device, x::T)::T where {T <: Tuple}
    if isbitstype(T)
        return x
    else
        return map(y->duplicate_for_device(device, y), x)::T
    end
end

function duplicate_for_device(device, x::T)::T where {T}
    if !isbitstype(T)
        error("MethodError: duplicate_for_device(device, ::$T) is not implemented")
    end
    return x
end

function duplicate_for_device(device, x::T)::T where {S, T <: DenseArray{S}}
    @assert !isbitstype(T)
    if isbitstype(S)
        return copy(x)::T
    else
        return map(y->duplicate_for_device(device,y), x)::T
    end
end
