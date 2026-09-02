# Adaption of the API presented in Ferrite.jl#1070 to general devices.
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

# Ferrite's own `Base.copy` IS the per-worker duplication these types need: it
# copies the mutable per-cell scratch, preserves the aliasing between a
# `FunctionValues`' `Nξ` and `Nx`, and returns the immutable quadrature rules
# and interpolations as they are.
const FerriteCopyDuplicable = Union{
    CellValues, FacetValues, Ferrite.FunctionValues, Ferrite.GeometryMapping,
    QuadratureRule, FacetQuadratureRule, Ferrite.Interpolation,
}
duplicate_for_device(device, x::FerriteCopyDuplicable) = copy(x)

function duplicate_for_device(device, x::T)::T where {T <: Tuple}
    if isbitstype(T)
        return x
    else
        return map(y->duplicate_for_device(device, y), x)::T
    end
end

function duplicate_for_device(device, x::T)::T where {T}
    isbitstype(T) || throw(MethodError(duplicate_for_device, (device, x)))
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
