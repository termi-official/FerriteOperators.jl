# Adaption of the API presented in Ferrite.jl#1070 to general devices.
duplicate_for_device(device, ::Nothing) = nothing

# Per-worker duplication of a Ferrite assembler: `K` and `f` are shared — atomic scatter resolves
# the concurrent writes, and sharing the scatter targets is the duplicate's purpose — and every
# other field is per-worker scratch, whatever this Ferrite release calls it, so it is duplicated
# generically rather than named. `typeof(asm)` carries the concrete type parameters.
function duplicate_assembler(device, asm)
    args = map(fieldnames(typeof(asm))) do name
        field = getfield(asm, name)
        name === :K || name === :f ? field : duplicate_for_device(device, field)
    end
    return typeof(asm)(args...)
end
for Assembler in (:CSCAssembler, :SymmetricCSCAssembler, :CSRAssembler)
    @eval duplicate_for_device(device, asm::Ferrite.$Assembler) = duplicate_assembler(device, asm)
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
