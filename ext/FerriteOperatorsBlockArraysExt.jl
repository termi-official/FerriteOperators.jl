module FerriteOperatorsBlockArraysExt

using FerriteOperators: FerriteOperators
using Ferrite: Ferrite

# Ferrite's `BlockAssembler` lives in its own BlockArrays extension, which cannot be referenced at
# precompile time from a sibling extension (loading order between extensions of different parents
# is not guaranteed). The method therefore dispatches on the assembler supertype and verifies the
# concrete type at runtime; every other Ferrite assembler has its own specific duplicate, so only
# the block assembler legitimately reaches this method.
function FerriteOperators.duplicate_for_device(device, asm::Ferrite.AbstractAssembler)
    FerriteBlockArrays = Base.get_extension(Ferrite, :FerriteBlockArrays)
    if FerriteBlockArrays === nothing || !(asm isa FerriteBlockArrays.BlockAssembler)
        throw(MethodError(FerriteOperators.duplicate_for_device, (device, asm)))
    end
    # `K` and `f` are shared — atomic scatter resolves the concurrent writes, and sharing the
    # scatter targets is the whole purpose of the duplicate, matching the CSC/CSR assemblers.
    # Every remaining field is per-worker scratch, whatever this Ferrite release calls it, so it is
    # copied generically rather than named here; `typeof(asm)` carries the concrete type parameters.
    args = map(fieldnames(typeof(asm))) do name
        field = getfield(asm, name)
        name === :K || name === :f ? field : FerriteOperators.duplicate_for_device(device, field)
    end
    return typeof(asm)(args...)
end

end
