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
    return FerriteOperators.duplicate_assembler(device, asm)
end

end
