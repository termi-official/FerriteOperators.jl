#FIXME: this needs to be revisted

## EAOperatorAssembler (@concrete) ##
# Passed as `task.inner_assembler` into the GPU kernel.
Adapt.adapt_structure(to, a::EAOperatorAssembler) =
    EAOperatorAssembler(
        a.device,
        Adapt.adapt(to, a.K_element),
        Adapt.adapt(to, a.f_element),
        Adapt.adapt(to, a.f),
    )

## Task structs (plain parametric — Adapt.@adapt_structure works) ##
Adapt.@adapt_structure AssembleBilinearTerm
Adapt.@adapt_structure AssembleLinearizationJR
Adapt.@adapt_structure AssembleLinearizationJ
Adapt.@adapt_structure AssembleLinearizationR
Adapt.@adapt_structure AssembleLinearTerm

## GenericIndexedData ##
# Generic `to` so it works both with our device types (setup-time) and
# KA's KernelAdaptor (kernel-launch-time).
Adapt.adapt_structure(to, gid::GenericIndexedData) =
    GenericIndexedData(
        Adapt.adapt(to, gid.data),
        Adapt.adapt(to, gid.index_structure),
    )

## EAVector ##
Adapt.adapt_structure(to, eavec::EAVector) =
    EAVector(
        Adapt.adapt(to, eavec.data),
        Adapt.adapt(to, eavec.dof_to_element_map),
    )

## EAOperator ##
# `device` and `device_cache` are CPU-only (not accessed in the kernel).
# Drop them to avoid adapting non-isbitstype fields (e.g. RocDevice with Union fields).
Adapt.adapt_structure(to, op::EAOperator) =
    EAOperator(
        nothing,
        nothing,
        Adapt.adapt(to, op.element_matrices),
        Adapt.adapt(to, op.vector_element_map),
        Adapt.adapt(to, op.element_vector_map),
    )
