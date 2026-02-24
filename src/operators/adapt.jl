#FIXME: this needs to be revisted

## GenericIndexedData ##
Adapt.adapt_structure(::AbstractCPUDevice, gid::GenericIndexedData) = gid
function Adapt.adapt_structure(to::AbstractGPUDevice, gid::GenericIndexedData)
    backend = default_backend(to)
    GenericIndexedData(
        Adapt.adapt(backend, gid.data),
        Adapt.adapt(backend, gid.index_structure),
    )
end

## EAVector ##
Adapt.adapt_structure(::AbstractCPUDevice, eavec::EAVector) = eavec
function Adapt.adapt_structure(to::AbstractGPUDevice, eavec::EAVector)
    EAVector(
        Adapt.adapt(to, eavec.data),
        Adapt.adapt(to, eavec.dof_to_element_map),
    )
end

## EAOperator ##
Adapt.adapt_structure(::AbstractCPUDevice, op::EAOperator) = op
function Adapt.adapt_structure(to::AbstractGPUDevice, op::EAOperator)
    EAOperator(
        op.device,
        op.device_cache,
        Adapt.adapt(to, op.element_matrices),
        Adapt.adapt(to, op.vector_element_map),
        Adapt.adapt(to, op.element_vector_map),
    )
end
