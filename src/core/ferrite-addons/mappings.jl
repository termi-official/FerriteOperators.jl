
struct ElementDofPair{IndexType}
    element_index::IndexType
    local_dof_index::IndexType
end

create_dof_to_element_map(dh::DofHandler) = create_dof_to_element_map(Int, dh::DofHandler)

function create_dof_to_element_map(::Type{IndexType}, dh::DofHandler) where IndexType
    # Preallocate storage
    dof_to_element_vs = [Set{ElementDofPair{IndexType}}() for _ in 1:ndofs(dh)]
    # Fill set
    for sdh in dh.subdofhandlers
        for cc in CellIterator(sdh)
            eid = Ferrite.cellid(cc)
            for (ldi,dof) in enumerate(celldofs(cc))
                s = dof_to_element_vs[dof]
                push!(s, ElementDofPair(eid, ldi))
            end
        end
    end
    #
    dof_to_element_vv = ElementDofPair{IndexType}[]
    offset = 1
    offsets = IndexType[]
    for dof in 1:ndofs(dh)
        append!(offsets, offset)
        s = dof_to_element_vs[dof]
        offset += length(s)
        append!(dof_to_element_vv, s)
    end
    append!(offsets, offset)
    #
    return GenericIndexedData(
        dof_to_element_vv,
        offsets,
    )
end
