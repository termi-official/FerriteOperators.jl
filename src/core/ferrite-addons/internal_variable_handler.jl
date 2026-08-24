"""
    InternalVariableHandler(internal_variable_offsets, item_variable_offsets, base_offset, ndofs)

Layout of the condensed internal-variable block appended to the solution
vector, `u = [ū; q_cells; q_items]` — cell-owned internal state first, then
the algebraic-item family's, both numbered into the SAME tail. A cell
declares its internal dof count through
`get_number_of_internal_dofs_per_element`, an algebraic item through
[`get_number_of_internal_dofs_per_algebraic_item`](@ref); the handler turns
either into ranges addressing the solution vector — cell ranges through
[`internal_variable_offset`](@ref)/[`internal_variable_range`](@ref) keyed by
cellid, item ranges through the [`internal_variable_range`](@ref) overload
keyed by an [`AlgebraicItem`](@ref). The two are DIFFERENT methods (dispatched
on the second argument's type, `Int` vs `AlgebraicItem`) rather than one
`Int`-keyed method shared by both families: a cellid and an item index are
both small positive integers with no relation to each other, so collapsing
them onto one dispatch would silently return the wrong range whenever a
cellid and an item index coincide.

Internal dofs stay out of the `Ferrite.DofHandler` deliberately: their count
varies per quadrature point, per material, and per item, which a field-based
numbering would have to pad.
"""
@concrete mutable struct InternalVariableHandler <: AbstractDofHandler
    # `ncells+1` offsets relative to the start of the cell block, with
    # `internal_variable_offsets[1] == 0`, such that cell `cid` owns the relative entries
    # `internal_variable_offsets[cid]+1 : internal_variable_offsets[cid+1]`. `nothing` when no
    # cell carries condensed internal state.
    internal_variable_offsets
    # `nitems+1` offsets relative to the start of the item block (which itself starts right
    # after the cell block), same cumsum shape as `internal_variable_offsets`. `nothing` when
    # the algebraic-item family carries no condensed internal state.
    item_variable_offsets
    # Where the cell block starts in the solution vector, i.e. `ndofs(dh)` of the handler it
    # was built for.
    base_offset <: Integer
    ndofs <: Integer
end
Ferrite.ndofs(lvh::InternalVariableHandler) = lvh.ndofs

"""
    internal_variable_offset(ivh, cellid) -> Int

Index in the solution vector immediately before cell `cellid`'s internal
variables — absolute, i.e. it already includes the field dofs preceding the
internal block.
"""
internal_variable_offset(lvh::InternalVariableHandler, cellid::Int) = lvh.base_offset + lvh.internal_variable_offsets[cellid]

"""
    internal_variable_range(ivh, cellid::Int) -> UnitRange{Int}
    internal_variable_range(ivh, item::AlgebraicItem) -> UnitRange{Int}

Range of a cell's (keyed by `cellid`) or an algebraic item's (keyed by
[`AlgebraicItem`](@ref), i.e. `args.item`) internal variables in the solution
vector, absolute like [`internal_variable_offset`](@ref). The item range sits
AFTER the whole cell block, `[ū | q_cells | q_items]`.
"""
internal_variable_range(lvh::InternalVariableHandler, cellid::Int)  = (internal_variable_offset(lvh, cellid)+1):(lvh.base_offset + lvh.internal_variable_offsets[cellid+1])
# An operator without condensed CELLS carries `internal_variable_offsets ===
# nothing`: every cell owns zero internal dofs, so its range is empty — the
# same answer the offset cumsum gives a non-condensed cell of a mixed
# operator. This is what lets an `InternalSource` slot gather harmlessly on
# subdomains that have no internal state.
internal_variable_range(lvh::InternalVariableHandler{Nothing}, cellid::Int) = 1:0

_cell_block_length(offsets::Nothing) = 0
_cell_block_length(offsets) = offsets[end]

function internal_variable_range(lvh::InternalVariableHandler, item::AlgebraicItem)
    base = lvh.base_offset + _cell_block_length(lvh.internal_variable_offsets)
    offs = lvh.item_variable_offsets
    return (base + offs[item.index] + 1):(base + offs[item.index + 1])
end
# Same placeholder rule as the cell overload, keyed on `item_variable_offsets`
# instead: no algebraic item carries condensed internal state, so every item's
# range is empty regardless of whether any CELL is condensed (the `where {C}`
# leaves the cell-offsets type parameter free).
internal_variable_range(lvh::InternalVariableHandler{C, Nothing}, item::AlgebraicItem) where {C} = 1:0

Ferrite.close!(lvh::InternalVariableHandler) = nothing

# Offsets are shared read-only data, so duplication just returns the same instance.
duplicate_for_device(device, ivh::InternalVariableHandler) = ivh
