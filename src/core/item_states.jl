####################################
## Item-lifetime state
####################################

"""
    ItemStates{S}(nitems)

Per-item state slots of element type `S`, indexed by any integer item index,
not only patch ids. Two uses, one mechanism, differing only in the caller's
invalidation policy: *item-lifetime state* that must survive across sweeps (a
retained factorization, a reduction snapshot), and the *solve→scatter payload
channel* within one sweep.

Freshness is the caller's: FO never writes and never invalidates a slot. Store
with [`set_item_state!`](@ref), test with [`has_item_state`](@ref), and drop
stale content with [`invalidate_item_state!`](@ref) when whatever the slot was
derived from changes. Slots are indexed by ITEM, so items processed by
different workers touch disjoint slots — item lifetime is not worker lifetime,
and a slot must never be handed to a worker-lifetime cache.
"""
struct ItemStates{S}
    slots::Vector{S}
    valid::Vector{Bool}
end
ItemStates{S}(nitems::Int) where {S} = ItemStates(Vector{S}(undef, nitems), fill(false, nitems))

Base.length(st::ItemStates) = length(st.valid)

"Is item `i`'s state slot filled and not invalidated?"
has_item_state(st::ItemStates, i::Int) = st.valid[i]

"""
    item_state(st, i)

Item `i`'s state. Throws when the slot is empty or invalidated — guard with
[`has_item_state`](@ref).
"""
function item_state(st::ItemStates, i::Int)
    st.valid[i] || throw(ArgumentError("item $i has no valid state; check `has_item_state` first"))
    return st.slots[i]
end

"Store `s` as item `i`'s state and mark it valid."
set_item_state!(st::ItemStates, i::Int, s) = (st.slots[i] = s; st.valid[i] = true; st)

"Drop item `i`'s state (the caller's invalidation trigger fired)."
invalidate_item_state!(st::ItemStates, i::Int) = (st.valid[i] = false; st)

"Drop every item's state."
invalidate_item_states!(st::ItemStates) = (fill!(st.valid, false); st)

# Entries are indexed by ITEM and the partition assigns each item to exactly
# one worker at a time, so per-worker copies would duplicate memory without
# adding safety — every worker shares the same backing arrays.
duplicate_for_device(device, st::ItemStates) = st
