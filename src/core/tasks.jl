@concrete struct SubdomainAssemblyTaskBuffer
    u
    p
    subdomain
end

@concrete struct GenericTaskBuffer
    # Static parts
    u # <: AbstractVector
    p # global parameters
    element # <: Abstract*ElementCache
    assembly_cache # BilinearAssemblyCache | NonlinearAssemblyCache | LinearAssemblyCache
    # pe
    geometry_cache # <: CellCache
    ivh # <: InternalVariableHandler
end
Ferrite.reinit!(buffer::GenericTaskBuffer, taskid) = reinit!(buffer.geometry_cache, taskid)

## Local cache accessors ##
get_Ke(lc::BilinearAssemblyCache)  = lc.Ke
get_Ke(lc::NonlinearAssemblyCache) = lc.Ke
get_Ke(::LinearAssemblyCache)      = error("LinearAssemblyCache does not have an element matrix (Ke).")

get_ue(::BilinearAssemblyCache)    = error("BilinearAssemblyCache does not have an element unknown vector (ue).")
get_ue(lc::NonlinearAssemblyCache) = lc.ue
get_ue(::LinearAssemblyCache)      = error("LinearAssemblyCache does not have an element unknown vector (ue).")

get_re(::BilinearAssemblyCache)    = error("BilinearAssemblyCache does not have an element residual vector (re).")
get_re(lc::NonlinearAssemblyCache) = lc.re
get_re(lc::LinearAssemblyCache)    = lc.re

function get_task_buffer_for_device(task, u, p, device_cache::SimpleAssemblyCache, chunkid)
    (; assembly_cache, cell, ivh, element) = device_cache
    GenericTaskBuffer(u, p, element, assembly_cache, cell, ivh)
end

function get_task_buffer_for_device(task, u, p, device_cache::ThreadedAssemblyCache, chunkid)
    cache = device_cache.task_local_caches[chunkid]
    (; assembly_cache, cell, ivh, element) = cache
    GenericTaskBuffer(u, p, element, assembly_cache, cell, ivh)
end

# GPU: index into each container by thread id + cell id, then wrap into GenericTaskBuffer.
# Uses the ImmutableCellCache functor cc[tid](taskid) to get a cache with the correct cellid
# and filled coords — equivalent to reinit! but works on immutable structs.
function get_task_buffer_for_device(task, u, p, device_cache::SimpleAssemblyCache, tid, taskid)
    assembly_cache = device_cache.assembly_cache[tid]
    cell           = device_cache.cell[tid](Int(taskid)) # functor accepts only cellid::Int
    element        = device_cache.element[tid]
    GenericTaskBuffer(u, p, element, assembly_cache, cell, device_cache.ivh)
end

function load_element_unknowns!(uₑ, buffer::GenericTaskBuffer)
    load_element_unknowns!(uₑ, buffer.u, buffer.geometry_cache, buffer.ivh, buffer.element)
end
function store_condensed_element_unknowns!(uₑ, buffer::GenericTaskBuffer)
    store_condensed_element_unknowns!(uₑ, buffer.u, buffer.geometry_cache, buffer.ivh, buffer.element)
end

query_element_matrix(b::GenericTaskBuffer) = get_Ke(b.assembly_cache)
query_element_residual_buffer(b::GenericTaskBuffer) = get_re(b.assembly_cache)
query_element_unknown_buffer(b::GenericTaskBuffer) = query_element_unknown_buffer(b.element, get_ue(b.assembly_cache))
query_element_unknown_buffer(element, ue) = ue
query_element_parameters(b::GenericTaskBuffer) = query_element_parameters(b.element, b.geometry_cache, b.ivh, b.p)
query_element_parameters(element, geometry_cache, ivh, p) = p
query_geometry_cache(b::GenericTaskBuffer) = b.geometry_cache
query_element(b::GenericTaskBuffer) = b.element

function get_items(task, cache)
    (; subdomain)      = cache
    (; strategy_cache) = subdomain
    return _get_items(task, subdomain, strategy_cache)
end

function _get_items(task, subdomain, strategy_cache::SequentialAssemblyStrategyCache)
    # This is a choice users can make if they know that their assembly is either atomic or not conflicting in first place.
    return (subdomain.sdh.cellset, )
end
function _get_items(task, subdomain, strategy_cache::ElementAssemblyStrategyCache)
    # Remember that we collapse the E-vector, so everything in the assembly is trivially parallel
    return (subdomain.sdh.cellset, )
end
function _get_items(task, subdomain, strategy_cache::PerColorAssemblyStrategyCache)
    return strategy_cache.colors
end

# TODO this is too convoluted right now. Needs probably a few rounds of refactoring.
function get_task_buffer(task, cache, chunkid)
    (; u, p, subdomain) = cache
    (; strategy_cache) = subdomain
    (; device_cache) = strategy_cache

    return get_task_buffer_for_device(task, u, p, device_cache, chunkid)
end
