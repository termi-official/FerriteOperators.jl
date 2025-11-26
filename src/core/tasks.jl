@concrete struct SubdomainAssemblyTaskBuffer
    u
    p
    subdomain
end

@concrete struct GenericTaskBuffer
    # Static parts
    u
    p
    element
    # Moving parts
    Ke
    ue
    re
    geometry_cache
end
Ferrite.reinit!(buffer::GenericTaskBuffer, taskid) = reinit!(buffer.geometry_cache, taskid)

function get_task_buffer_for_device(task, u, p, device_cache::SimpleAssemblyCache, chunkid)
    (; element, Ke, ue, re, cell) = device_cache
    GenericTaskBuffer(u, p, element, Ke, ue, re, cell)
end

function get_task_buffer_for_device(task, u, p, device_cache::ThreadedAssemblyCache, chunkid)
    (; task_local_caches)  = device_cache
    (; cell, element, Ke, re, ue) = task_local_caches[chunkid]
    GenericTaskBuffer(u, p, element, Ke, ue, re, cell)
end

function load_element_unknowns!(uₑ, buffer::GenericTaskBuffer)
    load_element_unknowns!(uₑ, buffer.u, buffer.geometry_cache, buffer.element)
end
function store_condensed_element_unknowns!(uₑ, buffer::GenericTaskBuffer)
    store_condensed_element_unknowns!(uₑ, buffer.u, buffer.geometry_cache, buffer.element)
end

query_element_matrix(b::GenericTaskBuffer) = b.Ke
query_element_residual_buffer(b::GenericTaskBuffer) = b.re
query_element_unknown_buffer(b::GenericTaskBuffer) = b.ue
query_element_parameters(b::GenericTaskBuffer) = b.p
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
