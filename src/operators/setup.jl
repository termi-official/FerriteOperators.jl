
create_system_matrix(system_matrix_type, dh) = allocate_matrix(system_matrix_type, dh)

function setup_assembled_operator(strategy::AbstractFullAssemblyStrategy, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler, system_matrix_type::Type = matrix_type(strategy))
    A  = create_system_matrix(system_matrix_type, dh)

    element_caches = [setup_element_cache(integrator, sdh) for sdh in dh.subdofhandlers]

    strategy_caches = [setup_strategy_cache(strategy, element_cache, sdh) for (element_cache, sdh) in zip(element_caches, dh.subdofhandlers)]

    return AssembledBilinearFerriteOperator(
        A,
        [SubdomainCache(
            sdh,
            element_cache,
            strategy_cache,
        ) for (sdh, element_cache, strategy_cache) in zip(dh.subdofhandlers, element_caches, strategy_caches)],
    )
end

function setup_assembled_operator(strategy::AbstractFullAssemblyStrategy, integrator::AbstractNonlinearIntegrator, dh::AbstractDofHandler, system_matrix_type::Type = matrix_type(strategy))
    J  = create_system_matrix(system_matrix_type, dh)

    element_caches = [setup_element_cache(integrator, sdh) for sdh in dh.subdofhandlers]

    strategy_caches = [setup_strategy_cache(strategy, element_cache, sdh) for (element_cache, sdh) in zip(element_caches, dh.subdofhandlers)]

    return AssembledLinearizedFerriteOperator(
        J,
        [SubdomainCache(
            sdh,
            element_cache,
            strategy_cache,
        ) for (sdh, element_cache, strategy_cache) in zip(dh.subdofhandlers, element_caches, strategy_caches)],
    )
end
