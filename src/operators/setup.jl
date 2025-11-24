
create_system_matrix(strategy, dh) = allocate_matrix(matrix_type(strategy), dh)

function setup_element_caches(integrator, dh)
    return [setup_element_cache(integrator, sdh) for sdh in dh.subdofhandlers]
end

function setup_strategy_caches(strategy, element_caches, dh)
    return [setup_strategy_cache(strategy, element_cache, sdh) for (element_cache, sdh) in zip(element_caches, dh.subdofhandlers)]
end

function setup_operator(strategy::AbstractFullAssemblyStrategy, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler)
    A               = create_system_matrix(strategy, dh)
    element_caches  = setup_element_caches(integrator, dh)
    strategy_caches = setup_strategy_caches(strategy, element_caches, dh)

    return BilinearFerriteOperator(
        A,
        strategy,
        [SubdomainCache(
            sdh,
            element_cache,
            strategy_cache,
        ) for (sdh, element_cache, strategy_cache) in zip(dh.subdofhandlers, element_caches, strategy_caches)],
    )
end

function setup_operator(strategy::AbstractFullAssemblyStrategy, integrator::AbstractNonlinearIntegrator, dh::AbstractDofHandler)
    J               = create_system_matrix(strategy, dh)
    element_caches  = setup_element_caches(integrator, dh)
    strategy_caches = setup_strategy_caches(strategy, element_caches, dh)

    return LinearizedFerriteOperator(
        J,
        strategy,
        [SubdomainCache(
            sdh,
            element_cache,
            strategy_cache,
        ) for (sdh, element_cache, strategy_cache) in zip(dh.subdofhandlers, element_caches, strategy_caches)],
    )
end
