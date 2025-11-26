
create_system_matrix(strategy, dh) = allocate_matrix(matrix_type(strategy), dh)
create_system_vector(strategy, dh) = allocate_vector(vector_type(strategy), dh)

function setup_elements(integrator, dh)
    return [setup_element_cache(integrator, sdh) for sdh in dh.subdofhandlers]
end

function setup_element_strategy_caches(strategy, element_caches, dh)
    return [setup_element_strategy_cache(strategy, element_cache, sdh) for (element_cache, sdh) in zip(element_caches, dh.subdofhandlers)]
end

function setup_subdomain_caches(strategy, integrator, dh)
    element_caches  = setup_elements(integrator, dh)
    strategy_caches = setup_element_strategy_caches(strategy, element_caches, dh)
    return [SubdomainCache(
            sdh,
            element_cache,
            strategy_cache,
        ) for (sdh, element_cache, strategy_cache) in zip(dh.subdofhandlers, element_caches, strategy_caches)]
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler)
    operator_strategy = setup_operator_strategy_cache(strategy, integrator, dh)
    A                 = create_system_matrix(operator_strategy, dh)
    subdomain_caches  = setup_subdomain_caches(operator_strategy, integrator, dh)

    return BilinearFerriteOperator(
        A,
        operator_strategy,
        subdomain_caches,
    )
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractNonlinearIntegrator, dh::AbstractDofHandler)
    operator_strategy = setup_operator_strategy_cache(strategy, integrator, dh)
    J                 = create_system_matrix(operator_strategy, dh)
    subdomain_caches  = setup_subdomain_caches(operator_strategy, integrator, dh)

    return LinearizedFerriteOperator(
        J,
        operator_strategy,
        subdomain_caches,
    )
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractLinearIntegrator, dh::AbstractDofHandler)
    operator_strategy = setup_operator_strategy_cache(strategy, integrator, dh)
    b                 = create_system_vector(operator_strategy, dh)
    subdomain_caches  = setup_subdomain_caches(operator_strategy, integrator, dh)

    return LinearFerriteOperator(
        b,
        operator_strategy,
        subdomain_caches
    )
end
