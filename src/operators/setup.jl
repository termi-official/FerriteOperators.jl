
create_system_matrix(strategy, dh) = allocate_matrix(matrix_type(strategy), dh)
create_system_vector(strategy, dh) = allocate_vector(vector_type(strategy), dh)

function setup_elements(integrator, dh)
    return [setup_element_cache(integrator, sdh) for sdh in dh.subdofhandlers]
end

function setup_internal_variable_handler(integrator::AbstractCondensedNonlinearIntegrator, element_caches, dh)
    num_dofs_per_element = zeros(Int, getncells(get_grid(dh))+1)
    for (sdh, cache) in zip(dh.subdofhandlers, element_caches)
        for (cellid, nidofs) in zip(sdh.cellset, get_number_of_internal_dofs_per_element(integrator, cache, sdh))
            num_dofs_per_element[1+cellid] = nidofs
        end
    end
    @assert all(num_dofs_per_element .≥ 0) "Number of internal dofs must be non-negative!"
    offsets = cumsum(num_dofs_per_element)
    return InternalVariableHandler(offsets[1:end-1] .+ ndofs(dh), offsets[end])
end

function setup_internal_variable_handler(integrator, element_caches, dh)
    return InternalVariableHandler(nothing, 0)
end

function setup_element_strategy_caches(strategy, integrator, element_caches, ivh, dh)
    return [setup_element_strategy_cache(strategy, integrator, element_cache, ivh, sdh) for (element_cache, sdh) in zip(element_caches, dh.subdofhandlers)]
end

function setup_element_strategy_caches(strategy::ElementAssemblyOperatorStrategy{<:AbstractGPUDevice}, integrator, element_caches, ivh, dh)
    backend = default_backend(strategy.device)
    dh_gpu  = Adapt.adapt(backend, dh) # HostDofHandler
    return [
        setup_element_strategy_cache(strategy, integrator, element_cache, ivh, device_sdh)
        for (element_cache, device_sdh) in zip(element_caches, dh_gpu.subdofhandlers)
    ]
end

function setup_subdomain_caches(strategy, integrator, dh)
    element_caches  = setup_elements(integrator, dh)
    ivh             = setup_internal_variable_handler(integrator, element_caches, dh)
    strategy_caches = setup_element_strategy_caches(strategy, integrator, element_caches, ivh, dh)
    return [SubdomainCache(
            sdh,
            ivh,
            element_cache,
            strategy_cache,
        ) for (sdh, element_cache, strategy_cache) in zip(dh.subdofhandlers, element_caches, strategy_caches)]
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler)
    # Check device availability
    (;device) = strategy
    KA.functional(device) || error("Device $(device) is not functional. Please check your device setup.")
    
    # this has the resolved device launch config.
    resolved_staregy = resolve_device_config(strategy, dh)

    operator_strategy = setup_operator_strategy_cache(resolved_staregy, integrator, dh)
    A                 = create_system_matrix(operator_strategy, dh)
    subdomain_caches  = setup_subdomain_caches(operator_strategy, integrator, dh)

    return BilinearFerriteOperator(
        A,
        operator_strategy, 
        subdomain_caches, 
    )
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractNonlinearIntegrator, dh::AbstractDofHandler)
    # Check device availability
    (;device) = strategy
    KA.functional(device) || error("Device $(device) is not functional. Please check your device setup.")
    
    # this has the resolved device launch config.
    resolved_staregy = resolve_device_config(strategy, dh)

    operator_strategy = setup_operator_strategy_cache(resolved_staregy, integrator, dh)
    J                 = create_system_matrix(operator_strategy, dh)
    subdomain_caches  = setup_subdomain_caches(operator_strategy, integrator, dh)

    return LinearizedFerriteOperator(
        J,
        operator_strategy,
        subdomain_caches,
    )
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractLinearIntegrator, dh::AbstractDofHandler)
    # Check device availability
    (;device) = strategy
    KA.functional(device) || error("Device $(device) is not functional. Please check your device setup.")

    # this has the resolved device launch config.
    resolved_staregy = resolve_device_config(strategy, dh)
    
    operator_strategy = setup_operator_strategy_cache(resolved_staregy, integrator, dh)
    b                 = create_system_vector(operator_strategy, dh)
    subdomain_caches  = setup_subdomain_caches(operator_strategy, integrator, dh)

    return LinearFerriteOperator(
        b,
        operator_strategy,
        subdomain_caches
    )
end
