
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

function setup_element_strategy_caches(strategy, element_caches, ivh, dh)
    return [setup_element_strategy_cache(strategy, element_cache, ivh, sdh) for (element_cache, sdh) in zip(element_caches, dh.subdofhandlers)]
end

function setup_subdomain_caches(strategy, integrator, dh)
    element_caches  = setup_elements(integrator, dh)
    ivh             = setup_internal_variable_handler(integrator, element_caches, dh)
    strategy_caches = setup_element_strategy_caches(strategy, element_caches, ivh, dh)
    return [SubdomainCache(
            sdh,
            ivh,
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

"""
    setup_transfer_operator(strategy, integrator, dh_row, dh_col)

Set up a [`TransferFerriteOperator`](@ref) for assembling a rectangular sparse matrix of
size `(ndofs(dh_row) × ndofs(dh_col))`.

`dh_row` and `dh_col` must live on the **same** grid and their subdomain lists must
correspond 1-to-1 (same length, same cellsets at each index).

`integrator` must be an [`AbstractTransferIntegrator`](@ref); its
`setup_transfer_element_cache(integrator, sdh_row, sdh_col)` method is called once per
subdomain pair.
"""
function setup_transfer_operator(
        strategy::AbstractAssemblyStrategy,
        integrator::AbstractTransferIntegrator,
        dh_row::DofHandler,
        dh_col::DofHandler,
    )
    @assert get_grid(dh_row) === get_grid(dh_col) "Both DofHandlers must share the same grid"
    @assert length(dh_row.subdofhandlers) == length(dh_col.subdofhandlers) "Mismatch in number of subdomains"

    # Build rectangular sparse matrix
    nrdofs = ndofs(dh_row)
    ncdofs = ndofs(dh_col)
    P = spzeros(value_type(strategy.device), nrdofs, ncdofs)

    # Build per-subdomain caches (one per SubDofHandler pair)
    subdomain_caches = TransferSubdomainCache[]
    for (sdh_row, sdh_col) in zip(dh_row.subdofhandlers, dh_col.subdofhandlers)
        element = setup_transfer_element_cache(integrator, sdh_row, sdh_col)
        push!(subdomain_caches, TransferSubdomainCache(sdh_row, sdh_col, element))
    end

    return TransferFerriteOperator(P, strategy, subdomain_caches)
end
