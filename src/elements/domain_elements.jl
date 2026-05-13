# The types in this file are merely helpers for the setup logic
struct NonlinearMultiDomainIntegrator <: AbstractNonlinearIntegrator
    subintegrators::Dict{<:SubDofHandler}
end
function setup_element_cache(element_model::NonlinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::NonlinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end


struct LinearMultiDomainIntegrator <: AbstractLinearIntegrator
    subintegrators::Dict{<:SubDofHandler}
end
function setup_element_cache(element_model::LinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::LinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end


struct BilinearMultiDomainIntegrator <: AbstractBilinearIntegrator
    subintegrators::Dict{<:SubDofHandler}
end
function setup_element_cache(element_model::BilinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::BilinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end
