# The types in this file are merely helpers for the setup logic
struct NonlinearMultiDomainIntegrator{DictType <: Dict{<:SubDofHandler}} <: AbstractNonlinearIntegrator
    subintegrators::DictType
end
function setup_element_cache(element_model::NonlinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::NonlinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end


struct LinearMultiDomainIntegrator{DictType <: Dict{<:SubDofHandler}} <: AbstractLinearIntegrator
    subintegrators::DictType
end
function setup_element_cache(element_model::LinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::LinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end


struct BilinearMultiDomainIntegrator{DictType <: Dict{<:SubDofHandler}} <: AbstractBilinearIntegrator
    subintegrators::DictType
end
function setup_element_cache(element_model::BilinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::BilinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end
