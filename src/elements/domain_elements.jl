"""
    NonlinearMultiDomainIntegrator(subintegrators::Dict{<:SubDofHandler})
    LinearMultiDomainIntegrator(subintegrators::Dict{<:SubDofHandler})
    BilinearMultiDomainIntegrator(subintegrators::Dict{<:SubDofHandler})

Integrator carrying one sub-integrator per `SubDofHandler`, so a single
operator hosts different physics per subdomain. Element and boundary cache
setup forward to the sub-integrator of the subdomain being set up; everything
downstream sees an ordinary per-subdomain cache.
"""
struct NonlinearMultiDomainIntegrator{DictType <: Dict{<:SubDofHandler}} <: AbstractNonlinearIntegrator
    subintegrators::DictType
end
function setup_element_cache(element_model::NonlinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::NonlinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end


@doc (@doc NonlinearMultiDomainIntegrator)
struct LinearMultiDomainIntegrator{DictType <: Dict{<:SubDofHandler}} <: AbstractLinearIntegrator
    subintegrators::DictType
end
function setup_element_cache(element_model::LinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::LinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end


@doc (@doc NonlinearMultiDomainIntegrator)
struct BilinearMultiDomainIntegrator{DictType <: Dict{<:SubDofHandler}} <: AbstractBilinearIntegrator
    subintegrators::DictType
end
function setup_element_cache(element_model::BilinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_element_cache(element_model.subintegrators[sdh], sdh)
end
function setup_boundary_cache(element_model::BilinearMultiDomainIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(element_model.subintegrators[sdh], sdh)
end
