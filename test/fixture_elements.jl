# Element doubles and testbeds shared by several test files. Definitions only,
# no `@testset`: `runtests.jl` drops this file from the discovered suite, and
# each consuming file pulls it in with `include`.

using FerriteOperators
using FerriteOperatorsExampleElements

# The per-worker workspace and the element cache the engine holds for the first
# subdomain. Assertions reaching for these read engine internals; the two
# spellings live here so the tests that need them read the same way.
first_workspace(op) = first(first(op.engine.subdomain_caches).device_cache)
first_element_cache(op) = first(op.engine.subdomain_caches).domain.element

# Compressible Neo-Hookean strain energy, the material the bundled
# hyperelasticity integrator is exercised with.
struct NeoHookean
    E::Float64
    ν::Float64
end
function (mat::NeoHookean)(F)
    (; E, ν) = mat
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    C = tdot(F)
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

####################################
## Transient-diffusion element double
####################################
# Cell values over one scalar field plus whatever scalars a kernel reads from
# `params`. `tag` keeps the kernels different test files write — the transient
# diffusion residual r(u, u̇) = ∫ (u̇ v + ∇u⋅∇v) dΩ and the analytic Jacobians
# built on it — on separate cache types.
struct CVIntegrator{tag, P} <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
    params::P
end
CVIntegrator{tag}(qrc, field_name, params = nothing) where {tag} =
    CVIntegrator{tag, typeof(params)}(qrc, field_name, params)

struct CVCache{tag, P, CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
    params::P
end
CVCache{tag}(cv, params) where {tag} = CVCache{tag, typeof(params), typeof(cv)}(cv, params)

function FerriteOperators.setup_element_cache(m::CVIntegrator{tag}, sdh::SubDofHandler) where {tag}
    qr     = getquadraturerule(m.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return CVCache{tag}(CellValues(qr, ip, ip_geo), m.params)
end
FerriteOperators.duplicate_for_device(device, c::CVCache{tag}) where {tag} =
    CVCache{tag}(FerriteOperators.duplicate_for_device(device, c.cv), c.params)
FerriteOperators.reinit_values!(c::CVCache, cell) = reinit!(c.cv, cell)

# Transient diffusion, r(u, u̇) = ∫ (u̇ v + ∇u⋅∇v) dΩ. `cache` is any struct
# exposing a scalar `cv`; test files register their own `assemble_cell!`
# dispatch on their own cache type and delegate here.
function transient_diffusion_residual!(re, cache, args)
    (; cv) = cache
    uₑ, duₑ = args.states.u, args.states.du
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        u̇  = function_value(cv, qp, duₑ)
        ∇u = function_gradient(cv, qp, uₑ)
        for i in 1:getnbasefunctions(cv)
            re[i] += (u̇ * shape_value(cv, qp, i) + ∇u ⋅ shape_gradient(cv, qp, i)) * dΩ
        end
    end
end

# The hand-fused SDIRK/BE scheme matrix W = w_du·M + w_u·K for the transient
# diffusion element above, over the same `cv`. `scale` detunes the kernel
# (exact for scale = 1) so the same body serves both a correct analytic
# provider and a derivative-checker witness of a wrong one.
function analytic_weighted_jacobian!(K, cv, weights, scale = 1)
    wu, wdu = weights.u, weights.du
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        for i in 1:getnbasefunctions(cv), j in 1:getnbasefunctions(cv)
            K[i, j] += scale * (wu * (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) +
                                 wdu * shape_value(cv, qp, i) * shape_value(cv, qp, j)) * dΩ
        end
    end
end

####################################
## Constant Neumann surface element double
####################################
# r(v) = w ∫_Γ v dΓ over `facetset`, so a linear operator's load vector sums to
# w·|Γ|. `param_scaled` switches w from the cache's own `scale` to the parameter
# the driver queried for this facet (`scale · p`) — the channel that makes
# per-inner parameter views observable through a composite. The integrator
# family differs by supertype, so the load carries two spellings and one cache.
struct NeumannProbeCache{param_scaled, FV <: FacetValues} <: FerriteOperators.AbstractSurfaceElementCache
    scale::Float64
    fv::FV
    facetset::Set{FacetIndex}
end
NeumannProbeCache{ps}(scale, fv, facetset) where {ps} =
    NeumannProbeCache{ps, typeof(fv)}(scale, fv, facetset)

struct LinearNeumannProbe{param_scaled, V} <: AbstractLinearIntegrator
    scale::Float64
    field_name::Symbol
    facetset::Set{FacetIndex}
    volumetric::V
end
LinearNeumannProbe(scale, field_name, facetset; param_scaled = false, volumetric = nothing) =
    LinearNeumannProbe{param_scaled, typeof(volumetric)}(scale, field_name, facetset, volumetric)

struct NonlinearNeumannProbe{param_scaled, V} <: AbstractNonlinearIntegrator
    scale::Float64
    field_name::Symbol
    facetset::Set{FacetIndex}
    volumetric::V
end
NonlinearNeumannProbe(scale, field_name, facetset; param_scaled = false, volumetric = nothing) =
    NonlinearNeumannProbe{param_scaled, typeof(volumetric)}(scale, field_name, facetset, volumetric)

const NeumannProbe = Union{LinearNeumannProbe, NonlinearNeumannProbe}

# Without a volumetric term the probe is boundary-only.
FerriteOperators.setup_element_cache(m::NeumannProbe, sdh::SubDofHandler) =
    _neumann_volumetric_cache(m.volumetric, sdh)
_neumann_volumetric_cache(::Nothing, ::SubDofHandler) = FerriteOperators.EmptyVolumetricElementCache()
_neumann_volumetric_cache(integrator, sdh::SubDofHandler) =
    FerriteOperators.setup_element_cache(integrator, sdh)

FerriteOperators.setup_boundary_cache(m::LinearNeumannProbe{ps}, sdh::SubDofHandler) where {ps} =
    _neumann_boundary_cache(m, sdh, ps)
FerriteOperators.setup_boundary_cache(m::NonlinearNeumannProbe{ps}, sdh::SubDofHandler) where {ps} =
    _neumann_boundary_cache(m, sdh, ps)
function _neumann_boundary_cache(m, sdh::SubDofHandler, param_scaled::Bool)
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    fqr    = FacetQuadratureRule{Ferrite.getrefshape(ip)}(2)
    return NeumannProbeCache{param_scaled}(m.scale, FacetValues(fqr, ip, ip_geo), m.facetset)
end

FerriteOperators.duplicate_for_device(device, c::NeumannProbeCache{ps}) where {ps} =
    NeumannProbeCache{ps}(c.scale, FerriteOperators.duplicate_for_device(device, c.fv), c.facetset)
FerriteOperators.is_facet_in_cache(idx::FacetIndex, cell, c::NeumannProbeCache) = idx ∈ c.facetset
FerriteOperators.query_facet_parameters(c::NeumannProbeCache{true}, cell, lfi, p) = c.scale * p

_neumann_load(c::NeumannProbeCache{false}, args) = c.scale
_neumann_load(::NeumannProbeCache{true}, args) = args.p
function FerriteOperators.assemble_facet!(req::ResidualRequest, c::NeumannProbeCache, args, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    w = _neumann_load(c, args)
    for qp in 1:getnquadpoints(c.fv)
        dΓ = getdetJdV(c.fv, qp)
        for i in 1:getnbasefunctions(c.fv)
            req.r[i] += w * shape_value(c.fv, qp, i) * dΓ
        end
    end
end

####################################
## Condensed-viscoelasticity testbed
####################################
# Vector displacement on a hex block plus a hidden per-QP εᵛ, slots (:u, :uprev).
# `transform` reshapes the reference grid before the dofs are distributed.
function visco_testbed(strategy, qrc, dims = (1, 1, 1); transform = nothing, kwargs...)
    grid = generate_grid(Hexahedron, dims)
    transform === nothing || Ferrite.transform_coordinates!(grid, transform)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(dh)
    integrator = SimpleCondensedLinearViscoelasticity(MaxwellParameters(), qrc, :u, :εᵛ)
    op = setup_operator(strategy, integrator, dh; slots = (:u, :uprev), kwargs...)
    return (; op, dh, grid)
end

####################################
## Condensed power-law relaxation testbed
####################################
# Scalar field on a quad grid plus a hidden per-QP internal state whose local
# stage problem is nonlinear, slots (:u, :uprev). `material` and
# `local_solver` are the element's configuration, both arriving through the
# integrator.
function relaxation_testbed(strategy, qrc, dims = (2, 2);
                            material = NortonRelaxationParameters(),
                            local_solver = LocalNewtonSettings(), kwargs...)
    grid = generate_grid(Quadrilateral, dims)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    integrator = SimpleCondensedPowerLawRelaxation(material, qrc, :u, :q; local_solver)
    op = setup_operator(strategy, integrator, dh; slots = (:u, :uprev), kwargs...)
    return (; op, dh, grid)
end
