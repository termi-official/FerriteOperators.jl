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
## Facet-item route switch for the Neumann probe
####################################
# The very same `NeumannProbeCache` a `NeumannProbe` hands the fused cell
# sweep, declared as facet items instead: `setup_boundary_cache` stays at its
# empty default and the facet set travels through `facet_items`. No cache and
# no kernel changes, which is what makes the two routes comparable.
#
# `declared` defaults to the cache's own facet set, so the two routes see the
# same facets. Passing a different set separates the ROUTE's declaration from
# the cache's `is_facet_in_cache` gate, which this route does not consult.

struct LinearFacetItemProbe{I} <: AbstractLinearIntegrator
    inner::I
    declared::Set{FacetIndex}
end
struct NonlinearFacetItemProbe{I} <: AbstractNonlinearIntegrator
    inner::I
    declared::Set{FacetIndex}
end
LinearFacetItemProbe(inner) = LinearFacetItemProbe(inner, inner.facetset)
NonlinearFacetItemProbe(inner) = NonlinearFacetItemProbe(inner, inner.facetset)
const FacetItemProbe = Union{LinearFacetItemProbe, NonlinearFacetItemProbe}

FerriteOperators.setup_element_cache(m::FacetItemProbe, sdh::SubDofHandler) =
    FerriteOperators.setup_element_cache(m.inner, sdh)
FerriteOperators.facet_items(m::FacetItemProbe, sdh::SubDofHandler) = m.declared
FerriteOperators.setup_facet_item_cache(m::FacetItemProbe, sdh::SubDofHandler) =
    FerriteOperators.setup_boundary_cache(m.inner, sdh)

####################################
## Parameter-scaled facet traction — the facet-item sensitivity double
####################################
# r(v) = θ ∫_Γ v dΓ over the declared facets, plus the ANALYTIC ∂r/∂θ a facet
# sensitivity sweep needs — facet kernels have no AD fallback, so the cache
# serves the request itself or the term cannot be differentiated at all.
# `with_parameter_jacobian = false` drops that kernel; declaring
# `ParameterJacobianKind` over it is what setup must reject.

struct TractionFacetCache{with_parameter_jacobian, FV <: FacetValues} <: FerriteOperators.AbstractSurfaceElementCache
    fv::FV
end
TractionFacetCache{wpj}(fv) where {wpj} = TractionFacetCache{wpj, typeof(fv)}(fv)

struct TractionFacetProbe{with_parameter_jacobian} <: AbstractNonlinearIntegrator
    field_name::Symbol
    facetset::Set{FacetIndex}
end
TractionFacetProbe(field_name, facetset; with_parameter_jacobian = true) =
    TractionFacetProbe{with_parameter_jacobian}(field_name, facetset)

FerriteOperators.setup_element_cache(::TractionFacetProbe, ::SubDofHandler) =
    FerriteOperators.EmptyVolumetricElementCache()
FerriteOperators.facet_items(m::TractionFacetProbe, ::SubDofHandler) = m.facetset
function FerriteOperators.setup_facet_item_cache(m::TractionFacetProbe{wpj}, sdh::SubDofHandler) where {wpj}
    ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    fqr    = FacetQuadratureRule{Ferrite.getrefshape(ip)}(2)
    return TractionFacetCache{wpj}(FacetValues(fqr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::TractionFacetCache{wpj}) where {wpj} =
    TractionFacetCache{wpj}(FerriteOperators.duplicate_for_device(device, c.fv))

function FerriteOperators.assemble_facet!(req::ResidualRequest, c::TractionFacetCache, args::FacetArgs, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    for qp in 1:getnquadpoints(c.fv)
        dΓ = getdetJdV(c.fv, qp)
        for i in 1:getnbasefunctions(c.fv)
            req.r[i] += args.p * shape_value(c.fv, qp, i) * dΓ
        end
    end
end

FerriteOperators.provides_analytic(::Type{<:TractionFacetCache{true}}, ::FerriteOperators.ParameterJacobianKind) = true
function FerriteOperators.assemble_facet!(req::ParameterJacobianRequest, c::TractionFacetCache{true}, args::FacetArgs, lfi::Int)
    reinit!(c.fv, args.cell, lfi)
    for qp in 1:getnquadpoints(c.fv)
        dΓ = getdetJdV(c.fv, qp)
        for i in 1:getnbasefunctions(c.fv)
            req.B[i, 1] += shape_value(c.fv, qp, i) * dΓ
        end
    end
    return req.B
end

####################################
## Condensed-viscoelasticity testbed
####################################
# Vector displacement on a hex block plus a hidden per-QP εᵛ, slots (:u, :q,
# :qprev) — :q the trial internal state (InternalSource over the trial `u`),
# :qprev its committed predecessor (InternalSource over a separate `uprev`
# vector). `transform` reshapes the reference grid before the dofs are
# distributed.
function visco_testbed(strategy, qrc, dims = (1, 1, 1); transform = nothing,
                       corrector = Stored(), kwargs...)
    grid = generate_grid(Hexahedron, dims)
    transform === nothing || Ferrite.transform_coordinates!(grid, transform)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(dh)
    integrator = SimpleCondensedLinearViscoelasticity(MaxwellParameters(), qrc, :u, :εᵛ; corrector)
    op = setup_operator(strategy, integrator, dh; slots = (:u, :q, :qprev), kwargs...)
    return (; op, dh, grid)
end

"States NamedTuple for a condensed testbed's trial `u`/committed `uprev` pair."
condensed_states(u, uprev) = (u = u, q = InternalSource(u), qprev = InternalSource(uprev))

####################################
## Condensed power-law relaxation testbed
####################################
# Scalar field on a quad grid plus a hidden per-QP internal state whose local
# stage problem is nonlinear, slots (:u, :q, :qprev) — see `visco_testbed`.
# `material`, `local_solver` and `corrector` are the element's configuration,
# all arriving through the integrator.
function relaxation_testbed(strategy, qrc, dims = (2, 2);
                            material = NortonRelaxationParameters(),
                            local_solver = LocalNewtonSettings(),
                            corrector = Stored(), kwargs...)
    grid = generate_grid(Quadrilateral, dims)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    integrator = SimpleCondensedPowerLawRelaxation(material, qrc, :u, :q; local_solver, corrector)
    op = setup_operator(strategy, integrator, dh; slots = (:u, :q, :qprev), kwargs...)
    return (; op, dh, grid)
end

####################################
## Stress-driven homogenization testbed
####################################
# Linear elasticity on an RVE whose macroscopic strain ε̄ is an unknown living
# in no cell (Ferrite's `AlgebraicVariable`), so the element's local system is
# `[celldofs(cell); the algebraic dofs of ε̄]`. The problem is the reference for
# elements with global dofs; it is defined only where Ferrite carries algebraic
# variables, and the consuming test files skip themselves otherwise.
#
# `analytic` picks the element that provides the Jacobian kernel or the one
# that provides only the residual, so the same problem exercises both routes.
if isdefined(Ferrite, :AlgebraicVariable)

    struct StressDrivenIntegrator{analytic, V} <: AbstractNonlinearIntegrator
        variable::V
        order::Int
        field_name::Symbol
        variable_name::Symbol
        E::SymmetricTensor{4, 2, Float64, 9}
        σ̄::SymmetricTensor{2, 2, Float64, 3}
    end
    StressDrivenIntegrator(variable, E, σ̄; analytic = true, order = 2,
                           field_name = :u, variable_name = :εbar) =
        StressDrivenIntegrator{analytic, typeof(variable)}(variable, order, field_name, variable_name, E, σ̄)

    struct StressDrivenCache{analytic, CV, AV} <: AbstractVolumetricElementCache
        cv::CV
        av::AV
        E::SymmetricTensor{4, 2, Float64, 9}
        σ̄::SymmetricTensor{2, 2, Float64, 3}
        range_u::UnitRange{Int}
        range_ε::UnitRange{Int}
    end
    StressDrivenCache{a}(cv, av, E, σ̄, ru, rε) where {a} =
        StressDrivenCache{a, typeof(cv), typeof(av)}(cv, av, E, σ̄, ru, rε)

    FerriteOperators.global_dofs(m::StressDrivenIntegrator, sdh::SubDofHandler) =
        algebraic_dofs(sdh.dh, m.variable_name)

    function FerriteOperators.setup_element_cache(m::StressDrivenIntegrator{a}, sdh::SubDofHandler) where {a}
        ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
        ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
        qr     = QuadratureRule{Ferrite.getrefshape(ip)}(m.order)
        return StressDrivenCache{a}(CellValues(qr, ip, ip_geo), AlgebraicValues(m.variable), m.E, m.σ̄,
                                    dof_range(sdh, m.field_name), global_dof_range(m, sdh))
    end

    FerriteOperators.duplicate_for_device(device, c::StressDrivenCache{a}) where {a} =
        StressDrivenCache{a}(FerriteOperators.duplicate_for_device(device, c.cv), c.av,
                             c.E, c.σ̄, c.range_u, c.range_ε)
    FerriteOperators.reinit_values!(c::StressDrivenCache, cell) = reinit!(c.cv, cell)

    # ε̄ read out of the tail of the local unknown vector — eltype-generic, so
    # the AD route seeds the algebraic dofs like any other.
    function macroscopic_strain(c::StressDrivenCache, uₑ)
        ε̄ = zero(SymmetricTensor{2, 2, eltype(uₑ)})
        for (jε, J) in pairs(c.range_ε)
            ε̄ += uₑ[J] * algebraic_basis_value(c.av, jε)
        end
        return ε̄
    end

    # F(u) = 0 is the stationarity of the RVE potential: the u rows carry
    # ∫ δε : σ dΩ, the ε̄ rows the constraint ⟨σ⟩ = σ̄ tested with the
    # algebraic basis.
    function FerriteOperators.assemble_cell!(req::ResidualRequest, c::StressDrivenCache, args)
        (; cv, av, E, σ̄, range_u, range_ε) = c
        uₑ = args.states.u
        ε̄  = macroscopic_strain(c, uₑ)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            σ  = E ⊡ (ε̄ + function_symmetric_gradient(cv, qp, uₑ, range_u))
            for (iu, I) in pairs(range_u)
                req.r[I] += (shape_symmetric_gradient(cv, qp, iu) ⊡ σ) * dΩ
            end
            for (iε, I) in pairs(range_ε)
                Eᵢ = algebraic_basis_value(av, iε)
                req.r[I] += (Eᵢ ⊡ σ - σ̄ ⊡ Eᵢ) * dΩ
            end
        end
    end

    FerriteOperators.provides_analytic(::Type{<:StressDrivenCache{true}}, ::JacobianKind{:u}) = true
    function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, c::StressDrivenCache{true}, args)
        (; cv, av, E, range_u, range_ε) = c
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for (iu, I) in pairs(range_u)
                δεi = shape_symmetric_gradient(cv, qp, iu)
                for (ju, J) in pairs(range_u)
                    req.K[I, J] += (δεi ⊡ E ⊡ shape_symmetric_gradient(cv, qp, ju)) * dΩ
                end
                for (jε, J) in pairs(range_ε)
                    v = (δεi ⊡ E ⊡ algebraic_basis_value(av, jε)) * dΩ
                    req.K[I, J] += v
                    req.K[J, I] += v
                end
            end
            for (iε, I) in pairs(range_ε)
                Eᵢ = algebraic_basis_value(av, iε)
                for (jε, J) in pairs(range_ε)
                    req.K[I, J] += (Eᵢ ⊡ E ⊡ algebraic_basis_value(av, jε)) * dΩ
                end
            end
        end
    end

    ####################################
    ## Facet-item tying testbed — a facet term coupling to a global dof
    ####################################
    # The RSAFDQ shape: a facet set whose local system is
    # `[celldofs(cell); the chamber pressure dof]`, carrying the coupling in
    # both the rows and the columns of the augmented tail,
    #
    #     r_u[i] += p ∫_Γ Nᵢ dΓ,      r_p += ∫_Γ u dΓ,
    #
    # with the matching symmetric off-diagonal blocks. There is deliberately no
    # volumetric term: what the fixture witnesses is the facet-item traversal
    # composed with the `global_dofs` machinery, nothing else.

    struct TyingFacetIntegrator <: AbstractNonlinearIntegrator
        field_name::Symbol
        variable_name::Symbol
        facetset::Set{FacetIndex}
    end

    struct TyingFacetCache{FV <: FacetValues} <: FerriteOperators.AbstractSurfaceElementCache
        fv::FV
        range_u::UnitRange{Int}
        range_p::UnitRange{Int}
    end

    FerriteOperators.global_dofs(m::TyingFacetIntegrator, sdh::SubDofHandler) =
        algebraic_dofs(sdh.dh, m.variable_name)
    FerriteOperators.setup_element_cache(::TyingFacetIntegrator, ::SubDofHandler) =
        FerriteOperators.EmptyVolumetricElementCache()
    FerriteOperators.facet_items(m::TyingFacetIntegrator, ::SubDofHandler) = m.facetset
    function FerriteOperators.setup_facet_item_cache(m::TyingFacetIntegrator, sdh::SubDofHandler)
        ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
        ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
        fqr    = FacetQuadratureRule{Ferrite.getrefshape(ip)}(2)
        return TyingFacetCache(FacetValues(fqr, ip, ip_geo),
                               dof_range(sdh, m.field_name), global_dof_range(m, sdh))
    end
    FerriteOperators.duplicate_for_device(device, c::TyingFacetCache) =
        TyingFacetCache(FerriteOperators.duplicate_for_device(device, c.fv), c.range_u, c.range_p)

    # One body for all three requests: which buffers it fills is decided on the
    # request TYPE, so the branches fold away per kernel.
    function _tying_facet!(req, c::TyingFacetCache, args, lfi)
        reinit!(c.fv, args.cell, lfi)
        uₑ = args.states.u
        P  = first(c.range_p)
        p  = uₑ[P]
        for qp in 1:getnquadpoints(c.fv)
            dΓ = getdetJdV(c.fv, qp)
            uq = function_value(c.fv, qp, uₑ, c.range_u)
            for (i, I) in pairs(c.range_u)
                Nᵢ = shape_value(c.fv, qp, i) * dΓ
                req isa Union{ResidualRequest, JacobianResidualRequest} && (req.r[I] += p * Nᵢ)
                if req isa Union{JacobianRequest{:u}, JacobianResidualRequest}
                    req.K[I, P] += Nᵢ
                    req.K[P, I] += Nᵢ
                end
            end
            req isa Union{ResidualRequest, JacobianResidualRequest} && (req.r[P] += uq * dΓ)
        end
        return nothing
    end

    FerriteOperators.assemble_facet!(req::ResidualRequest, c::TyingFacetCache, args::FacetArgs, lfi::Int) =
        _tying_facet!(req, c, args, lfi)
    FerriteOperators.assemble_facet!(req::JacobianRequest{:u}, c::TyingFacetCache, args::FacetArgs, lfi::Int) =
        _tying_facet!(req, c, args, lfi)
    FerriteOperators.assemble_facet!(req::JacobianResidualRequest, c::TyingFacetCache, args::FacetArgs, lfi::Int) =
        _tying_facet!(req, c, args, lfi)
    FerriteOperators.provides_analytic(::Type{<:TyingFacetCache}, ::Union{JacobianKind{:u}, JacobianResidualKind}) = true

    """
    Grid, DofHandler, coupling descriptor and integrator of the facet tying
    problem. The declared set is the union of two boundary facetsets, so the
    corner cell owns TWO declared facets — the grouping a facet item exists for.
    """
    function tying_facet_testbed(dims = (3, 3); field_name = :u, variable_name = :p)
        grid = generate_grid(Quadrilateral, dims)
        dh   = DofHandler(grid)
        add!(dh, field_name, Lagrange{RefQuadrilateral, 1}())
        add!(dh, variable_name, AlgebraicVariable())
        close!(dh)
        facets   = union(Set(getfacetset(grid, "right")), Set(getfacetset(grid, "top")))
        coupling = CellCoupling(1:getncells(grid); algebraic_coupling = ((field_name, variable_name),))
        return (; grid, dh, facets, coupling,
                  integrator = TyingFacetIntegrator(field_name, variable_name, facets),
                  pdof = only(algebraic_dofs(dh, variable_name)))
    end

    # The hand-rolled Ferrite loop the facet-item assembly is compared against:
    # `FacetIterator` over the declared set, one local system PER FACET with the
    # augmented dof vector spelled out — the shape Thunderbolt's tying term
    # hand-rolls today.
    function tying_facet_reference(testbed, u)
        (; dh, facets, coupling, pdof) = testbed
        sdh     = dh.subdofhandlers[1]
        ip      = Ferrite.getfieldinterpolation(sdh, :u)
        fv      = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), ip,
                              Ferrite.geometric_interpolation(Quadrilateral))
        nc      = ndofs_per_cell(sdh)
        range_u = dof_range(sdh, :u)
        P       = nc + 1
        dofs    = Vector{Int}(undef, nc + 1)
        dofs[P] = pdof
        Ke, re, uₑ = zeros(nc + 1, nc + 1), zeros(nc + 1), zeros(nc + 1)

        K = allocate_matrix(dh; algebraic_couplings = (coupling,))
        r = zeros(ndofs(dh))
        assembler = start_assemble(K, r)
        for facet in FacetIterator(sdh, facets)
            reinit!(fv, facet)
            copyto!(view(dofs, 1:nc), celldofs(facet))
            uₑ .= @view u[dofs]
            fill!(Ke, 0)
            fill!(re, 0)
            p = uₑ[P]
            for qp in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, qp)
                uq = function_value(fv, qp, uₑ, range_u)
                for (i, I) in pairs(range_u)
                    Nᵢ = shape_value(fv, qp, i) * dΓ
                    re[I] += p * Nᵢ
                    Ke[I, P] += Nᵢ
                    Ke[P, I] += Nᵢ
                end
                re[P] += uq * dΓ
            end
            assemble!(assembler, dofs, Ke, re)
        end
        return K, r
    end

    ####################################
    ## Reservoir testbed — cells plus algebraic items
    ####################################
    # A scalar diffusion field on a quad grid coupled to a lumped pressure `p1`
    # that lives in no cell, plus nonlinear 0D exchange rows between `p1` and a
    # second lumped pressure `p2`. The cell term carries `p1` through
    # `global_dofs`; the 0D rows are algebraic items — one per exchange path,
    # all of them on the SAME two dofs, which is what makes their scatter
    # collide.
    #
    # `analytic` picks whether the algebraic cache provides the Jacobian kernel
    # or leaves it to AD; `coupled` whether the cell term touches `p1` at all,
    # an uncoupled cell term declaring no global dofs and therefore admitting
    # a colored partition.

    struct ReservoirIntegrator{analytic, coupled} <: AbstractNonlinearIntegrator
        order::Int
        field_name::Symbol
        variable_names::Tuple{Symbol, Symbol}
        α::Float64
        conductances::Vector{Float64}
        sources::Vector{Float64}
    end
    ReservoirIntegrator(; analytic = true, coupled = true, order = 2, field_name = :u,
                        variable_names = (:p1, :p2), α = 0.7,
                        conductances = [1.5, -0.4], sources = [0.25, 0.6]) =
        ReservoirIntegrator{analytic, coupled}(order, field_name, variable_names, α, conductances, sources)

    struct ReservoirCellCache{CV} <: AbstractVolumetricElementCache
        cv::CV
        α::Float64
        range_u::UnitRange{Int}
        range_p::UnitRange{Int}   # where `p1` sits in the local system; empty when uncoupled
    end

    FerriteOperators.global_dofs(m::ReservoirIntegrator{<:Any, true}, sdh::SubDofHandler) =
        algebraic_dofs(sdh.dh, m.variable_names[1])

    function FerriteOperators.setup_element_cache(m::ReservoirIntegrator, sdh::SubDofHandler)
        ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
        ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
        qr     = QuadratureRule{Ferrite.getrefshape(ip)}(m.order)
        return ReservoirCellCache(CellValues(qr, ip, ip_geo), m.α,
                                  dof_range(sdh, m.field_name), global_dof_range(m, sdh))
    end

    FerriteOperators.duplicate_for_device(device, c::ReservoirCellCache) =
        ReservoirCellCache(FerriteOperators.duplicate_for_device(device, c.cv), c.α, c.range_u, c.range_p)
    FerriteOperators.reinit_values!(c::ReservoirCellCache, cell) = reinit!(c.cv, cell)

    # r_u = ∫ (∇u⋅∇v + α p₁ v) dΩ, and the symmetric partner ∫ α u dΩ in the
    # `p₁` row — the term that makes the lumped unknown see the field.
    function FerriteOperators.assemble_cell!(req::ResidualRequest, c::ReservoirCellCache, args::CellArgs)
        (; cv, α, range_u, range_p) = c
        uₑ = args.states.u
        p  = isempty(range_p) ? zero(eltype(uₑ)) : uₑ[first(range_p)]
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            ∇u = function_gradient(cv, qp, uₑ, range_u)
            uq = function_value(cv, qp, uₑ, range_u)
            for (i, I) in pairs(range_u)
                req.r[I] += (∇u ⋅ shape_gradient(cv, qp, i) + α * p * shape_value(cv, qp, i)) * dΩ
            end
            for I in range_p
                req.r[I] += α * uq * dΩ
            end
        end
    end

    FerriteOperators.provides_analytic(::Type{<:ReservoirCellCache}, ::JacobianKind{:u}) = true
    function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, c::ReservoirCellCache, args::CellArgs)
        (; cv, α, range_u, range_p) = c
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for (i, I) in pairs(range_u)
                for (j, J) in pairs(range_u)
                    req.K[I, J] += (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) * dΩ
                end
                v = α * shape_value(cv, qp, i) * dΩ
                for J in range_p
                    req.K[I, J] += v
                    req.K[J, I] += v
                end
            end
        end
    end

    FerriteOperators.functional_value_type(::FunctionalKind{:reservoir_volume}) = Float64
    function FerriteOperators.evaluate_cell_functional(::FunctionalKind{:reservoir_volume},
                                                       c::ReservoirCellCache, args::CellArgs)
        return sum(qp -> getdetJdV(c.cv, qp), 1:getnquadpoints(c.cv))
    end

    # The 0D rows. Item `k` exchanges between `p₁` and `p₂` with a cubic
    # characteristic and a parameter-scaled source, so the kernel needs both the
    # item's index and the sweep's parameter.
    struct ReservoirItemCache{analytic}
        conductances::Vector{Float64}
        sources::Vector{Float64}
    end

    FerriteOperators.algebraic_items(m::ReservoirIntegrator, dh::DofHandler) =
        [[only(algebraic_dofs(dh, m.variable_names[1])), only(algebraic_dofs(dh, m.variable_names[2]))]
         for _ in eachindex(m.conductances)]

    FerriteOperators.setup_algebraic_cache(m::ReservoirIntegrator{analytic}, dh::DofHandler) where {analytic} =
        ReservoirItemCache{analytic}(m.conductances, m.sources)

    FerriteOperators.duplicate_for_device(device, c::ReservoirItemCache) = c

    function FerriteOperators.assemble_algebraic!(req::ResidualRequest, c::ReservoirItemCache, args::AlgebraicArgs)
        k    = args.item.index
        Δ    = args.states.u[1] - args.states.u[2]
        flux = c.conductances[k] * Δ^3 - args.p * c.sources[k]
        req.r[1] += flux
        req.r[2] -= flux
    end

    FerriteOperators.provides_analytic(::Type{<:ReservoirItemCache{true}}, ::JacobianKind{:u}) = true
    function FerriteOperators.assemble_algebraic!(req::JacobianRequest{:u}, c::ReservoirItemCache{true}, args::AlgebraicArgs)
        Δ = args.states.u[1] - args.states.u[2]
        g = 3 * c.conductances[args.item.index] * Δ^2
        req.K[1, 1] += g
        req.K[1, 2] -= g
        req.K[2, 1] -= g
        req.K[2, 2] += g
    end

    "Grid, DofHandler and coupling descriptors of the reservoir problem."
    function reservoir_testbed(dims = (2, 2); variable_names = (:p1, :p2))
        grid = generate_grid(Quadrilateral, dims)
        dh   = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        add!(dh, variable_names[1], AlgebraicVariable())
        add!(dh, variable_names[2], AlgebraicVariable())
        close!(dh)
        cell_coupling = CellCoupling(1:getncells(grid); algebraic_coupling = ((:u, variable_names[1]),))
        item_coupling = AlgebraicCoupling(; algebraic_coupling = ((variable_names[1], variable_names[2]),))
        return (; grid, dh, cell_coupling, item_coupling,
                  item_dofs = [only(algebraic_dofs(dh, variable_names[1])),
                               only(algebraic_dofs(dh, variable_names[2]))])
    end

    # The hand-rolled Ferrite loop the FO assembly is compared against: one pass
    # over the cells with the augmented dof vector spelled out, one over the 0D
    # rows with their own dof vector.
    function reservoir_reference(testbed, m::ReservoirIntegrator{<:Any, coupled}, u, θ) where {coupled}
        (; dh, cell_coupling, item_coupling) = testbed
        ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], m.field_name)
        cv = CellValues(QuadratureRule{RefQuadrilateral}(m.order), ip,
                        Ferrite.geometric_interpolation(Quadrilateral))
        nc      = ndofs_per_cell(dh)
        gdofs   = coupled ? algebraic_dofs(dh, m.variable_names[1]) : Int[]
        dofs    = Vector{Int}(undef, nc + length(gdofs))
        dofs[(nc + 1):end] .= gdofs
        range_u = dof_range(dh, m.field_name)
        range_p = (nc + 1):(nc + length(gdofs))
        nl      = nc + length(gdofs)
        Ke, re, uₑ = zeros(nl, nl), zeros(nl), zeros(nl)

        K = allocate_matrix(dh; algebraic_couplings = (cell_coupling, item_coupling))
        r = zeros(ndofs(dh))
        assembler = start_assemble(K, r)
        for cell in CellIterator(dh)
            reinit!(cv, cell)
            copyto!(dofs, celldofs(cell))
            uₑ .= @view u[dofs]
            fill!(Ke, 0)
            fill!(re, 0)
            for qp in 1:getnquadpoints(cv)
                dΩ = getdetJdV(cv, qp)
                ∇u = function_gradient(cv, qp, uₑ, range_u)
                uq = function_value(cv, qp, uₑ, range_u)
                p  = isempty(range_p) ? 0.0 : uₑ[first(range_p)]
                for (i, I) in pairs(range_u)
                    re[I] += (∇u ⋅ shape_gradient(cv, qp, i) + m.α * p * shape_value(cv, qp, i)) * dΩ
                    for (j, J) in pairs(range_u)
                        Ke[I, J] += (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) * dΩ
                    end
                    v = m.α * shape_value(cv, qp, i) * dΩ
                    for J in range_p
                        Ke[I, J] += v
                        Ke[J, I] += v
                    end
                end
                for I in range_p
                    re[I] += m.α * uq * dΩ
                end
            end
            assemble!(assembler, dofs, Ke, re)
        end

        idofs = testbed.item_dofs
        Δ     = u[idofs[1]] - u[idofs[2]]
        Ki, ri = zeros(2, 2), zeros(2)
        for k in eachindex(m.conductances)
            fill!(Ki, 0)
            fill!(ri, 0)
            flux = m.conductances[k] * Δ^3 - θ * m.sources[k]
            ri[1] += flux
            ri[2] -= flux
            g = 3 * m.conductances[k] * Δ^2
            Ki[1, 1] += g
            Ki[1, 2] -= g
            Ki[2, 1] -= g
            Ki[2, 2] += g
            assemble!(assembler, idofs, Ki, ri)
        end
        return K, r
    end

    "Grid, DofHandler and the coupling descriptor of the stress-driven RVE problem."
    function stress_driven_testbed(dims = (3, 3); variable_name = :εbar)
        grid = generate_grid(Quadrilateral, dims)
        var  = AlgebraicVariable{SymmetricTensor{2, 2}}()
        dh   = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
        add!(dh, variable_name, var)
        close!(dh)
        coupling = CellCoupling(1:getncells(grid);
                                algebraic_coupling = ((:u, variable_name), (variable_name, variable_name)))
        E = 100.0 * one(SymmetricTensor{4, 2}) + 60.0 * (one(SymmetricTensor{2, 2}) ⊗ one(SymmetricTensor{2, 2}))
        σ̄ = SymmetricTensor{2, 2}((0.3, 1.0, -0.2))
        return (; grid, dh, var, coupling, E, σ̄)
    end

    # The hand-rolled Ferrite loop the FO assembly is compared against — the
    # `assemble_system!` of Ferrite's stress-driven homogenization tutorial,
    # over one material and with the augmented dof vector spelled out.
    function stress_driven_reference(testbed; variable_name = :εbar)
        (; dh, var, coupling, E, σ̄) = testbed
        ip  = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], :u)
        cv  = CellValues(QuadratureRule{RefQuadrilateral}(2), ip, Ferrite.geometric_interpolation(Quadrilateral))
        av  = AlgebraicValues(var)
        n   = ndofs_per_cell(dh)
        nε  = getnbasefunctions(av)
        dofs = Vector{Int}(undef, n + nε)
        dofs[(n + 1):end] .= algebraic_dofs(dh, variable_name)
        range_u = dof_range(dh, :u)
        range_ε = (n + 1):(n + nε)
        Ke = zeros(n + nε, n + nε)
        fe = zeros(n + nε)
        K  = allocate_matrix(dh; algebraic_couplings = (coupling,))
        f  = zeros(ndofs(dh))
        assembler = start_assemble(K, f)
        for cell in CellIterator(dh)
            reinit!(cv, cell)
            copyto!(dofs, celldofs(cell))
            fill!(Ke, 0)
            fill!(fe, 0)
            for qp in 1:getnquadpoints(cv)
                dΩ = getdetJdV(cv, qp)
                for (iu, I) in pairs(range_u)
                    δεi = shape_symmetric_gradient(cv, qp, iu)
                    for (ju, J) in pairs(range_u)
                        Ke[I, J] += (δεi ⊡ E ⊡ shape_symmetric_gradient(cv, qp, ju)) * dΩ
                    end
                    for (jε, J) in pairs(range_ε)
                        v = (δεi ⊡ E ⊡ algebraic_basis_value(av, jε)) * dΩ
                        Ke[I, J] += v
                        Ke[J, I] += v
                    end
                end
                for (iε, I) in pairs(range_ε)
                    Eᵢ = algebraic_basis_value(av, iε)
                    fe[I] += (σ̄ ⊡ Eᵢ) * dΩ
                    for (jε, J) in pairs(range_ε)
                        Ke[I, J] += (Eᵢ ⊡ E ⊡ algebraic_basis_value(av, jε)) * dΩ
                    end
                end
            end
            assemble!(assembler, dofs, Ke, fe)
        end
        return K, f
    end

    ####################################
    ## Condensed algebraic item testbed — chamber exchange, linear relaxation
    ####################################
    # A single-dof "chamber" pressure `p` (an algebraic dof, one per item, items
    # sharing no dofs) exchanging linearly with a hidden per-item internal state
    # `q` — the algebraic-item analogue of `SimpleCondensedPowerLawRelaxation`'s
    # exchange term, but LINEAR so the local stage problem solves in closed form:
    #
    #     r_p = β(p − q),   dq/dt = β(p − q)/τ
    #
    # canonical stage problem q = q₀ + γ̃·β(p−q)/τ, k = γ̃β/τ:
    #
    #     q = (q₀ + k·p)/(1+k)                          -- closed-form local solve
    #     dq/dp = k/(1+k)                                -- IFT corrector (Consistent Jacobian)
    #     dq/dβ = (γ̃/τ)(p−q)/(1+k)                       -- IFT corrector (∂F/∂θ)
    #
    # `cell_integrator` supplies the OTHER term of the operator — plain diffusion
    # (`PlainPoissonIntegrator`) for "condensed item alone", or a condensed cell
    # integrator (e.g. the lib's `SimpleCondensedPowerLawRelaxation`) for the
    # layout-collision proof — forwarded to via `setup_element_cache` only:
    # every other cell hook (`get_number_of_internal_dofs_per_element`,
    # `condense_cell!`, the kernels) dispatches on the CACHE type, so nothing
    # else needs forwarding.

    struct PlainPoissonIntegrator <: AbstractNonlinearIntegrator
        qrc::QuadratureRuleCollection
        field_name::Symbol
    end
    struct PlainPoissonCache{CV <: CellValues} <: AbstractVolumetricElementCache
        cv::CV
    end
    function FerriteOperators.setup_element_cache(m::PlainPoissonIntegrator, sdh::SubDofHandler)
        qr     = getquadraturerule(m.qrc, sdh)
        ip     = Ferrite.getfieldinterpolation(sdh, m.field_name)
        ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
        return PlainPoissonCache(CellValues(qr, ip, ip_geo))
    end
    FerriteOperators.duplicate_for_device(device, c::PlainPoissonCache) =
        PlainPoissonCache(FerriteOperators.duplicate_for_device(device, c.cv))
    FerriteOperators.reinit_values!(c::PlainPoissonCache, cell) = reinit!(c.cv, cell)
    function FerriteOperators.assemble_cell!(req::ResidualRequest, c::PlainPoissonCache, args::CellArgs)
        uₑ = args.states.u
        for qp in 1:getnquadpoints(c.cv)
            dΩ = getdetJdV(c.cv, qp)
            ∇u = function_gradient(c.cv, qp, uₑ)
            for i in 1:getnbasefunctions(c.cv)
                req.r[i] += (shape_gradient(c.cv, qp, i) ⋅ ∇u) * dΩ
            end
        end
    end

"""
Material parameters of the chamber's linear exchange: rate `β` and relaxation
time `τ`. Generic over `T` — the layout-collision testbed shares this GLOBAL
parameter object with a plain cell integrator that has no analytic
`ParameterJacobianRequest` kernel, so a Dual-valued `rebuild_parameters` must
be constructible even though the chamber cache itself never needs one (it
provides the analytic kernel).
"""
    struct ChamberRelaxationParameters{T}
        β::T
        τ::T
    end
    ChamberRelaxationParameters(; β = 1.3, τ = 0.9) = ChamberRelaxationParameters(promote(β, τ)...)
    # θ = (β,) only — τ stays fixed, mirroring `NortonRelaxationParameters`' κ.
    FerriteOperators.parameter_vector(p::ChamberRelaxationParameters) = [p.β]
    FerriteOperators.rebuild_parameters(p::ChamberRelaxationParameters, θ) =
        ChamberRelaxationParameters(promote(θ[1], p.τ)...)

    """
    The cache associated with the chamber exchange term: one corrector per item
    (`dq/dp`, read by the `Consistent` kernel) and one parameter corrector per
    item (`dq/dβ`, read by the `ParameterJacobianRequest` kernel), both
    populated by `condense_algebraic!`. Declares `has_internal_state`.
    """
    struct CondensedChamberCache
        params::ChamberRelaxationParameters
        correctors::ItemStates{Float64}
        param_correctors::ItemStates{Float64}
    end
    FerriteOperators.duplicate_for_device(device, c::CondensedChamberCache) =
        CondensedChamberCache(c.params, FerriteOperators.duplicate_for_device(device, c.correctors),
                              FerriteOperators.duplicate_for_device(device, c.param_correctors))
    FerriteOperators.has_internal_state(::Type{CondensedChamberCache}) = true
    function FerriteOperators.invalidate_correctors!(cache::CondensedChamberCache)
        FerriteOperators.invalidate_item_states!(cache.correctors)
        FerriteOperators.invalidate_item_states!(cache.param_correctors)
        return nothing
    end

    _chamber_params(cache::CondensedChamberCache, ::Nothing) = cache.params
    _chamber_params(cache::CondensedChamberCache, p::ChamberRelaxationParameters) = p

    FerriteOperators.get_number_of_internal_dofs_per_algebraic_item(
        m, ::CondensedChamberCache, items) = fill(1, length(items))

    """
        condense_algebraic!(cache::CondensedChamberCache, args, weights) -> CondensationReport

    Solve the item's local exchange problem in closed form (a direct solve —
    always converged, zero inner iterations), write the trial `q` and store
    `dq/dp`/`dq/dβ`. `worst_cell` reports `-args.item.index`, the family
    disambiguation `CondensationReport` documents.
    """
    function FerriteOperators.condense_algebraic!(cache::CondensedChamberCache, args::AlgebraicArgs, weights::NamedTuple)
        (; β, τ) = _chamber_params(cache, args.p)
        γ̃  = stage_scaling(args.ctx)
        k  = γ̃ * β / τ
        idx = args.item.index
        p  = args.states.u[1]
        q₀ = args.states.qprev[1]
        q  = (q₀ + k * p) / (1 + k)
        args.states.q[1] = q
        FerriteOperators.set_item_state!(cache.correctors, idx, k / (1 + k))
        FerriteOperators.set_item_state!(cache.param_correctors, idx, (γ̃ / τ) * (p - q) / (1 + k))
        return CondensationReport(true, 1, 0, 0, -idx, 0, 0.0, 1.0)
    end

    const _ChamberJacobianLike = Union{JacobianRequest{:u, Consistent}, JacobianRequest{:u, FrozenQ},
                                       JacobianResidualRequest{Consistent}, JacobianResidualRequest{FrozenQ}}
    const _ChamberFrozenLike = Union{JacobianRequest{:u, FrozenQ}, JacobianResidualRequest{FrozenQ}}

    # Pure evaluation at the FROZEN q the last `condense_internal!` wrote: no
    # solve, no write-back. `item_state` throws, naming the item, if
    # `condense_internal!` never condensed it (or it was invalidated since).
    function _chamber_assemble!(req, cache::CondensedChamberCache, args::AlgebraicArgs)
        (; β) = _chamber_params(cache, args.p)
        idx = args.item.index
        p = args.states.u[1]
        q = args.states.q[1]

        needs_jac = req isa _ChamberJacobianLike
        dqdp = (needs_jac && !(req isa _ChamberFrozenLike)) ? FerriteOperators.item_state(cache.correctors, idx) : nothing

        if req isa Union{ResidualRequest, JacobianResidualRequest}
            req.r[1] += β * (p - q)
        end
        if needs_jac
            slope = req isa _ChamberFrozenLike ? 0.0 : dqdp
            req.K[1, 1] += β * (1 - slope)
        end
    end
    FerriteOperators.assemble_algebraic!(req::ResidualRequest, cache::CondensedChamberCache, args::AlgebraicArgs) = _chamber_assemble!(req, cache, args)
    FerriteOperators.assemble_algebraic!(req::JacobianRequest{:u, Consistent}, cache::CondensedChamberCache, args::AlgebraicArgs) = _chamber_assemble!(req, cache, args)
    FerriteOperators.assemble_algebraic!(req::JacobianRequest{:u, FrozenQ}, cache::CondensedChamberCache, args::AlgebraicArgs) = _chamber_assemble!(req, cache, args)
    FerriteOperators.assemble_algebraic!(req::JacobianResidualRequest{Consistent}, cache::CondensedChamberCache, args::AlgebraicArgs) = _chamber_assemble!(req, cache, args)
    FerriteOperators.assemble_algebraic!(req::JacobianResidualRequest{FrozenQ}, cache::CondensedChamberCache, args::AlgebraicArgs) = _chamber_assemble!(req, cache, args)
    FerriteOperators.provides_analytic(::Type{CondensedChamberCache}, ::Union{JacobianKind, JacobianResidualKind}) = true

    # ∂F/∂q, the item's 1×1 block of the rectangular field × internal target
    # (`update_internal_jacobian!`). Analytic is the only route for this
    # family: an item's AD buffers are sized from a dof count, which builds no
    # `:q` configuration.
    function FerriteOperators.assemble_algebraic!(req::JacobianRequest{:q}, cache::CondensedChamberCache, args::AlgebraicArgs)
        (; β) = _chamber_params(cache, args.p)
        req.K[1, 1] += -β
        return req.K
    end

    # Analytic ∂F/∂θ (θ = (β,)): the exchange term's own partial ∂r/∂β|_q =
    # (p−q) plus the stored ∂r/∂q · dq/dβ correction (∂r/∂q|_(p,β) = -β).
    function FerriteOperators.assemble_algebraic!(req::ParameterJacobianRequest, cache::CondensedChamberCache, args::AlgebraicArgs)
        (; β) = _chamber_params(cache, args.p)
        idx = args.item.index
        p = args.states.u[1]
        q = args.states.q[1]
        dqdβ = FerriteOperators.item_state(cache.param_correctors, idx)
        req.B[1, 1] += (p - q) - β * dqdβ
        return req.B
    end
    FerriteOperators.provides_analytic(::Type{CondensedChamberCache}, ::FerriteOperators.ParameterJacobianKind) = true

    """
        ChamberIntegrator(cell_integrator; params = ChamberRelaxationParameters())

    Plain-or-condensed cell physics (`cell_integrator`, forwarded to via
    `setup_element_cache` only) plus TWO independent chamber items on their own
    dedicated algebraic dofs `:p1`/`:p2` (no shared dofs, no cell↔item
    coupling — the fixture proves dof-LAYOUT composition, not scatter
    collision, which `test_algebraic_items.jl` already covers). No coupling
    descriptor is needed: every entry this operator has is on the diagonal,
    always allocated.
    """
    struct ChamberIntegrator{C} <: AbstractNonlinearIntegrator
        cell_integrator::C
        params::ChamberRelaxationParameters
    end
    ChamberIntegrator(cell_integrator; params = ChamberRelaxationParameters()) =
        ChamberIntegrator(cell_integrator, params)

    FerriteOperators.setup_element_cache(m::ChamberIntegrator, sdh::SubDofHandler) =
        FerriteOperators.setup_element_cache(m.cell_integrator, sdh)
    FerriteOperators.algebraic_items(m::ChamberIntegrator, dh::DofHandler) =
        [[only(algebraic_dofs(dh, :p1))], [only(algebraic_dofs(dh, :p2))]]
    FerriteOperators.setup_algebraic_cache(m::ChamberIntegrator, dh::DofHandler) =
        CondensedChamberCache(m.params, ItemStates{Float64}(2), ItemStates{Float64}(2))

    """
        chamber_testbed(strategy, qrc; cell_integrator = PlainPoissonIntegrator(qrc, :u), dims = (2, 2), kwargs...)

    Grid, DofHandler and operator of the chamber-exchange problem: a scalar
    field `:u` plus the two `:p1`/`:p2` chamber dofs. `condensed_states(u,
    uprev)` (defined above) serves this testbed's states too — the mechanism is
    family-agnostic.
    """
    function chamber_testbed(strategy, qrc; cell_integrator = PlainPoissonIntegrator(qrc, :u),
                             dims = (2, 2), params = ChamberRelaxationParameters(), kwargs...)
        grid = generate_grid(Quadrilateral, dims)
        dh   = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        add!(dh, :p1, AlgebraicVariable())
        add!(dh, :p2, AlgebraicVariable())
        close!(dh)
        integrator = ChamberIntegrator(cell_integrator; params)
        op = setup_operator(strategy, integrator, dh; slots = (:u, :q, :qprev), kwargs...)
        return (; op, dh, grid, item_dofs = [only(algebraic_dofs(dh, :p1)), only(algebraic_dofs(dh, :p2))])
    end

end
