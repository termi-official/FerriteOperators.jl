"""
Material parameters of the power-law relaxation model used by
[`SimpleCondensedPowerLawRelaxation`](@ref): the diffusivity `κ`, the exchange
modulus `α` between field and internal state, the dashpot viscosity `η` and
the Norton exponent `n`. `n = 1` degenerates to a linear dashpot.

Generic over `T` and promoting, because [`rebuild_parameters`](@ref) must be
able to return a DUAL-valued object: every generic parameter path — the AD
frozen-q partial, and the `local_conditions!` route's `∂L/∂θ` — rebuilds the
bag from a seeded θ, and a monomorphic field type would reject it.
"""
struct NortonRelaxationParameters{T}
    κ::T
    α::T
    η::T
    n::T
end
NortonRelaxationParameters(; κ = 1.0, α = 1.0, η = 1.0, n = 3.0) =
    NortonRelaxationParameters(promote(κ, α, η, n)...)

# The differentiable-parameter view for parameter sensitivities: θ = (α, η,
# n). κ is pure diffusion, structurally independent of the local q-problem
# and of the exchange term's q-dependence, so it stays out of θ.
FerriteOperators.parameter_vector(p::NortonRelaxationParameters) = SVector(p.α, p.η, p.n)
FerriteOperators.rebuild_parameters(p::NortonRelaxationParameters, θ) =
    NortonRelaxationParameters(κ = p.κ, α = θ[1], η = θ[2], n = θ[3])

"""
    LocalNewtonSettings(; max_iterations = 25, tolerance = 1.0e-12)

Static configuration of the element-local Newton solver of
[`SimpleCondensedPowerLawRelaxation`](@ref): the per-quadrature-point
iteration budget, and the tightest absolute local residual tolerance the
element ever solves to. A solver may only LOOSEN that tolerance, per
[`local_solve_tolerance`](@ref). Exceeding the budget is reported through
[`CondensationReport`](@ref)`.converged`, not thrown.
"""
@kwdef struct LocalNewtonSettings
    max_iterations::Int = 25
    tolerance::Float64  = 1.0e-12
end

"""
    InexactLocalSolveContext(inner, local_tolerance)

Per-sweep context decorating `inner` with the local residual tolerance the
outer solver asks element-local problems to solve to this sweep — the
inexact-Newton forcing term, e.g. `ηₖ‖F(uₖ)‖`. Framework context handling
passes through to `inner` ([`evaluation_time`](@ref), `with_time`); elements
read the tolerance through [`local_solve_tolerance`](@ref). The tolerance is
consumed once per trial point, at [`condense_internal!`](@ref) time, rather
than on every sweep that happens to solve.
"""
struct InexactLocalSolveContext{C, T}
    inner::C
    local_tolerance::T
end

evaluation_time(ctx::InexactLocalSolveContext) = evaluation_time(ctx.inner)
with_time(ctx::InexactLocalSolveContext, t̃) =
    InexactLocalSolveContext(with_time(ctx.inner, t̃), ctx.local_tolerance)

"""
    local_solve_tolerance(ctx) -> tolerance or `nothing`

The local residual tolerance the outer solver requests for this sweep.
`nothing` — the default for every context type — leaves the element at its own
[`LocalNewtonSettings`](@ref) tolerance, which is also the floor: a requested
tolerance tighter than the element's own is ignored.
"""
local_solve_tolerance(ctx) = nothing
local_solve_tolerance(ctx::InexactLocalSolveContext) = ctx.local_tolerance

# InexactLocalSolveContext decorates a TimeIntegrationContext, so it forwards
# the stage-scaling accessor like it forwards evaluation_time.
stage_scaling(ctx::InexactLocalSolveContext) = stage_scaling(ctx.inner)

@doc raw"""
    SimpleCondensedPowerLawRelaxation(material_parameters, qrc, field_name, internal_name;
                                       local_solver = LocalNewtonSettings(),
                                       corrector = Stored())

Diffusion of a scalar field `u` exchanging with a condensed
per-quadrature-point internal state `q` across a power-law (Norton) dashpot:

```math
r(u, q) = \int_\Omega \kappa\, \nabla u \cdot \nabla \delta u
        + \alpha\, (u - q)\, \delta u \; \mathrm{d}\Omega ,
\qquad
\dot q = \frac{\sigma\, |\sigma|^{n-1}}{\eta},
\quad \sigma = \alpha\,(u - q) .
```

The element owns the LOCAL stage problem `q = q_prev + γ̃ · q̇(u, q)`, which is
nonlinear for `n ≠ 1` and solved per quadrature point by a Newton iteration
started at `q_prev` inside [`condense_cell!`](@ref) — [`condense_internal!`](@ref)
must run before any evaluation sweep. The implicit-function-theorem slopes
`dq/du` and (for the parameter-sensitivity path) `dq/dθ` are what the
`Consistent` kernels read, so the element is admissible under the
internal-state rule.

Two channels connect the local solver to the outer one:

  * outer → inner: the requested local tolerance, read from the context via
    [`local_solve_tolerance`](@ref) and clamped from below by the element's own
    [`LocalNewtonSettings`](@ref).
  * inner → outer: the [`CondensationReport`](@ref) [`condense_internal!`](@ref)
    returns.

`corrector` elects where those slopes come from ([`CorrectorElection`](@ref)):
`Stored()` keeps them per quadrature point, `Recompute()` keeps none and
re-derives them from the converged `(u, q)` in the kernel that needs them.
Both are implemented and produce the same numbers.
"""
struct SimpleCondensedPowerLawRelaxation{Corr <: CorrectorElection} <: AbstractCondensedNonlinearIntegrator
    material_parameters::NortonRelaxationParameters{Float64}
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
    internal_name::Symbol
    local_solver::LocalNewtonSettings
    corrector::Corr
end
function SimpleCondensedPowerLawRelaxation(material_parameters, qrc, field_name, internal_name;
        local_solver = LocalNewtonSettings(), corrector = Stored())
    corrector isa Union{Stored, Recompute} || corrector_election_error(corrector)
    return SimpleCondensedPowerLawRelaxation(material_parameters, qrc, field_name, internal_name,
                                              local_solver, corrector)
end
FerriteOperators.corrector_election(integrator::SimpleCondensedPowerLawRelaxation) = integrator.corrector

"""
The cache associated with [`SimpleCondensedPowerLawRelaxation`](@ref). Beyond
its `CellValues` and the local solver's static configuration it carries the
two per-quadrature-point corrector stores the `Consistent` kernels read —
`correctors` (`dq/du`, Tier 1) and `param_correctors` (`dq/dθ`, θ = (α, η, n),
the parameter-sensitivity path) — under a [`Stored`](@ref) election, and
`nothing` in both fields under [`Recompute`](@ref), where the kernels
re-derive the slopes instead. `weight` is the solver's `:u` chain-rule scalar
of the last condensation: `O(1)` per cache rather than per item, so it stays
whatever the election, and the recomputing path reads it back (a `Stored()`
corrector has it folded in already). Declares [`has_internal_state`](@ref).
"""
struct SimpleCondensedPowerLawRelaxationCache{NQP, CV <: CellValues, S, P} <: AbstractVolumetricElementCache
    material_parameters::NortonRelaxationParameters{Float64}
    cv::CV
    local_solver::LocalNewtonSettings
    correctors::S
    param_correctors::P
    weight::Base.RefValue{Float64}
end
# NQP is a kernel loop bound, and under `Recompute()` no field carries it, so
# it is spelled at construction instead of inferred from the store type.
SimpleCondensedPowerLawRelaxationCache{NQP}(mat, cv::CV, ls, c::S, pc::P, w) where {NQP, CV, S, P} =
    SimpleCondensedPowerLawRelaxationCache{NQP, CV, S, P}(mat, cv, ls, c, pc, w)

# The election, read off the cache: no store means the slopes are recomputed.
# The `CV` bound is spelled rather than left `<:Any`, which would make every
# method below ambiguous against its unparameterized sibling instead of more
# specific.
const _PLRRecomputing{NQP} = SimpleCondensedPowerLawRelaxationCache{NQP, <:CellValues, Nothing, Nothing}

Ferrite.getnquadpoints(e::SimpleCondensedPowerLawRelaxationCache) = getnquadpoints(e.cv)
reinit_values!(e::SimpleCondensedPowerLawRelaxationCache, cell) = Ferrite.reinit!(e.cv, cell)

# The corrector stores alias across workers (ItemStates' own duplication
# rule): items are disjoint per worker, so there is nothing to copy. `weight`
# is per-worker, each worker's condensation writing its own copy.
function duplicate_for_device(device, cache::SimpleCondensedPowerLawRelaxationCache{NQP}) where {NQP}
    return SimpleCondensedPowerLawRelaxationCache{NQP}(
        cache.material_parameters,
        duplicate_for_device(device, cache.cv),
        cache.local_solver,
        duplicate_for_device(device, cache.correctors),
        duplicate_for_device(device, cache.param_correctors),
        Ref(cache.weight[]),
    )
end

function get_number_of_internal_dofs_per_element(element_model, cache::SimpleCondensedPowerLawRelaxationCache, sdh)
    nqp = getnquadpoints(cache.cv)
    return [nqp for i in sdh.cellset]
end

provides_analytic(::Type{<:SimpleCondensedPowerLawRelaxationCache}, ::Union{JacobianKind, JacobianResidualKind}) = true
provides_analytic(::Type{<:SimpleCondensedPowerLawRelaxationCache}, ::FerriteOperators.ParameterJacobianKind) = true
has_internal_state(::Type{<:SimpleCondensedPowerLawRelaxationCache}) = true

function FerriteOperators.invalidate_correctors!(cache::SimpleCondensedPowerLawRelaxationCache)
    FerriteOperators.invalidate_item_states!(cache.correctors)
    FerriteOperators.invalidate_item_states!(cache.param_correctors)
    return nothing
end
# Nothing is stored, so nothing can be stale — the framework default.
FerriteOperators.invalidate_correctors!(::_PLRRecomputing) = nothing

# The element solves no tighter than its own configuration, and no tighter
# than the outer solver asked for.
@inline _plr_tolerance(cache::SimpleCondensedPowerLawRelaxationCache, ctx) =
    _plr_tolerance(cache.local_solver.tolerance, local_solve_tolerance(ctx))
@inline _plr_tolerance(tightest, ::Nothing) = tightest
@inline _plr_tolerance(tightest, requested) = max(tightest, requested)

_plr_params(cache::SimpleCondensedPowerLawRelaxationCache, ::Nothing) = cache.material_parameters
_plr_params(cache::SimpleCondensedPowerLawRelaxationCache, p::NortonRelaxationParameters) = p

# The local operator ∂R/∂q at (u, q) — the single quantity both correctors
# invert, and a closed form in the pair alone. That is what makes the
# `Recompute()` election exact: re-evaluating this at the converged pair is the
# same arithmetic the solve's last iterate ran.
@inline function _plr_dRdq(mat, u, q, γ̃)
    (; α, η, n) = mat
    σ = α * (u - q)
    return 1 + γ̃ * α * n * abs(σ)^(n - 1) / η
end

# The two implicit-function-theorem slopes at the converged pair, `dq/du`
# carrying the solver's chain-rule scalar `w` inside the local inverse.
@inline _plr_dqdu(dRdq, w) = ((dRdq - 1) / dRdq) * w
@inline _plr_dqdθ(mat, u, q, γ̃, dRdq) = (-_plr_dRdθ(mat, u, q, γ̃)) / dRdq

# Local stage problem  R(q) = q − q₀ − γ̃ σ|σ|ⁿ⁻¹/η  with  σ = α(u − q),
# solved by Newton from q₀. R is strictly increasing with R′ ≥ 1 and the flow
# is convex on each side of the origin, so the iteration converges
# monotonically from that start. Returns `(q, dR/dq, iterations, |R|, ok)`;
# `ok = false` at the iteration budget is DATA, not an exception — a device
# kernel can compute it and [`CondensationReport`](@ref).`converged` carries it.
@inline function _plr_local_solve(cache::SimpleCondensedPowerLawRelaxationCache, mat, u, q₀, γ̃, tol)
    (; α, η, n) = mat
    q = oftype(u, q₀)
    maxit = cache.local_solver.max_iterations
    iterations = 0
    while true
        σ    = α * (u - q)
        flow = σ * abs(σ)^(n - 1) / η
        R    = q - q₀ - γ̃ * flow
        dRdq = _plr_dRdq(mat, u, q, γ̃)
        ok   = abs(R) ≤ tol
        (ok || iterations == maxit) && return q, dRdq, iterations, abs(R), ok
        q -= R / dRdq
        iterations += 1
    end
end

# ∂R/∂θ at the converged (q, u), θ = (α, η, n) — the local model's own
# parameter sensitivity, element-internal AD-free (closed form; the local
# model is one scalar equation).
@inline function _plr_dRdθ(mat, u, q, γ̃)
    (; α, η, n) = mat
    σ = α * (u - q)
    absσ = abs(σ)
    powterm = absσ == 0 ? 0.0 : absσ^(n - 1)
    dRdα = -γ̃ / η * n * powterm * (u - q)
    dRdη = γ̃ * σ * powterm / η^2
    dRdn = absσ == 0 ? 0.0 : -γ̃ / η * σ * powterm * log(absσ)
    return SVector(dRdα, dRdη, dRdn)
end

# The corrector ACCESS POINT the `Consistent` kernels call — under `Stored()`
# what `condense_cell!` put in the store, under `Recompute()` the same slopes
# re-derived from the item's current `(u, q)` through the very expressions the
# condensation ran. The kernels below do not know which they got.
_plr_correctors(cache::SimpleCondensedPowerLawRelaxationCache, args) =
    FerriteOperators.item_state(cache.correctors, cellid(args.cell))
_plr_param_correctors(cache::SimpleCondensedPowerLawRelaxationCache, args) =
    FerriteOperators.item_state(cache.param_correctors, cellid(args.cell))

_plr_correctors(cache::_PLRRecomputing, args) =
    _plr_recompute((mat, u, q, γ̃, dRdq, w) -> _plr_dqdu(dRdq, w), cache, args)
_plr_param_correctors(cache::_PLRRecomputing, args) =
    _plr_recompute((mat, u, q, γ̃, dRdq, w) -> _plr_dqdθ(mat, u, q, γ̃, dRdq), cache, args)

# One quadrature-point walk over the converged pair, shared by both slopes.
@inline function _plr_recompute(slope, cache::_PLRRecomputing{NQP}, args) where {NQP}
    mat = _plr_params(cache, args.p)
    γ̃   = stage_scaling(args.ctx)
    w   = cache.weight[]
    cv  = cache.cv
    dₑ  = args.states.u
    qₑ  = args.states.q
    return SVector{NQP}(ntuple(Val(NQP)) do qp
        u = function_value(cv, qp, dₑ)
        q = qₑ[qp]
        slope(mat, u, q, γ̃, _plr_dRdq(mat, u, q, γ̃), w)
    end)
end

"""
    condense_cell!(cache::SimpleCondensedPowerLawRelaxationCache, args, weights) -> CondensationReport

Solve the local Norton-dashpot problem at every quadrature point and write the
trial `q`. Under a [`Stored`](@ref) election the two slopes `dq/du` and
`dq/dθ` are stored alongside; under [`Recompute`](@ref) they are not, and the
kernels re-derive them from the pair this hook converged.
"""
function FerriteOperators.condense_cell!(cache::SimpleCondensedPowerLawRelaxationCache{NQP}, args::CellArgs, weights::NamedTuple) where {NQP}
    cv  = cache.cv
    mat = _plr_params(cache, args.p)
    γ̃   = stage_scaling(args.ctx)
    tol = _plr_tolerance(cache, args.ctx)
    id  = cellid(args.cell)
    w   = get(weights, :u, 1.0)
    cache.weight[] = w

    dₑ     = args.states.u
    qₑ     = args.states.q
    qₑprev = args.states.qprev

    converged        = true
    total_iterations = 0
    worst_iterations = 0
    worst_cell       = 0
    worst_qp         = 0
    worst_residual   = 0.0
    dqdu  = MVector{NQP, Float64}(undef)
    dqdθ  = MVector{NQP, SVector{3, Float64}}(undef)

    @inbounds for qp in 1:NQP
        u = function_value(cv, qp, dₑ)
        q, dRdq, iterations, resid, ok = _plr_local_solve(cache, mat, u, qₑprev[qp], γ̃, tol)
        qₑ[qp]   = q
        dqdu[qp] = _plr_dqdu(dRdq, w)
        dqdθ[qp] = _plr_dqdθ(mat, u, q, γ̃, dRdq)

        converged &= ok
        total_iterations += iterations
        if iterations > worst_iterations
            worst_iterations = iterations
            worst_cell = id
            worst_qp = qp
        end
        worst_residual = max(worst_residual, resid)
    end

    _plr_store_correctors!(cache, id, SVector{NQP}(dqdu), SVector{NQP}(dqdθ))
    return CondensationReport(converged, NQP, total_iterations, worst_iterations, worst_cell, worst_qp, worst_residual, 1.0)
end

function _plr_store_correctors!(cache::SimpleCondensedPowerLawRelaxationCache, id, dqdu, dqdθ)
    FerriteOperators.set_item_state!(cache.correctors, id, dqdu)
    FerriteOperators.set_item_state!(cache.param_correctors, id, dqdθ)
    return nothing
end
_plr_store_correctors!(::_PLRRecomputing, id, dqdu, dqdθ) = nothing

# One concrete entry method per provided kernel (no blanket request method:
# it would satisfy every `hasmethod` probe in the setup-time validation).
assemble_cell!(req::ResidualRequest, cache::SimpleCondensedPowerLawRelaxationCache, args::CellArgs) = _plr_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u, Consistent}, cache::SimpleCondensedPowerLawRelaxationCache, args::CellArgs) = _plr_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u, FrozenQ}, cache::SimpleCondensedPowerLawRelaxationCache, args::CellArgs) = _plr_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest{Consistent}, cache::SimpleCondensedPowerLawRelaxationCache, args::CellArgs) = _plr_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest{FrozenQ}, cache::SimpleCondensedPowerLawRelaxationCache, args::CellArgs) = _plr_assemble!(req, cache, args)

"""
    assemble_cell!(req::JacobianRequest{:q}, cache::SimpleCondensedPowerLawRelaxationCache, args) -> K

∂F/∂q, the LOCAL `ndofs × nqp` block a Schur-complement consumer or a generic
corrector combination wants (see `condense_internal!`'s §6 payoff). A pure
function of `(u, q)` — no store needed, `Consistent` and `FrozenQ` coincide,
since `q` is the seed itself. `K` is sized by the caller: this is a
cell-local, never a global-sweep, quantity (`q`'s dofs are internal to the
cell, so there is no meaningful global scatter target for it).
"""
function assemble_cell!(req::JacobianRequest{:q}, cache::SimpleCondensedPowerLawRelaxationCache, args::CellArgs)
    (; α) = _plr_params(cache, args.p)
    cv = cache.cv
    ndofs = getnbasefunctions(cv)
    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∂r∂q = -α
        for i in 1:ndofs
            req.K[i, qp] += ∂r∂q * shape_value(cv, qp, i) * dΩ
        end
    end
    return req.K
end
provides_analytic(::Type{<:SimpleCondensedPowerLawRelaxationCache}, ::JacobianKind{:q}) = true

const _PLRJacobianLike = Union{JacobianRequest{:u, Consistent}, JacobianRequest{:u, FrozenQ}, JacobianResidualRequest{Consistent}, JacobianResidualRequest{FrozenQ}}
const _PLRFrozenLike   = Union{JacobianRequest{:u, FrozenQ}, JacobianResidualRequest{FrozenQ}}

# Pure evaluation at the FROZEN q the last `condense_internal!` wrote: no
# solve, no write-back. `req isa _PLRFrozenLike` reads `dq/du = 0` (the elected
# partial); otherwise it reads the consistent slope through the corrector
# access point, which under `Stored()` throws, naming the cell, if
# `condense_internal!` never ran for it.
function _plr_assemble!(req::Union{ResidualRequest, _PLRJacobianLike}, cache::SimpleCondensedPowerLawRelaxationCache, args::CellArgs)
    (; κ, α) = _plr_params(cache, args.p)
    cv = cache.cv
    dₑ = args.states.u
    qₑ = args.states.q
    ndofs = getnbasefunctions(cv)

    needs_jac = req isa _PLRJacobianLike
    dqdu = (needs_jac && !(req isa _PLRFrozenLike)) ? _plr_correctors(cache, args) : nothing

    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        u  = function_value(cv, qp, dₑ)
        ∇u = function_gradient(cv, qp, dₑ)
        q  = qₑ[qp]

        if req isa Union{ResidualRequest, JacobianResidualRequest}
            for i in 1:ndofs
                req.r[i] += (κ * (shape_gradient(cv, qp, i) ⋅ ∇u) +
                             α * (u - q) * shape_value(cv, qp, i)) * dΩ
            end
        end
        if needs_jac
            slope = req isa _PLRFrozenLike ? 0.0 : dqdu[qp]
            ∂σ∂u  = α * (1 - slope)
            for i in 1:ndofs
                for j in 1:ndofs
                    req.K[i, j] += (κ * (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) +
                                    ∂σ∂u * shape_value(cv, qp, i) * shape_value(cv, qp, j)) * dΩ
                end
            end
        end
    end
end

# Analytic ∂F/∂θ, θ = (α, η, n): the exchange term's own partial ∂r/∂α|_q
# (κ and n never appear directly in the residual) plus the stored
# ∂F/∂q · dq/dθ correction (∂r/∂q|_(u,θ) = -α·φ, the residual kernel's own
# q-dependence). The payoff of a corrector store: this kind was inadmissible
# for a condensed cache before it existed.
function assemble_cell!(req::ParameterJacobianRequest, cache::SimpleCondensedPowerLawRelaxationCache, args::CellArgs)
    (; α) = _plr_params(cache, args.p)
    cv = cache.cv
    dₑ = args.states.u
    qₑ = args.states.q
    ndofs = getnbasefunctions(cv)
    dqdθ = _plr_param_correctors(cache, args)

    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        u  = function_value(cv, qp, dₑ)
        q  = qₑ[qp]
        ∂r∂α_partial = (u - q)
        ∂r∂q         = -α
        correction   = ∂r∂q * dqdθ[qp]     # SVector{3}: (dα, dη, dn) contributions
        for i in 1:ndofs
            φi = shape_value(cv, qp, i)
            req.B[i, 1] += (∂r∂α_partial + correction[1]) * φi * dΩ
            req.B[i, 2] += correction[2] * φi * dΩ
            req.B[i, 3] += correction[3] * φi * dΩ
        end
    end
    return req.B
end

function setup_element_cache(element_model::SimpleCondensedPowerLawRelaxation, sdh::SubDofHandler)
    qr     = getquadraturerule(element_model.qrc, sdh)
    nqp    = getnquadpoints(qr)
    ip     = Ferrite.getfieldinterpolation(sdh, element_model.field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    ncells = getncells(Ferrite.get_grid(sdh.dh))

    correctors, param_correctors = _plr_corrector_stores(element_model.corrector, nqp, ncells)
    return SimpleCondensedPowerLawRelaxationCache{nqp}(
        element_model.material_parameters,
        CellValues(qr, ip, ip_geo),
        element_model.local_solver,
        correctors,
        param_correctors,
        Ref(1.0),
    )
end

# The election, spent: `Stored()` allocates the two per-item stores,
# `Recompute()` allocates neither.
_plr_corrector_stores(::Stored, nqp, ncells) = (
    ItemStates{SVector{nqp, Float64}}(ncells),
    ItemStates{SVector{nqp, SVector{3, Float64}}}(ncells),
)
_plr_corrector_stores(::Recompute, nqp, ncells) = (nothing, nothing)
