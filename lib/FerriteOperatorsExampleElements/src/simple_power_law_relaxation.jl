"""
Material parameters of the power-law relaxation model used by
[`SimpleCondensedPowerLawRelaxation`](@ref): the diffusivity `κ`, the exchange
modulus `α` between field and internal state, the dashpot viscosity `η` and
the Norton exponent `n`. `n = 1` degenerates to a linear dashpot.
"""
@kwdef struct NortonRelaxationParameters
    κ::Float64 = 1.0
    α::Float64 = 1.0
    η::Float64 = 1.0
    n::Float64 = 3.0
end

"""
    LocalNewtonSettings(; max_iterations = 25, tolerance = 1.0e-12)

Static configuration of the element-local Newton solver of
[`SimpleCondensedPowerLawRelaxation`](@ref): the per-quadrature-point
iteration budget, and the tightest absolute local residual tolerance the
element ever solves to. A solver may only LOOSEN that tolerance, per
[`local_solve_tolerance`](@ref). Exceeding the budget throws
[`LocalSolveNotConvergedError`](@ref).
"""
@kwdef struct LocalNewtonSettings
    max_iterations::Int = 25
    tolerance::Float64  = 1.0e-12
end

"""
    LocalSolveStatistics()

Mutable accumulator of element-local Newton solves: how many ran, their total
and worst per-quadrature-point iteration count, and the `(cell, qp)` attaining
the worst count. One instance per worker, created by `duplicate_for_device`;
[`local_solve_statistics`](@ref) merges them, `reset_local_solve_statistics!`
clears them.

Counts local solves PERFORMED, not quadrature points visited: a Jacobian sweep
and a residual sweep over the same cells count twice.
"""
mutable struct LocalSolveStatistics
    solves::Int
    iterations::Int
    worst_iterations::Int
    worst_cell::Int
    worst_qp::Int
end
LocalSolveStatistics() = LocalSolveStatistics(0, 0, 0, 0, 0)

@inline function record_local_solve!(s::LocalSolveStatistics, iterations::Int, cell::Int, qp::Int)
    s.solves += 1
    s.iterations += iterations
    if iterations > s.worst_iterations
        s.worst_iterations = iterations
        s.worst_cell = cell
        s.worst_qp = qp
    end
    return nothing
end

"""
    LocalSolveNotConvergedError(cell, qp, iterations, residual, tolerance)

Thrown when the element-local Newton solve of
[`SimpleCondensedPowerLawRelaxation`](@ref) exhausts its iteration budget at
one quadrature point. An exception is the CPU spelling of local
non-convergence; it cannot cross a device boundary.
"""
struct LocalSolveNotConvergedError{R, T} <: Exception
    cell::Int
    qp::Int
    iterations::Int
    residual::R
    tolerance::T
end

function Base.showerror(io::IO, e::LocalSolveNotConvergedError)
    print(io, "LocalSolveNotConvergedError: local Newton solve at quadrature point $(e.qp) of ",
        "cell $(e.cell) reached $(e.iterations) iterations at |R| = $(e.residual), ",
        "requested tolerance $(e.tolerance).")
    return
end

"""
    InexactLocalSolveContext(inner, local_tolerance)

Per-sweep context decorating `inner` with the local residual tolerance the
outer solver asks element-local problems to solve to this sweep — the
inexact-Newton forcing term, e.g. `ηₖ‖F(uₖ)‖`. Framework context handling
passes through to `inner` ([`evaluation_time`](@ref), `with_time`); elements
read the tolerance through [`local_solve_tolerance`](@ref).
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
    SimpleCondensedPowerLawRelaxation

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
started at `q_prev`; the previous state arrives through the `uprev` slot and
the trial state is written back into the element-local `u` buffer per the
condensation contract. The Jacobian kernels carry the implicit-function-theorem
slope `dq/du` of that local solve, so the element is admissible under the
internal-state rule.

Two channels connect the local solver to the outer one:

  * outer → inner: the requested local tolerance, read from the context via
    [`local_solve_tolerance`](@ref) and clamped from below by the element's own
    [`LocalNewtonSettings`](@ref).
  * inner → outer: [`local_solve_statistics`](@ref) over the operator, plus a
    thrown [`LocalSolveNotConvergedError`](@ref) when a quadrature point
    exhausts the iteration budget.
"""
struct SimpleCondensedPowerLawRelaxation <: AbstractCondensedNonlinearIntegrator
    material_parameters::NortonRelaxationParameters
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
    internal_name::Symbol
    local_solver::LocalNewtonSettings
end
SimpleCondensedPowerLawRelaxation(material_parameters, qrc, field_name, internal_name;
        local_solver = LocalNewtonSettings()) =
    SimpleCondensedPowerLawRelaxation(material_parameters, qrc, field_name, internal_name, local_solver)

"""
The cache associated with [`SimpleCondensedPowerLawRelaxation`](@ref). Beyond
the element-local dof ranges of the field and of the condensed internal state
it carries the local solver's static configuration and this worker's
[`LocalSolveStatistics`](@ref), and declares [`has_internal_state`](@ref).
"""
struct SimpleCondensedPowerLawRelaxationCache{CV <: CellValues} <: AbstractVolumetricElementCache
    material_parameters::NortonRelaxationParameters
    field_range::UnitRange{Int}
    internal_range::UnitRange{Int}
    cv::CV
    local_solver::LocalNewtonSettings
    statistics::LocalSolveStatistics
end

Ferrite.getnquadpoints(e::SimpleCondensedPowerLawRelaxationCache) = getnquadpoints(e.cv)
reinit_values!(e::SimpleCondensedPowerLawRelaxationCache, cell) = Ferrite.reinit!(e.cv, cell)

# Every worker gets its own statistics: the accumulator is mutated by the
# kernel, so sharing one across workers would race.
function duplicate_for_device(device, cache::SimpleCondensedPowerLawRelaxationCache)
    return SimpleCondensedPowerLawRelaxationCache(
        cache.material_parameters,
        cache.field_range,
        cache.internal_range,
        duplicate_for_device(device, cache.cv),
        cache.local_solver,
        LocalSolveStatistics(),
    )
end

function get_number_of_internal_dofs_per_element(element_model, cache::SimpleCondensedPowerLawRelaxationCache, sdh)
    return [length(cache.internal_range) for i in sdh.cellset]
end

provides_analytic(::Type{<:SimpleCondensedPowerLawRelaxationCache}, ::Union{JacobianKind, JacobianResidualKind}) = true
has_internal_state(::Type{<:SimpleCondensedPowerLawRelaxationCache}) = true

# Local stage problem  R(q) = q − q₀ − γ̃ σ|σ|ⁿ⁻¹/η  with  σ = α(u − q),
# solved by Newton from q₀. R is strictly increasing with R′ ≥ 1 and the flow
# is convex on each side of the origin, so the iteration converges
# monotonically from that start. Returns `(q, dq/du, iterations)`, the slope
# being the implicit-function-theorem derivative the consistent tangent needs.
@inline function _plr_local_solve(cache::SimpleCondensedPowerLawRelaxationCache, u, q₀, γ̃, tol, cell, qp)
    (; α, η, n) = cache.material_parameters
    q = oftype(u, q₀)
    iterations = 0
    while true
        σ    = α * (u - q)
        flow = σ * abs(σ)^(n - 1) / η
        R    = q - q₀ - γ̃ * flow
        dRdq = 1 + γ̃ * α * n * abs(σ)^(n - 1) / η
        abs(R) ≤ tol && return q, (dRdq - 1) / dRdq, iterations
        iterations == cache.local_solver.max_iterations &&
            throw(LocalSolveNotConvergedError(cell, qp, iterations, abs(R), tol))
        q -= R / dRdq
        iterations += 1
    end
end

# The element solves no tighter than its own configuration, and no tighter
# than the outer solver asked for.
@inline _plr_tolerance(cache::SimpleCondensedPowerLawRelaxationCache, ctx) =
    _plr_tolerance(cache.local_solver.tolerance, local_solve_tolerance(ctx))
@inline _plr_tolerance(tightest, ::Nothing) = tightest
@inline _plr_tolerance(tightest, requested) = max(tightest, requested)

# One concrete entry method per provided kernel (no blanket request method:
# it would satisfy every `hasmethod` probe in the setup-time validation).
assemble_cell!(req::ResidualRequest, cache::SimpleCondensedPowerLawRelaxationCache, args) = _plr_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u}, cache::SimpleCondensedPowerLawRelaxationCache, args) = _plr_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest, cache::SimpleCondensedPowerLawRelaxationCache, args) = _plr_assemble!(req, cache, args)

function _plr_assemble!(req::Union{ResidualRequest, JacobianRequest{:u}, JacobianResidualRequest}, cache::SimpleCondensedPowerLawRelaxationCache, args)
    (; field_range, internal_range, cv, statistics) = cache
    (; κ, α) = cache.material_parameters
    γ̃   = stage_scaling(args.ctx)
    tol = _plr_tolerance(cache, args.ctx)
    id  = cellid(args.cell)
    uₑ     = args.states.u
    uₑprev = args.states.uprev

    ndofs = getnbasefunctions(cv)
    dₑ     = @view uₑ[field_range]
    qₑ     = @view uₑ[internal_range]
    qₑprev = @view uₑprev[internal_range]

    @inbounds for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        u  = function_value(cv, qp, dₑ)
        ∇u = function_gradient(cv, qp, dₑ)

        q, dqdu, iterations = _plr_local_solve(cache, u, qₑprev[qp], γ̃, tol, id, qp)
        # Trial write-back into the element-local buffer (the condensation
        # contract; the framework's store step propagates it into global u).
        qₑ[qp] = q
        record_local_solve!(statistics, iterations, id, qp)

        if req isa Union{ResidualRequest, JacobianResidualRequest}
            for i in 1:ndofs
                req.r[i] += (κ * (shape_gradient(cv, qp, i) ⋅ ∇u) +
                             α * (u - q) * shape_value(cv, qp, i)) * dΩ
            end
        end
        if req isa Union{JacobianRequest{:u}, JacobianResidualRequest}
            ∂σ∂u = α * (1 - dqdu)
            for i in 1:ndofs
                for j in 1:ndofs
                    req.K[i, j] += (κ * (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) +
                                    ∂σ∂u * shape_value(cv, qp, i) * shape_value(cv, qp, j)) * dΩ
                end
            end
        end
    end
end

function setup_element_cache(element_model::SimpleCondensedPowerLawRelaxation, sdh::SubDofHandler)
    qr     = getquadraturerule(element_model.qrc, sdh)
    nqp    = getnquadpoints(qr)
    ip     = Ferrite.getfieldinterpolation(sdh, element_model.field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)

    field_range    = dof_range(sdh, element_model.field_name)
    internal_range = (field_range[end]+1):(field_range[end]+nqp)

    return SimpleCondensedPowerLawRelaxationCache(
        element_model.material_parameters,
        field_range,
        internal_range,
        CellValues(qr, ip, ip_geo),
        element_model.local_solver,
        LocalSolveStatistics(),
    )
end

function get_element_internal_index_range(cell, ivh, element::SimpleCondensedPowerLawRelaxationCache)
    nqp    = getnquadpoints(element.cv)
    offset = internal_variable_offset(ivh, cellid(cell))
    return (offset+1):(offset+nqp)
end

function load_element_unknowns!(uₑ, u, cell, ivh, element::SimpleCondensedPowerLawRelaxationCache)
    internal_range                      = get_element_internal_index_range(cell, ivh, element)
    @views uₑ[element.field_range]    .= u[celldofs(cell)]
    @views uₑ[element.internal_range] .= u[internal_range]
    return nothing
end

function store_condensed_element_unknowns!(uₑ, u, cell, ivh, element::SimpleCondensedPowerLawRelaxationCache)
    internal_range     = get_element_internal_index_range(cell, ivh, element)
    u[internal_range] .= uₑ[element.internal_range]
    return nothing
end

allocate_element_unknown_vector(element::SimpleCondensedPowerLawRelaxationCache, sdh) =
    zeros(getnbasefunctions(element.cv) + getnquadpoints(element.cv))

# FerriteOperators exposes no accessor for the per-worker element caches, so
# the statistics helpers walk the engine's subdomain caches.
function _relaxation_caches(op)
    caches = SimpleCondensedPowerLawRelaxationCache[]
    for sc in op.engine.subdomain_caches, ws in sc.device_cache
        ws.element isa SimpleCondensedPowerLawRelaxationCache && push!(caches, ws.element)
    end
    return caches
end

"""
    local_solve_statistics(op) -> LocalSolveStatistics

The element-local Newton statistics of every
[`SimpleCondensedPowerLawRelaxation`](@ref) cache in `op`, merged over workers
and subdomains, accumulated since the last
[`reset_local_solve_statistics!`](@ref).
"""
function local_solve_statistics(op)
    total = LocalSolveStatistics()
    for cache in _relaxation_caches(op)
        s = cache.statistics
        total.solves     += s.solves
        total.iterations += s.iterations
        if s.worst_iterations > total.worst_iterations
            total.worst_iterations = s.worst_iterations
            total.worst_cell       = s.worst_cell
            total.worst_qp         = s.worst_qp
        end
    end
    return total
end

"""
    reset_local_solve_statistics!(op)

Clear the per-worker [`LocalSolveStatistics`](@ref) of every
[`SimpleCondensedPowerLawRelaxation`](@ref) cache in `op`.
"""
function reset_local_solve_statistics!(op)
    for cache in _relaxation_caches(op)
        s = cache.statistics
        s.solves = 0
        s.iterations = 0
        s.worst_iterations = 0
        s.worst_cell = 0
        s.worst_qp = 0
    end
    return nothing
end
