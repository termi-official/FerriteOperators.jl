"""
Material parameters of the relaxing bar used by [`SimpleRelaxingBar`](@ref) and,
as the micro material, by [`SimpleNestedHomogenization`](@ref): equilibrium
modulus `E₀`, relaxing-branch modulus `E₁`, dashpot viscosity `η` and Norton
exponent `n` (`n = 1` degenerates to a linear dashpot).
"""
struct RelaxingBarParameters
    E₀::Float64
    E₁::Float64
    η::Float64
    n::Float64
end
RelaxingBarParameters(; E₀ = 1.0, E₁ = 0.6, η = 1.0, n = 3.0) = RelaxingBarParameters(E₀, E₁, η, n)

@doc raw"""
    SimpleRelaxingBar(material_parameters, qrc, field_name, internal_name;
                      moduli = Float64[], local_solver = LocalNewtonSettings())

A one-dimensional bar with the viscous strain `q` as a condensed
per-quadrature-point internal variable:

```math
r(u, q) = \int_\Omega \sigma(\varepsilon, q)\, \delta\varepsilon \; \mathrm{d}x,
\qquad
\sigma = c\,(E_0 \varepsilon + E_1 (\varepsilon - q)),
\qquad
q = q_{prev} + \tilde\gamma \frac{\sigma_v |\sigma_v|^{n-1}}{\eta},
\quad \sigma_v = c\,E_1 (\varepsilon - q),
```

with `ε = ∂u/∂x` and `c` the cell's entry of `moduli` (`1` for every cell where
`moduli` is empty — the homogeneous bar). The local `q` problem is nonlinear for
`n ≠ 1` and solved by a Newton iteration per quadrature point in
[`condense_cell!`](@ref); [`condense_internal!`](@ref) must run before any
evaluation sweep. `Consistent` reads the stored implicit-function-theorem slope
`dq/dε`, `FrozenQ` drops it.

This is the MICRO element of [`SimpleNestedHomogenization`](@ref), and — on a
macroscopic mesh, with `moduli` empty — the single-material law that element's
homogenized response reduces to.
"""
struct SimpleRelaxingBar <: AbstractCondensedNonlinearIntegrator
    material_parameters::RelaxingBarParameters
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
    internal_name::Symbol
    moduli::Vector{Float64}
    local_solver::LocalNewtonSettings
end
SimpleRelaxingBar(material_parameters, qrc, field_name, internal_name;
        moduli = Float64[], local_solver = LocalNewtonSettings()) =
    SimpleRelaxingBar(material_parameters, qrc, field_name, internal_name, moduli, local_solver)

"""
The cache associated with [`SimpleRelaxingBar`](@ref). `correctors` holds the
per-quadrature-point slope `dq/dε` that [`condense_cell!`](@ref) wrote and the
`Consistent` kernel reads; `moduli` is immutable problem structure, aliased
across workers. Declares [`has_internal_state`](@ref).
"""
struct SimpleRelaxingBarCache{NQP, CV <: CellValues} <: AbstractVolumetricElementCache
    material_parameters::RelaxingBarParameters
    cv::CV
    moduli::Vector{Float64}
    local_solver::LocalNewtonSettings
    correctors::ItemStates{SVector{NQP, Float64}}
end
SimpleRelaxingBarCache{NQP}(mat, cv::CV, m, ls, c) where {NQP, CV} =
    SimpleRelaxingBarCache{NQP, CV}(mat, cv, m, ls, c)

Ferrite.getnquadpoints(e::SimpleRelaxingBarCache) = getnquadpoints(e.cv)
reinit_values!(e::SimpleRelaxingBarCache, cell) = Ferrite.reinit!(e.cv, cell)

# `moduli` is immutable problem structure and `correctors` is item-keyed, so
# both are shared; only the values object is per-worker.
duplicate_for_device(device, cache::SimpleRelaxingBarCache{NQP}) where {NQP} =
    SimpleRelaxingBarCache{NQP}(cache.material_parameters, duplicate_for_device(device, cache.cv),
                                cache.moduli, cache.local_solver, cache.correctors)

get_number_of_internal_dofs_per_element(element_model, cache::SimpleRelaxingBarCache, sdh) =
    [getnquadpoints(cache.cv) for _ in sdh.cellset]

provides_analytic(::Type{<:SimpleRelaxingBarCache}, ::Union{JacobianKind{:u}, JacobianResidualKind}) = true
has_internal_state(::Type{<:SimpleRelaxingBarCache}) = true

FerriteOperators.invalidate_correctors!(cache::SimpleRelaxingBarCache) =
    (FerriteOperators.invalidate_item_states!(cache.correctors); nothing)

# The cell's modulus factor: an empty declaration is the homogeneous bar.
@inline _bar_modulus(cache::SimpleRelaxingBarCache, cellid::Int) =
    isempty(cache.moduli) ? 1.0 : cache.moduli[cellid]

@inline _bar_stress(mat::RelaxingBarParameters, c, ε, q) = c * (mat.E₀ * ε + mat.E₁ * (ε - q))
@inline _bar_viscous_stress(mat::RelaxingBarParameters, c, ε, q) = c * mat.E₁ * (ε - q)

# ∂R/∂q of the local stage problem at (ε, q) — the quantity the local Newton
# steps with and the slope inverts, a closed form in the pair alone.
@inline function _bar_dRdq(mat::RelaxingBarParameters, c, ε, q, γ̃)
    σᵥ = _bar_viscous_stress(mat, c, ε, q)
    return 1 + γ̃ * c * mat.E₁ * mat.n * abs(σᵥ)^(mat.n - 1) / mat.η
end

# ∂R/∂ε = −(∂R/∂q − 1) for this flow law, so the implicit-function-theorem
# slope needs no second derivative.
@inline _bar_dqdε(dRdq, w) = ((dRdq - 1) / dRdq) * w

@inline _bar_tangent(mat::RelaxingBarParameters, c, dqdε) = c * (mat.E₀ + mat.E₁ * (1 - dqdε))

# Local stage problem  R(q) = q − q₀ − γ̃ σᵥ|σᵥ|ⁿ⁻¹/η  with  σᵥ = cE₁(ε − q),
# solved by Newton from q₀. Returns `(q, ∂R/∂q, iterations, |R|, ok)`; `ok =
# false` at the budget is DATA, not an exception.
@inline function _bar_local_solve(cache::SimpleRelaxingBarCache, c, ε, q₀, γ̃)
    mat   = cache.material_parameters
    tol   = cache.local_solver.tolerance
    maxit = cache.local_solver.max_iterations
    q = oftype(ε, q₀)
    iterations = 0
    while true
        σᵥ   = _bar_viscous_stress(mat, c, ε, q)
        R    = q - q₀ - γ̃ * σᵥ * abs(σᵥ)^(mat.n - 1) / mat.η
        dRdq = _bar_dRdq(mat, c, ε, q, γ̃)
        ok   = abs(R) ≤ tol
        (ok || iterations == maxit) && return q, dRdq, iterations, abs(R), ok
        q -= R / dRdq
        iterations += 1
    end
end

# Both elements of this file own a local stage problem scaled by γ̃, so neither
# has a meaningful stationary reading. The macro element only hands the context
# down and never reads the scaling itself, which is the whole difference between
# the two spellings.
@inline function _bar_stage_scaling(ctx, name)
    _require_stage_context(ctx, name)
    return stage_scaling(ctx)
end
@inline function _require_stage_context(ctx, name)
    ctx === nothing && throw(ArgumentError(
        "$name requires a TimeIntegrationContext: its local stage problem scales by " *
        "stage_scaling(ctx)."))
    return nothing
end

"""
    condense_cell!(cache::SimpleRelaxingBarCache, args, weights) -> CondensationReport

Solve the local viscous-strain problem at every quadrature point, write the
trial `q` and store the slope `dq/dε` the `Consistent` kernel reads.
"""
function FerriteOperators.condense_cell!(cache::SimpleRelaxingBarCache{NQP}, args::CellArgs, weights::NamedTuple) where {NQP}
    cv = cache.cv
    γ̃  = _bar_stage_scaling(args.ctx, "SimpleRelaxingBar")
    id = cellid(args.cell)
    c  = _bar_modulus(cache, id)
    w  = get(weights, :u, 1.0)

    converged        = true
    total_iterations = 0
    worst_iterations = 0
    worst_qp         = 0
    worst_residual   = 0.0
    dqdε = MVector{NQP, Float64}(undef)

    @inbounds for qp in 1:NQP
        ε = function_gradient(cv, qp, args.states.u)[1]
        q, dRdq, iterations, resid, ok = _bar_local_solve(cache, c, ε, args.states.qprev[qp], γ̃)
        args.states.q[qp] = q
        dqdε[qp] = _bar_dqdε(dRdq, w)

        converged &= ok
        total_iterations += iterations
        if iterations > worst_iterations
            worst_iterations = iterations
            worst_qp = qp
        end
        worst_residual = max(worst_residual, resid)
    end

    FerriteOperators.set_item_state!(cache.correctors, id, SVector{NQP}(dqdε))
    return CondensationReport(converged, NQP, total_iterations, worst_iterations,
                              worst_iterations > 0 ? id : 0, worst_qp, worst_residual, 1.0)
end

# One concrete entry method per provided kernel (no blanket request method:
# it would satisfy every `hasmethod` probe in the setup-time validation).
assemble_cell!(req::ResidualRequest, cache::SimpleRelaxingBarCache, args::CellArgs) = _bar_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u, Consistent}, cache::SimpleRelaxingBarCache, args::CellArgs) = _bar_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u, FrozenQ}, cache::SimpleRelaxingBarCache, args::CellArgs) = _bar_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest{Consistent}, cache::SimpleRelaxingBarCache, args::CellArgs) = _bar_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest{FrozenQ}, cache::SimpleRelaxingBarCache, args::CellArgs) = _bar_assemble!(req, cache, args)

const _BarJacobianLike = Union{JacobianRequest{:u, Consistent}, JacobianRequest{:u, FrozenQ},
                               JacobianResidualRequest{Consistent}, JacobianResidualRequest{FrozenQ}}
const _BarFrozenLike   = Union{JacobianRequest{:u, FrozenQ}, JacobianResidualRequest{FrozenQ}}

# Pure evaluation at the FROZEN q the last `condense_internal!` wrote: no solve,
# no write-back. The corrector read throws, naming the cell, if
# `condense_internal!` never ran for it.
function _bar_assemble!(req::Union{ResidualRequest, _BarJacobianLike}, cache::SimpleRelaxingBarCache{NQP}, args::CellArgs) where {NQP}
    cv    = cache.cv
    mat   = cache.material_parameters
    c     = _bar_modulus(cache, cellid(args.cell))
    ndofs = getnbasefunctions(cv)

    needs_jac = req isa _BarJacobianLike
    dqdε = (needs_jac && !(req isa _BarFrozenLike)) ?
        FerriteOperators.item_state(cache.correctors, cellid(args.cell)) : nothing

    @inbounds for qp in 1:NQP
        dΩ = getdetJdV(cv, qp)
        ε  = function_gradient(cv, qp, args.states.u)[1]
        q  = args.states.q[qp]

        if req isa Union{ResidualRequest, JacobianResidualRequest}
            σ = _bar_stress(mat, c, ε, q)
            for i in 1:ndofs
                req.r[i] += σ * shape_gradient(cv, qp, i)[1] * dΩ
            end
        end
        if needs_jac
            ∂σ∂ε = _bar_tangent(mat, c, req isa _BarFrozenLike ? 0.0 : dqdε[qp])
            for i in 1:ndofs, j in 1:ndofs
                req.K[i, j] += ∂σ∂ε * shape_gradient(cv, qp, i)[1] * shape_gradient(cv, qp, j)[1] * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::SimpleRelaxingBar, sdh::SubDofHandler)
    qr     = getquadraturerule(element_model.qrc, sdh)
    nqp    = getnquadpoints(qr)
    ip     = Ferrite.getfieldinterpolation(sdh, element_model.field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    ncells = getncells(Ferrite.get_grid(sdh.dh))
    # `moduli` is indexed by cellid in the kernels, so a short declaration would
    # surface as a bounds error on some later cell instead of here.
    isempty(element_model.moduli) || length(element_model.moduli) == ncells || throw(ArgumentError(
        "SimpleRelaxingBar carries $(length(element_model.moduli)) moduli for a grid of $ncells " *
        "cells; declare one per cell, or none at all for the homogeneous bar."))
    return SimpleRelaxingBarCache{nqp}(
        element_model.material_parameters,
        CellValues(qr, ip, ip_geo),
        element_model.moduli,
        element_model.local_solver,
        ItemStates{SVector{nqp, Float64}}(ncells),
    )
end

####################################
## The macro element: a nested multilevel Newton
####################################

"""
    MicroBarProblem

Immutable problem structure of one [`SimpleNestedHomogenization`](@ref)
quadrature point's micro problem — mesh, handler, the driven/free dof split and
the micro domain's measure. Built once in `setup_element_cache` from what the
integrator carries and deliberately ALIASED across workers; nothing here is
written during a sweep.

The driven dofs are the bar's two ends, loaded affinely by the macroscopic
strain (`u_d = ε̄ x_d`), which is what makes the reaction
`σ̄ = Σ_d r_d x_d / |Ω_micro|` the work-conjugate homogenized stress.
"""
struct MicroBarProblem{I, DH <: DofHandler}
    integrator::I
    dh::DH
    driven::Vector{Int}
    free::Vector{Int}
    driven_coordinates::Vector{Float64}
    volume::Float64
end

"""
    MicroBarWorkspace

Per-worker mutable solve workspace of [`SimpleNestedHomogenization`](@ref): the
worker's OWN micro operator — a full `FerriteOperators` operator over the micro
problem, built by `duplicate_for_device` so no two workers share its matrix,
element caches or corrector stores — and the micro state, history and residual
buffers its Newton runs on.
"""
struct MicroBarWorkspace{OP}
    op::OP
    x::Vector{Float64}       # trial micro state, [micro ū; micro q]
    xprev::Vector{Float64}   # committed micro state of the previous macro step
    r::Vector{Float64}       # micro residual
end

# The micro operator is sequential whatever the macro device is: the macro
# device already parallelizes over macro cells, and a worker's micro problem is
# its own.
function _micro_bar_workspace(problem::MicroBarProblem)
    op = setup_operator(AssemblyStrategy(SequentialCPUDevice()), problem.integrator, problem.dh;
                        slots = (:u, :q, :qprev))
    return MicroBarWorkspace(op, zeros(unknown_size(op)), zeros(unknown_size(op)), zeros(residual_size(op)))
end

# One dof per node for the linear Lagrange bar, so the dofs carry coordinates
# and the driven set is the two ends.
function _micro_bar_problem(integrator::SimpleRelaxingBar, nelements::Int)
    nelements ≥ 2 || throw(ArgumentError(
        "A micro bar needs at least two elements: with one element both dofs are driven by the " *
        "macroscopic strain and there is no micro problem left to solve."))
    grid = generate_grid(Line, (nelements,))
    dh   = DofHandler(grid)
    add!(dh, integrator.field_name, Lagrange{RefLine, 1}())
    close!(dh)

    x = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        for (i, d) in pairs(celldofs(cell))
            x[d] = getcoordinates(cell)[i][1]
        end
    end
    xmin, xmax = extrema(x)
    driven = findall(xi -> xi == xmin || xi == xmax, x)
    free   = setdiff(1:ndofs(dh), driven)
    return MicroBarProblem(integrator, dh, driven, free, x[driven], xmax - xmin)
end

@doc raw"""
    SimpleNestedHomogenization(micro, micro_elements, qrc, field_name, internal_name;
                               local_solver = LocalNewtonSettings())

A macroscopic bar whose stress at every quadrature point is the homogenized
response of a MICRO finite element problem that itself carries condensed
internal variables — the two-stage protocol nested inside itself:

```math
r(\bar u, \bar q) = \int_{\bar\Omega} \bar\sigma\, \delta\bar\varepsilon \;\mathrm{d}x,
\qquad
\bar\sigma = \frac{1}{|\Omega_{micro}|} \sum_{d} r_d(x)\, x_d ,
```

where `x` solves the micro problem of `micro` (a [`SimpleRelaxingBar`](@ref) on
`micro_elements` elements) driven by `u_d = ε̄ x_d` on its two ends. Phase one
([`condense_cell!`](@ref)) runs that micro Newton — whose every iteration
condenses the micro problem's own internal variables first — and forms the
homogenized tangent by the implicit function theorem on the converged micro
tangent, i.e. its Schur complement onto the driven dofs. Phase two is a pure
evaluation: the residual reads the stored micro state, the Jacobian the stored
homogenized tangent.

The whole micro state `[micro ū; micro q]` of every macro quadrature point rides
the macro `[ū; q]` vector, so trial write-back, [`rollback_state!`](@ref) and
[`commit_state!`](@ref) carry it with no second mechanism.
`local_solver` budgets the MICRO NEWTON; the micro material's own local solver
budgets the level below it, and a failure at either level folds into this
element's [`CondensationReport`](@ref).

Analytic-only, for both [`CorrectionMode`](@ref)s: no generic or AD route
reaches through two levels — a kernel that runs another operator's sweeps is not
eltype-generic — and the sweep's parameter bag is not passed down, the micro
problem's configuration being the micro integrator's own.
"""
struct SimpleNestedHomogenization <: AbstractCondensedNonlinearIntegrator
    micro::SimpleRelaxingBar
    micro_elements::Int
    # Every integrator needs these
    qrc::QuadratureRuleCollection
    field_name::Symbol
    internal_name::Symbol
    local_solver::LocalNewtonSettings
end
SimpleNestedHomogenization(micro, micro_elements, qrc, field_name, internal_name;
        local_solver = LocalNewtonSettings()) =
    SimpleNestedHomogenization(micro, micro_elements, qrc, field_name, internal_name, local_solver)

"""
The cache associated with [`SimpleNestedHomogenization`](@ref), carrying one
field per storage class: `micro` is immutable problem structure (aliased),
`workspace` the per-worker micro operator and its buffers (duplicated), and
`tangents` the per-item [`ItemStates`](@ref) store holding each quadrature
point's `(frozen, consistent)` homogenized tangent pair (aliased; item slots are
disjoint per worker). Declares [`has_internal_state`](@ref).
"""
struct SimpleNestedHomogenizationCache{NQP, CV <: CellValues, MP, MW} <: AbstractVolumetricElementCache
    cv::CV
    micro::MP
    workspace::MW
    local_solver::LocalNewtonSettings
    tangents::ItemStates{SVector{NQP, SVector{2, Float64}}}
end
SimpleNestedHomogenizationCache{NQP}(cv::CV, m::MP, w::MW, ls, t) where {NQP, CV, MP, MW} =
    SimpleNestedHomogenizationCache{NQP, CV, MP, MW}(cv, m, w, ls, t)

Ferrite.getnquadpoints(e::SimpleNestedHomogenizationCache) = getnquadpoints(e.cv)
reinit_values!(e::SimpleNestedHomogenizationCache, cell) = Ferrite.reinit!(e.cv, cell)

function duplicate_for_device(device, cache::SimpleNestedHomogenizationCache{NQP}) where {NQP}
    return SimpleNestedHomogenizationCache{NQP}(
        duplicate_for_device(device, cache.cv),
        cache.micro,                            # immutable structure: aliased
        _micro_bar_workspace(cache.micro),      # a micro operator of this worker's own
        cache.local_solver,
        cache.tangents,                         # item-keyed: aliased
    )
end

# Route A: the macro internal block carries every quadrature point's whole micro
# state, so the counts vary with the micro problem rather than with the macro
# field.
get_number_of_internal_dofs_per_element(element_model, cache::SimpleNestedHomogenizationCache, sdh) =
    [getnquadpoints(cache.cv) * length(cache.workspace.x) for _ in sdh.cellset]

provides_analytic(::Type{<:SimpleNestedHomogenizationCache}, ::Union{JacobianKind{:u}, JacobianResidualKind}) = true
has_internal_state(::Type{<:SimpleNestedHomogenizationCache}) = true

FerriteOperators.invalidate_correctors!(cache::SimpleNestedHomogenizationCache) =
    (FerriteOperators.invalidate_item_states!(cache.tangents); nothing)

# The micro operator's slots, over the worker's own buffers. `x` and `xprev` are
# never reassigned, so one spelling stays valid for every call.
_micro_states(ws::MicroBarWorkspace) = (u = ws.x, q = InternalSource(ws.x), qprev = InternalSource(ws.xprev))

# A dense block of the micro tangent. The micro system is a handful of dofs, so
# the readable extraction costs less than the machinery that would avoid it.
_dense_block(A, rows, cols) = [A[i, j] for i in rows, j in cols]

# The macroscopic strain drives the bar's ends affinely; everything else is the
# committed micro state, which makes phase one a pure function of the trial `ū`
# and the committed history — warm-starting from the CURRENT trial slice would
# make it path-dependent instead.
function _load_micro_state!(cache::SimpleNestedHomogenizationCache, ε̄, xprev_slice)
    (; x, xprev) = cache.workspace
    (; driven, driven_coordinates) = cache.micro
    xprev .= xprev_slice
    x .= xprev
    @inbounds for (k, d) in pairs(driven)
        x[d] = ε̄ * driven_coordinates[k]
    end
    return x
end

# The micro Newton on the free dofs: every iteration solves the micro problem's
# own internal variables first (`condense_internal!`) and then linearizes at
# that state, so the matrix it steps with is the CONDENSED micro tangent.
# Returns `(iterations, ‖r_free‖, ok)` and leaves the converged state in `x`,
# the micro residual in `r` and the condensed micro tangent in `op.J`.
function _micro_newton!(cache::SimpleNestedHomogenizationCache, ctx)
    (; op, x, r) = cache.workspace
    free   = cache.micro.free
    states = _micro_states(cache.workspace)
    maxit  = cache.local_solver.max_iterations
    tol    = cache.local_solver.tolerance

    iterations = 0
    while true
        report = condense_internal!(op, states, nothing, ctx)
        update_linearization!(op, r, states, nothing, ctx)
        resid = norm(view(r, free))
        (resid ≤ tol || iterations == maxit) && return iterations, resid, (resid ≤ tol) & report.converged
        view(x, free) .-= _dense_block(op.J, free, free) \ view(r, free)
        iterations += 1
    end
end

# The homogenized stress at the state `x` currently holds — a pure evaluation of
# the micro residual, no solve and no write-back, which is what makes it legal
# in a phase-two kernel.
function _homogenized_stress(cache::SimpleNestedHomogenizationCache, ctx)
    (; op, r) = cache.workspace
    (; driven, driven_coordinates, volume) = cache.micro
    evaluate!(op, r, _micro_states(cache.workspace), nothing, ctx)
    return dot(view(r, driven), driven_coordinates) / volume
end

# The homogenized tangents at the converged micro state, from the condensed
# micro tangent `op.J` left by `_micro_newton!`:
#
#   dσ̄/dε̄ = xᵈ ⋅ (K_dd − K_df K_ff⁻¹ K_fd) xᵈ / |Ω_micro|
#
# — the implicit function theorem on the micro problem's own equilibrium,
# composed with the one already inside `K` (the micro internal variables'
# response). The frozen-q partial is the same contraction over the micro
# problem's FROZEN tangent, which is why the correction mode composes across the
# two levels; `w` is the solver's chain-rule scalar, and it multiplies exactly
# the part the elimination adds.
function _homogenized_tangents(cache::SimpleNestedHomogenizationCache, w, ctx)
    (; op) = cache.workspace
    (; driven, free, driven_coordinates, volume) = cache.micro
    xᵈ = driven_coordinates
    # The factorization of the converged micro tangent's free block, the one
    # both eliminations stand on; the constructor's two-element floor is what
    # guarantees there is a free block at all.
    F = lu(_dense_block(op.J, free, free))
    S = _dense_block(op.J, driven, driven) -
        _dense_block(op.J, driven, free) * (F \ _dense_block(op.J, free, driven))
    consistent = dot(xᵈ, S * xᵈ) / volume

    # `op.J` is this worker's scratch and the consistent block is already read
    # out of it, so the frozen sweep may overwrite it.
    assemble_slot_jacobian!(op.J, op, JacobianKind{:u, FrozenQ}(), _micro_states(cache.workspace), nothing, ctx)
    frozen = dot(xᵈ, _dense_block(op.J, driven, driven) * xᵈ) / volume
    return SVector(frozen, frozen + w * (consistent - frozen))
end

"""
    condense_cell!(cache::SimpleNestedHomogenizationCache, args, weights) -> CondensationReport
    condense_cell!(cache::SimpleNestedHomogenizationCache, args, ::Nothing) -> CondensationReport

Run every quadrature point's micro Newton to convergence, write the converged
micro state into the macro `[ū; q]` tail and store the homogenized tangents. The
residual-only election skips the two tangent sweeps and stores nothing; the
micro states it writes are bit-identical, the election governing only what is
formed after the solve.

The report is this level's: `solves` counts the macro quadrature points,
`iterations` the micro Newton iterations, `worst_residual` the largest micro
equilibrium residual at exit. A micro condensation that did not converge makes
the macro-local problem unconverged, so a failure at either level surfaces here.
"""
FerriteOperators.condense_cell!(cache::SimpleNestedHomogenizationCache, args::CellArgs, weights::NamedTuple) =
    _nested_condense!(cache, args, get(weights, :u, 1.0))
FerriteOperators.condense_cell!(cache::SimpleNestedHomogenizationCache, args::CellArgs, ::Nothing) =
    _nested_condense!(cache, args, nothing)

function _nested_condense!(cache::SimpleNestedHomogenizationCache{NQP}, args::CellArgs, w) where {NQP}
    cv = cache.cv
    id = cellid(args.cell)
    nmicro = length(cache.workspace.x)
    qₑ     = reshape(args.states.q, (nmicro, NQP))
    qₑprev = reshape(args.states.qprev, (nmicro, NQP))
    # The stage scaling is read one level down, by the micro material's own
    # local problem; this level only guarantees there is a context to read.
    _require_stage_context(args.ctx, "SimpleNestedHomogenization")

    converged        = true
    total_iterations = 0
    worst_iterations = 0
    worst_qp         = 0
    worst_residual   = 0.0
    tangents = MVector{NQP, SVector{2, Float64}}(undef)

    @inbounds for qp in 1:NQP
        ε̄ = function_gradient(cv, qp, args.states.u)[1]
        _load_micro_state!(cache, ε̄, view(qₑprev, :, qp))
        iterations, resid, ok = _micro_newton!(cache, args.ctx)
        (@view qₑ[:, qp]) .= cache.workspace.x
        w === nothing || (tangents[qp] = _homogenized_tangents(cache, w, args.ctx))

        converged &= ok
        total_iterations += iterations
        if iterations > worst_iterations
            worst_iterations = iterations
            worst_qp = qp
        end
        worst_residual = max(worst_residual, resid)
    end

    w === nothing || FerriteOperators.set_item_state!(cache.tangents, id, SVector{NQP}(tangents))
    return CondensationReport(converged, NQP, total_iterations, worst_iterations,
                              worst_iterations > 0 ? id : 0, worst_qp, worst_residual, 1.0)
end

assemble_cell!(req::ResidualRequest, cache::SimpleNestedHomogenizationCache, args::CellArgs) = _nested_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u, Consistent}, cache::SimpleNestedHomogenizationCache, args::CellArgs) = _nested_assemble!(req, cache, args)
assemble_cell!(req::JacobianRequest{:u, FrozenQ}, cache::SimpleNestedHomogenizationCache, args::CellArgs) = _nested_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest{Consistent}, cache::SimpleNestedHomogenizationCache, args::CellArgs) = _nested_assemble!(req, cache, args)
assemble_cell!(req::JacobianResidualRequest{FrozenQ}, cache::SimpleNestedHomogenizationCache, args::CellArgs) = _nested_assemble!(req, cache, args)

const _NestedJacobianLike = Union{JacobianRequest{:u, Consistent}, JacobianRequest{:u, FrozenQ},
                                  JacobianResidualRequest{Consistent}, JacobianResidualRequest{FrozenQ}}
const _NestedFrozenLike   = Union{JacobianRequest{:u, FrozenQ}, JacobianResidualRequest{FrozenQ}}

# Pure evaluation at the micro state the last `condense_internal!` wrote: the
# stress re-evaluates the micro residual there (no solve), the tangent is read
# from the store, which throws — naming the cell — if the condensation never ran.
function _nested_assemble!(req::Union{ResidualRequest, _NestedJacobianLike},
                           cache::SimpleNestedHomogenizationCache{NQP}, args::CellArgs) where {NQP}
    cv     = cache.cv
    nmicro = length(cache.workspace.x)
    qₑ     = reshape(args.states.q, (nmicro, NQP))
    ndofs  = getnbasefunctions(cv)

    needs_jac = req isa _NestedJacobianLike
    tangents = needs_jac ? FerriteOperators.item_state(cache.tangents, cellid(args.cell)) : nothing

    @inbounds for qp in 1:NQP
        dΩ = getdetJdV(cv, qp)
        ε̄  = function_gradient(cv, qp, args.states.u)[1]

        if req isa Union{ResidualRequest, JacobianResidualRequest}
            _load_micro_state!(cache, ε̄, view(qₑ, :, qp))
            σ̄ = _homogenized_stress(cache, args.ctx)
            for i in 1:ndofs
                req.r[i] += σ̄ * shape_gradient(cv, qp, i)[1] * dΩ
            end
        end
        if needs_jac
            ∂σ̄∂ε̄ = tangents[qp][req isa _NestedFrozenLike ? 1 : 2]
            for i in 1:ndofs, j in 1:ndofs
                req.K[i, j] += ∂σ̄∂ε̄ * shape_gradient(cv, qp, i)[1] * shape_gradient(cv, qp, j)[1] * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::SimpleNestedHomogenization, sdh::SubDofHandler)
    qr     = getquadraturerule(element_model.qrc, sdh)
    nqp    = getnquadpoints(qr)
    ip     = Ferrite.getfieldinterpolation(sdh, element_model.field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    ncells = getncells(Ferrite.get_grid(sdh.dh))
    micro  = _micro_bar_problem(element_model.micro, element_model.micro_elements)
    return SimpleNestedHomogenizationCache{nqp}(
        CellValues(qr, ip, ip_geo),
        micro,
        _micro_bar_workspace(micro),
        element_model.local_solver,
        ItemStates{SVector{nqp, SVector{2, Float64}}}(ncells),
    )
end
