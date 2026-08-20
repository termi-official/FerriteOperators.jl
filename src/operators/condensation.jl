####################################
## Condensation phase
####################################
#
# The phase that turns a condensed element's per-kernel local solve into one
# up-front traversal: `condense_internal!` solves every quadrature point's
# local problem, writes the trial `q`, and stores each item's corrector; every
# evaluation sweep afterwards is a pure function of `(u, q, p, t)` at frozen
# `q`. See the design's structural statement:
#
#     W = Σₛ wₛ ∂F/∂s|_q  +  ∂F/∂q · dq/du
#
# — the existing weighted-Jacobian machinery folding frozen-q partials, plus
# ONE additive correction block. `CorrectionMode` (`Consistent`/`FrozenQ`,
# requests.jl) names which of the two a Jacobian-shaped sweep computes.

"""
    CondensationReport{T}

Per-item summary of a [`condense_internal!`](@ref) sweep, isbits by
construction so the same mechanism serves a device sweep. `converged`
replaces a thrown non-convergence exception with data a caller can act on
without crossing a device boundary.

| field | monoid op | meaning |
|---|---|---|
| `converged` | `&` | did every local problem reach its tolerance |
| `solves` | `+` | local problems attempted |
| `iterations` | `+` | total inner iterations |
| `worst_iterations` | `max` | worst per-quadrature-point iteration count |
| `worst_cell` | argmax carrier | the cell attaining `worst_iterations` |
| `worst_qp` | argmax carrier | the quadrature point attaining `worst_iterations` |
| `worst_residual` | `max` | largest local-residual magnitude at exit |
| `dt_factor` | `min` | 1 means "no reduction requested"; < 1 is a request |

Reports combine with `+`: a commutative monoid except the argmax tie-break,
which keeps the FIRST contribution in fold order — the same determinism
statement [`fold_items`](@ref) already makes for a fixed worker count. The
identity is `zero(CondensationReport{T})`.
"""
struct CondensationReport{T}
    converged::Bool
    solves::Int
    iterations::Int
    worst_iterations::Int
    worst_cell::Int
    worst_qp::Int
    worst_residual::T
    dt_factor::T
end

Base.zero(::Type{CondensationReport{T}}) where {T} =
    CondensationReport{T}(true, 0, 0, 0, 0, 0, zero(T), one(T))

function Base.:+(a::CondensationReport{T}, b::CondensationReport{T}) where {T}
    take_b = b.worst_iterations > a.worst_iterations
    return CondensationReport{T}(
        a.converged & b.converged,
        a.solves + b.solves,
        a.iterations + b.iterations,
        take_b ? b.worst_iterations : a.worst_iterations,
        take_b ? b.worst_cell : a.worst_cell,
        take_b ? b.worst_qp : a.worst_qp,
        max(a.worst_residual, b.worst_residual),
        min(a.dt_factor, b.dt_factor),
    )
end

"""
    condense_cell!(cache, args::CellArgs, weights::NamedTuple) -> CondensationReport

Element hook run once per item by [`condense_internal!`](@ref): solve every
quadrature point's local problem, write the trial state into `args.states.q`
(never into `args.states.u`), and store the corrector — whatever compact
per-quadrature-point quantity the cache's own `Consistent` kernel later reads
(Tier 1), or the completed local block a generic combination would need
(Tier 2). `weights` are the solver's chain-rule scalars for reconstructed
slots participating in the local model (e.g. a rate slot under Newmark),
chained into the corrector INSIDE the local inverse — see the design's
Newmark/multilevel-Newton witness.

There is no default; only condensed elements implement it.
"""
function condense_cell! end

"""
    invalidate_correctors!(cache)

Drop every item's condensation corrector, so the cache's `Consistent` kernel
throws on its next read until [`condense_internal!`](@ref) runs again. The
default is a no-op; only caches carrying a corrector store override it.
Called by [`rollback_state!`](@ref).
"""
invalidate_correctors!(cache) = nothing

"""
    CondensationKind{T}(weights)

The kind [`condense_internal!`](@ref) rides: a `FunctionalFamily` kind whose
kernel ALSO writes back — the one combination the three built-in driver
bodies ([`primal_cell_sweep!`](@ref), [`sensitivity_cell_sweep!`](@ref),
[`functional_cell_sweep`](@ref)) don't have. `weights` are
[`condense_cell!`](@ref)'s per-slot scalars; `T` fixes the report's
[`functional_value_type`](@ref) so a device reduction can preallocate typed
partials.
"""
struct CondensationKind{T, W <: NamedTuple}
    weights::W
end
CondensationKind(weights::NamedTuple) = CondensationKind{Float64, typeof(weights)}(weights)

functional_value_type(::CondensationKind{T}) where {T} = CondensationReport{T}
sweep_family(::Type{<:CondensationKind}) = FunctionalFamily()
# Served by `condense_cell!`, not `assemble_cell!` — there is no cell request
# to validate, exactly like `FunctionalKind`.
has_cell_request(::Type{<:CondensationKind}) = false

execute_kind!(kind::CondensationKind, task, ws) = condensation_cell_sweep!(kind, task, ws)

# The `:q` slot is condense_internal!'s write-back target, so it must be
# present and InternalSource-backed — a plain vector or an AffineRate source
# would give the write-back nowhere meaningful to land.
function _q_source(states::NamedTuple{names}) where {names}
    :q in names || throw(ArgumentError(
        "condense_internal! needs a `:q` slot in `states` (sourced by `InternalSource`); " *
        "got slots $names."))
    src = states.q
    src isa InternalSource || throw(ArgumentError(
        "The `:q` slot must be sourced by `InternalSource(u)` for condense_internal! to have " *
        "a global vector to write the trial state into, got $(typeof(src))."))
    return src
end

"""
    condensation_cell_sweep!(kind::CondensationKind, task, ws) -> CondensationReport

The fourth driver body: value-returning WITH write-back. Gathers the declared
slots, hands them to [`condense_cell!`](@ref), and copies the element-local
`q` buffer it filled into the item's slice of the `:q` slot's global vector —
exactly where the pre-phase per-kernel trial write-back used to write, from
exactly the same data.
"""
function condensation_cell_sweep!(kind::CondensationKind, task, ws)
    reinit_values!(ws.element, ws.cell, kind)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    args = _cell_args(ws, statesₑ, pₑ, task.ctx)
    report = condense_cell!(ws.element, args, kind.weights)
    qsrc = _q_source(task.states)
    range = internal_variable_range(ws.ivh, cellid(ws.cell))
    qsrc.u[range] .= statesₑ.q
    return report
end

"""
    condense_internal!(op, weights::NamedTuple, states::NamedTuple, p, ctx) -> CondensationReport
    condense_internal!(op, states, p, ctx)

Solve every condensed element's local problem over the WHOLE domain, write the
trial `q` into the `[ū; q]` tail, store each item's corrector, and report what
happened — the only writer of `q` (no evaluation sweep writes back). Must run
before any `Consistent`-mode sweep that would read a never-condensed or
invalidated ([`rollback_state!`](@ref)) item: reading one throws, naming the
item, through [`item_state`](@ref)'s own freshness contract on the element's
corrector store. This catches "never condensed" and "invalidated since"; it
does NOT catch the same vector mutated in place without going through
[`rollback_state!`](@ref) — see the design's staleness concession, §10.2.

`weights` are the solver's chain-rule scalars passed through to
[`condense_cell!`](@ref); `condense_internal!(op, states, p, ctx)` defaults to
`(u = 1.0,)`.

Value-returning WITH write-back — reuses [`fold_items`](@ref)/
[`reduce_on_device`](@ref)/[`reduce_on_subdomains`](@ref) unchanged, riding
`FunctionalFamily` so the existing deterministic fold order and
`_check_reduction_domain` structural checks apply as they stand.

Errors loudly, naming itself, if the operator's integrator elected anything
other than `Separate()` condensation ([`condensation_election`](@ref)) —
`FusedWithResidual()` is a construction-time seam, not yet implemented.
"""
function condense_internal!(op, weights::NamedTuple, states::NamedTuple, p, ctx)
    election = condensation_election(op.integrator)
    election === Separate() || condensation_election_error(election)
    _check_declared_slots(op.engine, states)
    _check_rate_slots(states)
    kind = CondensationKind(weights)
    _check_reduction_domain(kind, op.engine)
    task = AssemblyTask(kind, nothing, states, p, ctx)
    report = reduce_on_subdomains(task, op.engine)
    return report === nothing ? zero(CondensationReport{Float64}) : report
end
condense_internal!(op, states::NamedTuple, p, ctx) = condense_internal!(op, (u = 1.0,), states, p, ctx)

"""
    condensed_update_linearization!(op, residual, weights, states, p, ctx) -> CondensationReport

The fused convenience entry point a Newton loop calls once per trial point:
condenses via [`condense_internal!`](@ref) and, only if every local problem
converged, calls [`update_linearization!`](@ref) to fill `op.J`/`residual`.
Returns the report EARLY on `!report.converged`, without evaluating — the
same "one call, route decided inside" move
[`assemble_weighted_jacobian!`](@ref) already makes. Forgetting to condense
requires deliberately dropping to the lower-level `condense_internal!` +
`update_linearization!` pair, which is what makes the correct sequence the
convenient one.
"""
function condensed_update_linearization!(op, residual, weights::NamedTuple, states::NamedTuple, p, ctx)
    report = condense_internal!(op, weights, states, p, ctx)
    report.converged || return report
    update_linearization!(op, residual, states, p, ctx)
    return report
end

"""
    rollback_state!(op, u, committed)

Discard a rejected trial: copy the committed solution back into `u` and
invalidate every condensation corrector the operator's element caches carry —
they were computed for the discarded trial `q`, which `u` no longer carries.
The next `Consistent` sweep needs [`condense_internal!`](@ref) again. This is
the one mutation FO itself sees: the solver's own `u .+= Δu` outside FO is
invisible to it (see the design's staleness concession, §10.2).
"""
function rollback_state!(op, u::AbstractVector, committed::AbstractVector)
    u .= committed
    for sc in op.engine.subdomain_caches, ws in sc.device_cache
        invalidate_correctors!(ws.element)
    end
    return u
end

"""
    commit_state!(op, u, committed)

Accept a converged trial: copy `u` into the committed solution `committed`.
Correctors stay valid — the committed point is the last condensed point, so
no invalidation is needed.
"""
function commit_state!(op, u::AbstractVector, committed::AbstractVector)
    committed .= u
    return committed
end

####################################
## Construction-time elections
####################################

"""
    CondensationElection
    Separate <: CondensationElection
    FusedWithResidual <: CondensationElection

How [`condense_internal!`](@ref) is scheduled relative to the first
evaluation sweep — an OPERATOR-CONSTRUCTION election
([`condensation_election`](@ref)), never a runtime branch, so it cannot
become the fused/composed call-time fork the operator algebra already moved
out of. `Separate()` (default) runs condensation as its own domain traversal.
`FusedWithResidual()` would fuse it with the first residual sweep to recover
today's single-traversal cost (the design's §10.1 mitigation for the
fused-analytic-Jacobian-residual user, who otherwise pays one extra item
traversal for no local-solve saving) — a construction-time seam, not
implemented in this slice: selecting it errors loudly, naming itself, the
moment condensation would run.
"""
abstract type CondensationElection end
@doc (@doc CondensationElection) struct Separate <: CondensationElection end
@doc (@doc CondensationElection) struct FusedWithResidual <: CondensationElection end

"""
    CorrectorElection
    Stored <: CorrectorElection
    Recompute <: CorrectorElection

Whether a condensed element's Jacobian correction is read from a stored
per-quadrature-point corrector (`Stored()`, default) or recomputed inline on
every Jacobian-shaped sweep ([`corrector_election`](@ref)) — an
operator-construction election trading the corrector's memory (the design's
§10.4 binding constraint) against re-solving, which recovers today's memory
profile at today's time cost for large forward-only solves. `Recompute()` is
a construction-time seam, not implemented in this slice: selecting it errors
loudly, naming itself, at construction.
"""
abstract type CorrectorElection end
@doc (@doc CorrectorElection) struct Stored <: CorrectorElection end
@doc (@doc CorrectorElection) struct Recompute <: CorrectorElection end

"""
    condensation_election(integrator) -> CondensationElection

Default `Separate()`; a condensed integrator overrides it to report its own
constructor-elected value.
"""
condensation_election(integrator) = Separate()

"""
    corrector_election(integrator) -> CorrectorElection

Default `Stored()`; a condensed integrator overrides it to report its own
constructor-elected value.
"""
corrector_election(integrator) = Stored()

"""
    condensation_election_error(election)

The loud, self-naming rejection every non-`Separate()` [`CondensationElection`](@ref)
must raise — a condensed integrator's constructor calls this for any election
it does not implement, so the seam exists without silently accepting a
selection it cannot honor.
"""
condensation_election_error(election) = throw(ArgumentError(
    "$(nameof(typeof(election))) condensation is not implemented yet; construct the " *
    "integrator with `condensation = Separate()` (the default)."))

"""
    corrector_election_error(election)

The loud, self-naming rejection every non-`Stored()` [`CorrectorElection`](@ref)
must raise — see [`condensation_election_error`](@ref).
"""
corrector_election_error(election) = throw(ArgumentError(
    "$(nameof(typeof(election))) correctors are not implemented yet; construct the " *
    "integrator with `corrector = Stored()` (the default)."))
