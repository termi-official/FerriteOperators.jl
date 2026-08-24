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
| `worst_cell` | argmax carrier | the cell (or, negated, the algebraic item) attaining `worst_iterations` |
| `worst_qp` | argmax carrier | the quadrature point attaining `worst_iterations`; `0` for an algebraic item, which has no quadrature points |
| `worst_residual` | `max` | largest local-residual magnitude at exit |
| `dt_factor` | `min` | 1 means "no reduction requested"; < 1 is a request |

`worst_cell` is FAMILY-DISAMBIGUATED by sign, since [`condense_cell!`](@ref)
and [`condense_algebraic!`](@ref) reports fold into the same total and a
cellid and an item index are both small positive integers: a cell reports its
`cellid` (`≥ 1`), [`condense_algebraic!`](@ref) reports `-item.index` (`≤ -1`,
`AlgebraicItem` indices being 1-based too), and `0` means neither — every
quadrature point (or item) converged in zero iterations, which is what the
type's `zero` and every closed-form local solve reports.

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
(never into `args.states.u`), and — under a [`Stored`](@ref) election — store
the corrector, whatever compact per-quadrature-point quantity the cache's own
`Consistent` kernel later reads (Tier 1), or the completed local block a
generic combination would need (Tier 2). Under [`Recompute`](@ref) nothing is
stored and the corrector is re-derived at every point of use. `weights` are
the solver's chain-rule scalars for reconstructed slots participating in the
local model (e.g. a rate slot under Newmark), chained into the corrector
INSIDE the local inverse, where post-hoc weighting of separated partial
Jacobians cannot put them. They are per-sweep solver data rather than item
data, so a recomputing element retains them per cache.

This is the only hook allowed to EVOLVE a condensed element's state:
`assemble_cell!` kernels are pure evaluations at the `q` this hook wrote and
never write back. An inline local solve inside a kernel stays legitimate when
it is history-free — a pure function of the gathered `args`, recomputable
outside a sweep — but re-solving from committed history inside a kernel has
nowhere to write the result and silently freezes that history instead of
advancing it.

There is no default; only condensed elements implement it.
"""
function condense_cell! end

"""
    condense_algebraic!(cache, args::AlgebraicArgs, weights::NamedTuple) -> CondensationReport

The [`condense_cell!`](@ref) counterpart for the algebraic-item family: solve
the item's local problem, write the trial state into `args.states.q` (never
into `args.states.u`), and store the corrector the cache's own `Consistent`
kernel reads — item-keyed (an `ItemStates` store indexed by `args.item.index`,
same mechanism a condensed cell cache uses, indexed by cellid), or nothing at
all under a [`Recompute`](@ref) election. `weights` are the solver's
chain-rule scalars, exactly as for [`condense_cell!`](@ref).

Report the item in `worst_cell` as `-args.item.index` (see
[`CondensationReport`](@ref)'s family-disambiguation convention), never the
raw index — a positive value there is read as a cellid by any consumer folding
this report together with a condensed cell's.

There is no default; only a condensed algebraic cache implements it. Called
only when the cache declares [`has_internal_state`](@ref) — a stateless
algebraic cache's condensation sweep never reaches this hook.

Analytic-first: a condensed algebraic cache admits `Consistent`
sensitivity/Jacobian kinds only by serving them analytically or by declaring
[`internal_state_insensitive`](@ref) — there is no generic AD `Consistent`
bootstrap for this family (unlike [`ADElementCache`](@ref)'s
`condensed_corrector` combination for condensed cells): an algebraic item has
no cellid to key a corrector store by AD would need one keyed on.
"""
function condense_algebraic! end

"""
    local_conditions!(L, cache, args) -> L

The element's LOCAL CONDITIONS `L(u, q, θ, t) = 0` for the item `args` stands
on: the implicit equations [`condense_cell!`](@ref) solves for `q`, EVALUATED
(never solved) at the args' current state and written into `L`, whose length
is the item's condensed internal dof count and whose ordering is the `:q`
slot's.

Optional, and the third acceptance branch of the internal-state admissibility
rule ([`assert_sensitivity_admissible`](@ref)). Without an analytic kernel or
an [`internal_state_insensitive`](@ref) declaration, a condensed cache is
refused `ParameterJacobianKind`/`ParameterVJPKind`/`TimeSensitivityKind`,
because AD of the residual kernel computes only the frozen-q partial. Given
this hook, [`ADElementCache`](@ref) completes the total generically: it
differentiates `local_conditions!` for `∂L/∂q` (factorized once per item),
`∂L/∂θ` and `∂L/∂t`, and closes the implicit function theorem against the
`∂F/∂q` block [`JacobianKind{:q}`](@ref JacobianKind) already gives —

    dq/dθ = −(∂L/∂q)⁻¹ ∂L/∂θ,      dF/dθ = ∂F/∂θ|_q + ∂F/∂q · dq/dθ

and likewise with `t` in place of `θ`. An analytic kernel still wins where the
cache declares one; the hook is what a cache without one falls back to.

Contract: `L` is the residual form of exactly the equations `condense_cell!`
converged, a PURE function of `(args.states, args.p, args.ctx)` with no solve
and no write-back, and eltype-generic — it is what gets differentiated, so `q`,
the parameters and the evaluation time all reach it Dual-valued. Cell caches
only: the `∂F/∂q` block the combination multiplies is sized from
`getnquadpoints`, which the algebraic-item family has no counterpart for.

!!! warning "Experimental surface"
    The local-model seam is a CANDIDATE contract, not a frozen one: this
    signature may change in a minor release. The assembled results of the
    kinds it admits are not affected.
"""
function local_conditions! end

"""
    invalidate_correctors!(cache)

Drop every item's condensation corrector, so the cache's `Consistent` kernel
throws on its next read until [`condense_internal!`](@ref) runs again. The
default is a no-op; only caches carrying a corrector store override it — a
[`Recompute`](@ref) cache has none and keeps the default. Called by
[`rollback_state!`](@ref).
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

# Trait-gated: a plain cell subdomain sharing an operator with a condensed one
# (a condensed algebraic item's cell physics, a plain subdomain of a
# multi-domain integrator) has no `condense_cell!` method and contributes
# nothing to the report, exactly like a stateless algebraic cache's gate below
# — `condense_internal!`'s per-subdomain reduction reaches every subdomain
# unconditionally, so this is what keeps a mixed operator from a MethodError
# on the subdomain that never had a local problem to solve.
execute_kind!(kind::CondensationKind, task, ws) =
    has_internal_state(typeof(ws.element)) ? condensation_cell_sweep!(kind, task, ws) : nothing

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

The cell family's condensation driver body: value-returning WITH write-back.
Gathers the declared slots, hands them to [`condense_cell!`](@ref), and copies
the element-local `q` buffer it filled into the item's slice of the `:q` slot's
global vector, through the cell's [`internal_variable_range`](@ref).
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
trial `q` into the `[ū; q]` tail, store each item's corrector (under a
[`Stored`](@ref) election), and report what happened — the only writer of `q`
(no evaluation sweep writes back). Must run before any `Consistent`-mode
sweep, whatever the election: the sweep is a pure evaluation at whatever `q`
the tail currently holds.

Under `Stored()`, reading a never-condensed or invalidated
([`rollback_state!`](@ref)) item throws, naming the item, through
[`item_state`](@ref)'s own freshness contract on the element's corrector
store. This catches "never condensed" and "invalidated since"; it does NOT
catch the same vector mutated in place without going through
[`rollback_state!`](@ref) — see [what the phase
concedes](devdocs/rationale.md#What-the-phase-concedes). Under
[`Recompute`](@ref) there is no store to stamp, so none of these are detected
and the ordering requirement is the caller's alone.

`weights` are the solver's chain-rule scalars passed through to
[`condense_cell!`](@ref); `condense_internal!(op, states, p, ctx)` defaults to
`(u = 1.0,)`.

Value-returning WITH write-back: it rides `FunctionalFamily` through
[`fold_items`](@ref)/[`reduce_on_device`](@ref)/[`reduce_on_subdomains`](@ref),
so the deterministic fold order and the `_check_reduction_domain` structural
checks hold for it too.

Unconditionally its own domain traversal, run before the evaluation sweeps it
feeds.
"""
function condense_internal!(op, weights::NamedTuple, states::NamedTuple, p, ctx)
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
the one mutation FerriteOperators itself sees: the solver's own `u .+= Δu`
happens outside the package and is invisible to it (see [what the phase
concedes](devdocs/rationale.md#What-the-phase-concedes)).

A [`Recompute`](@ref) cache carries no correctors, so the invalidation pass is
a no-op for it: `u` and its `q` tail are restored together, and a corrector
re-derived from that pair is the committed point's own.
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
## ∂F/∂q — the rectangular field × internal block
####################################
#
# `JacobianKind{:q}` is the one Jacobian kind whose target is not the
# operator's square matrix: `q` lives in the `[ū | q_cells | q_items]` tail, so
# the block is `ndofs(dh) × ndofs(ivh)` and its per-item contribution is
# `celldofs(cell) × internal_variable_range(ivh, cellid)`. Columns are disjoint
# between items by construction — an item owns its internal range alone — so
# only the rows ever collide, exactly like every other dof-scattered sweep.

"""
    internal_jacobian_cell_sweep!(kind::JacobianKind{:q}, task, ws)

The ∂F/∂q driver body of the cell family: [`primal_cell_sweep!`](@ref) over a
RECTANGULAR local block instead of the square `ws.Ke`, scattered by the
two-index `assemble!(assembler, rowdofs, coldofs, Kqₑ)`. An item owning no
internal dofs contributes nothing and never reaches a kernel, which is what
lets a plain subdomain share an operator with a condensed one.
"""
function internal_jacobian_cell_sweep!(kind::JacobianKind{:q, C}, task, ws) where {C}
    range = internal_variable_range(ws.ivh, cellid(ws.cell))
    isempty(range) && return nothing
    Kqₑ = internal_sweep_buffers!(ws.sensitivity, length(range)).Kqₑ
    fill!(Kqₑ, zero(eltype(Kqₑ)))
    reinit_values!(ws.element, ws.cell, kind)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    @timeit_debug "assemble internal jacobian" assemble_cell!(
        JacobianRequest{:q, C}(Kqₑ), ws.element, _cell_args(ws, statesₑ, pₑ, task.ctx))
    assemble!(task.inner_assembler, item_dofs(ws), _internal_columns(ws.ivh, range), Kqₑ)
    return nothing
end

execute_kind!(kind::JacobianKind{:q}, task, ws::AssemblyWorkspace) =
    internal_jacobian_cell_sweep!(kind, task, ws)

# The tail range as COLUMNS of the rectangular target: the range addresses the
# solution vector, whose internal block starts at `base_offset`.
@inline _internal_columns(ivh::InternalVariableHandler, range) = (range .- ivh.base_offset)

"""
    init_internal_jacobian_sparsity_pattern(engine) -> SparsityPattern

The `ndofs(dh) × ndofs(ivh)` pattern of ∂F/∂q: for every item carrying
condensed internal state, the product of the dofs its local system spans with
its own [`internal_variable_range`](@ref). Cell subdomains and condensed
algebraic items both contribute, their column blocks lying where the
`[ū | q_cells | q_items]` layout puts them.
"""
function init_internal_jacobian_sparsity_pattern(engine)
    ivh = engine.ivh
    # `nnz_per_row` is a growth hint only — the pattern grows as entries are
    # added, and a row's real width is (items touching the dof) × (internal
    # dofs per item), which no single number covers across mixed subdomains.
    sp = SparsityPattern(ndofs(engine.dh), ndofs(ivh); nnz_per_row = 8)
    for sc in engine.subdomain_caches
        _add_internal_jacobian_entries!(sp, sc, ivh)
    end
    return sp
end

# Families that never own condensed internal state (facet items, transfer)
# contribute no entries.
_add_internal_jacobian_entries!(sp, sc, ivh) = nothing

function _add_internal_jacobian_entries!(sp, sc::SubdomainCache{<:AssemblyDomain}, ivh)
    sdh = sc.domain.sdh
    # The rows an item's local system spans are its scatter address, which the
    # workspace already resolves: `celldofs`, or the augmented
    # `[celldofs; declared global dofs]` vector.
    tail = _declared_global_dofs(first(sc.device_cache))
    dofs = Int[]
    for cellid in sdh.cellset
        range = internal_variable_range(ivh, cellid)
        isempty(range) && continue
        resize!(dofs, ndofs_per_cell(sdh))
        celldofs!(dofs, sdh.dh, cellid)
        for col in _internal_columns(ivh, range)
            for rdof in dofs
                Ferrite.add_entry!(sp, rdof, col)
            end
            for rdof in tail
                Ferrite.add_entry!(sp, rdof, col)
            end
        end
    end
    return nothing
end

function _add_internal_jacobian_entries!(sp, sc::SubdomainCache{<:AlgebraicDomain}, ivh)
    for (index, dofs) in pairs(sc.domain.items)
        range = internal_variable_range(ivh, AlgebraicItem(index, dofs))
        isempty(range) && continue
        for col in _internal_columns(ivh, range), rdof in dofs
            Ferrite.add_entry!(sp, rdof, col)
        end
    end
    return nothing
end

"""
    allocate_internal_jacobian(op) -> SparseMatrixCSC

Allocate the rectangular ∂F/∂q target of a condensed operator,
`residual_size(op) × ndofs(op.engine.ivh)`, over the pattern
[`init_internal_jacobian_sparsity_pattern`](@ref) builds. Pass the result to
[`update_internal_jacobian!`](@ref), or hand that entry point any matrix
sharing the pattern.
"""
function allocate_internal_jacobian(op)
    _assert_condensed_operator(op, "allocate_internal_jacobian")
    Tv = value_type(op.engine.strategy.device)
    return allocate_matrix(SparseMatrixCSC{Tv, Int}, init_internal_jacobian_sparsity_pattern(op.engine))
end

"""
    update_internal_jacobian!(Kq, op, states, p, ctx)

Assemble ∂F/∂q into `Kq` — the `residual_size(op) × ndofs(op.engine.ivh)`
block coupling the residual to the CONDENSED internal state, evaluated at the
trial state. `Kq` must share the pattern
[`allocate_internal_jacobian`](@ref) builds. Never writes back into `states`.

This is the block a Schur-complement consumer wants, and the `∂F/∂q` factor of
the corrector combination `∂F/∂· |_q + ∂F/∂q · dq/d·`. Elements serve it
through the analytic `assemble_cell!(::JacobianRequest{:q}, …)` kernel or by
ForwardDiff seeding of the `:q` slot — which needs no admissibility guard,
`q` being the seed itself, so `Consistent` and `FrozenQ` coincide here.

An operator without condensed unknowns has no such block and is rejected:
`ndofs(ivh) == 0` means there is no column space to assemble into.
"""
function update_internal_jacobian!(Kq::AbstractMatrix, op, states::NamedTuple, p, ctx)
    _assert_condensed_operator(op, "update_internal_jacobian!")
    expected = (residual_size(op), ndofs(op.engine.ivh))
    size(Kq) == expected || throw(DimensionMismatch(
        "expected Kq of size $(expected), got $(size(Kq))"))
    assembler = start_assemble(op.engine.strategy, Kq; fillzero = true)
    run_sweep!(JacobianKind{:q}(), assembler, op, states, p, ctx)
    return Kq
end

function _assert_condensed_operator(op, entry)
    ndofs(op.engine.ivh) > 0 || throw(ArgumentError(
        "`$(entry)` assembles the ∂F/∂q block over the operator's CONDENSED internal state, " *
        "and this operator carries none (`ndofs(op.engine.ivh) == 0`): no element cache " *
        "declares `has_internal_state` together with an internal dof count, so `q` has no " *
        "column space. `JacobianKind{:q}` is meaningful only for a condensed operator — see " *
        "`condense_internal!`."))
    return nothing
end

####################################
## Construction-time elections
####################################

"""
    CorrectorElection
    Stored <: CorrectorElection

Whether a condensed element's Jacobian correction is READ from a per-item
corrector the condensation sweep stored (`Stored()`, the default) or
RE-DERIVED from the item's current `(u, q)` wherever a `Consistent`-mode
kernel needs it ([`Recompute`](@ref)) — an operator-construction election
([`corrector_election`](@ref)) trading the store's per-quadrature-point memory
against one extra evaluation of the element's local slopes per Jacobian-shaped
sweep.

The election is invisible to kernels: a `Consistent` kernel reads its
corrector through ONE access point, which either reads the store or
recomputes. [`condensed_corrector`](@ref) is that access point for the AD
decorator's generic combination, and it receives the args record for exactly
this reason.
"""
abstract type CorrectorElection end
@doc (@doc CorrectorElection) struct Stored <: CorrectorElection end

"""
    Recompute <: CorrectorElection

Keeps NO per-item corrector storage — [`condense_cell!`](@ref) writes the
trial `q` and nothing else — and re-derives the corrector from the item's
current `(u, q)` at every point of use.

Targets memory-bound ASSEMBLED sweeps at scale, where per-quadrature-point
corrector storage is the binding cost. For matrix-free/action-style use —
repeated operator actions at a fixed state, e.g. a Krylov `mul!`/JVP sequence
— [`Stored`](@ref) is the right election, since every action would otherwise
re-derive the same corrector.

Recomputation is EXACT, not approximate: the corrector is a closed-form
function of the converged pair `(u, q)` — the implicit-function-theorem slopes
of the element's local conditions — so it is the same quantity
[`condense_cell!`](@ref) would have stored, evaluated at the same point. The
solver's chain-rule scalars are not item data; an element chaining `weights`
into its corrector retains them per cache, which is `O(1)` and not the storage
this election trades away.

Freshness: the corrector staleness class disappears — nothing is stored, so
nothing can go stale, and [`rollback_state!`](@ref) has no correctors to drop
(restoring `u` restores its `q` tail with it). The q-freshness contract
REMAINS unchanged: `q` is written only by [`condense_internal!`](@ref), so a
`Consistent` sweep still requires a condensation at the current trial point.
What is lost is the DETECTION of a missing one — [`Stored`](@ref) throws
through [`item_state`](@ref)'s freshness contract on a never-condensed or
invalidated item, while a recomputing kernel silently derives a corrector from
whatever `q` the tail currently holds.
"""
struct Recompute <: CorrectorElection end

"""
    corrector_election(integrator) -> CorrectorElection

Default `Stored()`; a condensed integrator overrides it to report its own
constructor-elected value.
"""
corrector_election(integrator) = Stored()

"""
    corrector_election_error(election)

The loud, self-naming rejection a condensed integrator's constructor raises
for a [`CorrectorElection`](@ref) it does not implement, so the seam exists
without silently accepting a selection the element cannot honor.
"""
corrector_election_error(election) = throw(ArgumentError(
    "$(nameof(typeof(election))) correctors are not implemented by this integrator; construct " *
    "it with an election it implements — `Stored()` (the default) or `Recompute()`, which " *
    "both shipped condensed elements accept."))
