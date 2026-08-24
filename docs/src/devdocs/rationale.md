```@meta
CurrentModule = FerriteOperators
```

# Design rationale

This page answers *why is it this way*. It is the one sanctioned home for
reasoning in the documentation: the decisions that shaped the current design,
the load-bearing reasons behind them, and the alternatives that were rejected
and on what grounds.

Everything else states the contract. [The layer contract](design.md) says which
layer owns which piece of information and what the extension points are;
[Writing elements](../elements.md) and [Operators and entry points](../operators.md)
say what an author must write. If a statement here disagrees with those pages,
those pages are right and this one is stale — report it.

Entries are organized by **design question**, not by date. Each closes with the
date it was ruled and a pointer into the verbatim record.

## The archived record

The deliberation these decisions came out of is archived in the repository,
outside the docs build, as unedited agent output — dissents and concessions
included. Paths below are relative to the repository root.

| file under `references/design-reviews-2026-08-19-kernelargs/` | what it is |
|---|---|
| `README.md` | index of the round |
| `report1-architect-operator-algebra.md` | the worked proposal: operator algebra, capability records, closed `CellArgs`, quantitative fate inventory |
| `report2-adversarial-review.md` | the attack: ten angles, the measured cost of naive operator-typed grouping, the extensibility inversion |
| `report3-control-arm-minimal-path.md` | the control arm: what the minimal subtractive path buys, measured in lines |
| `report4-architect-second-pass.md` | the revision after the adversary: AD-as-decorator (W1), scratch verdict (W2), plan-reconciliation ledger (W3) |
| `condensation-phase-design.md` | the condensation phase: contract, corrector storage, witnesses, self-adversarial section |
| `devdocs-draft.md` | this page's ancestor, written *before* implementation and superseded by it |

Reading the draft against this page is instructive: several of its claims did
not survive contact with the implementation, and the corrections are called out
below where they matter.

## Argument bundles

### Why the kernel argument bundle is a closed, annotatable record

**Decision.** There is one concrete argument type per item family —
[`CellArgs`](@ref) for cell kernels, [`FacetArgs`](@ref) for facet kernels, a
concrete `InterfaceArgs{N}` reserved for interface items — with no common
supertype, and element authors *may* annotate `args::CellArgs`. The bundle is
not an open structural protocol a downstream operator family re-implements.

**Why.** Almost all of the machinery that policed the bundle's form existed to
support *openness*, not to support the bundle: a per-operator-family declaration
of which argument type it builds, threaded through setup into method lookups; a
type parameter on the setup-time validator, because `hasmethod` against an
abstract queried type misses concretely-annotated methods; a setup-time
reflection warning that inspected element method signatures and advised authors
*not* to annotate — a warning whose only job was to make the previous two
mechanisms unnecessary; three rebuild functions promoted to public seams with a
rule that every family implement all three and none have a fallback; and a
documented channel table calling struct fields a protocol.

Measured against the repository at the time: the open-args axis had **no
end-to-end proof anywhere**. No operator, test, or example ever built a foreign
argument object and ran a sweep with it. Closing the type deleted the whole
apparatus at the cost of a capability nobody had exercised.

**Rejected: an open channel protocol.** What a downstream family genuinely needs
— an extra per-sweep quantity — is served by `ctx`, which is open by design and
costs nothing to keep open. Openness in the *bundle* bought a second, redundant
extension axis and paid for it in validation surface.

The three rebuild functions survive as ordinary private constructors. They are
not seams; they are named record updates, and they are what keeps the field
order written once instead of at every AD seeding site.

*Ruled 2026-08-19; executed 2026-08-20. Record: `report4-architect-second-pass.md`
Part A1, `report3-control-arm-minimal-path.md` §1.*

### Why a record and not positional arguments

**Decision.** Kernels receive one record, not a flattened argument list.

**Why.** Three reasons, in increasing order of weight.

- **Transposition.** `states` and `p` are opaque bags. Positional arguments of
  opaque type produce silent wrong-argument errors instead of `MethodError`s —
  exactly the failure mode the request types were introduced to cure. The
  pre-0.4 interface discriminated the operator induced by a bilinear form from
  the load vector of a linear form by `AbstractMatrix` versus `AbstractVector`
  in argument one, and residual from Jacobian by the shapes of arguments one and
  two. A downstream package needing a differently-shaped kernel set had to
  invent new function names, because the shapes were dispatch-ambiguous.
- **Frozen signatures.** A fixed positional signature promises the argument list
  never grows. When it must grow, the pressure goes into whichever argument is a
  bag — the "smuggle it through `p`" failure described below, arrived at from
  the other direction.
- **Seeding arithmetic, which is what settles it.** Seven derivative sweeps seed
  through exactly three channels: states, parameters, context. The framework
  therefore needs *three* rebuild methods per argument family, not one per
  sweep. That ratio is a property of the record. Under positional arguments the
  framework cannot express "the same call with one argument replaced"
  generically, and the count becomes (sweeps × signature shapes).

*Ruled 2026-08-19, explicitly not to be reopened. Record:
`report2-adversarial-review.md` Angle 5.*

## Where information lives

### The principle: decorate capabilities, pass data

> **Structure is decorated; data is passed.** Anything fixed before a sweep
> starts and constant across it belongs to a type — an operator, a cache, or a
> decorator around a cache. Anything that varies per evaluation is an argument.

`ctx` is an argument because a time is per evaluation. Weights are arguments
because a step size changes. The element cache is structure. Whether a Jacobian
is computed analytically or by differentiating the residual is *structure* — and
that is the observation the AD decorator rests on.

**The cautionary tale this principle is drawn from.** The disease the 0.4
redesign cured was a scheme expressed as a *wrapper around the element
interface*. A first-order time protocol shipped as a parameter type carrying one
history level, with its own forwarding layer; a second-order scheme downstream
then copied the whole protocol file — a parallel parameter type, a parallel
`Union` encoding the fork in the type system, the element assembly methods
duplicated per protocol, and half a dozen methods whose only job was to strip
the wrapper back off at facets. Every new scheme meant a new copy, and no copy
composed with the derivative machinery.

The lesson is not "do not decorate". It is **decorate the right layer**. That
protocol decorated the *scheme* onto the element, which is data (per step:
`uprev`, `Δt`) masquerading as structure. The AD decorator decorates a
*capability* onto a cache, which is genuinely structure: whether this element
can produce this derivative does not change between steps.

*Named 2026-08-19. Record: `report4-architect-second-pass.md` Part A1 (R3).*

### Why `ctx` is an argument — and not `p`, and not operator state

**Decision.** [`TimeIntegrationContext`](@ref) reaches kernels as a field of the
args record. It is neither folded into the parameter bag nor stored on the
operator.

**Rejected: time inside `p`.** This was the 0.3 design, and it fails four ways
at once. The framework cannot see `t` where it must — the `∂F/∂t` sweep seeds a
Dual through it and the engine threads `γ̃` to element-local stage problems — so
every wrapper needs an unwrapping convention; a user's own
[`query_cell_parameters`](@ref) override silently drops that wrapper; and
`∂F/∂θ` and `∂F/∂t` become indistinguishable, because they seed the same
channel. That last one was a **live silent-zero defect**: the time sweep seeded
bare time through the parameter channel, so an element reading its context per
the documented contract got a derivative of exactly zero, with no error
anywhere.

**Rejected: per-sweep scalars as operator state (the "solver as a decorator"
shape).** A number stored on a structure has to be invalidated, and invalidation
is solver knowledge that no operator can infer. This re-creates the
update-semantics ambiguity that Rule C below exists to forbid. The downstream
witness is a backward-Euler path guarded by `Δt ≈ Δt_last`: an `isapprox`
predicate, so a step size drifting within tolerance silently reuses a stale
matrix.

Per-item contexts — the reservation for local time stepping — do not contradict
this. A context *source* is structure and lives on the operator; the value it
yields for an item is call data and reaches the kernel as the argument.

Elements read the context only through accessors ([`evaluation_time`](@ref),
[`stage_scaling`](@ref)), which is what makes a richer downstream context type
substitutable. Before the ruling could be acted on, `γ̃` had no accessor at all —
elements read `ctx.γ̃` as a field, so any context decoration would have broken
every condensed element. That gap was closed first; it is a small illustration of
why "open channel" only means something if the channel has a reader.

*Ruled 2026-08-19. Record: `report4-architect-second-pass.md` Part A1 (R3).*

### Why scratch is a cache field and not a channel

**Decision.** There is no scratch channel. An element needing a working buffer
declares a cache field and allocates it in `setup_element_cache`;
`duplicate_for_device` gives each worker its own.

**Why.** Nothing a scratch channel can express is beyond a cache field, and the
channel adds two failure modes a field does not have. Name collisions between
the inners of a composite meant a merged namespace silently shared a buffer — a
latent defect that the field spelling makes *unexpressable*. And the shipped
`args.scratch` had **zero consumers** anywhere in the repository.

**Correction to the draft.** The draft credits this retraction with also solving
Dual-eltype pinning, via an element hook `duplicate_with_eltype(cache, T)` that
would reconstruct a buffer-carrying inner at the Dual eltype. **That hook does
not exist** — not in `src/`, not in `lib/`, not in `test/`. It is a recorded
intention, and the actual state is narrower: the AD decorator carries its *own*
Dual seeds and ForwardDiff configurations (`ADElementBuffers`), so the seeded
buffers are correctly typed, but a `Float64` working buffer inside an element
cache is still eltype-pinned and would truncate Duals if a kernel wrote a seeded
value into it. Spelling the buffer as a cache field removes the *collision*
hazard; it does not by itself remove the *retyping* hazard.

*Ruled 2026-08-19 (W2), confirmed 2026-08-20 by a stress-test element whose five
would-be scratch uses all mapped to a cache field, to `ctx`, or to a thrown
value. Record: `report4-architect-second-pass.md` Part B.*

### Why there is no structural unknowns/data split

**Decision.** `states` is one flat bundle of slots. There is no grouping of
slots into "unknowns" versus "data".

**Why.** Roles are **request-relative**: a request kind seeds what it names, and
everything else in `states` is data for that sweep. `q` is data in a residual
sweep and an unknown in a `∂F/∂q` sweep — the same slot, two roles, decided by
the request. A structural grouping would force kernels to read one slot from
different groups across phases, making the kernel phase-aware, and would
re-create precisely the slot-tuple-carrier problem the flat bundle removed.

The two use cases usually cited for such a split are served elsewhere and
better: **IMEX** is term-split operators issuing different request sets, and
**local time stepping** is item restriction.

*Ruled 2026-08-20. Record: `condensation-phase-design.md` §11 (L11).*

## Differentiation

### Why AD is a decorator cache rather than engine machinery

**Decision.** [`ADElementCache`](@ref) wraps an element cache that does not
serve some request analytically, implements the derivative requests by seeding
the inner's residual kernel, and forwards everything else. Wrapping happens once
at `setup_operator` time. The engine has no differentiation in it.

**Why.** The choice of route is structure, so under the principle above it
belongs in a type. Making it one has four consequences, and together they are
what pays for the type.

- **The per-cell fork disappears.** The fork used to live in the engine as a
  branch inside every kernel-dispatch method, consulted per cell and answered by
  a trait. Post-decoration every resolved cache is analytic from the engine's
  point of view, and the four specialized `cell_kernel!` methods collapse to one
  generic body; the five `sensitivity_kernel!` methods collapse to one.
- **Differentiation machinery becomes structurally absent when unused.** The
  workspace no longer carries an optional derivative member selected by a
  declaration and reached through a family accessor. An operator that never
  differentiates carries no differentiation machinery because no cache was
  wrapped — the declaration mechanism that engineered this property became
  unnecessary rather than being satisfied.
- **Composition becomes per-inner.** A composite whose inners are partly
  analytic used to fall back to differentiating the *whole* composite residual,
  because capability was all-or-nothing across the fan-out. Wrapping is
  per-cache, so analytic blocks come from their own kernels and only the
  non-analytic ones are differentiated. The Jacobian of a sum is the sum of the
  Jacobians, so this is exact. The constructor's policy is binding: wrap the
  non-analytic inners **as one maximal sub-composite**, never individually —
  naive per-inner wrapping costs one full seeding pass per wrapped inner and is
  *worse than not wrapping at all* once two or more inners need it.
- **The backend becomes an extension seam.** [`ForwardDiffAD`](@ref) is the
  default, but the decorator's contract is what a backend implements, so a
  reverse-mode backend is a package extension rather than a fork of the engine.

**What deliberately did not move into the decorator:** the admissibility rules
and [`check_derivatives`](@ref) stay framework-side. The decorator *implements*;
the constructor *enforces*. Likewise the rejection of gather-time-reconstructed
slots in a differentiated position stays a per-call check, because a slot's
source is call data.

*Ruled 2026-08-19, implemented 2026-08-20. Record:
`report4-architect-second-pass.md` §A2.*

### What the decorator turned out not to be

The draft describes the decorator as designed; four things came out differently
under implementation, and each is a constraint a future contributor will trip
over.

**AD output buffers live on the workspace, not in a task-owned bag.** The design
sketched splitting seeds and configurations into the decorator and outputs into a
task-owned bag. Outputs instead live per worker on the workspace as
[`SensitivityBuffers`](@ref), because `AssemblyTask` is reused across subdomains
of differing `ndofs_per_cell` while `AssemblyWorkspace` is correctly
per-subdomain-per-worker. A task-owned output bag would be sized for the wrong
subdomain. The workspace therefore lands at nine fields, not the seven the
draft predicts — the ninth being `dofs`, the augmented dof vector a
[`global_dofs`](@ref) declaration needs.

**The decorator does not broaden [`provides_analytic`](@ref) at all.** It first
did, and the trait then answered two questions at once: "*has a real analytic
kernel*", which the `AffineRate`-under-AD rejection and the fused-weighted route
select on, and "*this cache serves the kind*", which the condensed-state
admissibility rule needs. Keeping both meanings in one trait cost per-kind
non-broadening overrides wherever a check wanted the first. The questions are
now two predicates: `provides_analytic` is the hand-kernel one and forwards
through the decorators unchanged, `FerriteOperators.serves_kind` is the
served-capability one, and the overrides went with the split.

**`FusedFromSplit` is required, not an optimization.** It is the mini-decorator
that serves a fused Jacobian+residual request by issuing the split kernels back
to back. Without it, [`check_derivatives`](@ref) false-negatives on a *wrong*
analytic Jacobian, because AD would silently substitute the correct Jacobian
into the fused path and the check would pass.

**Buffer sizes did not all become construction knowledge.** The draft argues
that a parameter-derivative operator, being constructed against a parameter
space, knows nθ at construction, and that the last reason for mutable
lazily-reallocating buffers therefore goes. It has not: there is no
parameter-space-parameterized operator constructor in the tree, nθ still arrives
with `p` at call time, and the parameter-sized members of
[`SensitivityBuffers`](@ref) are still reallocated on first use. The
state-and-time sweeps *are* allocation-free per cell; the parameter sweeps are
not.

The finite-difference decorator sketched as a third backend was **not** built,
and that is the settled state rather than a deferral. A cache-level FD
decorator would differentiate the volumetric kernel, exactly like the AD one,
and would therefore lose the single property that makes operator-level
[`FiniteDifferenceSensitivity`](@ref) worth keeping: it differences
`evaluate!`, so **boundary terms enter**. Its remaining niche — kernels that
cannot carry `Dual`s — has no consumer in the tree. The two mechanisms are
final and their division of labour is documented where the routes are
described: the operator-level method is the boundary-inclusive, Dual-free
route, the decorator is the per-cache one where analytic kernels win cache by
cache.

*Implemented 2026-08-20 (W1). Record: `report4-architect-second-pass.md` §A2 for
the design, and this page for the deviations.*

## Condensation and internal variables

### Why condensation is an explicit phase

**Decision.** [`condense_internal!`](@ref) is a traversal of its own. It solves
every quadrature point's local problem, writes the trial `q` into the `[ū; q]`
tail, and stores each item's corrector. Every assembly sweep afterwards is a
pure evaluation at frozen `q`.

**Why: model/solver decoupling, completed at the local level.** The global level
was already decoupled — slots, weights and `ctx` between them let a solver drive
BDF or Newmark without an element knowing which. The local level was not: a
condensed element owned *both* its local model and its local time
discretization, and ran both inside every kernel. The consequence was visible in
the elements' own docstrings, which had to explain that a Jacobian sweep and a
residual sweep over the same cells count their local solves twice.

Splitting the phase off buys three things that were not reachable before:

- **Derivative machinery becomes generic.** With the kernel pure, a plain AD
  fallback computes a well-defined `∂F/∂·|_q`, and the correction is a stored
  matrix rather than an iteration to differentiate through. Condensed elements
  went from "cannot do parameter sensitivities without a hand-written kernel" to
  "can, generically" — the payoff case, `dF/dθ = ∂F/∂θ|_q + ∂F/∂q·dq/dθ`, now
  matches a finite-difference referee to ~1e-10 on the shipped power-law
  element. That capability did not exist.
- **Solver families become caller sequencing.** Exact condensed Newton,
  multilevel Newton, frozen-`q` modified Newton, and line search without
  re-condensing are all orderings of two calls. None of them requires element
  cooperation.
- **Non-convergence becomes data.** [`CondensationReport`](@ref) is isbits and
  folds as a monoid, so it crosses a device boundary — which an exception, as
  the replaced element's own docstring noted, cannot. Its shape was not
  invented: five of its eight fields already existed hand-rolled in an element,
  together with a hand-rolled inner-to-outer channel that walked engine
  internals because no accessor for per-worker element caches existed.

The `weights` argument of the phase is where the solver's chain-rule slopes
enter. The corrector is defined as `dq/dū` — with respect to the *primary*
unknown, with slot reconstructions pre-chained — so the weighted-Jacobian
machinery is untouched, keeps folding frozen-`q` partials with the solver's
scalars, and nothing double-counts. The slopes must reach the element *during*
condensation, because that is where they are chained in.

*Ruled and implemented 2026-08-20. Record: `condensation-phase-design.md` §1–2.*

### Why partial versus total is explicit vocabulary

**Decision.** Jacobian-shaped kinds carry a [`CorrectionMode`](@ref).
[`Consistent`](@ref) is the default and means the total; [`FrozenQ`](@ref) means
the partial and must always be spelled.

**Why.** Purity is mechanical, not semantic. `L(q; ū, θ, t) = 0` remains an
*implicit function*, so plain AD on a pure kernel yields the **partial at fixed
`q`** — a correct number, answering a question the caller usually did not ask.
Every total is one pattern with three seeds:

```
dF/dū = ∂F/∂ū|_q + ∂F/∂q · dq/dū
dF/dθ = ∂F/∂θ|_q + ∂F/∂q · dq/dθ
dF/dt = ∂F/∂t|_q + ∂F/∂q · dq/dt
```

The new silent-wrong class is therefore the **missing combination**: nothing
throws, nothing is `NaN`, the number is simply a different derivative than the
caller believes. A vocabulary that leaves partial-versus-total to the reader
hands that class to every user. Naming it in the request kind puts it where
elements dispatch on it, which is also where a consistent tangent is naturally
formed — at the quadrature point.

**Defaults fail in the safe direction.** The unsafe direction here is the
missing combination, so the default is the total.

**`FrozenQ` is refused for the sensitivity kinds — structurally.** A frozen
tangent is a legitimate and valuable election for an *iteration matrix*
(modified Newton, multilevel-Newton outer loops), where a wrong tangent costs
convergence rate and nothing else. It is never legitimate for a *gradient*,
where the wrong number *is* the answer. The refusal is not a runtime check: the
five sensitivity kinds simply carry no `CorrectionMode` parameter, so electing
one is a `TypeError`. The correction-mode parameter is also placed *between* the
slot and buffer type parameters of the kind family, so bare annotations stay
`UnionAll` — which is why the four non-condensed example elements needed zero
changes.

*Ruled 2026-08-20 (Dennis's correction to the round). Record:
`condensation-phase-design.md` §1.1, §3, §10.3.*

### Why `q` is a slot

**Decision.** A condensed element's internal state arrives as an ordinary slot
whose source, [`InternalSource`](@ref), carries the internal-dof range as its
restriction. Global `[ū; q]` storage stands; this is a change at the gather
level only.

**Why.** It collapses five element-side hooks and two cache fields into one
already-existing hook. It makes `∂F/∂q` fall out for free as an ordinary
per-slot Jacobian — load-bearing twice, as the block the generic corrector
combination needs and as the block a Schur-complement consumer wants. And it
retires a documented artifact: the note that "the condensed tail of a
reconstructed slot must not be interpreted by elements" becomes an
impossibility, because a reconstructed slot's source is the field space and
structurally cannot touch `q`.

This **supersedes** the earlier ruling that internal degrees are never mixed
into slots. That ruling's stated reason was that `q` is solved inside the kernel
and so is not an independent variable of the residual — which the phase makes
false. Worth noting the ruling was never implemented as written either: `q` used
to ride *inside* the `ū` slot through an element hook, a third arrangement that
was neither the ruling nor a separate component.

*Ruled and implemented 2026-08-20. Record: `condensation-phase-design.md` §4.*

### Why kernels are pure, and why the element-level fused solve was eliminated

**Decision.** Condensation is *unconditionally* its own traversal. The election
axis that would have let an operator fuse condensation into the residual sweep
was built and then deleted the same day.

This is the one entry where the honest accounting matters more than the
conclusion, so both sides are recorded.

**What the deletion buys.**

- It deletes an **unimplemented dual-path obligation**: a second driver body,
  dual freshness semantics, and dual reporting — including the once-per-item
  problem, since a fused sweep enters the element more than once per item under
  split-Jacobian-residual and AD chunk passes.
- **Kernel purity becomes an invariant** rather than a convention. The
  admissibility question turns from "did this kernel differentiate through an
  iteration" into "is a corrector present", which is structurally checkable. The
  hazard where AD runs through a *deliberately inexact* local solve — wrong by a
  solver-controlled amount, unlike a converged loop which is asymptotically
  right — dies structurally rather than by rule.
- There is exactly **one writer of `q`**.
- **Solver families need zero element cooperation** — frozen-`q`, multilevel
  Newton, and line-search policies are all caller sequencing.
- **No capability was lost**, because the fused path was structurally
  primal-only.

**What the deletion costs.**

- **One extra item traversal per exact-Newton iterate.** For a fused-analytic
  user this saves no local solves at all and pays gather, geometry reinit and
  values reinit for nothing. The design round put this at **+10–20% per Newton
  iterate** — and that figure is a **design-round estimate, never measured**. It
  hits cheap-local-solve elements hardest; modified-Newton and multilevel-Newton
  callers can net win, and a line search is neutral. Reinstating a fused
  schedule later is additive and non-breaking if a measurement ever demands it.
- **Condensed-residual exactness becomes a sequencing obligation.** See the
  freshness concession below.
- **Stateful do-it-yourself inline elements fail silently.** This is the
  documented boundary. A kernel receives `q` as an already-gathered slot and
  never writes the global vector, so an element that re-solves from committed
  history inside `assemble_cell!` has *nowhere to put the new state*: it freezes
  the committed state across steps instead of advancing it, with no error to
  catch it. State that must survive a step goes through
  [`condense_cell!`](@ref).

  A *stateless* inline implicit solve stays legitimate and is unaffected: if its
  root is a pure function of the gathered args, it is recomputable at any later
  time, and AD through it is approximately-total, limited by the inner
  tolerance. The clean upgrade, when that limit matters, is an analytic tangent
  via the implicit function theorem — the same shape a corrector has.

*Ruled 2026-08-20 after a gain/loss analysis, with the one downstream consumer
explicitly rewritable. Record: `condensation-phase-design.md` §10.1 for the cost
model.*

### What the phase concedes

Recorded so they are not rediscovered as surprises.

**Freshness is not structurally closable, and the residual hazard is named.**
The solver writes `u .+= Δu` outside the package entirely, so no hook sees it.
Four independent guards narrow the window: correctors are stamped and reading an
unstamped or stale one throws naming the cell;
[`condensed_update_linearization!`](@ref) makes the correct sequence the
convenient one; [`rollback_state!`](@ref) invalidates while
[`commit_state!`](@ref) does not (the committed point *is* the last condensed
point); and [`check_derivatives`](@ref) re-condenses at every trial point, so a
stale corrector surfaces as a failing check. **What remains uncovered is the
same vector mutated in place between condensation and a sensitivity sweep.** For
a *solve* that is benign in kind — the residual is still exact, so Newton stalls
visibly rather than converging to a wrong answer. For a *sensitivity* it is
silently wrong, and an optimizer will consume the gradient happily. The
documentation names that case rather than implying the guards are complete.

There is no literal generation counter in the tree; the corrector store's own
validity mask carries the same information at O(1) invalidation cost, and the
docstrings describe what is there.

**Per-quadrature-point corrector storage is the binding constraint at scale.** A
scalar corrector is cheap; a 6×6 slope on a viscoelastic element is ~2.3 GB at
10⁶ cells, and the generic block bigger still. [`Recompute`](@ref) is therefore
a **first-class election**, not a fallback, and it is targeted: memory-bound
*assembled* sweeps. For matrix-free, action-style use — Krylov `mul!`/JVP
sequences at a fixed state — recompute-per-action is not the intended profile
and [`Stored`](@ref) is the election.

The election is exact rather than approximate, which is what makes it an
election rather than a trade of accuracy: the corrector is a closed form in the
converged `(u, q)` — the implicit-function-theorem slopes of the local
conditions — so recomputing it re-runs the arithmetic the condensation already
ran, at the same point. Both shipped condensed elements implement both
elections and produce bitwise-identical matrices. What the election *does* cost
is the freshness guard: with no store there is no stamp, so a `Consistent`
sweep on a never-condensed item silently uses whatever `q` the tail holds
instead of throwing. The q-ordering contract is unchanged and the detection of
its violation is what is traded away, alongside the memory.

The election is not invisible to a kernel's *inputs* either. A recomputing
access point evaluates the closed form, so whatever that form reads becomes a
requirement of every sweep that reads the corrector: the viscoelastic element's
`Recompute()` Jacobian needs `stage_scaling(args.ctx)` and therefore a context,
where its `Stored()` Jacobian reads the retained factorization and runs without
one. The election is a construction-time choice with a call-time consequence,
which is why it lives on the integrator rather than on the sweep.

**The phase is a global barrier.** It must complete over the whole domain before
any evaluation sweep begins. On a shared-memory engine that is one join; on a
future distributed backend it is a real barrier that forecloses overlapping
condensation with anything asynchronous. This is intrinsic to the decoupling —
the decoupling *is* "solve everything, then evaluate" — not an implementation
artifact.

**The generic bootstrap reaches θ and `t` only through a declared local model.**
The decorator's generic [`Consistent`](@ref) path handles `∂F/∂ū` (and its
fused sibling) out of the corrector alone. `dq/dθ = −(∂L/∂q)⁻¹ ∂L/∂θ` needs
something the corrector does not carry: the local operator itself, against a
different right-hand side. The framework cannot see an element's local
equations, so the extension could only ever be an element declaration — which
is what [`local_conditions!`](@ref) is. With it the decorator differentiates
the hook for `∂L/∂q`, `∂L/∂θ` and `∂L/∂t`, factorizes the local operator once
per item, and completes the total against the `∂F/∂q` block; without it those
kinds keep the admissibility rejection, now naming the hook as the third
remedy.

The seam is spelled against the package's own kernel convention —
`local_conditions!(L, cache, args)`, per ITEM, filling the item's stacked local
residual — rather than the design draft's per-quadrature-point argument list.
The draft's form cannot be driven by the framework: it asks for the field value
AT a quadrature point, which only the element's own `CellValues` can produce,
and for a `qprev` the framework has no vocabulary for. The `args` form carries
both without the framework interpreting either, and it is what makes one
factorization per item — the whole cost argument for the route — well defined.
It is marked experimental for that reason: the contract is validated, not
frozen.

**∂F/∂q is a target, not a slot Jacobian.** The block is field × internal
shaped, so it fits neither the operator's square matrix nor the per-slot
component bag. It gets a rectangular sparse target of its own
([`allocate_internal_jacobian`](@ref)/[`update_internal_jacobian!`](@ref)) whose
pattern is the cell loop's `celldofs × internal range`, scattered through the
two-index `assemble!` the transfer operators already use. Columns are disjoint
between items by construction — an item owns its internal range alone — so the
sweep needs no new assembler and no new race analysis. [`assemble_slot_jacobian!`](@ref)
refuses `:q` outright rather than silently scattering a rectangular block into
a square pattern.

*Record: `condensation-phase-design.md` §5.2, §6, §7, §10.2, §10.4, §10.6.*

## The operator layer

### Why the request kinds stayed the extension axis

**Decision.** Request kinds remain public, and a downstream kind is defined
entirely outside this package — see [Extension points](design.md#Extension-points).

This is a **correction to the draft**, which anticipated kinds demoting to
internal tags once operators became the public vocabulary. They did not, because
the operator-algebra step that would have made an operator type the unit of
extension has not been taken and its decision gate is still open. Meanwhile the
adversary's central objection stands empirically: the kind axis is *open and
proven by test* — a downstream kind can be defined with zero source edits,
validating at setup, allocating nothing, and constant-folding — and replacing it
with records of functions would have **inverted** the extensibility claim,
closing a proven-open axis in exchange for making an already-cheap axis
marginally cheaper.

Two kinds did disappear, and for a reason worth keeping: a state-independent
matrix sweep and a state-independent vector sweep are not distinct computations
from their state-dependent counterparts. They are the same computations with an
empty slot set. The operator induced by a bilinear form is assembled by a
Jacobian sweep whose plan gathers nothing; the load vector of a linear form is a
residual sweep whose plan gathers nothing.

`reinit_values!(cache, cell, kind)` is the other reason kinds cannot be removed:
it exists precisely so an element can reinitialize only the values a given sweep
shape needs, and that requires a name for the shape.

*Ruled 2026-08-19; the demotion did not happen. Record:
`report2-adversarial-review.md` Angles 1 and 3.*

### Eager by default, lazy only with a consumer

**Decision.** Materialization is a **constructor choice**, stated in the type,
never inferred. The default is eager: components are assembled into matrices
sharing one sparsity pattern and folded by an `O(nnz)` combination. Eager is the
form that has no invalidation semantics to get wrong.

One lazy form exists because a downstream consumer has it: a
[`StageBlockOperator`](@ref) that keeps its blocks separate and applies the
weighted sum at application time, so that a step-size change costs a field write
instead of a re-fold. That operator stores its scalar — a deliberate exception,
confined to the lazy variant.

The prohibition that shaped this: the one downstream package living inside the
SciML ecosystem carries a written rule against subtyping the ecosystem's lazy
operator root, because the branch it would land in drops the matrix-reuse flags.
Concrete-versus-lazy confusion is not a hypothetical cost.

**The pattern group is checked, not assumed.** Components of one operator share
a sparsity pattern — for the shared-memory compressed-column backend, by
aliasing the index arrays and giving each component private values. Two
downstream packages and three separate code paths formed a scheme matrix by
combining nonzero arrays under an *unstated* assumption that the patterns
matched, one of them across two different storage formats.
[`share_pattern`](@ref) and [`combine!`](@ref)'s pattern assertion are what turn
that assumption into an invariant. Square components share the group;
rectangular members are legal in a bag and are never combined.

*Ruled 2026-08-19 under the adversary's constraints. Record:
`report2-adversarial-review.md` Angle 6, `report1-architect-operator-algebra.md` §1.*

### Three rules reserved for the operator algebra

The operator-type algebra — `jacobian(op, :u)`, `weighted_jacobian(op, slots)`,
`op₁ + op₂` and siblings — is a **ruled direction that has not been
implemented**, and its decision gate is still open. The draft states it as
though it were contract; it is not. What is settled are the three rules that
would govern it, each answering a specific failure observed in general-purpose
operator libraries, and they are worth recording because they already constrain
what may be added.

- **Rule A — bounded depth.** Derived operators are built from a base operator,
  never from other derived operators, with one exception: a sum over derived
  members. A weighted combination `w₁·∂ᵤ + w₂·∂_du` is *one* type parameterized
  by its slot set, not a nesting of scaled and added wrappers. The failure this
  avoids is type explosion: deep composition types produce inference cliffs and
  unreadable errors. A general-purpose library cannot impose this rule because
  its operator universe is open; here the set is small and
  finite-element-specific, so it can.
- **Rule B — scalars are payload.** The slot set is a type parameter; the
  weights are an argument to the update. A change of step size therefore never
  changes a type, and weight/slot agreement is a matter of dispatch rather than
  runtime validation.
- **Rule C — one mutating entry point.** Every derived operator is written by an
  update call and by nothing else; every query is free of side effects and never
  triggers an update. The failure this avoids is update-semantics ambiguity:
  libraries offering an in-place update, an out-of-place update, and a lazy call
  form make "is this matrix current?" unanswerable. Staleness is solver
  knowledge and stays with the solver.

**The guard on sums.** A sum is where a term-declaration layer would try to
reappear, so the boundary is stated where the type is defined: a sum carries
*structure* — that it is a sum of opaque operators — and never term-shaped
*data*. No per-term context maps as construction data, no domain algebra, no
weight symbolism inside a type. Terms evaluated at different times
(generalized-α is the canonical case) pass their contexts as a per-summand
argument, because a context is call data. If a sum ever needs to *store* a
per-term mapping, that is the rejected declarative-form layer arriving in a new
costume, and the answer is separate sweeps.

**The accepted cost of this direction**, recorded at ruling time: derived
operators share the base operator's engine, so two of them cannot be swept
concurrently, which makes the reserved "one plan, many caches" object model
necessary rather than optional.

*Ruled 2026-08-19; not implemented. Record:
`report1-architect-operator-algebra.md` §1, `report2-adversarial-review.md` §Verdict.*

## Scope

### Why the hub principle bounds what the package owns

FerriteOperators is an **interface package**. It owns contracts and the seams
that make them extensible; it does not own every implementation of them. This is
why the AD backend is a seam rather than a fork of the engine, why a downstream
request kind needs zero source edits here, and why a solver-shaped need is
answered by asking which *seam* is missing rather than by moving the downstream
code upstream.

The rule has teeth in both directions. It licensed deleting the open-args
apparatus (an unused generality is not a contract) and it licensed *keeping* the
kind axis open (a used generality is). It is also why the honest cost of the
proposed algebra direction was recorded rather than argued away: narrowing a
proven zero-source-edit extension axis is a real loss, accepted on the grounds
that an operator is the better-shaped unit — and mitigated by *exporting* the
seams involved rather than leaving downstream code to reach for qualified
internal names.

### Why the package speaks the MFEM/libCEED decomposition

The design follows the fundamental finite-element operator decomposition
`A = Pᵀ Gᵀ Bᵀ D B G P` popularized by MFEM and libCEED, and says so
deliberately: the vocabulary is what keeps *how much of the operator is
materialized* separate from *what the physics is*.

| block | meaning | where it lives here |
|---|---|---|
| `P` | local↔global including constraints | constraint handling around the element contribution |
| `G` | element restriction (gather/scatter) | slot loaders and element-assembly dof maps |
| `B` | basis evaluation at quadrature points | element-owned `CellValues` |
| `D` | pointwise quadrature-point operation | the quadrature-point kernel tier |

MFEM's **assembly levels** — FULL, ELEMENT, PARTIAL, NONE — name the strategy
axis, orthogonal to the device axis. Full sparse assembly and stored element
matrices are the two shipped levels; the [`QVector`](@ref) is exactly the qdata
store a partial-assembly level needs, which is why it is described as
matrix-free precomputation.

**The deliberate deviation from libCEED:** an element may own the whole `Bᵀ D B`
block, because condensed materials with element-level local solves do not
decompose into a pointwise `D`. The quadrature-point tier is opt-in, not
mandatory. The same frame explains why kernel-level AD is the right sensitivity
granularity: `∂F/∂p` acts on `D`, the small dense pointwise map.

One nuance the phase changed: post-condensation, the *evaluation* sweeps of a
condensed element **are** per-quadrature-point decomposable, so the pointwise
tier becomes reachable for them. The condensation phase itself is not.

*Vocabulary adopted with the v2 target architecture; the decomposability note is
2026-08-20. Record: `condensation-phase-design.md` §11 (L10).*
