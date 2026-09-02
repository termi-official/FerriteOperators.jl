```@meta
CurrentModule = FerriteOperators
```

# Design rationale

This page answers *why is it this way*. It is the one sanctioned home for
reasoning in the documentation: the load-bearing reason behind each design
decision, and the alternatives that are rejected and on what grounds. Entries
are organized by **design question**.

Everything else states the contract. [The layer contract](design.md) says which
layer owns which piece of information and what the extension points are;
[Writing elements](../elements.md) and [Operators and entry points](../operators.md)
say what an author must write. If a statement here disagrees with those pages,
those pages are right and this one is stale — report it.

## Argument bundles

### Why the kernel argument bundle is a closed, annotatable record

**Question.** Is the bundle a kernel receives an open structural interface that
a downstream operator family re-implements, or one fixed type per item family?

**Answer.** One concrete argument type per item family — [`CellArgs`](@ref) for
cell kernels, [`FacetArgs`](@ref) for facet kernels, [`AlgebraicArgs`](@ref) for
algebraic items — with no common supertype, and element authors *may* annotate
`args::CellArgs`.

**Why.** An open bundle is a second extension axis, and it is the redundant one.
What a downstream family genuinely needs — an extra per-sweep quantity — is
served by `ctx`, which is open by design and costs nothing to keep open.
Openness in the *bundle* buys nothing beyond that and pays for it in validation
surface: a per-family declaration of which argument type it builds threaded
through setup, a type-parameterized validator (because `hasmethod` against an
abstract queried type misses concretely annotated methods), a reflection warning
whose only job is to make those unnecessary, and every record rebuild promoted
to a public seam.

Closing the type is what makes annotation legal, and it turns those rebuilds
into ordinary private constructors: [`with_states`](@ref),
[`with_parameters`](@ref) and [`with_context`](@ref) are named record updates
that keep the field order written once instead of at every AD seeding site.

### Why a record and not positional arguments

**Question.** Why does a kernel take one record rather than a flattened
argument list?

**Why.** Three reasons, in increasing order of weight.

- **Transposition.** `states` and `p` are opaque bags. Positional arguments of
  opaque type produce silent wrong-argument errors instead of `MethodError`s —
  exactly the failure mode the request types exist to cure. Discriminating one
  sweep from another by the *shape* of an argument is dispatch-ambiguous on top
  of that, so a downstream package needing a differently-shaped kernel set has
  to invent new function names.
- **Frozen signatures.** A fixed positional signature promises the argument list
  never grows. When it must grow, the pressure goes into whichever argument is a
  bag — the "smuggle it through `p`" failure described below, arrived at from
  the other direction.
- **Seeding arithmetic, which is what settles it.** Every derivative sweep seeds
  through exactly three channels: states, parameters, context — and there are
  many more sweeps than channels. The framework therefore needs *three* rebuild
  methods per argument family, not one per sweep. That ratio is a property of
  the record. Under positional arguments the framework cannot express "the same
  call with one argument replaced" generically, and the count becomes
  (sweeps × signature shapes).

## Where information lives

### The principle: decorate capabilities, pass data

> **Structure is decorated; data is passed.** Anything fixed before a sweep
> starts and constant across it belongs to a type — an operator, a cache, or a
> decorator around a cache. Anything that varies per evaluation is an argument.

`ctx` is an argument because a time is per evaluation. Weights are arguments
because a step size changes. The element cache is structure. Whether a Jacobian
is computed analytically or by differentiating the residual is *structure* — and
that is the observation the AD decorator rests on.

**The failure the principle rules out is decorating the wrong layer**: a time
discretization wrapped around the element interface. A scheme is data (per step:
`uprev`, `Δt`) masquerading as structure, so every new scheme becomes a new copy
of the wrapper — a parallel parameter type, a parallel `Union` encoding the fork
in the type system, the element assembly methods duplicated per wrapper — and no
copy composes with the derivative machinery. The AD decorator decorates a
*capability* onto a cache, which is genuinely structure: whether this element
can produce this derivative does not change between steps.

### Why `ctx` is an argument — and not `p`, and not operator state

**Answer.** [`TimeIntegrationContext`](@ref) reaches kernels as a field of the
args record. It is neither folded into the parameter bag nor stored on the
operator.

**Rejected: time inside `p`.** It fails four ways at once. The framework must
see `t` where a parameter bag hides it — the `∂F/∂t` sweep seeds a Dual through
it and the engine threads `γ̃` to element-local stage problems — so every wrapper
needs an unwrapping convention; a user's own [`query_cell_parameters`](@ref)
override silently drops that wrapper; and `∂F/∂θ` and `∂F/∂t` become
indistinguishable, because they seed the same channel. That last one is a
silent-zero defect: a time sweep seeding bare time through the parameter channel
hands an element reading its context per the documented contract a derivative of
exactly zero, with no error anywhere.

**Rejected: per-sweep scalars as operator state (the "solver as a decorator"
shape).** A number stored on a structure has to be invalidated, and invalidation
is solver knowledge that no operator can infer. Staleness stays with the solver,
which is also why every derived quantity here is written by an update call and
never by a query. The witness for the alternative is a backward-Euler path
guarded by `Δt ≈ Δt_last`: an `isapprox` predicate, so a step size drifting
within tolerance silently reuses a stale matrix.

Per-item contexts — the reservation for local time stepping — do not contradict
this. A context *source* is structure and lives on the operator; the value it
yields for an item is call data and reaches the kernel as the argument.

Elements read the context only through accessors ([`evaluation_time`](@ref),
[`stage_scaling`](@ref)), which is what makes a richer downstream context type
substitutable. An "open channel" means nothing unless the channel has a reader:
a quantity elements reach as a struct field cannot be decorated.

### Why there is no structural unknowns/data split

**Answer.** `states` is one flat bundle of slots. There is no grouping of slots
into "unknowns" versus "data".

**Why.** Roles are **request-relative**: a request kind seeds what it names, and
everything else in `states` is data for that sweep. `q` is data in a residual
sweep and an unknown in a `∂F/∂q` sweep — the same slot, two roles, decided by
the request. A structural grouping would force kernels to read one slot from
different groups across phases, making the kernel phase-aware, and would
re-create precisely the slot-tuple-carrier problem the flat bundle removes.

The two use cases usually cited for such a split are served elsewhere and
better: **IMEX** is term-split operators issuing different request sets, and
**local time stepping** is item restriction.

## Differentiation

### Why AD is a decorator cache rather than engine machinery

**Answer.** [`ADElementCache`](@ref) wraps an element cache that does not serve
some request analytically, implements the derivative requests by seeding the
inner's residual kernel, and forwards everything else. Wrapping happens once at
`setup_operator` time. The engine has no differentiation in it.

**Why.** The choice of route is structure, so under the principle above it
belongs in a type. Making it one has four consequences, and together they are
what pays for the type.

- **There is no per-cell fork.** A fork living in the engine is a branch inside
  every kernel-dispatch method, consulted per cell and answered by a trait.
  Post-decoration every resolved cache is analytic from the engine's point of
  view, so one generic kernel-dispatch body serves every case.
- **Differentiation machinery is structurally absent when unused.** An operator
  that never differentiates carries none, because no cache was wrapped — a
  property that has to be engineered and maintained when the machinery instead
  sits on the workspace behind a declaration.
- **Composition is per-inner.** Capability that is all-or-nothing across a
  fan-out forces a composite whose inners are partly analytic to differentiate
  the *whole* composite residual. Wrapping is per-cache, so analytic blocks come
  from their own kernels and only the non-analytic ones are differentiated; the
  Jacobian of a sum is the sum of the Jacobians, so this is exact. The
  constructor's policy is binding: wrap the non-analytic inners **as one maximal
  sub-composite**, never individually — naive per-inner wrapping costs one full
  seeding pass per wrapped inner and is *worse than not wrapping at all* once
  two or more inners need it.
- **The backend is an extension seam.** [`ForwardDiffAD`](@ref) is the default,
  but the decorator's contract is what a backend implements, so a reverse-mode
  backend is a package extension rather than a fork of the engine.

**What deliberately does not live in the decorator:** the admissibility rules
and [`check_derivatives`](@ref) are framework-side. The decorator *implements*;
the constructor *enforces*. Likewise the rejection of gather-time-reconstructed
slots in a differentiated position is a per-call check, because a slot's source
is call data.

AD **output** buffers live per worker on the workspace as
[`SensitivityBuffers`](@ref) rather than in a task-owned bag, because an
`AssemblyTask` is reused across subdomains of differing `ndofs_per_cell` while
an `AssemblyWorkspace` is correctly per-subdomain-per-worker; a task-owned
output bag would be sized for the wrong subdomain.

The decorator does not broaden [`provides_analytic`](@ref), because two
questions need two predicates: `provides_analytic` is "*is there a hand-written
kernel*" and forwards through the decorators unchanged, while
`FerriteOperators.serves_kind` is "*does the resolved cache serve this kind, by
kernel or by a decorator's generic route*". One trait answering both needs
per-kind non-broadening overrides wherever a check wants the first.

### Why `FusedFromSplit` is required rather than an optimization

**Question.** An element with analytic split kernels but no fused kernel could
fall back to AD for the fused request. Why the extra decorator?

**Answer.** [`FusedFromSplit`](@ref) serves a fused Jacobian+residual request by
issuing the split kernels back to back. Without it, [`check_derivatives`](@ref)
false-negatives on a *wrong* analytic Jacobian: AD would silently substitute the
correct Jacobian into the fused path, and the check would pass.

### Why finite differences are operator-level and AD is per cache

**Question.** Both are derivative routes. Why is one a cache decorator and the
other an operator entry point?

**Answer.** Operator-level [`FiniteDifferenceSensitivity`](@ref) differences
`evaluate!`, so **every term enters** — including a facet term whose cache
carries no analytic sensitivity kernel — and no kernel ever sees a `Dual`. The
[`ADElementCache`](@ref) decorator is the per-cache route: analytic kernels win
cache by cache, and it is volumetric only.

There is no cache-level finite-difference decorator, and that is the settled
division rather than a gap. It would differentiate the volumetric kernel exactly
like the AD one, and would therefore lose the single property that makes the
operator-level method worth keeping. Its remaining niche — kernels that cannot
carry `Dual`s — has no consumer.

## Condensation and internal variables

### Why condensation is an explicit phase

**Answer.** [`condense_internal!`](@ref) is a traversal of its own. It solves
every quadrature point's local problem, writes the trial `q` into the `[ū; q]`
tail, and stores each item's corrector. Every assembly sweep afterwards is a
pure evaluation at frozen `q`.

**Why: model/solver decoupling, completed at the local level.** The global level
is decoupled by slots, weights and `ctx` between them, which let a solver drive
BDF or Newmark without an element knowing which. A condensed element that owns
*both* its local model and its local time discretization, and runs both inside
every kernel, is not decoupled at the local level: a Jacobian sweep and a
residual sweep over the same cells then count their local solves twice.

Splitting the phase off buys three things that a fused local solve cannot have:

- **Derivative machinery becomes generic.** With the kernel pure, a plain AD
  fallback computes a well-defined `∂F/∂·|_q`, and the correction is a stored
  matrix rather than an iteration to differentiate through. Condensed elements
  get parameter sensitivities without a hand-written kernel: the payoff case,
  `dF/dθ = ∂F/∂θ|_q + ∂F/∂q·dq/dθ`, matches a finite-difference referee to
  ~1e-10 on the shipped power-law element.
- **Solver families become caller sequencing.** Exact condensed Newton,
  multilevel Newton, frozen-`q` modified Newton, and line search without
  re-condensing are all orderings of two calls. None of them requires element
  cooperation.
- **Non-convergence becomes data.** [`CondensationReport`](@ref) is isbits and
  folds as a monoid, so it crosses a device boundary — which an exception
  cannot.

The `weights` argument of the phase is where the solver's chain-rule slopes
enter. The corrector is defined as `dq/dū` — with respect to the *primary*
unknown, with slot reconstructions pre-chained — so the weighted-Jacobian
machinery is untouched, keeps folding frozen-`q` partials with the solver's
scalars, and nothing double-counts. The slopes must reach the element *during*
condensation, because that is where they are chained in.

### Why partial versus total is explicit vocabulary

**Answer.** Jacobian-shaped kinds carry a [`CorrectionMode`](@ref).
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

The silent-wrong class this creates is the **missing combination**: nothing
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
one is a `TypeError`. The correction-mode parameter also sits *between* the slot
and buffer type parameters of the kind family, so bare annotations stay
`UnionAll` and a non-condensed element needs no changes to accommodate it.

### Why `q` is a slot

**Answer.** A condensed element's internal state arrives as an ordinary slot
whose source, [`InternalSource`](@ref), carries the internal-dof range as its
restriction. Global `[ū; q]` storage is unaffected; this is a statement about
the gather level only.

**Why.** It collapses a family of dedicated element-side hooks and cache fields
into one already-existing hook. It makes `∂F/∂q` fall out for free as an ordinary
per-slot Jacobian — load-bearing twice, as the block the generic corrector
combination needs and as the block a Schur-complement consumer wants. And it
makes an otherwise necessary caveat unexpressable: the condensed tail of a
reconstructed slot cannot be misinterpreted by an element, because a
reconstructed slot's source is the field space and structurally cannot touch
`q`.

The obvious alternative — keeping internal degrees out of slots entirely, on
the grounds that `q` is solved inside the kernel and so is not an independent
variable of the residual — rests on a premise the condensation phase makes
false. After the phase, `q` *is* an independent variable of a pure residual.

### Why kernels are pure

**Answer.** Condensation is unconditionally its own traversal. No election lets
an operator fuse it into the residual sweep.

**Why.**

- **Kernel purity is an invariant** rather than a convention. The admissibility
  question turns from "did this kernel differentiate through an iteration" into
  "is a corrector present", which is structurally checkable. The hazard where AD
  runs through a *deliberately inexact* local solve — wrong by a
  solver-controlled amount, unlike a converged loop which is asymptotically
  right — dies structurally rather than by rule.
- There is exactly **one writer of `q`**.
- **Solver families need zero element cooperation** — frozen-`q`, multilevel
  Newton, and line-search policies are all caller sequencing.
- A fused route would be structurally primal-only, so purity costs no
  capability. What it would buy is a second driver body, dual freshness
  semantics and dual reporting, including a once-per-item problem: a fused sweep
  enters the element more than once per item under split-Jacobian-residual and
  AD chunk passes.

**What it costs.** One extra item traversal per exact-Newton iterate. For a
fused-analytic user that saves no local solves at all and pays gather, geometry
reinit and values reinit for nothing; it hits cheap-local-solve elements
hardest, while modified-Newton and multilevel-Newton callers can net win and a
line search is neutral.

**The documented boundary: stateful do-it-yourself inline elements fail
silently.** A kernel receives `q` as an already-gathered slot and never writes
the global vector, so an element that re-solves from committed history inside
`assemble_cell!` has *nowhere to put the new state*: it freezes the committed
state across steps instead of advancing it, with no error to catch it. State
that must survive a step goes through [`condense_cell!`](@ref).

A *stateless* inline implicit solve stays legitimate and is unaffected: if its
root is a pure function of the gathered args, it is recomputable at any later
time, and AD through it is approximately-total, limited by the inner tolerance.
The clean upgrade, when that limit matters, is an analytic tangent via the
implicit function theorem — the same shape a corrector has.

### What the phase concedes

Recorded so they are not rediscovered as surprises.

**Freshness is not structurally closable, and the residual hazard is named.**
The solver writes `u .+= Δu` outside the package entirely, so no hook sees it.
Three independent guards narrow the window — corrector stamping,
[`rollback_state!`](@ref) invalidating where [`commit_state!`](@ref) does not,
and [`check_derivatives`](@ref) re-condensing at every trial point — but **what
remains uncovered is the same vector mutated in place between condensation and a
sensitivity sweep.** For a *solve* that is benign in kind: the residual is still
exact, so Newton stalls visibly rather than converging to a wrong answer. For a
*sensitivity* it is silently wrong, and an optimizer will consume the gradient
happily. The documentation names that case rather than implying the guards are
complete.

**Per-quadrature-point corrector storage is the binding constraint at scale**, and
it is why [`Recompute`](@ref) is a first-class election rather than a fallback:
a scalar corrector is cheap, but a 6×6 slope on a viscoelastic element is
~2.3 GB at 10⁶ cells, and the generic block is bigger still.

The election is exact rather than approximate — the corrector is a closed form
in the converged `(u, q)`, so recomputing re-runs the arithmetic the
condensation already ran, at the same point — which is what makes it an
election rather than a trade of accuracy. What it costs instead is the freshness
guard (with no store there is no stamp) and, since a recomputing access point
evaluates that closed form, whatever the form reads becomes an input requirement
of every sweep that reads the corrector. Both consequences are documented with
the election in [Condensed elements](../elements.md#Condensed-elements-(internal-variables)).

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
residual — rather than per quadrature point. A per-quadrature-point argument
list cannot be driven by the framework: it asks for the field value AT a
quadrature point, which only the element's own `CellValues` can produce, and for
a `qprev` the framework has no vocabulary for. The `args` form carries both
without the framework interpreting either, and it is what makes one
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

## The operator layer

### Why request kinds are the extension axis

**Answer.** Request kinds are public, and a downstream kind is defined entirely
outside this package — see [Extension points](design.md#Extension-points).

**Why.** The kind axis is open and proven by test: a downstream kind can be
defined with zero source edits, validating at setup, allocating nothing, and
constant-folding. Replacing it with records of functions would **invert** the
extensibility claim — closing a proven-open axis in exchange for making an
already-cheap axis marginally cheaper.

`reinit_values!(cache, cell, kind)` is the second reason kinds cannot be
removed: it exists precisely so an element can reinitialize only the values a
given sweep shape needs, and that requires a name for the shape.

State-independence is *not* a kind of its own, though: the operator induced by a
bilinear form is assembled by a Jacobian sweep whose plan gathers nothing, and
the load vector of a linear form is a residual sweep whose plan gathers nothing.
They are the same computations with an empty slot set, so they need no separate
names.

### Eager by default, lazy only with a consumer

**Answer.** Materialization is a **constructor choice**, stated in the type,
never inferred. The default is eager: components are assembled into matrices
sharing one sparsity pattern and folded by an `O(nnz)` combination. Eager is the
form that has no invalidation semantics to get wrong.

One lazy form exists because a downstream consumer has it: a
[`StageBlockOperator`](@ref) that keeps its blocks separate and applies the
weighted sum at application time, so that a step-size change costs a field write
instead of a re-fold. That operator stores its scalar — a deliberate exception,
confined to the lazy variant.

The prohibition that shapes this: the one downstream package living inside the
SciML ecosystem carries a written rule against subtyping the ecosystem's lazy
operator root, because the branch it would land in drops the matrix-reuse flags.
Concrete-versus-lazy confusion is not a hypothetical cost.

**The pattern group is checked, not assumed.** Components of one operator share
a sparsity pattern — for the shared-memory compressed-column backend, by
aliasing the index arrays and giving each component private values. Two
downstream packages and three separate code paths form a scheme matrix by
combining nonzero arrays under an *unstated* assumption that the patterns match,
one of them across two different storage formats. [`share_pattern`](@ref) and
[`combine!`](@ref)'s pattern assertion are what turn that assumption into an
invariant. Square components share the group; rectangular members are legal in a
bag and are never combined.

## Scope

### Why the hub principle bounds what the package owns

FerriteOperators is an **interface package**. It owns contracts and the seams
that make them extensible; it does not own every implementation of them. This is
why the AD backend is a seam rather than a fork of the engine, why a downstream
request kind needs zero source edits here, and why a solver-shaped need is
answered by asking which *seam* is missing rather than by moving the downstream
code upstream.

The rule has teeth in both directions: an unused generality is not a contract
and may be removed, while a used one is a contract and stays open. Where a
direction does narrow a proven extension axis, the honest cost is recorded
rather than argued away, and the seams involved are *exported* rather than left
for downstream code to reach as qualified internal names.

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
axis, orthogonal to the device axis. Full sparse assembly is the shipped level;
the [`QVector`](@ref) is exactly the qdata store a partial-assembly level needs,
which is why it is described as matrix-free precomputation.

**The deliberate deviation from libCEED:** an element may own the whole `Bᵀ D B`
block, because condensed materials with element-level local solves do not
decompose into a pointwise `D`. The quadrature-point tier is opt-in, not
mandatory. The same frame explains why kernel-level AD is the right sensitivity
granularity: `∂F/∂p` acts on `D`, the small dense pointwise map.

One nuance the condensation phase adds: the *evaluation* sweeps of a condensed
element **are** per-quadrature-point decomposable, so the pointwise tier is
reachable for them. The condensation phase itself is not.
