```@meta
CurrentModule = FerriteOperators
```

# The layer contract

Three layers share the assembly pipeline, and every design question about
"where does this belong" is answered by which layer owns the information.

| layer | owns | in this package |
|---|---|---|
| **term** | scheme-agnostic integrands over one work item | element caches and their request-typed kernels |
| **operator** | evaluating a set of terms at `(states, p, ctx)` | the engine, its workspaces, and the entry points |
| **scheme** | composing evaluations into a discretization | solver code: history vectors, slot sources, weights, contexts |

## The term layer

An element expresses an integrand. It reads slot *values*, a parameter view,
and a per-sweep context — and nothing else. It does not know which time
discretization produced the values in its slots, whether a Jacobian will be
derived from it by automatic differentiation, or how many other terms share
its cell.

The consequence that makes the layering worth having: one element serves every
scheme. A residual kernel that reads `args.states.u` and `args.states.du`
serves backward Euler, BDF-k, SDIRK stages, and a fully implicit Runge-Kutta
stage row without a line of change, because each of those merely supplies
different slot values.

A *hand-fused* integrator — one that derives its own discretization from
`uprev` and `ctx`, or that computes a scheme's combined matrix directly — is a
scheme-layer object living in an element cache. That is a legitimate authoring
choice (it is how a deliberately manual first-order discretization, or a
multilevel-Newton element with a rate-coupled local problem, is written), and
the framework serves it through the same request protocol. The distinction is
one of authorship, not of mechanism.

## The operator layer

An operator evaluates its term set at `(states, p, ctx)` and scatters the
result. It owns the item loop, the per-worker workspaces, the geometry cache
reinitialization, the slot gathers, the request materialization, and the
choice between an analytic kernel and its derivative fallback.

It owns no time discretization and no scheme coefficients. `states` arrive as
whatever the solver assembled them to be; `ctx` carries the scalars of *this*
evaluation; weights are per-request payload.

[`evaluate!`](@ref) is layer-polymorphic by design. It evaluates whatever the
integrator encodes — a nonlinear residual, the action of the linear operator a
bilinear form induces (its element matrix acting on the element vector), or a
hand-fused scheme residual — and the name says exactly that much and no more.

## The scheme layer

The solver owns the discretization. It holds history vectors and tableau
coefficients, decides what each slot contains for this evaluation, supplies
the chain-rule weights that fold per-slot Jacobians into the matrix it solves
with, and constructs the context.

Where a scheme needs several *evaluation times* — generalized-α evaluating
stiffness at `tₙ₊₁₋αf` and inertia at `tₙ₊₁₋αm` — it splits the problem into
term-subset operators and runs one sweep per term at its own context,
combining the results as components. One monolithic kernel cannot host two
evaluation times, and the context is not the place to smuggle a second one:
this is a modeling requirement on the operator split, not a gap in the context
channel.

## The channel decision table

Every piece of information a kernel might want has one channel. The decision
is made by the *shape* of the information, not by which layer produced it.

| shape | channel | notes |
|---|---|---|
| dof-shaped, one value per dof | a **slot** (`args.states.<name>`) | histories, rates, adjoint directions, stochastic realizations — anything the operator can gather with a dof map. Slots may be plain vectors, [`AffineRate`](@ref) reconstructions, or — for a condensed element's internal state `q` — [`InternalSource`](@ref) restrictions (see [Condensed elements](../elements.md#Condensed-elements-(internal-variables))). |
| point-shaped, one value per quadrature point | quadrature storage and the **query seams** | [`QVector`](@ref) for stored per-QP data; [`query_cell_parameters`](@ref) / [`query_facet_parameters`](@ref) for element-owned gathers, including parameter fields. |
| a scalar of *this* sweep | **`args.ctx`** | `t`, `Δt`, `γ̃` in [`TimeIntegrationContext`](@ref). A scheme with richer per-sweep scalars passes its own context type; framework code touches contexts only through [`evaluation_time`](@ref), [`with_time`](@ref) and [`stage_scaling`](@ref). |
| configuration, constant across the sweep | **`args.p`** | material parameters and the user's bag. Never time, never history. `p` stays opaque: a solver-side wrapper is unwrapped by the cache's own [`query_cell_parameters`](@ref). |
| per-worker mutable working memory | **element cache fields** | duplicated — not aliased — per worker by `duplicate_for_device`, see [storage classes for elements with local problems](../elements.md). |
| a scheme scalar attached to a slot | request payload | rides on the request instead of the args bundle — that is what [`WeightedJacobianKind`](@ref) does with its weights. |

Two consequences worth stating explicitly.

**Time reaches elements through `ctx`, and only through `ctx`.** If `t` could
hide inside `p`, every wrapper would need an unwrapping convention, and the
framework itself must see `t` to seed the ∂F/∂t sweep. A kernel that reads its
time from `args.p` gets a silently zero time sensitivity.

**`γ̃` is not a rate slope.** It is the normalized local stage interval of the
element-local internal-variable problem, fixed by `q = q_ref + γ̃·g(·, q)`.
Under backward Euler `γ̃` and a rate slope happen to be reciprocals, which
makes `1/γ̃` accidentally right there and wrong everywhere else. Rate slopes
belong to the slot that carries the reconstruction.

## Extension points

The framework is extended by adding methods to a small number of generic
functions rather than by subclassing drivers.

**New request kinds** — a kind is defined entirely outside this package. The
built-in kind families are trait *defaults*, not a closed world: what a sweep
does with the workspace is a set of overloadable predicates, so a downstream
kind reuses the built-in driver bodies rather than reimplementing them.

The complete recipe:

```julia
struct MyKind end
struct MyRequest{M <: AbstractMatrix} <: FerriteOperators.AbstractAssemblyRequest
    K::M
end

# 1. The kind → request association (the pure form and the executing form).
FerriteOperators.request_type(::MyKind) = MyRequest
FerriteOperators.materialize_request(::MyKind, ws) = MyRequest(ws.Ke)

# 2. What the sweep does with the workspace. Return literals: these guard the
#    driver's branches and must fold away.
FerriteOperators.assembles_matrix(::MyKind) = true
# assembles_vector / depends_on_unknowns default to `false` for a new kind.

# 3. The driver, per ITEM FAMILY. Annotate the workspace: `execute_kind!` is
#    looked up as `(kind, task, workspace)`, so an unannotated method also
#    catches the facet and algebraic workspaces, which a cell body cannot drive.
FerriteOperators.execute_kind!(kind::MyKind, task, ws::FerriteOperators.AssemblyWorkspace) =
    FerriteOperators.primal_cell_sweep!(kind, task, ws)
```

The annotation is not optional on an operator carrying facet or algebraic
items. `execute_single_task!` dispatches on the workspace, and the built-in
kinds get away with unannotated cell methods only because the facet and
algebraic methods are more specific *on the kind argument*
(`execute_kind!(::PrimalKind, task, ws::FacetItemWorkspace)`). A downstream
kind is outside those unions, so it has no such method to lose to: an
unannotated `execute_kind!(::MyKind, task, ws)` is the only candidate for every
family and hands a `FacetItemWorkspace` to `primal_cell_sweep!`. Give the kind
one method per family it must serve, and an explicit `= nothing` for each
family it deliberately does not — which is exactly how the built-in kinds
declare that facet items carry no condensed internal state.

A kind whose sweep needs per-worker scratch beyond `ws.Ke`/`ws.re` reads
[`SensitivityBuffers`](@ref) through `ws.sensitivity` — structurally present
whenever the operator's integrator [`needs_ad_decoration`](@ref) (any
`AbstractNonlinearIntegrator`) and `nothing` otherwise, no family declaration
required. `materialize_request(::MyKind, ws, task)` (the 3-arg form) is where
a sensitivity-shaped kind binds it; see the five built-in sensitivity kinds
for the pattern.

A kind computing a global scalar or tensor rather than something scattered
declares the item families it integrates over instead, which routes it to the
**value-returning** driver: the per-item kernel *returns* its contribution and
the sweep folds the returned values, so there is no request type, no
assembler, and no workspace state at either end.

That one declaration replaces the per-family spelling above.
[`reduction_families`](@ref) answers `execute_kind!` for every family (the
named ones run that family's reduction driver body, the rest contribute
nothing), `sweep_family` (a declaring kind is value-returning) and the
structural precondition that decides which subdomains a reduction requires and
traverses.

```julia
struct MyFunctionalKind end
FerriteOperators.reduction_families(::Type{<:MyFunctionalKind}) = (:cells,)
FerriteOperators.has_cell_request(::Type{<:MyFunctionalKind}) = false

# The type the reduction accumulates in. Optional on a sequential device, where
# the first contributing item fixes it; REQUIRED on a parallel one, whose
# per-worker partials are allocated before the batch runs.
FerriteOperators.functional_value_type(::MyFunctionalKind) = Float64

# the kernel hook the provided body calls
function FerriteOperators.evaluate_cell_functional(::MyFunctionalKind, cache::MyCache, args)
    # ... return this cell's contribution, or `nothing` for none ...
end

value = FerriteOperators.run_reduction(MyFunctionalKind(), op, states, p, ctx)
```

Declaring the value type seeds every worker's fold with `zero(T)`, so a worker
that sees no contribution hands back the reduction's additive identity rather
than a "nothing yet" marker, and a kernel returning some other type is an
`ArgumentError` naming the declaration instead of a silently widened
accumulator.

A reduction whose family bodies are NOT the provided ones keeps its own
`execute_kind!` methods — they are strictly more specific than the derived
route — and still declares `reduction_families` for the structural half.
[`CondensationKind`](@ref) is that case: value-returning over cells and
algebraic items, with a write-back its driver bodies do and no provided body
has.

[`FunctionalKind`](@ref) is exactly this with a tag, and
[`evaluate_functional`](@ref) is its entry point.

Elements then serve it like any built-in kind — `provides_analytic(::Type{<:MyCache}, ::MyKind) = true`
plus an `assemble_cell!(req::MyRequest, cache::MyCache, args)` method — and the
operator issues it through `assemble_into!(MyKind(), (A,), op, states, p, ctx)`.
Declaring it (`setup_operator(...; requests = (MyKind,))`, or a protocol whose
`get_declared_kinds` names it) selects its sweep-state family and runs its
setup-time trait ↔ kernel validation.

Thirteen provided bodies exist, across the four workspace types:

| item family | workspace | provided bodies |
|---|---|---|
| cells | `AssemblyWorkspace` | [`primal_cell_sweep!`](@ref) (buffer zeroing, values reinit, slot gather, cell and facet kernels, scatter — no write-back: [`condense_internal!`](@ref) is the only writer of `q`); [`sensitivity_cell_sweep!`](@ref) (trial gather, no write-back, dispatch to `sensitivity_kernel!`); [`functional_cell_sweep`](@ref) (slot gather, no write-back, RETURN what the kernel hook gives); [`condensation_cell_sweep!`](@ref) (slot gather, dispatch to [`condense_cell!`](@ref), RETURN the [`CondensationReport`](@ref) AND write the trial `q` back — the one combination the others don't have); [`internal_jacobian_cell_sweep!`](@ref) (the rectangular ∂F/∂q block) |
| facet items | `FacetItemWorkspace` | [`primal_facet_item_sweep!`](@ref); [`sensitivity_facet_item_sweep!`](@ref); [`functional_facet_item_sweep`](@ref) (slot gather, no write-back, fold what the facet hook gives over the item's declared facets). Condensation, `JacobianKind{:q}` and quadrature evaluation are explicit `nothing` methods — the family has no body for them |
| algebraic items | `AlgebraicWorkspace` | [`primal_algebraic_sweep!`](@ref); [`sensitivity_algebraic_sweep!`](@ref); [`functional_algebraic_sweep`](@ref); [`condensation_algebraic_sweep!`](@ref); [`internal_jacobian_algebraic_sweep!`](@ref) |
| patches | `PatchAssemblyWorkspace` | the `PatchCallbackKind` body, reached through [`foreach_patch`](@ref) rather than the operator entry points; the per-patch assembly itself is [`assemble_patch_target!`](@ref), called by the callback |

A kind riding `primal_cell_sweep!` without its own `cell_kernel!` method gets
the plain analytic route.

Declarations carry kind *types*, normalized to their `UnionAll` base, while
sweeps carry instances. Two hooks bridge that for validation:
[`validation_instance`](@ref) supplies the placeholder instance traits are
queried on — a kind whose payload is a type parameter must overload it, since
the default `K()` cannot construct one — and [`has_cell_request`](@ref) is
`false` for a kind reaching the element through a hook other than
`assemble_cell!`. [`requires_admissibility_check`](@ref) opts a kind into the
internal-state admissibility rule at setup. That rule reads
`FerriteOperators.serves_kind` — "does the RESOLVED cache answer this kind,
by kernel or by a decorator's generic route" — and not
[`provides_analytic`](@ref), which is reserved for "is there a hand-written
kernel" and forwards through the decorators unchanged.

**New per-worker state** — a workspace is a fixed core (geometry cache, slot
buffers, `Ke`/`re`) plus [`SensitivityBuffers`](@ref), present exactly when
[`needs_ad_decoration`](@ref) says so — structural, by integrator family, not
by protocol declaration. The workspace itself is immutable, so a sweep fills
buffers and never rebinds a field. There is no third, downstream-openable
family: an element cache wanting its own per-worker scratch carries it as an
ordinary cache field, duplicated per worker by its own `duplicate_for_device`
(see [storage classes for elements with local problems](../elements.md)).

**New AD backends** — [`ADElementCache`](@ref)'s `backend` field is the seam:
[`ForwardDiffAD`](@ref) is the default, and a downstream extension implements
its own buffer struct plus the same eight `assemble_cell!` methods for its own
backend marker type, activated via `setup_operator(...; ad_backend =
MyBackend())`.

**New devices and scheduling** — `execute_on_device!`,
`setup_device_instances` and `compute_partition` are the three hooks a device
or scheduling policy implements; the item loop and the workspaces are shared.
