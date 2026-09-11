```@meta
CurrentModule = FerriteOperators
```

# The layer contract

This package owns two layers of the assembly pipeline — the **term** and the
**operator**. Every design question about "where does this belong" is answered
by which of the two owns the information, or by the finding that neither does
and it belongs to the caller.

| owner | owns | in this package |
|---|---|---|
| **term** layer | scheme-agnostic integrands over one work item | element caches and their request-typed kernels |
| **operator** layer | evaluating a set of terms at `(states, p, ctx)` | the engine, its workspaces, and the entry points |
| *the caller* — solver code | composing evaluations into a discretization | nothing: there is no scheme-facing API here. The solver holds the history vectors and tableau coefficients, fills the slots, supplies the weights, and constructs the context |

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
`uprev` and `ctx`, or that computes a scheme's combined matrix directly — puts
solver-owned discretization inside an element cache by deliberate authorship.
That is a legitimate choice (it is how a deliberately manual first-order
discretization, or a multilevel-Newton element with a rate-coupled local
problem, is written), and the framework serves it through the same
request-typed kernels. The distinction is one of authorship, not of mechanism.

## The operator layer

An operator evaluates its term set at `(states, p, ctx)` and scatters the
result. It owns the item loop, the per-worker workspaces, the geometry cache
reinitialization, the slot gathers, the request materialization, and the
choice between an analytic kernel and its derivative fallback.

It owns no time discretization and no scheme coefficients. `states` arrive as
whatever the solver assembled them to be; `ctx` carries the scalars of *this*
evaluation; weights are per-request payload.

[`evaluate!`](@ref) is deliberately polymorphic. It evaluates whatever the
integrator encodes — a nonlinear residual, the action of the linear operator a
bilinear form induces (its element matrix acting on the element vector), or a
hand-fused scheme residual — and the name says exactly that much and no more.

## The caller's side of the contract

The solver owns the discretization, and it owns it *outside* this package:
nothing here declares, registers or hosts a scheme. The solver holds the
history vectors and tableau coefficients, decides what each slot contains for
this evaluation, supplies the chain-rule weights that fold per-slot Jacobians
into the matrix it solves with, and constructs the context.

Where a scheme needs several *evaluation times* — generalized-α evaluating
stiffness at `tₙ₊₁₋αf` and inertia at `tₙ₊₁₋αm` — the solver splits the problem
into term-subset operators and runs one sweep per term at its own context,
combining the results as components. One monolithic kernel cannot host two
evaluation times, and the context is not the place to smuggle a second one:
this is a modeling requirement on the operator split, not a gap in the context
channel.

## The channel decision table

Every piece of information a kernel might want has one channel. The decision
is made by the *shape* of the information, not by who produced it.

| shape | channel | notes |
|---|---|---|
| dof-shaped, one value per dof | a **slot** (`args.states.<name>`) | histories, rates, adjoint directions, stochastic realizations — anything the operator can gather with a dof map. Slots may be plain vectors, [`AffineRate`](@ref) reconstructions, or — for a condensed element's internal state `q` — [`InternalSource`](@ref) restrictions (see [Condensed elements](../elements.md#Condensed-elements-(internal-variables))). |
| point-shaped, one value per quadrature point | quadrature storage and the **query seams** | [`QVector`](@ref) for stored per-QP data; [`query_cell_parameters`](@ref) / [`query_facet_parameters`](@ref) for element-owned gathers, including parameter fields. |
| a scalar of *this* sweep | **`args.ctx`** | `t`, `Δt`, `γ̃` in [`TimeIntegrationContext`](@ref). A scheme with richer per-sweep scalars passes its own context type; framework code touches contexts only through [`evaluation_time`](@ref), [`with_time`](@ref) and [`stage_scaling`](@ref). |
| configuration, constant across the sweep | **`args.p`** | material parameters and the user's bag. Never time, never history. `p` stays opaque: a solver-side wrapper is unwrapped by the cache's own [`query_cell_parameters`](@ref). |
| per-worker mutable working memory | **element cache fields** | duplicated — not aliased — per worker by `duplicate_for_device`, see [storage classes for elements with local problems](../elements.md#Storage-classes-for-elements-with-local-problems). |
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

That one declaration stands in for the per-family spelling above.
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
Declaring it (`setup_operator(...; requests = (MyKind,))`) runs its setup-time
trait ↔ kernel validation.

Thirteen provided bodies exist, across the four workspace types:

| item family | registration | workspace | provided bodies |
|---|---|---|---|
| cells | [`CellFamily`](@ref) | `AssemblyWorkspace` | [`primal_cell_sweep!`](@ref) (buffer zeroing, values reinit, slot gather, cell kernel, scatter — no write-back: [`condense_internal!`](@ref) is the only writer of `q`); [`sensitivity_cell_sweep!`](@ref) (trial gather, no write-back, dispatch to `sensitivity_kernel!`); [`functional_cell_sweep`](@ref) (slot gather, no write-back, RETURN what the kernel hook gives); [`condensation_cell_sweep!`](@ref) (slot gather, dispatch to [`condense_cell!`](@ref), RETURN the [`CondensationReport`](@ref) AND write the trial `q` back — the one combination the others don't have); [`internal_jacobian_cell_sweep!`](@ref) (the rectangular ∂F/∂q block) |
| facet items | [`FacetItemFamily`](@ref) | `FacetItemWorkspace` | [`primal_facet_item_sweep!`](@ref); [`sensitivity_facet_item_sweep!`](@ref); [`functional_facet_item_sweep`](@ref) (slot gather, no write-back, fold what the facet hook gives over the item's declared facets). Condensation, `JacobianKind{:q}` and quadrature evaluation are explicit `nothing` methods — the family has no body for them |
| algebraic items | [`AlgebraicItemFamily`](@ref) | `AlgebraicWorkspace` | [`primal_algebraic_sweep!`](@ref); [`sensitivity_algebraic_sweep!`](@ref); [`functional_algebraic_sweep`](@ref); [`condensation_algebraic_sweep!`](@ref); [`internal_jacobian_algebraic_sweep!`](@ref) |
| patches | — (see [`setup_family_caches`](@ref)) | `PatchAssemblyWorkspace` | the `PatchCallbackKind` body, reached through [`foreach_patch`](@ref) rather than the operator entry points; the per-patch assembly itself is [`assemble_patch_target!`](@ref), called by the callback |

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
by declaration. The workspace itself is immutable, so a sweep fills
buffers and never rebinds a field. There is no third, downstream-openable
family: an element cache wanting its own per-worker scratch carries it as an
ordinary cache field, duplicated per worker by its own `duplicate_for_device`
(see [storage classes for elements with local
problems](../elements.md#Storage-classes-for-elements-with-local-problems)).

**New AD backends** — [`ADElementCache`](@ref)'s `backend` field is the seam:
[`ForwardDiffAD`](@ref) is the default, and a downstream extension implements
its own buffer struct plus the same eight `assemble_cell!` methods for its own
backend marker type, activated via `setup_operator(...; ad_backend =
MyBackend())`.

**New devices and scheduling** — `execute_on_device!`,
`setup_device_instances` and `compute_partition` are the three hooks a device
or scheduling policy implements; the item loop and the workspaces are shared. A
device whose per-worker state is a struct of arrays rather than an array of
structs — [`KernelAbstractionsDevice`](@ref) is the shipped one — additionally
implements [`device_worker_view`](@ref) (the in-kernel slice),
[`n_workers`](@ref), [`adapt_partition`](@ref) and, where its geometry cache
needs a device-resident handler, [`setup_device_handler`](@ref). It answers
`allocate_vector(device, dh)` for the global vector, and
[`allocate_operator_matrix`](@ref) for the global matrix whose type the
operator specification names.

A device that cannot serve an item family says so at setup through
`assert_device_supported` rather than failing inside the first sweep; the GPU
method there is the list of what the device kernel covers today.

**New item iterators** — [`assembly_iterator`](@ref)`(kind, element_cache,
sdh)` decides what a sweep of `kind` positions on one item of the HOST
`SubDofHandler` `sdh`, and what rides `args.cell` while the element kernels
run; the default is Ferrite's `CellCache`. Three accessors are REQUIRED by the
framework and no more:

```julia
struct MyIterator
    # ...
end
Ferrite.reinit!(it::MyIterator, item) = (# position in place; return it)

Ferrite.cellid(it::MyIterator)                    = ...  # a representative cell id
FerriteOperators.iterator_dofs(it::MyIterator)    = ...  # the item's global dof indices
FerriteOperators.iterator_handler(it::MyIterator) = ...  # the SubDofHandler it was built over

FerriteOperators.assembly_iterator(kind, ::MyCache, sdh) = MyIterator(sdh)
```

Everything else a shipped element kernel reaches for through `args.cell` —
`Ferrite.getcoordinates`, `Ferrite.getnodes`, `Ferrite.reinit!(cv, it)`, … — is
CONVENTIONAL between the iterator and the elements written for it: the
framework never calls them, so an iterator author implements whichever subset
its own elements need.

An iterator positioned by CONSTRUCTION (an immutable value, as the device
cursor is) rather than in place overloads
[`position_iterator`](@ref)`(it, item, flags)` instead of `Ferrite.reinit!`,
returning the new value; [`position_item`](@ref) — the only thing a sweep
calls — carries whichever value came back into the workspace, so both
positioning styles compose with the rest of the engine unchanged.
[`item_update_flags`](@ref)`(kind, element_cache)` lets an iterator that stages
some of what it carries answer, per `(kind, cache)` pair, which members a
positioning refreshes; an iterator that stages nothing ignores it, and an
under-declaring pair reads a stale buffer rather than erroring, the same
contract Ferrite's own `UpdateFlags` carries.

The DEVICE shape is a fourth hook,
[`device_assembly_iterator`](@ref)`(kind, element_cache, sdh, device_sdh)`,
whose default forwards to `assembly_iterator` over the device handler — a
downstream iterator needing no host-only setup fact writes only the host
method; one that does either overloads this instead (narrowing the CACHE, per
the rule below) or answers [`decorate_device_iterator`](@ref) on its own
iterator type, which is where the matrix-free action's uniform-dof-stride check
lives. `reinit_values!`'s setup-time admissibility probe is validated against
the subdomain's RESOLVED iterator type, so a cache author who annotates it
against a custom iterator still passes.

Once constructed, a device iterator reaches the workspace through a FIFTH hook:
the batching [`setup_device_instances`](@ref)`(device, it, n)` moves whatever it
STAGES onto the device, and [`device_worker_view`](@ref)`(it, worker)` is the
in-kernel slice. The workspace's iterator slot routes through a private,
`ext`-level `_batch_iterator(device, it, n)` whose GENERIC default is
`setup_device_instances(device, it, n)` itself, so a downstream device iterator
writes only that 3-arg hook and needs no `ext`-private method.
`Ferrite.CellCache` and the KA extension's `DeviceCellCursor` are the two
shipped iterators with a more specific answer — Ferrite's own struct-of-arrays
batching, and the cooperative mapping's positioned-by-construction fork.

**New item SETS** — an iterator says how to POSITION on an item; it does not say
what the items ARE. That is the second seam,
[`item_provider`](@ref)`(kind, element_cache, sdh)`, whose answer
[`compute_partition`](@ref) turns into the barriers and chunks a sweep walks.
The default is [`CellItems`](@ref)`(sdh)`.

The two vary independently, which is why they are two seams: the facet family
runs a custom provider over the stock cell iterator, and the matrix-free action
a custom iterator over the stock provider. A family that is BOTH — a two-sided
interface traversal whose item is a pair of cells and whose local system is
indexed by both cells' dofs — is these six methods and nothing else:

```julia
FerriteOperators.assembly_iterator(kind, ::MyCache, sdh) = MyIterator(sdh, …)
FerriteOperators.item_provider(kind, ::MyCache, sdh)     = MyItems(sdh, …)

Ferrite.reinit!(it::MyIterator, item::Int) = …   # position it; `position_iterator`'s default calls this
FerriteOperators.compute_partition(::SequentialScheduling, p::MyItems) = (collect(eachindex(…)),)
FerriteOperators.compute_partition(::ColoredScheduling,    p::MyItems) = …  # see below
FerriteOperators.duplicate_for_device(::AbstractCPUDevice, it::MyIterator) = MyIterator(…)
```

plus the three required accessors above. `Ferrite.reinit!` is what positions the
iterator on an item — [`position_iterator`](@ref)'s default is exactly that call,
and an iterator positioned by CONSTRUCTION instead (a device cursor) overloads
`position_iterator` and writes no `reinit!` at all.

**Both seams narrow the CACHE argument, and that is a rule.** A declaration may
narrow the sweep kind on top of it; it must never narrow ONLY the kind. The
decorators forward these seams with methods that narrow the cache and leave the
kind open ([`AbstractElementCacheDecorator`](@ref)), so a kind-narrow/cache-open
method ties with those forwards over every decorated cache and Julia reports the
call ambiguous. The rule covers [`item_update_flags`](@ref) and
[`reinit_values!`](@ref) for the same reason. A KIND-level default — the
matrix-free action's device cursor — goes on
[`default_assembly_iterator`](@ref) instead, which sits BELOW every cache
declaration, so a cache that names its own iterator keeps it under every kind.

Overloading one seam and forgetting the other is not an error and not a
`MethodError`: the other answers with its default, and an interface iterator left
with `CellItems` is positioned on CELL ids.

Half of that hazard is checked and half is not. A method written against a
signature the engine does not call — the wrong sweep kind, the wrong handler
type, the wrong arity — is rejected at setup by
[`assert_iteration_signatures`](@ref), which takes the ELEMENT CACHE as its
subject: a hook with any method narrowing that argument to a type this
subdomain's cache conforms to must have one the engine's own call resolves to.
A method that is simply ABSENT has no drift to detect and stays invisible.
Assert the item COUNT a sweep visits; no framework check can see that one.

A provider carries its family's whole partition safety argument, and the wording
the package uses is exact: a partition is **safe given a valid partition**,
never *thread-safe*. Under [`ColoredScheduling`](@ref) the provider promises that
no two items of one inner chunk share a SCATTER DOF — that promise, and nothing
else, is what makes the scatter race-free without atomics. Under
[`SequentialScheduling`](@ref) it promises nothing and the atomic scatter
resolves the collisions. The three shipped providers each argue it in their own
terms ([`FacetItems`](@ref) colors the owning cells, [`AlgebraicItems`](@ref)
puts one item per barrier, [`PatchItems`](@ref) refuses to color at all).

Two structural consequences of a local system spanning more than one cell:

- The dof window [`iterator_dofs`](@ref) returns is the scatter's address, and
  Ferrite's assembler requires it to be DUPLICATE FREE. Two cells of a
  continuous space share the dofs on their common facet, so a two-sided item
  over one repeats them; the family's space is discontinuous, or the item's
  sides do not touch.
- The entries that window addresses are not in the `DofHandler`'s cell pattern,
  and the framework never infers them. Declare them through the operator
  specification's `sparsity_entries`
  ([`StandardOperatorSpecification`](@ref)) — the same doctrine
  [`global_dofs`](@ref) states for its tail.

**New item FAMILIES** — the two seams above narrow what the CELL family
positions on and enumerates, which is all a family riding the cell route needs.
Registering a family of its OWN is the third seam, and it is what an operator
carrying more than one traversal at a time needs:
[`item_families`](@ref)`(integrator, dh)` names the families the engine carries,
in TRAVERSAL order, and [`setup_family_caches`](@ref)`(family, strategy,
integrator, dh, shared)` builds one family's `SubdomainCache`s.

```julia
struct MyFamily end

FerriteOperators.item_families(::MyIntegrator, dh) = (CellFamily(), MyFamily())

FerriteOperators.setup_family_caches(::MyFamily, strategy, integrator, dh, shared) =
    # one SubdomainCache per subdomain this family serves; `()` declines
```

The tuple REPLACES the default, which derives `CellFamily()` always,
[`FacetItemFamily`](@ref) where a subdomain declares [`facet_items`](@ref) and
[`AlgebraicItemFamily`](@ref) where the handler declares
[`algebraic_items`](@ref). So an integrator returning only its own marker
carries only that family — the cells are not swept — and one returning
`(CellFamily(), MyFamily())` carries both, cells first. That order is the order
of `engine.subdomain_caches`, which the reduction determinism contract rests on.

`shared` carries what `setup_engine` resolved before any family ran, and every
field is available to every family; [`setup_family_caches`](@ref) tabulates
them.

**No ENGINE-REGISTERED family has a privileged setup path.** The three shipped
families are three `setup_family_caches` methods and nothing else — the cell one
in `operators/setup.jl`, the facet-item one in `core/facet-task.jl`, the
algebraic one in `core/algebraic-task.jl` — reached by the same dispatch a
downstream marker reaches, and `setup_engine` iterates `item_families` rather
than calling any of them by name.

**Two families deliberately do not register**, for reasons that are properties
of those families rather than of this seam. [`foreach_patch`](@ref) is
sequential-CPU-only because the callback's collectors are the CALLER's, so this
package cannot duplicate them per worker, and `PatchAssemblyWorkspace` positions
through a `Ref` resolved against its provider. [`setup_transfer_operator`](@ref)
restricts to sequential full assembly by design and assembles a RECTANGULAR
matrix through a driver of its own. Both adopt the iteration seams above and
keep their own entry points.

Two scope limits. The composite and multi-domain wrappers forward
[`facet_items`](@ref) and [`algebraic_items`](@ref), so a wrapped
sub-integrator's BUILT-IN families are carried through the default above — but
they do not forward [`item_families`](@ref): a wrapper's own answer is the
operator's. And a family whose subdomain caches are not cell-shaped builds them
from names this package does not export; a downstream family with cell-shaped
caches answers with the [`CellFamily`](@ref) method instead.

!!! warning "Experimental surface"
    `item_families`, `setup_family_caches` and the three family markers are
    experimental and may change in a minor release, as the whole iteration
    protocol's registration half.

**New assembly levels** — a form member decides what `setup_operator` returns
and, through [`operator_specification`](@ref), whether the global-storage walls
apply to it at all. [`MatrixFreeAction`](@ref) is the second member: it
allocates nothing, and its operator's `mul!` rides the ordinary sweep
(`run_sweep!` → `execute_on_subdomains!` → `execute_on_device!`) under its own
kind, whose driver body ([`matrix_free_cell_sweep!`](@ref)) differs from
[`primal_cell_sweep!`](@ref) only in gathering fixed-width.

A form choice a DEVICE has to realize — the element mapping — resolves onto the
device at setup through [`with_element_mapping`](@ref), because
[`n_workers`](@ref), [`setup_device_instances`](@ref) and
[`execute_on_device!`](@ref) receive the device and never the form. A choice the
ELEMENT has to realize — the `storage` election separating the ELEMENT, PARTIAL
and NONE levels — resolves onto the caches through
[`with_assembly_form`](@ref) for the mirrored reason: `setup_element_cache`
receives the subdomain and never the form. Its two per-quadrature-point members
reach the element's own hook ([`with_action_storage`](@ref)); the ELEMENT member
is element-AGNOSTIC and wraps whatever cache the integrator built in an
[`ElementAssemblyCache`](@ref), which is why a cache with no matrix-free kernel
serves it. What either keeps is filled by a sweep of its own kind
([`QuadratureDataKind`](@ref)), which scatters nothing and carries no assembler.
