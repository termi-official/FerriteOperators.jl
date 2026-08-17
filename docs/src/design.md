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

An element expresses an integrand. It reads slot *values*, a parameter view, a
per-sweep context, and its scratch — and nothing else. It does not know which
time discretization produced the values in its slots, whether a Jacobian will
be derived from it by automatic differentiation, or how many other terms share
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
| dof-shaped, one value per dof | a **slot** (`args.states.<name>`) | histories, rates, adjoint directions, stochastic realizations — anything the operator can gather with a dof map. Slots may be plain vectors or [`AffineRate`](@ref) reconstructions. |
| point-shaped, one value per quadrature point | quadrature storage and the **query seams** | [`QVector`](@ref) for stored per-QP data; [`query_cell_parameters`](@ref) / [`query_facet_parameters`](@ref) for element-owned gathers, including parameter fields. |
| a scalar of *this* sweep | **`args.ctx`** | `t`, `Δt`, `γ̃` in [`TimeIntegrationContext`](@ref). A scheme with richer per-sweep scalars passes its own context type; framework code touches contexts only through [`evaluation_time`](@ref) and [`with_time`](@ref). |
| configuration, constant across the sweep | **`args.p`** | material parameters and the user's bag. Never time, never history. `p` stays opaque: [`unwrap_parameters`](@ref) is the one place a solver-side wrapper is unwrapped. |
| per-worker mutable working memory | **`args.scratch`** | declared by the solver (`scratch = (…)`) and/or the element ([`declare_scratch`](@ref)), instantiated once per worker. |
| a property *of a slot* | reserved vocabulary | an args family may carry per-slot metadata; [`KernelArgs`](@ref) carries none, and no in-repo kernel reads any. A scheme scalar attached to a slot rides as request payload instead — that is what [`WeightedJacobianKind`](@ref) does with its weights. |

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

# 3. The per-worker family the sweep reads, queried on the TYPE. Declaring
#    `DerivativeFamily()` is what builds the `ADWorkspace` at setup.
FerriteOperators.sweep_family(::Type{<:MyKind}) = FerriteOperators.NoFamily()

# 4. The driver. Reuse a provided body, or write a bespoke one.
FerriteOperators.execute_kind!(kind::MyKind, task, ws) =
    FerriteOperators.primal_cell_sweep!(kind, task, ws)
```

A kind computing a global scalar or tensor rather than something scattered
declares `FunctionalFamily()` instead, which routes it to the
**value-returning** driver: the per-item kernel *returns* its contribution and
the sweep folds the returned values, so there is no request type, no
assembler, and no workspace state at either end.

```julia
struct MyFunctionalKind end
FerriteOperators.sweep_family(::Type{<:MyFunctionalKind}) = FerriteOperators.FunctionalFamily()
FerriteOperators.has_cell_request(::Type{<:MyFunctionalKind}) = false
FerriteOperators.execute_kind!(kind::MyFunctionalKind, task, ws) =
    FerriteOperators.functional_cell_sweep(kind, task, ws)

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

[`FunctionalKind`](@ref) is exactly this with a tag, and
[`evaluate_functional`](@ref) is its entry point.

Elements then serve it like any built-in kind — `provides_analytic(::Type{<:MyCache}, ::MyKind) = true`
plus an `assemble_cell!(req::MyRequest, cache::MyCache, args)` method — and the
operator issues it through `assemble_into!(MyKind(), (A,), op, states, p, ctx)`.
Declaring it (`setup_operator(...; requests = (MyKind,))`, or a protocol whose
`get_declared_kinds` names it) selects its sweep-state family and runs its
setup-time trait ↔ kernel validation.

Three provided bodies exist: [`primal_cell_sweep!`](@ref) (buffer zeroing, slot
gather, cell and facet kernels, condensed write-back, scatter),
[`sensitivity_cell_sweep!`](@ref) (trial gather, no write-back, dispatch to
`sensitivity_kernel!`), and [`functional_cell_sweep`](@ref) (slot gather, no
write-back, RETURN what the kernel hook gives). A kind riding
`primal_cell_sweep!` without its own `cell_kernel!` method gets the plain
analytic route.

Declarations carry kind *types*, normalized to their `UnionAll` base, while
sweeps carry instances. Two hooks bridge that for validation:
[`validation_instance`](@ref) supplies the placeholder instance traits are
queried on — a kind whose payload is a type parameter must overload it, since
the default `K()` cannot construct one — and [`has_cell_request`](@ref) is
`false` for a kind reaching the element through a hook other than
`assemble_cell!`. [`requires_admissibility_check`](@ref) opts a kind into the
internal-state admissibility rule at setup.

**New per-worker state** — [`sweep_state`](@ref) is the accessor for a
workspace's kind-family members. A workspace is a fixed core plus those
members, all of them bound at construction from the protocol's declarations
routed through [`sweep_family`](@ref); the workspace itself is immutable, so a
sweep fills buffers and never rebinds a field. Membership in the existing
families is open; adding a new family *type* is not.

**New operator families** — [`mandatory_kinds`](@ref) states the kinds an
integrator family always issues regardless of what a protocol declares, which
is what keeps declarations additive: an operator is never *less* capable for
having declared nothing.

**New args families** — an operator family building its own kernel-args type
implements [`with_states`](@ref), [`with_parameters`](@ref) and
[`with_context`](@ref) for it, and declares it through
[`get_declared_args_type`](@ref) so setup-time method lookups query against the
right type. Elements written with an unannotated `args` parameter serve every
family unchanged.

**New devices and scheduling** — `execute_on_device!`,
`setup_device_instances` and `compute_partition` are the three hooks a device
or scheduling policy implements; the item loop and the workspaces are shared.
