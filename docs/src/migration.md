# Migrating from 0.3.x to the v2 interface

This guide maps every removed or changed 0.3.x API to its v2 replacement.
Breaking changes are clustered into this one transition — there are no
deprecation shims; old signatures are gone, and (important!) some old element
methods are **silently never called** rather than erroring, so grep for the
patterns marked ⚠ below.

## Quick map

| 0.3.x | v2 |
|---|---|
| `assemble_element!(Kₑ, [rₑ,] [uₑ,] cell, cache, t)` (5 arities) | `assemble_cell!(req, cache, args)` — request-typed |
| `assemble_facet!(Kₑ, …, cell, lfi, cache, t)` ⚠ | `assemble_facet!(req, cache, args, lfi)` |
| `assemble_element_gto1!(…, uₑprev, …, p, t, Δt)` | kernel reads `args.states.uprev`, `args.ctx` |
| `GenericFirstOrderTimeParameters(p, t, Δt, uprev)` | `slots = (:u, :uprev)` at setup + `TimeIntegrationContext(t, Δt, γ̃)` |
| `AbstractGenericFirstOrderTime*ElementCache` | plain `AbstractVolumetricElementCache`/`AbstractSurfaceElementCache` |
| bare `t` as the parameter object (`evaluate!(op, r, u, t)`) | `p` is user parameters; time rides in `ctx` and is read as `evaluation_time(args.ctx)` |
| `query_element_parameters(cache, cell, ivh, p)` | `query_cell_parameters(cache, cell, p)` (no `ivh`) |
| — (volumetric `pₑ` reused on facets) | `query_facet_parameters(cache, cell, lfi, p)` per facet |
| `query_element_unknown_buffer(cache, ue)` | removed — slot buffers are workspace-owned |
| `SequentialAssemblyStrategy{Dev}` as a **type** ⚠ | `AssemblyStrategy{<:FullAssembly, SequentialScheduling, Dev}` |
| `ElementAssemblyOperatorStrategy` | `AssemblyStrategy{<:ElementAssemblyData}` |
| `op.dh`, `op.strategy`, `op.subdomain_caches` | `op.engine.dh`, `op.engine.strategy`, `op.engine.subdomain_caches` |
| `op.J` / `op.A` / `op.b`, `residual_size`, `unknown_size` | unchanged |
| `residual!(op, r, u, p)` | `evaluate!(op, r, u, p)` |
| `setup_quadrature_operator` / `FerriteQuadratureOperator` | any operator works: `evaluate_quadrature!(q, op, u, p, f)` |
| silent `setup_element_cache` fallback | missing method **throws at setup** |
| `reinit!` inside every cell-kernel body | engine calls `reinit_values!(cache, cell, kind)` once per cell and sweep |
| `Ferrite.getnquadpoints`/`reinit!` via `.cv`/`.fv` field fallback | define `Ferrite.getnquadpoints` and `reinit_values!` explicitly on your cache |
| `FerriteOperators.Simple*` example elements | `FerriteOperatorsExampleElements` — a separate package under `lib/`, exporting the integrators |
| `*MultiDomainIntegrator(Dict(sdh => integrator))` | `*MultiDomainIntegrator(Dict("cellset_name" => integrator))` — volumetric cellset names, validated at setup |

Constructor *calls* like `SequentialAssemblyStrategy(device)` still work — the
names are convenience constructors for the common strategy compositions. Only
**dispatch on them as types** breaks.

## Example elements

`SimpleBilinearDiffusionIntegrator`, `SimpleLinearIntegrator`,
`SimpleBilinearMassIntegrator`, `SimpleHyperelasticityIntegrator`,
`SimpleCondensedLinearViscoelasticity`, their caches and `MaxwellParameters`
are no longer part of FerriteOperators. They live in
`FerriteOperatorsExampleElements`, which is a test-time and example-time
dependency, not a runtime one. Code using them adds

```julia
Pkg.add(url = "https://github.com/termi-official/FerriteOperators.jl",
        subdir = "lib/FerriteOperatorsExampleElements")
```

to the environment that needs them (typically `test/`) and replaces
`FerriteOperators.Simple…` with `using FerriteOperatorsExampleElements` plus
the bare name — the subpackage exports all of them.

The composition machinery is *not* affected: `CompositeVolumetricElementCache`,
`CompositeSurfaceElementCache`, the `*MultiDomainIntegrator` family and the
transfer integrators `MassProlongatorIntegrator` /
`NestedMassProlongatorIntegrator` remain in FerriteOperators.

The `*ElementCache` types are internal to the example package. Code dispatching
on one reaches it as `FerriteOperatorsExampleElements.Simple…ElementCache`.

## Composition

`NonlinearCompositeIntegrator`, `BilinearCompositeIntegrator` and
`LinearCompositeIntegrator` build the composite caches, which previously had to
be assembled by hand. One behavioural change reaches existing hand-built
composites: an inner cache's `query_cell_parameters` / `query_facet_parameters`
override was documented as bypassed and is now honoured — each inner receives
its own parameter view. An inner that relied on seeing the outer view must take
that view from its own query.

## Multi-domain routing

`NonlinearMultiDomainIntegrator`, `BilinearMultiDomainIntegrator` and
`LinearMultiDomainIntegrator` are keyed by the **name of a volumetric cellset**
instead of by `SubDofHandler`:

```julia
# before
BilinearMultiDomainIntegrator(Dict(sdh_right => a, sdh_left => b))
# now
BilinearMultiDomainIntegrator(Dict("right_cells" => a, "left_cells" => b))
```

A name claims the subdomain whose cells lie inside its cellset, and resolves
both the element cache and the boundary cache of that subdomain. Setup throws
an `ArgumentError` for a subdomain claimed by no name, a subdomain claimed by
several names, and a declared name claiming no subdomain — so a mistyped name
fails at `setup_operator` rather than assembling nothing. The claim is read
from each subdomain's first cell; `FerriteOperators.debug_mode` upgrades that
sample to an exhaustive per-cell check.

## Element kernels

One request-typed entry point replaces the arity family. The residual kernel
is mandatory (validated at `setup_operator`); Jacobians and every sensitivity
are derived from it by ForwardDiff unless you declare analytic kernels.

```julia
# 0.3.x — three near-identical bodies
function assemble_element!(Kₑ, rₑ, uₑ, cell, cache::MyCache, p) ... end
function assemble_element!(Kₑ, uₑ, cell, cache::MyCache, p) ... end
function assemble_element!(rₑ, uₑ, cell, cache::MyCache, p) ... end

# v2 — reinit lives in the per-cache hook …
FerriteOperators.reinit_values!(c::MyCache, cell) = reinit!(c.cv, cell)

# … one mandatory residual kernel (pure evaluation, no reinit) …
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::MyCache, args)
    uₑ = args.states.u
    pₑ = args.p
    # accumulate into req.r
end

# … and optional analytic kernels, declared via a trait
FerriteOperators.provides_analytic(::Type{<:MyCache}, ::FerriteOperators.JacobianKind) = true
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, cache::MyCache, args)
    # accumulate into req.K
end
# fused Newton path: JacobianResidualRequest (req.K and req.r), kind JacobianResidualKind
```

Requirements on the residual kernel: eltype-generic in `eltype(args.states.*)`,
`eltype(args.p)` and the context time (the AD contract); never write global
state; treat `args.cell` as read-only; no reinit inside cell kernels — the
engine calls `reinit_values!` once per cell and sweep (elements carrying
several values objects specialize the kind-dispatched form to reinitialize
only what a request needs). Facet kernels reinitialize their `FacetValues` per
facet themselves.

**Leave the `args` parameter unannotated.** Kernels select on the
`(request, cache)` pair; the third argument is a *channel protocol*
(`args.states`, `args.cell`, `args.p`, `args.scratch`, `args.ctx`) with no
supertype, plus the three rebuild seams `with_states`, `with_parameters` and
`with_context` that derivative sweeps re-seed channels through. `KernelArgs`
is the type this package's operators build — annotating `args::KernelArgs`
pins the element to that one family and makes it unusable to an operator
family building its own args type. `setup_operator` emits an advisory warning
per element cache whose residual kernel carries a concrete args annotation.

## Time protocols (GTO1, Newmark)

The parameter-wrapper protocols are gone. Solvers pass named slot vectors and
a context; elements read values.

```julia
# 0.3.x
update_linearization!(op, r, u, GenericFirstOrderTimeParameters(p, t, Δt, uprev))
# element: assemble_element_gto1!(Kₑ, rₑ, uₑ, uₑprev, cell, cache, p, t, Δt)

# v2
op = setup_operator(strategy, integrator, dh; slots = (:u, :uprev))
update_linearization!(op, r, (u = u, uprev = uprev), p, TimeIntegrationContext(t, Δt, γ̃))
# element: uₑprev = args.states.uprev;  t = evaluation_time(args.ctx);  γ̃ = args.ctx.γ̃
```

**Time reaches elements through `ctx`, and only through `ctx`.** `p` is the
user parameter bag: passing a bare `t` as the parameter object, or hiding `t`
inside `p`, is not a supported convention — every wrapper would then need an
unwrapping rule, and the framework itself must see `t` to seed ∂F/∂t. A kernel
that reads its time from `args.p` gets no time sensitivity at all.

`γ̃` is the *normalized local stage interval* of the element-local
internal-variable problem (`q = q_ref + γ̃·g` is the normalization; the element
may integrate with any consistent rule over it, including exponential
updates). For a backward-Euler local state under any one-step global scheme,
`γ̃ = Δt`. **`γ̃` is not a rate slope** — see the `TimeIntegrationContext`
docstring for the trap.

Slot names are free: a Newmark-style protocol passes whatever it needs, e.g.
`slots = (:u, :v, :a)`. Rate-like slots do not have to be materialized by the
solver — an [`AffineRate`](@ref) source reconstructs them from the primary
unknown at gather time:

```julia
op = setup_operator(strategy, integrator, dh; slots = (:u, :v, :a))
update_linearization!(op, r,
    (u = u, v = AffineRate(γ/(β*Δt), uᵥ), a = AffineRate(1/(β*Δt^2), ũ)),
    p, TimeIntegrationContext(t, Δt, Δt))
# element: vₑ = args.states.v
```

A kernel reads slot *values* and nothing else — the reconstruction is a
solver-side statement about where those values came from. Do not carry
reconstruction slopes in `p`, and do not put them into `ctx`; a scheme scalar
that must reach a kernel rides as request payload (see weighted Jacobians
below).

The `:u` slot must be declared and must precede any reconstructed slot in the
states NamedTuple — the sweep throws otherwise. The assembled Jacobian is
∂F/∂u at frozen slot values (AD seeds the `:u` buffer only), so the
chain-rule contribution `slope · ∂F/∂v` remains the solver's, applied through
its per-slot weights.

Condensed elements: declare `FerriteOperators.has_internal_state(::Type{<:MyCache}) = true`.
The previous state arrives through your chosen slot; the local solve scales by
`args.ctx.γ̃`; the trial result is written into the element-local `u` buffer,
and the framework propagates it — that per-evaluation write-back is the
condensation contract.

## Facets ⚠

The framework owns the facet loop. `is_facet_in_cache` is unchanged; kernels
become request-typed and facet parameters are queried per facet:

```julia
# 0.3.x
function assemble_facet!(rₑ, uₑ, cell, lfi, cache::MyFacetCache, p) ... end

# v2
function FerriteOperators.assemble_facet!(req::ResidualRequest, cache::MyFacetCache, args, lfi::Int)
    reinit!(cache.fv, args.cell, lfi)
    # accumulate into req.r; args.p came from query_facet_parameters(cache, cell, lfi, p)
end
```

**⚠ Old-signature `assemble_facet!` methods are never called and produce no
error — the boundary contribution silently vanishes.** Grep every downstream
`assemble_facet!`/`assemble_element!` method definition and port it; put at
least one boundary integral under a test with an analytic reference (see
`test/test_element_api.jl`, "Facet driver with a real Neumann kernel").

## Strategies and operators

```julia
# 0.3.x type dispatch                       # v2
f(s::SequentialAssemblyStrategy)            f(s::AssemblyStrategy{<:FullAssembly, SequentialScheduling})
f(s::PerColorAssemblyStrategy)              f(s::AssemblyStrategy{<:FullAssembly, <:ColoredScheduling})
f(s::ElementAssemblyStrategy)               f(s::AssemblyStrategy{ElementAssembly})   # pre-setup
                                            f(s::AssemblyStrategy{<:ElementAssemblyData}) # post-setup
```

Or dispatch on a single axis: `s.form`, `s.scheduling`, `s.device`. Operators
are payload + engine + integrator; anything that read `op.dh`/`op.strategy`/
`op.subdomain_caches` reads `op.engine.*`. `getJ(op) = op.J` style accessors
keep working.

The engine's setup-time declarations live on its scheme protocol:
`declared_slots`, `declared_kinds` and `declared_args_type` of
`op.engine.protocol`.

## Solver-owned scratch

The pattern of smuggling solver state into element caches (model-tree
rewrites, parameter-bag payloads) is replaced by declared scratch:

```julia
op = setup_operator(strategy, integrator, dh;
                    scratch = (local_solver = () -> MyLocalSolverCache(...),))
# element side: FerriteOperators.declare_scratch(cache::MyCache) = (buf = () -> zeros(6),)
# kernels: args.scratch.local_solver, args.scratch.buf — instantiated per worker
```

## New capabilities worth adopting during the port

- **Scheme protocols**: `setup_operator(strategy, integrator, dh, protocol)` is
  the positional form for scheme operators; the keyword form is sugar whose
  keywords are `DefaultProtocol`'s constructor arguments, so
  `setup_operator(strategy, integrator, dh; slots, requests, scratch, args_type)`
  keeps working verbatim. Declarations move admissibility failures from first
  use to `setup_operator`, and select which per-worker sweep-state families
  exist — a bilinear or linear operator carries no AD machinery.
- **Sensitivities**: `update_parameter_jacobian!(B, op, states, p, ctx)`,
  `parameter_vjp!(g, op, λ, states, p, ctx)`,
  `time_sensitivity!(g, op, states, p, ctx)` (AD by default, analytic kernels
  win per cache, `FiniteDifferenceSensitivity` for condensed time
  derivatives). ∂F/∂t seeds through the context — the AD sweep hands the
  kernel a Dual-timed context and the FD method perturbs the context time — so
  `time_sensitivity!` takes the same `(states, p, ctx)` triple as every other
  entry point, reads `t` from `evaluation_time(ctx)`, and throws when `ctx` is
  `nothing`.
- **Matrix-free state actions**: `state_jvp!` (`J·v` without a matrix),
  `state_vjp!` (`Jᵀλ` — the adjoint action).
- **Components and stage operators**: `allocate_components` +
  `assemble_slot_jacobian!(J, op, JacobianKind{:du}(), …)` + `combine!` replace
  hand-matched `M`/`K` pairs and the `op.A`/`op.J` reach-through — one shared
  sparsity pattern, weights applied by the solver, complex targets supported
  (transformed Radau). `StageBlockOperator`/`assemble_stages!` carry the same
  components into fully implicit Runge-Kutta.
- **Weighted Jacobians**: `assemble_weighted_jacobian!(W, op, weights, states, p, ctx)`
  is the scheme matrix `Σₛ wₛ ∂F/∂s` in one call. A hand-fused `W` kernel
  (`M/(γΔt) + K`) ports as an analytic provider of `WeightedJacobianKind`
  reading `req.weights` — NOT as a `JacobianRequest{:u}` kernel, which the
  Jacobian checks legitimately reject against the AD referee.
- **Derivative verification**: `check_derivatives(op, states, p, ctx)`
  cross-checks every analytic kernel and AD path against finite differences
  of the operator's own residual — run it once per ported element. Its time
  check runs only with a context, its weighted-Jacobian checks only with
  `weights = (…)`; the rest are skipped with the reason recorded.
- **Functionals**: `evaluate_functional(op, FunctionalKind(:energy), states, p, ctx)`
  reduces per-cell contributions returned by `evaluate_cell_functional`
  kernels — global scalars/tensors without hand-rolled loops.
- Admissibility with internal state is per cache and per kind: analytic
  kernel, `internal_state_insensitive` declaration, or FD — never a silent
  wrong adjoint, never a blanket rejection.

## Porting checklist

1. Grep for `assemble_element!`, `assemble_facet!`, `assemble_element_gto1!`
   method *definitions* — port each to request kernels (⚠ facet methods fail
   silently, everything else errors loudly).
2. Grep for strategy **type** dispatch and `op.dh`/`op.strategy`/`op.subdomain_caches`.
3. Replace `GenericFirstOrderTimeParameters` call sites with slots + ctx;
   declare `slots` at `setup_operator`.
4. Declare `has_internal_state` for condensed caches; add explicit
   `Ferrite.getnquadpoints` and `reinit_values!` methods for every cache,
   and delete `reinit!` calls from cell-kernel bodies.
5. Replace parameter-wrapper unwrapping hacks with one `unwrap_parameters`
   method; move facet-specific parameters into `query_facet_parameters`.
6. Replace solver-state smuggling with `scratch` declarations.
7. Drop `::KernelArgs` annotations from kernel signatures (setup warns about
   the ones you miss), and move every kernel that read time out of `p` onto
   `evaluation_time(args.ctx)`.
8. Rename `residual!` call sites to `evaluate!`.
9. Run your suite; every port failure except facets is a loud
   `MethodError`/`ArgumentError` at setup or first assembly.
