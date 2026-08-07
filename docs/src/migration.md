# Migrating from 0.3.x to the v2 interface

This guide maps every removed or changed 0.3.x API to its v2 replacement.
Breaking changes were deliberately clustered into this one transition — there
are no deprecation shims; old signatures are gone, and (important!) some old
element methods would be **silently never called** rather than erroring, so
grep for the patterns marked ⚠ below.

## Quick map

| 0.3.x | v2 |
|---|---|
| `assemble_element!(Kₑ, [rₑ,] [uₑ,] cell, cache, t)` (5 arities) | `assemble_cell!(req, cache, args::KernelArgs)` — request-typed |
| `assemble_facet!(Kₑ, …, cell, lfi, cache, t)` ⚠ | `assemble_facet!(req, cache, args, lfi)` |
| `assemble_element_gto1!(…, uₑprev, …, p, t, Δt)` | kernel reads `args.states.uprev`, `args.ctx` |
| `GenericFirstOrderTimeParameters(p, t, Δt, uprev)` | `slots = (:u, :uprev)` at setup + `TimeIntegrationContext(t, Δt, γ̃)` |
| `AbstractGenericFirstOrderTime*ElementCache` | plain `AbstractVolumetricElementCache`/`AbstractSurfaceElementCache` |
| `query_element_parameters(cache, cell, ivh, p)` | `query_cell_parameters(cache, cell, p)` (no `ivh`) |
| — (volumetric `pₑ` reused on facets) | `query_facet_parameters(cache, cell, lfi, p)` per facet |
| `query_element_unknown_buffer(cache, ue)` | removed — slot buffers are workspace-owned |
| `SequentialAssemblyStrategy{Dev}` as a **type** ⚠ | `AssemblyStrategy{<:FullAssembly, SequentialScheduling, Dev}` |
| `ElementAssemblyOperatorStrategy` | `AssemblyStrategy{<:ElementAssemblyData}` |
| `op.dh`, `op.strategy`, `op.subdomain_caches` | `op.engine.dh`, `op.engine.strategy`, `op.engine.subdomain_caches` |
| `op.J` / `op.A` / `op.b`, `residual_size`, `unknown_size` | unchanged |
| `setup_quadrature_operator` / `FerriteQuadratureOperator` | any operator works: `evaluate_quadrature!(q, op, u, p, f)` |
| silent `setup_element_cache` fallback | missing method now **throws at setup** |
| `Ferrite.getnquadpoints`/`reinit!` via `.cv`/`.fv` field fallback | define both explicitly on your cache |

Constructor *calls* like `SequentialAssemblyStrategy(device)` still work — the
names survive as convenience constructors. Only **dispatch on them as types**
breaks.

## Element kernels

One request-typed entry point replaces the arity family. The residual kernel
is mandatory (validated at `setup_operator`); Jacobians and every sensitivity
are derived from it by ForwardDiff unless you declare analytic kernels.

```julia
# 0.3.x — three near-identical bodies
function assemble_element!(Kₑ, rₑ, uₑ, cell, cache::MyCache, p) ... end
function assemble_element!(Kₑ, uₑ, cell, cache::MyCache, p) ... end
function assemble_element!(rₑ, uₑ, cell, cache::MyCache, p) ... end

# v2 — one mandatory residual kernel …
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::MyCache, args::KernelArgs)
    uₑ = args.states.u
    pₑ = args.p
    reinit!(cache.cv, args.cell)   # elements own reinit! of their values
    # accumulate into req.r
end

# … and optional analytic kernels, declared via a trait
FerriteOperators.provides_analytic(::Type{<:MyCache}, ::FerriteOperators.JacobianKind) = true
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, cache::MyCache, args::KernelArgs)
    # accumulate into req.K
end
# fused Newton path: JacobianResidualRequest (req.K and req.r), kind JacobianResidualKind
```

Requirements on the residual kernel: eltype-generic in `eltype(args.states.*)`,
`eltype(args.p)` and the context time (the AD contract); never write global
state; treat `args.cell` as read-only.

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
# element: uₑprev = args.states.uprev;  γ̃ = args.ctx.γ̃
```

`γ̃` is the *normalized local stage interval* of the element-local
internal-variable problem (`q = q_ref + γ̃·g` is the normalization; the element
may integrate with any consistent rule over it, including exponential
updates). For a backward-Euler local state under any one-step global scheme,
`γ̃ = Δt`. **`γ̃` is not a rate slope** — see the `TimeIntegrationContext`
docstring for the trap.

Slot names are free: a Newmark-style protocol passes whatever it needs, e.g.
`slots = (:u, :uprev, :vanchor)`. *Interim note until the phase-2 `AffineRate`
and slot-metadata contract land*: rate-reconstruction **slopes** (`∂v/∂u =
γ/(βΔt)` etc.) have no framework channel yet — carry them in your parameter
object `p` (define `unwrap_parameters` for your wrapper so plain elements see
through it), and migrate to slot metadata when phase 2 ships. Do not put
slopes into `ctx`.

Condensed elements: declare `FerriteOperators.has_internal_state(::Type{<:MyCache}) = true`.
The previous state arrives through your chosen slot; the local solve scales by
`args.ctx.γ̃`; the trial result is written into the element-local `u` buffer
(the framework propagates it — the condensation contract is unchanged).

## Facets ⚠

The framework owns the facet loop now. `is_facet_in_cache` is unchanged;
kernels become request-typed and facet parameters are queried per facet:

```julia
# 0.3.x
function assemble_facet!(rₑ, uₑ, cell, lfi, cache::MyFacetCache, p) ... end

# v2
function FerriteOperators.assemble_facet!(req::ResidualRequest, cache::MyFacetCache, args::KernelArgs, lfi::Int)
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
`op.subdomain_caches` now reads `op.engine.*`. `getJ(op) = op.J` style
accessors keep working.

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

- **Sensitivities**: `update_parameter_jacobian!`, `parameter_vjp!`,
  `time_sensitivity!` (AD by default, analytic kernels win per cache,
  `FiniteDifferenceSensitivity` for condensed time derivatives). Declare the
  kinds you will use at setup (`requests = (ParameterVJPKind, …)`) to move
  admissibility failures from first use to `setup_operator`.
- **Matrix-free state actions**: `state_jvp!` (`J·v` without a matrix),
  `state_vjp!` (`Jᵀλ` — the adjoint action).
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
   `Ferrite.getnquadpoints`/`Ferrite.reinit!` methods for every cache.
5. Replace parameter-wrapper unwrapping hacks with one `unwrap_parameters`
   method; move facet-specific parameters into `query_facet_parameters`.
6. Replace solver-state smuggling with `scratch` declarations.
7. Run your suite; every port failure except facets is a loud
   `MethodError`/`ArgumentError` at setup or first assembly.
