```@meta
CurrentModule = FerriteOperators
```

# FerriteOperators

*A SciML compatible high performance parallel assembly system for [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)*.

!!! note
    For an assembly framework in Ferrite.jl style we refer users for now to [FerriteAssembly.jl](https://github.com/KnutAM/FerriteAssembly.jl).

!!! warning
    This package is under heavy development. Expect regular breaking changes
    for now. If you are interested in joining development, then either comment
    an issue or reach out via julialang.zulipchat.com, via mail or via
    julialang.slack.com. Alternatively open a discussion if you have something
    specific in mind.

!!! note
    If you are interested in using this package, then I am also happy to
    to get some constructive feedback, especially if things don't work out
    in the current design. This can be done via julialang.slack.com,
    julialang.zulipchat.com or via mail.

## What this package is

FerriteOperators sits between Ferrite modeling code and solver code. Its
design follows the fundamental finite-element operator decomposition
popularized by MFEM and libCEED: element restriction, basis evaluation,
pointwise physics, and global scatter are separate concerns, and how much of
the operator is materialized (full sparse matrix, stored element matrices,
matrix-free action) is a *strategy axis*, not a property of the physics.

Elements express scheme-agnostic integrands. Operators evaluate a set of them
at a given state, parameter bag, and per-sweep context. Solvers own the time
discretization and compose operator evaluations into a scheme. [The layer
contract](devdocs/design.md) states that division of labour precisely.

## Quickstart

```julia
using FerriteOperators

strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
op = setup_operator(strategy, MyIntegrator(qrc, :u), dh; slots = (:u, :uprev))

r = zeros(ndofs(dh))
update_linearization!(op, r, (u = u, uprev = uprev), p, TimeIntegrationContext(t, Δt, γ̃))
Δu = op.J \ r
```

An element supplies one mandatory residual kernel; the assembled Jacobian, the
fused Newton path, and every sensitivity follow from it by ForwardDiff unless
analytic kernels are declared.

```julia
function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::MyCache, args)
    (; cv) = cache
    uₑ = args.states.u
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # ... accumulate into req.r ...
    end
end
```

## The assembly strategy

Which machinery an operator is built on is one composite choice, and the three
axes are orthogonal: the *operator form* ([`FullAssembly`](@ref) /
[`ElementAssembly`](@ref) — the MFEM assembly level), the *scheduling policy*
([`SequentialScheduling`](@ref) / [`ColoredScheduling`](@ref) — how parallel
work is made race-safe), and the *device* (sequential CPU, threaded via
Polyester). `SequentialAssemblyStrategy(device)`,
`PerColorAssemblyStrategy(device)` and `ElementAssemblyStrategy(device)` are
convenience constructors for the common compositions.

[`FullAssembly`](@ref) assembles the global matrix and vector and serves every
operator family. [`ElementAssembly`](@ref) accumulates per-element vector
contributions and collapses them into the global vector at the end of the
sweep, so it serves linear operators only — it holds no matrix, and a bilinear
or nonlinear operator under it is rejected at setup.

All operator entry points funnel into one task body executed by a shared
device loop:

```
for chunk in partition
    parfor item in chunk
        reinit!(workspace, item)                # geometry cache
        reinit_values!(cache, cell, kind)       # element values, once per sweep
        execute_single_task!(task, workspace)
    end
end
```

[The layer contract](devdocs/design.md) has the layer table that names who owns
what along that path — requests, protocols, engines and workspaces included.

## Where to read on

- [Writing elements](elements.md) — request-typed kernels, the cell/facet
  argument bundle, values reinitialization, parameter queries, analytic
  opt-ins, condensed elements, functionals.
- [Operators and entry points](operators.md) — setup and scheme protocols,
  the assembly entry points, slots and rate reconstruction, sensitivities,
  weighted Jacobians, component bags and stage operators, derivative
  verification, quadrature data, transfer operators.
- [Patch items](patches.md) — multi-cell work items with patch-local scatter
  (experimental).
- [Migrating from 0.3.x](migration.md) — the map from the old element and
  operator API to the current one.

API reference:

- [Element API reference](element-api.md) — the contracts an element cache
  implements, and the request types its kernels take.
- [Provided integrators and caches](provided-elements.md) — composition,
  multi-domain routing, the AD decorator, the transfer prolongators.
- [Example elements](example-elements.md) — the worked implementations in
  `FerriteOperatorsExampleElements`.
- [Operator API reference](operator-api.md) — the operator types and every
  assembly, sensitivity and condensation entry point.
- [Assembly engine API reference](engine-api.md) — kinds, drivers, strategies,
  devices, workspaces and the quadrature layer.

Developer documentation:

- [The layer contract](devdocs/design.md) — term / operator / scheme layers and
  their ownership boundaries, the channel decision table, and the framework's
  extension points.
- [Design rationale](devdocs/rationale.md) — why the design is the way it is:
  the decisions, the alternatives that were rejected, and what they cost.
