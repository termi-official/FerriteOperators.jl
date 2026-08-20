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

## The pipeline

1. **Requests** encode *what a kernel computes* for one work item: the
   residual, a Jacobian, a fused Jacobian+residual, a weighted Jacobian, or a
   sensitivity (`∂F/∂θ`, adjoint pullbacks, `∂F/∂t`). Elements implement
   request-typed kernels; solvers and operators speak in buffer-less request
   *kinds*.
2. **The assembly strategy** is the composition of three orthogonal axes:
   the *operator form* ([`FullAssembly`](@ref) / [`ElementAssembly`](@ref) —
   the MFEM assembly level), the *scheduling policy*
   ([`SequentialScheduling`](@ref) / [`ColoredScheduling`](@ref) — how
   parallel work is made race-safe), and the *device* (sequential CPU,
   threaded via Polyester, GPUs in the future). The names
   `SequentialAssemblyStrategy(device)`, `PerColorAssemblyStrategy(device)`
   and `ElementAssemblyStrategy(device)` are convenience constructors for the
   common compositions.
3. **The scheme protocol** carries the setup-time declarations — slot names
   and request kinds.
4. **The assembly engine** ([`AssemblyEngine`](@ref)) holds the strategy, the
   per-subdomain caches (workspaces + partitions), the dof handler, and the
   protocol. Operators are a payload (matrix/vector) plus an engine plus their
   integrator.
5. **Workspaces** hold pre-allocated per-worker data: a fixed core of local
   matrices and residuals, one state buffer per declared slot, the geometry
   cache and the element caches — plus the sweep-state families the
   declarations call for.

All operator entry points funnel into one task body executed by a shared
device loop:

```
for chunk in partition
    parfor item in chunk
        reinit!(workspace, item)               # geometry cache
        reinit_values!(element, item, kind)    # element values, once per sweep
        execute_single_task!(task, workspace)
    end
end
```

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

Developer documentation:

- [The layer contract](devdocs/design.md) — term / operator / scheme layers and
  their ownership boundaries, the channel decision table, and the framework's
  extension points.
- [Design rationale](devdocs/rationale.md) — why the design is the way it is:
  the decisions, the alternatives that were rejected, and what they cost.
