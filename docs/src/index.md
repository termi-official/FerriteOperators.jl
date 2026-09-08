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

strategy = AssemblyStrategy(SequentialCPUDevice())
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
axes are orthogonal: the *operator form* ([`AbstractAssemblyForm`](@ref) — the
MFEM assembly level), the *scheduling policy* ([`SequentialScheduling`](@ref) /
[`ColoredScheduling`](@ref) — how parallel work is made race-safe), and the
*device* (sequential CPU, threaded via Polyester, GPU via
KernelAbstractions.jl).
[`AssemblyStrategy`](@ref)`(device; form, scheduling)` is the keyword
convenience constructor for the common compositions: `AssemblyStrategy(device)`
and `AssemblyStrategy(device; scheduling = ColoredScheduling())`.

[`PolyesterDevice`](@ref) lives behind a package extension: `using Polyester`
activates it, and without that load the type exists with no execution route.
[`default_strategy`](@ref) resolves that at call time — the Polyester device
where the extension is loaded, [`SequentialCPUDevice`](@ref) otherwise.

## GPU assembly

[`KernelAbstractionsDevice`](@ref) assembles bilinear and linear forms on a
KernelAbstractions.jl backend. This package depends on no GPU vendor package:
the backend object and the device matrix type both come from the caller.

```julia
using CUDA, Adapt, KernelAbstractions   # `using CUDA` loads all of these
import CUDA: CUSPARSE.CuSparseMatrixCSC

device   = KernelAbstractionsDevice(CUDABackend(); value_type = Float32, index_type = Int32)
spec     = StandardOperatorSpecification(; matrix_type = CuSparseMatrixCSC{Float32, Int32})
strategy = AssemblyStrategy(FullAssembly(spec), ColoredScheduling(), device)

integrator = MyIntegrator(QuadratureRuleCollection(Float32, 2), :u)  # element precision
op = setup_operator(strategy, integrator, dh)   # op.A lives on the device
update_operator!(op, p)
```

`value_type` governs the GLOBAL system alone. The precision the element caches
evaluate in is the INTEGRATOR's, elected through its quadrature collection
(`QuadratureRuleCollection(Float32, 2)` — see
[Evaluation precision](elements.md#Evaluation-precision)), so a device run
elects it there as well. The two are free to differ: the scatter converts.
Build the grid with `Float32` coordinates too: a `Float64` grid assembles
correctly, but its coordinates are what the geometry mapping computes in, so
the device pays `Float64` memory and arithmetic for it. The linear operator's
vector is allocated on the device too, and the assembled matrix stays there — a
sweep transfers nothing.

What the device covers is CELL items under [`ColoredScheduling`](@ref), which
is REQUIRED: Ferrite's device matrix assembler accumulates without atomics.
Rejected at setup, each with a message naming the limitation:
[`SequentialScheduling`](@ref), facet items, algebraic items, patch and
transfer operators, condensed internal state, nonlinear integrators, a
[`BlockedOperatorSpecification`](@ref), constraints declared on the operator
specification, [`global_dofs`](@ref) declarations, value-returning sweeps
(functionals, quadrature evaluation), and a device matrix type Ferrite has no
assembler for.

An element cache reaches the device by declaring which of its fields are
batched per worker and which are shared, through
[`setup_device_instances`](@ref) and [`device_worker_view`](@ref):

```julia
setup_device_instances(dev::AbstractGPUDevice, c::MyCache, n) =
    MyCache(c.D, setup_device_instances(dev, c.cellvalues, n))
device_worker_view(c::MyCache, w) = MyCache(c.D, device_worker_view(c.cellvalues, w))
```

The cache's field type parameters have to admit the batched layout — a
`cellvalues::CV` field holds a struct-of-arrays container over `n` workers on
the device, not a `CellValues`.

[`FullAssembly`](@ref) assembles the global matrix and vector and serves every
operator family. It is the form axis' sole member; the axis and the `form`
keyword are the extension point a further assembly level (element assembly,
matrix-free) is added at.

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
what along that path — requests, engines and workspaces included.

## Where to read on

- [Writing elements](elements.md) — request-typed kernels, the cell/facet
  argument bundle, values reinitialization, parameter queries, analytic
  opt-ins, condensed elements, functionals.
- [Operators and entry points](operators.md) — setup and its declarations,
  the assembly entry points, slots and rate reconstruction, sensitivities,
  weighted Jacobians, component bags and stage operators, derivative
  verification, quadrature data, transfer operators.
- [Patch items](patches.md) — multi-cell work items with patch-local scatter
  (experimental).

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

- [The layer contract](devdocs/design.md) — the term and operator layers, their
  ownership boundaries and what the calling solver owns instead, the channel
  decision table, and the framework's extension points.
- [Design rationale](devdocs/rationale.md) — why the design is the way it is:
  the decisions, the alternatives that were rejected, and what they cost.
