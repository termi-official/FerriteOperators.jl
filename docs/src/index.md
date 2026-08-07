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

## Architecture Overview

FerriteOperators sits between Ferrite modeling code and solver code. Its
design follows the fundamental finite-element operator decomposition
popularized by MFEM and libCEED: element restriction, basis evaluation,
pointwise physics, and global scatter are separate concerns, and how much of
the operator is materialized (full sparse matrix, stored element matrices,
matrix-free action) is a *strategy axis*, not a property of the physics.

The pipeline is built around these pieces:

1. **Requests** encode *what a kernel computes* for one work item: the
   residual, a Jacobian, a fused Jacobian+residual, or a sensitivity
   (`∂F/∂θ`, adjoint pullbacks, `∂F/∂t`). Elements implement request-typed
   kernels; solvers and operators speak in buffer-less request *kinds*.
2. **The assembly strategy** is the composition of three orthogonal axes:
   the *operator form* ([`FullAssembly`](@ref) / [`ElementAssembly`](@ref) —
   the MFEM assembly level), the *scheduling policy*
   ([`SequentialScheduling`](@ref) / [`ColoredScheduling`](@ref) — how
   parallel work is made race-safe), and the *device* (sequential CPU,
   threaded via Polyester, GPUs in the future). The historical names
   (`SequentialAssemblyStrategy(device)`, `PerColorAssemblyStrategy(device)`,
   `ElementAssemblyStrategy(device)`) remain as convenience constructors.
3. **The assembly engine** (`AssemblyEngine`) holds the strategy, the
   per-subdomain caches (workspaces + partitions), and the dof handler.
   Operators are a payload (matrix/vector) plus an engine plus their
   integrator.
4. **Workspaces** hold pre-allocated per-worker data: local matrices and
   residuals, one state buffer per declared slot, the geometry cache, the
   element caches, and declared scratch space.

All operator entry points funnel into one task body executed by a shared
device loop:

```
for chunk in partition
    parfor item in chunk
        reinit!(workspace, item)          # geometry cache only
        execute_single_task!(task, workspace)
    end
end
```

Elements own the `reinit!` of their values objects (`CellValues` etc.),
selecting what to reinitialize per request kind — an element may carry
several values objects, and not every request needs all of them.

## Writing an element

An element consists of an integrator (setup-time description), a cache, and
request-typed kernels. The **residual kernel is mandatory** (validated at
setup); everything else is derived from it by ForwardDiff unless an analytic
kernel is declared. A minimal nonlinear element:

```julia
struct MyIntegrator <: AbstractNonlinearIntegrator
    qrc::QuadratureRuleCollection
    field_name::Symbol
end

struct MyCache{CV <: CellValues} <: AbstractVolumetricElementCache
    cv::CV
end

function FerriteOperators.setup_element_cache(m::MyIntegrator, sdh::SubDofHandler)
    qr = getquadraturerule(m.qrc, sdh)
    ip = Ferrite.getfieldinterpolation(sdh, m.field_name)
    ip_geo = FerriteOperators.geometric_subdomain_interpolation(sdh)
    return MyCache(CellValues(qr, ip, ip_geo))
end
FerriteOperators.duplicate_for_device(device, c::MyCache) =
    MyCache(FerriteOperators.duplicate_for_device(device, c.cv))

function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::MyCache, args::KernelArgs)
    (; cv) = cache
    uₑ = args.states.u
    reinit!(cv, args.cell)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # ... accumulate into req.r ...
    end
end
```

This alone buys the assembled Jacobian, the fused Newton path, and all
sensitivities through the AD fallback. Analytic kernels are an opt-in
optimization:

```julia
FerriteOperators.provides_analytic(::Type{<:MyCache}, ::FerriteOperators.JacobianKind) = true
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, cache::MyCache, args::KernelArgs)
    # ... accumulate into req.K ...
end
```

The residual kernel must be eltype-generic in `eltype(args.states.*)`,
`eltype(args.p)`, and the context time — that is the entire AD contract.
Kernels never write global state; the geometry cache in `args.cell` is
read-only.

### KernelArgs

- `args.states` — NamedTuple of element-local state buffers, one per slot
  declared at setup (`setup_operator(...; slots = (:u, :uprev))`). Slots are
  gathered through the element-overridable `load_element_unknowns!`, so
  condensed elements receive their full `[ū; q]` local layout for every slot.
- `args.p` — the user parameter bag, produced by the overridable
  [`query_cell_parameters`](@ref) (facets get their own
  [`query_facet_parameters`](@ref) per facet).
- `args.ctx` — the [`TimeIntegrationContext`](@ref) `(t, Δt, γ̃)`, or
  `nothing` for stationary problems. `γ̃` is the *normalized* local stage
  interval of the element-local internal-variable problem — see its
  docstring for the exact contract and for why it is **not** a rate slope.
- `args.scratch` — per-worker scratch declared by the solver
  (`setup_operator(...; scratch = (name = () -> ...,))`) and/or the element
  (`declare_scratch(cache)`).

### Condensed elements (internal variables)

Elements with per-quadrature-point internal state append their unknowns after
the FE dofs (`u = [ū; q]`, managed by the [`InternalVariableHandler`](@ref))
and own their local stage problem: the previous state arrives through a slot
(e.g. `uprev`), the local solve scales by `args.ctx.γ̃`, and the trial result
is written into the element-local `u` buffer — the framework propagates it
into the global trial vector (the condensation contract: `q(ū)` is refreshed
at every trial evaluation, line search included). Declare
[`has_internal_state`](@ref) for such caches — it governs the sensitivity
admissibility rules below.

## Operators and entry points

```julia
strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
op = setup_operator(strategy, integrator, dh; slots = (:u, :uprev))

# canonical states/ctx forms; u-vector conveniences exist for stationary use
update_linearization!(op, residual, (u = u, uprev = uprev), p, TimeIntegrationContext(t, Δt, γ̃))
residual!(op, residual, (u = u,), p, nothing)
mul!(y, op.J, v)
```

Time discretization of the global unknowns is solver-owned: solvers pass slot
*values* (reconstructed histories, rates) and contexts; elements never encode
a scheme. The hand-derived first-order path (an element reading `uprev` and
`ctx` and owning its discretization) remains a supported opt-in pattern.

### Sensitivities

```julia
update_parameter_jacobian!(B, op, states, p, ctx)   # ∂F/∂θ, dense
parameter_vjp!(g, op, λ, states, p, ctx)            # (∂F/∂θ)ᵀλ, matrix-free
time_sensitivity!(g, op, states, t, ctx)            # ∂F/∂t
time_sensitivity!(g, op, u, t; method = FiniteDifferenceSensitivity())
```

θ is the flat view defined by [`parameter_vector`](@ref) /
[`rebuild_parameters`](@ref). Per cache, analytic sensitivity kernels win;
otherwise ForwardDiff differentiates the residual kernel. Sensitivity sweeps
**never** write back into the caller's state.

Admissibility with internal state: AD through an element-local solve is wrong
in principle, so a cache with [`has_internal_state`](@ref) is admissible for
a sensitivity kind only if it (a) provides the analytic kernel, (b) declares
[`internal_state_insensitive`](@ref) (asserting the local equations do not
depend on the seeded quantity — then AD is exact), or (c) for time
sensitivities, the caller selects [`FiniteDifferenceSensitivity`](@ref)
(primal central differences on a protected copy — exact local solves, but it
bypasses analytic sensitivity kernels).

### Declaring request kinds at setup

```julia
op = setup_operator(strategy, integrator, dh;
                    requests = (ParameterVJPKind, TimeSensitivityKind))
```

Declared sensitivity kinds run their trait ↔ kernel and internal-state
admissibility checks eagerly at `setup_operator` instead of on first use —
an inadmissible adjoint fails when the operator is built, not mid-solve.
Undeclared kinds stay fully usable (the declaration is a hint, not a
capability restriction); their checks run at the call-time entry points.

State and time derivative sweeps (`update_linearization!` via AD,
`state_jvp!`, `state_vjp!`, `time_sensitivity!`) run over per-worker
preallocated buffers and ForwardDiff configurations and are allocation-free
per cell. The parameter sweeps preallocate their output buffers but rebuild
their ForwardDiff configurations per call — their seed dimension nθ arrives
with `p` and becomes setup-time knowledge once parameter layouts land.

## Quadrature data

Per-quadrature-point evaluation runs through the same engine as assembly:

```julia
q = setup_qvector(Float64, dh, qrc)
evaluate_quadrature!(q, op, u, p, (uₑ, qp, cell, cache, pₑ) -> ...)
```

with [`QVector`](@ref) as the flat storage, cell-set filtering, query/store
hooks for element-owned layouts, and the VTK export layer
([`VTKQuadratureGrid`](@ref), [`VTKQuadratureFile`](@ref),
[`write_quadrature_data`](@ref)) for visualization at quadrature points.

## Transfer operators

Rectangular transfer (prolongation/restriction) operators between two
DofHandlers — same-grid (p-multigrid) and nested-grid (geometric multigrid)
variants — currently live in their own small hierarchy
([`setup_transfer_operator`](@ref), [`MassProlongatorIntegrator`](@ref),
[`NestedMassProlongatorIntegrator`](@ref)) and will fold into the unified
engine as the two-DofHandler item family.

## API Reference

```@index
```

```@autodocs
Modules = [FerriteOperators]
```
