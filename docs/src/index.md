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
        reinit!(workspace, item)               # geometry cache
        reinit_values!(element, item, kind)    # element values, once per sweep
        execute_single_task!(task, workspace)
    end
end
```

Elements own their values objects (`CellValues` etc.) and implement
[`reinit_values!`](@ref): the mandatory two-arg method reinitializes all of
them; specializing the kind-dispatched three-arg form reinitializes only what
that request needs — an element may carry several values objects, and not
every request needs all of them. Kernels are pure evaluation: repeated kernel
invocations within one sweep (AD chunk passes, split Jacobian-then-residual
fallbacks) do not reinitialize again.

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

FerriteOperators.reinit_values!(c::MyCache, cell) = reinit!(c.cv, cell)

function FerriteOperators.assemble_cell!(req::ResidualRequest, cache::MyCache, args)
    (; cv) = cache
    uₑ = args.states.u
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
function FerriteOperators.assemble_cell!(req::JacobianRequest{:u}, cache::MyCache, args)
    # ... accumulate into req.K ...
end
```

The residual kernel must be eltype-generic in `eltype(args.states.*)`,
`eltype(args.p)`, and the context time — that is the entire AD contract.
Kernels never write global state; the geometry cache in `args.cell` is
read-only. The `args` parameter stays **unannotated**: kernels select on the
`(request, cache)` pair alone, and an open parameter lets the element serve
any operator family's args type (setup warns about a concrete annotation).

### The kernel-args channel protocol

`args` is any object carrying the channels below; [`KernelArgs`](@ref) is what
this package's operators build, not the contract.

- `args.states` — NamedTuple of element-local state buffers, one per slot
  declared at setup (`setup_operator(...; slots = (:u, :uprev))`). Slots are
  gathered through the element-overridable `load_element_unknowns!`, so
  condensed elements receive their full `[ū; q]` local layout for every slot.
- `args.cell` — the geometry cache of the current item, read-only.
- `args.p` — the user parameter bag, produced by the overridable
  [`query_cell_parameters`](@ref) (facets get their own
  [`query_facet_parameters`](@ref) per facet). Configuration only: time lives
  in `ctx`, history in slots.
- `args.scratch` — per-worker scratch declared by the solver
  (`setup_operator(...; scratch = (name = () -> ...,))`) and/or the element
  (`declare_scratch(cache)`).
- `args.ctx` — the per-sweep solver scalars, i.e. the
  [`TimeIntegrationContext`](@ref) `(t, Δt, γ̃)` read through
  `evaluation_time(args.ctx)` and `args.ctx.γ̃`, or `nothing` for stationary
  problems. `γ̃` is the *normalized* local stage interval of the
  element-local internal-variable problem — see its docstring for the exact
  contract and for why it is **not** a rate slope.

Per-slot metadata is reserved protocol vocabulary: a future args family may
carry a per-slot property, and `KernelArgs` carries none.

An operator family may build its own args type; it then implements the three
rebuild seams the framework re-seeds channels through —
`FerriteOperators.with_states`, `with_parameters`, and `with_context` — as
plain methods on that type. There is no abstract fallback: a family missing a
seam gets a `MethodError` on the sweep that needs it, never a silently
unseeded derivative.

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

### Unit-testing a kernel

Kernels are pure evaluation, so they can be called directly on a single cell
without an operator. Building the cell cache and the [`KernelArgs`](@ref) by
hand is the supported testing seam:

```julia
cache = FerriteOperators.setup_element_cache(MyIntegrator(qrc, :u), sdh)

cc = CellCache(dh)
reinit!(cc, 1)                      # geometry for cell 1
reinit_values!(cache, cc)           # the element's own values objects

uₑ = rand(ndofs_per_cell(sdh))
rₑ = zeros(ndofs_per_cell(sdh))
args = KernelArgs((u = uₑ,), cc, p, nothing, nothing)
assemble_cell!(ResidualRequest(rₑ), cache, args)
```

`KernelArgs` is constructed positionally as `(states, cell, p, scratch, ctx)`;
`scratch` and `ctx` are whatever the kernel reads (`nothing` when it reads
neither). Pass further slots as additional entries of the states NamedTuple.

## Operators and entry points

```julia
strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
op = setup_operator(strategy, integrator, dh; slots = (:u, :uprev))

# canonical states/ctx forms; u-vector conveniences exist for stationary use
update_linearization!(op, residual, (u = u, uprev = uprev), p, TimeIntegrationContext(t, Δt, γ̃))
evaluate!(op, residual, (u = u,), p, nothing)
mul!(y, op.J, v)
```

Time discretization of the global unknowns is solver-owned: solvers pass slot
*values* (reconstructed histories, rates) and contexts; elements never encode
a scheme. The hand-derived first-order path (an element reading `uprev` and
`ctx` and owning its discretization) remains a supported opt-in pattern.

A rate-like slot can be reconstructed from the primary unknown instead of
being materialized by the solver: an [`AffineRate`](@ref) source gives the
slot the cell-local value `slope · (u − anchor)`, e.g.
`update_linearization!(op, r, (u = u, du = AffineRate(1/Δt, uprev)), p, ctx)`
for backward Euler. The `:u` slot must precede the reconstructed one. Kernels
read the reconstructed values through `args.states.du` and nothing else, so an
element stays scheme-agnostic. The assembled Jacobian is ∂F/∂u at frozen slot
values; the chain-rule term through the reconstruction is contributed by the
solver's per-slot weights.

### Components and stage operators

A multi-slot linearization is assembled one slot at a time and folded by the
solver, so a scheme's matrix never needs its own kernel:

```julia
comps = allocate_components(op, (:Ju, :Jdu))          # one shared sparsity pattern
assemble_slot_jacobian!(comps.Ju,  op, JacobianKind{:u}(),  states, p, ctx)
assemble_slot_jacobian!(comps.Jdu, op, JacobianKind{:du}(), states, p, ctx)
combine!(W, comps, (Jdu = 1 / Δt, Ju = 1.0))          # backward Euler Newton matrix
```

[`allocate_components`](@ref) hands out square system matrices that share one
sparsity pattern (aliased `colptr`/`rowval`, private `nzval`), which makes
[`combine!`](@ref) a pure values operation and `apply_zero!` safe on any
member; structural mutation of a component breaks the bag and is not
supported. Components are plain system matrices — every existing assembly
entry point fills them. `combine!` is eltype-generic: real components with
complex weights combine into a complex target from
`share_pattern(A, ComplexF64)`.

The differentiated slot must carry a plain vector source. An
[`AffineRate`](@ref) slot is reconstructed at gather time and frozen under AD,
so `JacobianKind{:du}()` against it is rejected — assemble the components
against plain sources and let the reconstruction slope enter as a weight.

Fully implicit Runge-Kutta assembles `s` stage pairs and applies the s×s
Newton block `δᵢⱼ Jdu⁽ⁱ⁾ + Δt aᵢⱼ Ju⁽ⁱ⁾` without ever building it:

```julia
sbop = StageBlockOperator(op, A, c, Δt)
assemble_stages!(sbop, op, stage_states, p, ctxs)     # 2s sweeps, one per stage and slot
mul!(y, sbop, x)                                      # x, y stage-stacked, length s·n
```

The transformed (simplified-Newton) variant needs no stage-block machinery:
diagonalized Radau uses stage-*independent* Jacobians, i.e. a single
`(Ju, Jdu)` bag plus one complex `combine!(W_λ, comps, (Jdu = 1.0, Ju = Δt*λ))`
per eigenvalue of `A⁻¹`.

### Functionals

```julia
FerriteOperators.evaluate_cell_functional(::FunctionalKind{:energy}, cache::MyCache, args) =
    # return this cell's ∫ contribution (a Number or a Tensors tensor)

Φ = evaluate_functional(op, FunctionalKind(:energy), states, p, ctx)
```

Global reductions (energies for line searches, dissipation, quantities of
interest) are request kinds whose kernels *return* their cell contribution;
the engine sums per worker and reduces in a fixed order (deterministic for a
fixed worker count). Volumetric contributions only.

### Sensitivities

```julia
update_parameter_jacobian!(B, op, states, p, ctx)   # ∂F/∂θ, dense
parameter_vjp!(g, op, λ, states, p, ctx)            # (∂F/∂θ)ᵀλ, matrix-free
time_sensitivity!(g, op, states, p, ctx)            # ∂F/∂t at evaluation_time(ctx)
time_sensitivity!(g, op, states, p, ctx; method = FiniteDifferenceSensitivity())
```

θ is the flat view defined by [`parameter_vector`](@ref) /
[`rebuild_parameters`](@ref). Per cache, analytic sensitivity kernels win;
otherwise ForwardDiff differentiates the residual kernel. Sensitivity sweeps
**never** write back into the caller's state.

∂F/∂t seeds through the context channel: the AD sweep hands the kernel a
context whose evaluation time is Dual-valued, and the finite-difference method
evaluates the primal residual at contexts with perturbed times. An element
therefore reads time as `evaluation_time(args.ctx)`, and `time_sensitivity!`
requires a context — passing `nothing` is an `ArgumentError`.

Admissibility with internal state: AD through an element-local solve is wrong
in principle, so a cache with [`has_internal_state`](@ref) is admissible for
a sensitivity kind only if it (a) provides the analytic kernel, (b) declares
[`internal_state_insensitive`](@ref) (asserting the local equations do not
depend on the seeded quantity — then AD is exact), or (c) for time
sensitivities, the caller selects [`FiniteDifferenceSensitivity`](@ref)
(primal central differences on a protected copy — exact local solves, but it
bypasses analytic sensitivity kernels).

### Verifying derivative implementations

```julia
res = check_derivatives(op, states, p, ctx)
res.passed                      # conjunction of all non-skipped checks
res.checks.jacobian.err         # per-check relative error / skip reason
```

[`check_derivatives`](@ref) cross-checks every derivative path — the
assembled Jacobian, fused-vs-split residual, parameter Jacobian/VJP, state
JVP/VJP, time sensitivity — against central finite differences of the
operator's own residual, through the public entry points. The time check runs
only with a context and is recorded as a skip without one. A wrong analytic
kernel fails its check against the FD referee; inadmissible or unsupported
checks are skipped with the reason recorded. The parameter checks respect
the differentiable/static split: only the entries exposed by
[`parameter_vector`](@ref) are probed.

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
