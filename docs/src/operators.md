```@meta
CurrentModule = FerriteOperators
```

# Operators and entry points

```julia
strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
op = setup_operator(strategy, integrator, dh; slots = (:u, :uprev))

# canonical states/ctx forms; u-vector conveniences exist for stationary use
update_linearization!(op, residual, (u = u, uprev = uprev), p, TimeIntegrationContext(t, Δt, γ̃))
evaluate!(op, residual, (u = u,), p, nothing)
mul!(y, op.J, v)
```

An operator is a payload (matrix/vector) plus an [`AssemblyEngine`](@ref) plus
its integrator. Every entry point funnels into one sweep over the engine's
subdomain caches, so the difference between a residual, a Jacobian, and a
sensitivity is the request kind, not a separate driver.

[`evaluate!`](@ref) is layer-polymorphic by design: it evaluates whatever the
integrator encodes — a nonlinear residual, the action of the linear operator a
bilinear form induces, a hand-fused scheme residual.

## Scheme protocols

What a scheme asks of an operator is setup-time knowledge, and a *protocol* is
where it is declared:

```julia
struct SDIRKWProtocol <: AbstractSchemeProtocol end
FerriteOperators.get_declared_slots(::SDIRKWProtocol) = (:u, :du)
FerriteOperators.get_declared_kinds(::SDIRKWProtocol) = (WeightedJacobianKind, ResidualKind)

op = setup_operator(strategy, integrator, dh, SDIRKWProtocol())
```

Protocols are **declarations only**: slot names and request kinds. They carry
no coefficients — γ, tableaus and weights are per-evaluation solver data —
and nothing term-shaped; a term needing its own context or sink is its own
sweep.

The keyword form is sugar whose keywords are the [`DefaultProtocol`](@ref)
constructor arguments, so both forms build the same operator:

```julia
op = setup_operator(strategy, integrator, dh;
                    requests = (ParameterVJPKind, TimeSensitivityKind))
```

Declaring a kind runs its trait ↔ kernel and internal-state admissibility
checks eagerly at `setup_operator` instead of on first use — an inadmissible
adjoint fails when the operator is built, not mid-solve. Which element caches
carry [`ADElementCache`](@ref) decoration, and whether the workspace carries
[`SensitivityBuffers`](@ref) at all, is decided separately and structurally by
the integrator family ([`needs_ad_decoration`](@ref)) — a bilinear or linear
operator carries no AD/sensitivity machinery whatever an element cache does or
does not implement analytically, whatever the protocol declares. The
workspace is immutable — every field is bound at `setup_operator`, and a sweep
works by filling the buffers those fields point at.

Declaring a [`FunctionalKind`](@ref) builds **nothing**, which is the feature
rather than an omission: a functional sweep's kernel returns the cell's
contribution and the sweep folds the returned values, so it has no per-worker
state to allocate and nothing to reset between evaluations. What such a kind
does declare is its reduction's value type, on the kind rather than on the
protocol — [`FerriteOperators.functional_value_type`](@ref), required under a
parallel device and described with the kernel hook in
[Writing elements](@ref Functionals).

Declaring stays a hint, never a capability restriction: an operator always
builds what its own integrator family issues ([`mandatory_kinds`](@ref)).
Undeclared kinds run their checks at the call-time entry points.

## The assembled matrix

What the operator's global matrix looks like is the *operator specification*
carried by [`FullAssembly`](@ref). [`StandardOperatorSpecification`](@ref) — the
default — gives a monolithic `SparseMatrixCSC`, and both specifications take the
same two pattern declarations:

- `algebraic_couplings`, Ferrite coupling descriptors (`CellCoupling`,
  `FacetCoupling`, `AlgebraicCoupling`) for the entries an element's
  [`global_dofs`](@ref) couple into — see
  [Elements with global dofs](elements.md#Elements-with-global-dofs) — and for
  those an [`algebraic_items`](@ref) declaration needs, see
  [Algebraic terms](elements.md#Algebraic-terms-(items-with-no-mesh-support));
- `constraint_handler`, which adds the constraint entries so condensation has
  room to write. Sparsity **only**: applying the constraints to the assembled
  system stays with the caller, through Ferrite's `apply!`/`apply_assemble!`.

### Blocked matrices with CSR blocks

[`BlockedOperatorSpecification`](@ref) assembles into a `BlockMatrix` instead —
the layout a fieldwise preconditioner or a Schur-complement consumer wants, and
the way to give the blocks a storage format the monolithic path does not offer.
The matrix type is the caller's: this package depends on neither BlockArrays nor
SparseMatricesCSR, so the user loads them and names the type.

```julia
using BlockArrays, SparseMatricesCSR

spec     = BlockedOperatorSpecification([nu, nalg],
                                        BlockMatrix{Float64, Matrix{SparseMatrixCSR{1, Float64, Int}}};
                                        algebraic_couplings = (coupling,))
strategy = AssemblyStrategy(FullAssembly(spec), SequentialScheduling(), SequentialCPUDevice())
op       = setup_operator(strategy, integrator, dh)

update_linearization!(op, r, u, p)   # the fused Newton path
```

The residual stays a plain `Vector` — `create_system_vector` is unchanged, and
Ferrite's `BlockAssembler` takes a non-blocked `f`. The fused,
matrix-only and residual-only sweeps all work, so a bilinear operator assembles
into a blocked target just as a nonlinear one does. A *linear* operator holds no
matrix at all, and a blocked specification on one is rejected at
`setup_operator` rather than silently dropped.

## Slots and rate reconstruction

Time discretization of the global unknowns is solver-owned: solvers pass slot
*values* (reconstructed histories, rates) and contexts; elements never encode
a scheme. The hand-derived first-order path — an element reading `uprev` and
`ctx` and owning its discretization — is a supported opt-in pattern, expressed
through the same slot interface as everything else.

A rate-like slot can be reconstructed from the primary unknown instead of
being materialized by the solver: an [`AffineRate`](@ref) source gives the
slot the cell-local value `slope · (u − anchor)`, e.g.
`update_linearization!(op, r, (u = u, du = AffineRate(1/Δt, uprev)), p, ctx)`
for backward Euler. The `:u` slot must precede the reconstructed one. Kernels
read the reconstructed values through `args.states.du` and nothing else, so an
element stays scheme-agnostic. The assembled Jacobian is ∂F/∂u at frozen slot
values; the chain-rule term through the reconstruction is contributed by the
solver's per-slot weights.

## Components and stage operators

A multi-slot linearization can be assembled one slot at a time and folded by
the solver, so a scheme's matrix never needs its own kernel:

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
supported. Components are plain system matrices — every assembly entry point
fills them. `combine!` is eltype-generic: real components with complex weights
combine into a complex target from `share_pattern(A, ComplexF64)`.

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

## Weighted Jacobians

The fold itself is an entry point, so a scheme's matrix is one call:

```julia
W = share_pattern(op.J)
assemble_weighted_jacobian!(W, op, (u = 1.0, du = 1/(γ*Δt)), states, p, ctx)
```

[`assemble_weighted_jacobian!`](@ref) assembles `W = Σₛ wₛ ∂F/∂s` over the
slots the `weights` NamedTuple names, at frozen values of every other slot.
The weights are *request payload*: a fused element kernel reads them from
[`WeightedJacobianRequest`](@ref) and the composed fallback folds the same
NamedTuple with `combine!`, so the two routes cannot disagree about the
scheme's scalars.

Which route runs is a capability of the element caches, not a caller choice.
An element opts into the **fused** route by declaring the analytic kernel —
this is where a hand-derived scheme matrix belongs, since it computes a
combination that no single-slot Jacobian does:

```julia
FerriteOperators.provides_analytic(::Type{<:MyCache}, ::WeightedJacobianKind) = true
function FerriteOperators.assemble_cell!(req::WeightedJacobianRequest, cache::MyCache, args)
    wu, wdu = req.weights.u, req.weights.du
    # ... accumulate wu·∂F/∂u + wdu·∂F/∂du into req.K ...
end
```

Without it the fused route seeds every participating slot in the residual
kernel with its weight-scaled Duals — one sweep, one seed dimension. The
**composed** route (per-slot sweeps into operator-held components, folded by
`combine!`) runs for complex weights, since the element matrix and the Dual
machinery are real, and wherever condensed internal state makes the AD-seeded
route inadmissible; there each participating [`JacobianKind`](@ref) applies
its own guards, so the weighted kind is servable exactly when every
participating slot Jacobian is.

An [`AffineRate`](@ref) slot may participate only through an analytic weighted
kernel: reconstructed slots are frozen under AD, while a kernel forming the
combination itself sees the slope through its weights. That exemption is what
a multilevel-Newton element with a rate-coupled local problem needs — its
condensed tangent carries the slope inside the local inverse, where post-hoc
weighting of separated partial Jacobians cannot put it.

## Sensitivities

```julia
update_parameter_jacobian!(B, op, states, p, ctx)   # ∂F/∂θ, dense
parameter_vjp!(g, op, λ, states, p, ctx)            # (∂F/∂θ)ᵀλ, matrix-free
state_jvp!(Jv, op, v, states, p, ctx)               # (∂F/∂u)·v, no matrix
state_vjp!(g, op, λ, states, p, ctx)                # (∂F/∂u)ᵀλ, the adjoint action
time_sensitivity!(g, op, states, p, ctx)            # ∂F/∂t at evaluation_time(ctx)
time_sensitivity!(g, op, states, p, ctx; method = FiniteDifferenceSensitivity())
```

θ is the flat view defined by [`parameter_vector`](@ref) /
[`rebuild_parameters`](@ref); entries it does not expose are static and no
sensitivity cost scales with them. Per cache, analytic sensitivity kernels
win; otherwise the resolved cache — automatically wrapped in
[`ADElementCache`](@ref) at `setup_operator` time — differentiates the
residual kernel. The engine itself never forks between the two: it always
calls `assemble_cell!` on the resolved cache, analytic-or-decorated.
Sensitivity sweeps **never** write back into the caller's state.

!!! warning "Boundary terms are not differentiated"
    A sensitivity sweep runs the **volumetric** kernel only. Boundary
    contributions are omitted from `∂F/∂θ`, `∂F/∂t` and the matrix-free state
    products, so these results are correct exactly when the boundary terms are
    independent of the seeded quantity — θ for the parameter kinds, `t` for the
    time sensitivity, `u` for the state products. A θ-dependent traction or a
    time-dependent flux therefore yields a silently incomplete sensitivity.

    An operator declaring a sensitivity kind while carrying a non-empty
    boundary cache warns once at `setup_operator`, and
    [`check_derivatives`](@ref) detects the dependent case: its
    finite-difference referee evaluates the full residual *including* boundary
    terms, so a failing parameter, time, or state-product check on such an
    operator is the signature of this omission.

∂F/∂t seeds through the context channel: the AD sweep hands the kernel a
context whose evaluation time is Dual-valued, and the finite-difference method
evaluates the primal residual at contexts with perturbed times. An element
therefore reads time as `evaluation_time(args.ctx)`, and `time_sensitivity!`
requires a context — passing `nothing` is an `ArgumentError`.

Admissibility with internal state: a condensed element's residual kernel is
PURE (it reads the frozen `q` a prior [`condense_internal!`](@ref) wrote), so
a plain AD fallback computes a genuine `∂F/∂·|_q` partial rather than
differentiating through an iteration — but the sensitivity kinds carry no
[`CorrectionMode`](@ref) (they are always the total), so that partial is
missing the `∂F/∂q · dq/d·` correction. A cache with [`has_internal_state`](@ref)
is therefore admissible for a sensitivity kind only if it (a) provides the
analytic kernel (the correction, folded in — the payoff a corrector store
unlocks: exact parameter/state sensitivities on a condensed element,
generically, once the store exists), (b) declares
[`internal_state_insensitive`](@ref) (asserting the local equations do not
depend on the seeded quantity, so there is nothing to correct — then AD is
exact), or (c) for time sensitivities, the caller selects
[`FiniteDifferenceSensitivity`](@ref) (primal evaluations on a protected
copy, condensing at each — the total, but it bypasses analytic sensitivity
kernels).

State and time derivative sweeps (`update_linearization!` via AD,
`state_jvp!`, `state_vjp!`, `time_sensitivity!`) run over per-worker
preallocated buffers and ForwardDiff configurations and are allocation-free
per cell. The parameter sweeps preallocate their output buffers and build
their ForwardDiff configurations per call, since the seed dimension nθ arrives
with `p`.

## Verifying derivative implementations

```julia
res = check_derivatives(op, states, p, ctx)
res.passed                      # conjunction of all non-skipped checks
res.checks.jacobian.err         # per-check relative error / skip reason
```

[`check_derivatives`](@ref) cross-checks every derivative path — the
assembled Jacobian, fused-vs-split residual, parameter Jacobian/VJP, state
JVP/VJP, time sensitivity — against central finite differences of the
operator's own residual, through the public entry points. A wrong analytic
kernel fails its check against the FD referee; inadmissible or unsupported
checks are skipped with the reason recorded. The parameter checks respect the
differentiable/static split: only the entries exposed by
[`parameter_vector`](@ref) are probed.

The FD referee evaluates the operator's **full** residual, boundary terms
included, while the sensitivity sweeps it is checking cover the volumetric
kernel only. A parameter, time, or state-product check that fails on an
operator with boundary terms is therefore the diagnostic for a boundary term
that depends on the seeded quantity.

The time check runs only with a context and is recorded as a skip without one.
Passing `weights = (u = …, du = …)` adds the two weighted-Jacobian checks: the
fused `Σₛ wₛ ∂F/∂s` against per-slot finite differences, and
fused-against-composed equality — the caller cannot tell which route an
operator takes, so the two must be interchangeable.

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
variants — have their own constructors
([`setup_transfer_operator`](@ref), [`setup_nested_transfer_operator`](@ref))
and their own integrators ([`MassProlongatorIntegrator`](@ref),
[`NestedMassProlongatorIntegrator`](@ref)). They assemble sequentially on the
CPU into a rectangular sparse matrix.

!!! warning "Experimental surface"
    Transfer operators are scheduled to be folded into the unified assembly
    engine. The constructors and the operator types may change in a minor
    release; the assembled matrix and its sparsity are not affected.
