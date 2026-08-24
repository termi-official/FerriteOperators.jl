# What a single assembly sweep computes. Kinds select which request is
# materialized over the workspace buffers and which kernels run.
"""
    JacobianResidualKind{C <: CorrectionMode}
    JacobianResidualKind()

State-dependent J(u) and r(u), fused (the Newton hot path). `C` is the
[`CorrectionMode`](@ref) of the Jacobian half; `JacobianResidualKind()`
defaults it to `Consistent` — see [`JacobianKind`](@ref).
"""
struct JacobianResidualKind{C <: CorrectionMode} end
JacobianResidualKind() = JacobianResidualKind{Consistent}()
"""
    ResidualKind()

Assembly of the state-dependent residual `r(u)` into the operator's vector.
The kernel it reaches is [`ResidualRequest`](@ref) — the one kernel every
element must implement, and the one every AD fallback differentiates.
"""
struct ResidualKind end

struct BilinearKind end             # u-independent matrix
struct LinearKind end               # u-independent vector

"""
    ParameterJacobianKind()

Assembly of `∂F/∂θ` into a dense `residual_size × nθ` target, θ being the flat
view [`parameter_vector`](@ref) defines. Issued by
[`update_parameter_jacobian!`](@ref); served by an analytic
[`ParameterJacobianRequest`](@ref) kernel or by ForwardDiff over the residual
kernel. The seed dimension arrives with `p`, so the sweep builds its
differentiation configuration per call.
"""
struct ParameterJacobianKind end

"""
    ParameterVJPKind(λ)

The matrix-free pullback `(∂F/∂θ)ᵀλ` — the adjoint gradient, without ever
materializing `∂F/∂θ`. `λ` is the global adjoint vector, gathered per item.
Issued by [`parameter_vjp!`](@ref); the kernel is
[`ParameterVJPRequest`](@ref).
"""
struct ParameterVJPKind{L}; λ::L; end

"""
    TimeSensitivityKind()

Assembly of `∂F/∂t` at [`evaluation_time`](@ref)`(ctx)`. Time reaches elements
through the context alone, so this kind seeds through that channel: the AD
sweep hands the kernel a Dual-timed context, and
[`FiniteDifferenceSensitivity`](@ref) evaluates the primal residual at
perturbed context times instead. A kernel reading its time from `args.p` gets
a silently zero result. Issued by [`time_sensitivity!`](@ref).
"""
struct TimeSensitivityKind end

"""
    StateJVPKind(v)

The matrix-free action `(∂F/∂u)·v` — one directional-Dual sweep per item under
the AD fallback, no matrix anywhere. Issued by [`state_jvp!`](@ref); the
kernel is [`StateJVPRequest`](@ref).
"""
struct StateJVPKind{V}; v::V; end

"""
    StateVJPKind(λ)

The matrix-free pullback `(∂F/∂u)ᵀλ` — the action adjoint time stepping
applies, computed per item as the gradient of `λₑ·rₑ` under the AD fallback.
Issued by [`state_vjp!`](@ref); the kernel is [`StateVJPRequest`](@ref).
"""
struct StateVJPKind{L}; λ::L; end

"""
    JacobianKind{slot, C <: CorrectionMode}
    JacobianKind{slot}()
    JacobianKind()

Assembly of the Jacobian ∂F/∂slot into the operator's matrix. `slot` names the
state slot differentiated against; `JacobianKind()` is
`JacobianKind{:u, Consistent}()`, the Newton path. Every other slot is a
*component* of a multi-slot linearization (`JacobianKind{:du}()` for the DAE
mass block, `JacobianKind{:v}()`/`JacobianKind{:a}()` for structural
dynamics); the chain-rule weights that fold components into a Newton matrix
are the solver's, not the framework's. `slot = :q` is the block a
Schur-complement consumer or the generic corrector combination wants — see
[`condense_internal!`](@ref).

`C` is the [`CorrectionMode`](@ref); `JacobianKind{slot}()` defaults it to
`Consistent`, so `FrozenQ` must be spelled: `JacobianKind{slot, FrozenQ}()`.

The differentiated slot must carry a plain vector source — [`AffineRate`](@ref)
slots are reconstructed at gather time and frozen under AD, so the entry point
rejects them.

Elements serve the kind analytically (`assemble_cell!` on
`JacobianRequest{slot, C}`, declared through [`provides_analytic`](@ref)) or
through ForwardDiff seeding of the named slot buffer, which computes exactly
the `FrozenQ` partial — correct as-is for `C = FrozenQ`, and admissible for
`C = Consistent` only when the element has no condensed internal state to miss
a correction for.
"""
struct JacobianKind{slot, C <: CorrectionMode} end
JacobianKind{slot}() where {slot} = JacobianKind{slot, Consistent}()
JacobianKind() = JacobianKind{:u, Consistent}()

"""
    WeightedJacobianKind(weights::NamedTuple)
    WeightedJacobianKind{slots}(weights)

Assembly of the weighted Jacobian `W = Σₛ wₛ ∂F/∂s` in ONE sweep, over the
slots `weights` names and at frozen values of every other slot — the matrix a
scheme actually solves with (`W = M/(γΔt) + K` for SDIRK/backward Euler,
`M/(βΔt²) + γ/(βΔt) C + K` for Newmark). The slot set is the type parameter
`slots`; the weights are runtime payload and eltype-generic.

Weights are REQUEST payload: the kernel reads them from
[`WeightedJacobianRequest`](@ref) and the composed fallback folds the same
NamedTuple with [`combine!`](@ref), so the two routes cannot disagree about the
scheme's scalars. Solvers issue the kind through
[`assemble_weighted_jacobian!`](@ref), which selects the route.

`C` is the [`CorrectionMode`](@ref) of every participating slot;
`WeightedJacobianKind(weights)` defaults it to `Consistent`, and
`WeightedJacobianKind{slots, FrozenQ}(weights)` spells the partial.

An element opts into the fused route with

    provides_analytic(::Type{<:MyCache}, ::WeightedJacobianKind) = true
    assemble_cell!(req::WeightedJacobianRequest, cache::MyCache, args) = # reads req.weights

Without it the sweep derives `W` from the residual kernel by seeding every
participating slot with its weight-scaled Duals — real weights only, since the
element matrix and the Dual machinery are real. Complex weights (transformed
Radau) go through the composed route.

!!! note "AffineRate participation"
    The AD route freezes [`AffineRate`](@ref) slots at gather time, so a
    reconstructed slot cannot participate and the sweep rejects one — the same
    rule as [`JacobianKind`](@ref). An ANALYTIC weighted kernel is exempt: it
    forms the combination internally, which is what a multilevel-Newton
    element with a rate-coupled local problem needs (its condensed tangent
    `−(∂L/∂q)⁻¹(∂L/∂ε + slope·∂L/∂ε̇)` carries the slope inside the local
    inverse and cannot be recovered by weighting separated partials).
"""
struct WeightedJacobianKind{slots, C <: CorrectionMode, W <: NamedTuple}
    weights::W
end
function WeightedJacobianKind(weights::NamedTuple{slots}) where {slots}
    isempty(slots) && throw(ArgumentError(
        "A `WeightedJacobianKind` needs at least one weighted slot, got empty weights."))
    return WeightedJacobianKind{slots, Consistent, typeof(weights)}(weights)
end
WeightedJacobianKind{slots}(weights::NamedTuple{slots}) where {slots} = WeightedJacobianKind(weights)
function WeightedJacobianKind{slots, C}(weights::NamedTuple{slots}) where {slots, C <: CorrectionMode}
    isempty(slots) && throw(ArgumentError(
        "A `WeightedJacobianKind` needs at least one weighted slot, got empty weights."))
    return WeightedJacobianKind{slots, C, typeof(weights)}(weights)
end

"""
    FunctionalKind{tag}
    FunctionalKind(tag::Symbol)

Names a functional (reduction) query: a global scalar or Tensors tensor
integrated from per-cell contributions (energy, dissipation, …). Elements
implement [`evaluate_cell_functional`](@ref) dispatching on the tag; solvers
evaluate through [`evaluate_functional`](@ref).
"""
struct FunctionalKind{tag} end
FunctionalKind(tag::Symbol) = FunctionalKind{tag}()

"""
    functional_value_type(kind) -> Type

The type a value-returning sweep of `kind` reduces to, or `Nothing` when the
kind does not declare one. Queried on the INSTANCE, because every consumer is
a running sweep holding the kind it was issued with.

Declaring it types the worker's fold from the start: the accumulator is seeded
with `zero(T)` instead of with the first contributing item, so the partials are
a concretely typed array and a kernel returning something else fails loudly
instead of widening the reduction. REQUIRED under a parallel device, whose
per-worker partials are allocated before the batch runs; the sequential fold
can still infer the type from the first contribution.

    FerriteOperators.functional_value_type(::FunctionalKind{:energy}) = Float64
    FerriteOperators.functional_value_type(::FunctionalKind{:gradient_volume}) = Vec{2, Float64}

There is exactly one root method, so an overload is always strictly more
specific and cannot create an ambiguity. Return a literal — the fold's
accumulator type folds out of it.
"""
functional_value_type(kind) = Nothing

const MatrixAssemblyKind  = Union{JacobianResidualKind, JacobianKind, WeightedJacobianKind, BilinearKind}
const VectorAssemblyKind  = Union{JacobianResidualKind, ResidualKind, LinearKind}
const UnknownDependentKind = Union{JacobianResidualKind, JacobianKind, WeightedJacobianKind, ResidualKind}
const PrimalKind = Union{JacobianResidualKind, JacobianKind, WeightedJacobianKind, ResidualKind, BilinearKind, LinearKind}
const SensitivityKind = Union{ParameterJacobianKind, ParameterVJPKind, TimeSensitivityKind, StateJVPKind, StateVJPKind}

"""
    assembles_matrix(kind) -> Bool
    assembles_vector(kind) -> Bool
    depends_on_unknowns(kind) -> Bool

What a sweep of `kind` does with the workspace: which of `ws.Ke`/`ws.re` it
zeroes and scatters, and whether it gathers the state slots. The driver bodies
consult these instead of testing kind membership, so a downstream kind joins
the built-in driver by declaring them.

The defaults read the built-in family unions, so the answers are compile-time
constants and the branches they guard are eliminated. An overload must return
a literal for the same reason:

    FerriteOperators.assembles_matrix(::MyKind) = true

There is exactly one root method each, so an overload is always strictly more
specific and cannot create an ambiguity.
"""
assembles_matrix(kind) = kind isa MatrixAssemblyKind
@doc (@doc assembles_matrix) assembles_vector(kind) = kind isa VectorAssemblyKind
@doc (@doc assembles_matrix) depends_on_unknowns(kind) = kind isa UnknownDependentKind

"""
    NoFamily
    FunctionalFamily

The per-worker driver-shape routing. `NoFamily` marks a sweep that scatters
its result through an assembler (every request-carrying kind); `FunctionalFamily`
selects the VALUE-RETURNING driver instead: its kernel returns the item's
contribution and the sweep reduces the returned values, so it needs no
per-worker state at all.

A sensitivity kind's OUTPUT buffers ([`SensitivityBuffers`](@ref)) are
engine/workspace-owned unconditionally for a nonlinear operator (see
[`needs_ad_decoration`](@ref)) rather than gated by a third family — the
ForwardDiff seeding machinery lives on the resolved element cache
([`ADElementCache`](@ref)) instead of a family-gated workspace member.
"""
struct NoFamily end
@doc (@doc NoFamily) struct FunctionalFamily end

"""
    sweep_family(::Type{K}) -> family singleton

The per-worker family a sweep of kind `K` reads. Queried on the TYPE, because
declarations carry kind types (normalized to their `UnionAll` base) while
sweeps carry instances — one overload point serves both. `FunctionalFamily()`
routes the kind to the value-returning driver ([`run_reduction`](@ref)); every
other kind is `NoFamily()`. The default derives the answer from the built-in
kind and folds to a constant.
"""
sweep_family(::Type{K}) where {K} = K <: FunctionalKind ? FunctionalFamily() : NoFamily()

"""
    request_type(kind) -> Type
    materialize_request(kind, ws) -> AbstractAssemblyRequest

The single kind → request association. `request_type` is the pure form used
wherever a kernel method is looked up (the setup-time trait ↔ kernel checks);
`materialize_request` is the executing form, binding the workspace buffers a
sweep of `kind` accumulates into. Drivers — cell, facet, patch — and the
validation tables all go through these two, so a new kind enters the framework
by adding one pair of methods here.

Buffers are zeroed by the driver before the request is materialized, not here.
"""
function request_type end

request_type(::ResidualKind)                     = ResidualRequest
request_type(::LinearKind)                       = ResidualRequest
request_type(::JacobianKind{slot, C}) where {slot, C}  = JacobianRequest{slot, C}
request_type(::BilinearKind)                     = JacobianRequest{:u}
request_type(::JacobianResidualKind{C}) where {C} = JacobianResidualRequest{C}
request_type(::WeightedJacobianKind{slots, C}) where {slots, C} = WeightedJacobianRequest{C}
request_type(::ParameterJacobianKind)            = ParameterJacobianRequest
request_type(::ParameterVJPKind)                 = ParameterVJPRequest
request_type(::TimeSensitivityKind)              = TimeSensitivityRequest
request_type(::StateJVPKind)                     = StateJVPRequest
request_type(::StateVJPKind)                     = StateVJPRequest

# Placeholder instances for the payload-carrying kinds: setup-time validation
# reads only their types (see [`validation_instance`](@ref)).
validation_instance(::Type{<:ParameterVJPKind})     = ParameterVJPKind(nothing)
validation_instance(::Type{<:StateJVPKind})         = StateJVPKind(nothing)
validation_instance(::Type{<:StateVJPKind})         = StateVJPKind(nothing)
validation_instance(::Type{<:WeightedJacobianKind}) = WeightedJacobianKind((u = 1.0,))

# The kinds whose AD fallback would silently miss a condensed cache's
# ∂F/∂q·dq/d· correction — always for the sensitivity kinds, which carry no
# `CorrectionMode`. `JacobianKind`/`JacobianResidualKind` are mode-aware: a
# `Consistent` AD fallback needs the check, a `FrozenQ` one IS the requested
# partial and never does.
requires_admissibility_check(::Union{ParameterJacobianKind, ParameterVJPKind, StateJVPKind, StateVJPKind}) = true
requires_admissibility_check(::JacobianKind{slot, Consistent}) where {slot} = true
requires_admissibility_check(::JacobianKind{slot, FrozenQ}) where {slot} = false
requires_admissibility_check(::JacobianResidualKind{Consistent}) = true
requires_admissibility_check(::JacobianResidualKind{FrozenQ}) = false

# Functional kernels return their contribution through `evaluate_cell_functional`
# rather than filling a request, so there is no cell request to validate.
has_cell_request(::Type{<:FunctionalKind}) = false

materialize_request(::ResidualKind, ws)                    = ResidualRequest(ws.re)
materialize_request(::LinearKind, ws)                      = ResidualRequest(ws.re)
materialize_request(::JacobianKind{slot, C}, ws) where {slot, C} = JacobianRequest{slot, C}(ws.Ke)
materialize_request(::BilinearKind, ws)                    = JacobianRequest{:u}(ws.Ke)
materialize_request(::JacobianResidualKind{C}, ws) where {C} = JacobianResidualRequest{C}(ws.Ke, ws.re)
materialize_request(kind::WeightedJacobianKind{slots, C}, ws) where {slots, C} = WeightedJacobianRequest{C}(ws.Ke, kind.weights)

# The five sensitivity kinds take the 3-arg form: their destination buffers
# live on `ws.sensitivity`, and the parameter kinds size themselves from
# `task.p`. The destination is zeroed unconditionally — cheap, and correct
# whether the resolved cache accumulates (analytic) or overwrites (AD), so the
# engine needs no fork between the two.
function materialize_request(::ParameterJacobianKind, ws, task)
    s = parameter_sweep_buffers!(ws.sensitivity, length(parameter_vector(task.p)))
    fill!(s.Bₑ, zero(eltype(s.Bₑ)))
    return ParameterJacobianRequest(s.Bₑ, task.p)
end
function materialize_request(kind::ParameterVJPKind, ws, task)
    s = ws.sensitivity
    λₑ = _gather_residual_dofs!(s.λₑ, kind.λ, item_dofs(ws))
    parameter_sweep_buffers!(s, length(parameter_vector(task.p)))
    fill!(s.gθ, zero(eltype(s.gθ)))
    return ParameterVJPRequest(s.gθ, λₑ, task.p)
end
function materialize_request(::TimeSensitivityKind, ws, task)
    fill!(ws.sensitivity.gₜ, zero(eltype(ws.sensitivity.gₜ)))
    return TimeSensitivityRequest(ws.sensitivity.gₜ)
end
function materialize_request(kind::StateJVPKind, ws, task)
    s = ws.sensitivity
    s.vₑ .= @view kind.v[item_dofs(ws)]
    fill!(s.Jvₑ, zero(eltype(s.Jvₑ)))
    return StateJVPRequest(s.Jvₑ, s.vₑ)
end
function materialize_request(kind::StateVJPKind, ws, task)
    s = ws.sensitivity
    λₑ = _gather_residual_dofs!(s.λₑ, kind.λ, item_dofs(ws))
    fill!(s.gu, zero(eltype(s.gu)))
    return StateVJPRequest(s.gu, λₑ)
end

"""
    scatter_request!(req, assembler, address)

Hand a sensitivity request's payload to the assembler — the sensitivity
counterpart of [`scatter_local!`](@ref), dispatching on the request type
instead of the kind since the destination buffer IS the request's own field.
`address` is what [`scatter_address`](@ref) resolved for the item.
"""
scatter_request!(req::ParameterJacobianRequest, assembler, address) = assemble!(assembler, address, req.B)
scatter_request!(req::ParameterVJPRequest, assembler, address)      = assemble!(assembler, address, req.g)
scatter_request!(req::TimeSensitivityRequest, assembler, address)   = assemble!(assembler, address, req.g)
scatter_request!(req::StateJVPRequest, assembler, address)          = assemble!(assembler, address, req.Jv)
scatter_request!(req::StateVJPRequest, assembler, address)          = assemble!(assembler, address, req.g)

"""
    AssemblyTask(kind, inner_assembler, states, p, ctx)

The single per-cell assembly sweep shared by all operators. `kind` selects
what is computed (and thereby which element kernel is called), `states` is
the NamedTuple of global slot sources (empty for state-independent kinds),
`p` the user parameters, `ctx` the time-integration context (or `nothing`).
"""
@concrete struct AssemblyTask
    kind
    inner_assembler
    states
    p
    ctx
end
duplicate_for_device(device, task::AssemblyTask) =
    AssemblyTask(task.kind, duplicate_for_device(device, task.inner_assembler), task.states, task.p, task.ctx)

execute_single_task!(task::AssemblyTask, ws::AssemblyWorkspace) = execute_kind!(task.kind, task, ws)

# Loud once-per-sweep check instead of a raw NamedTuple field error per cell.
function _check_declared_slots(engine, states::NamedTuple{names}) where {names}
    slots = get_declared_slots(engine.protocol)
    issubset(names, slots) || throw(ArgumentError(
        "States pass slots $names but the operator declared slots $(slots). " *
        "Declare every slot at setup: `setup_operator(...; slots = $(Tuple(union(slots, names))))`."))
    return nothing
end

# Rate reconstruction reads the gathered `:u` buffer, so a `:u` slot must
# exist and be gathered before any AffineRate slot.
function _check_rate_slots(states::NamedTuple{names}) where {names}
    iu = findfirst(==(:u), names)
    for (i, name) in enumerate(names)
        states[i] isa AffineRate || continue
        (iu === nothing || iu >= i) && throw(ArgumentError(
            "Slot `$name` is an `AffineRate` source, whose value is `slope·(u − anchor)`. " *
            "The states NamedTuple must carry a plain `:u` slot BEFORE it, got $names."))
    end
    return nothing
end

"""
    item_dofs(ws) -> AbstractVector{Int}

The global dof indices of the current item's local system: `celldofs(cell)`
where the integrator declares no [`global_dofs`](@ref), and the augmented
`[celldofs(cell); global dofs]` vector the workspace carries otherwise. Every
gather of a sweep addresses through this, so the augmented tail reaches the
slot buffers and the adjoint payloads. On an [`AlgebraicWorkspace`](@ref) it is
the item's own dof vector, that family having no cell dofs to start from.
"""
@inline item_dofs(ws) = _item_dofs(ws.dofs, ws.cell)
@inline _item_dofs(::Nothing, cell) = celldofs(cell)
@inline _item_dofs(dofs, cell) = dofs

"""
    scatter_address(ws)

What a scatter of the current item addresses. Without declared
[`global_dofs`](@ref) it is the geometry cache, which every assembler in the
package takes — the element-indexed [`ElementAssembly`](@ref) one reads
`cellid` from it, the dof-scattered ones read `celldofs`. With them the local
system spans dofs no cell owns, so the augmented dof vector is the only address
that describes it, and `ElementAssembly` is rejected at setup for that reason.
An algebraic item is addressed by its dof vector, there being no cell to name
it by.
"""
@inline scatter_address(ws) = _scatter_address(ws.dofs, ws.cell)
@inline _scatter_address(::Nothing, cell) = cell
@inline _scatter_address(dofs, cell) = dofs

# Gather every task slot into the workspace's slot buffers, returning the
# element-local states NamedTuple. A slot's SOURCE decides how it gathers: a
# plain vector reads `item_dofs`, `AffineRate` reconstructs over that same
# gather, and `InternalSource` restricts to the cell's condensed internal-dof
# range, resizing the buffer to fit. A reconstructed slot's source is therefore
# structurally the field space and cannot reach `q`.
function load_slots!(ws, states::NamedTuple{names}) where {names}
    return map(NamedTuple{names}(ws.slot_buffers), states) do buf, src
        load_slot!(buf, src, ws)
        buf
    end
end
function load_slot!(buf, src::AbstractVector, ws)
    dofs = item_dofs(ws)
    resize!(buf, length(dofs))
    buf .= @view src[dofs]
    return buf
end
# The anchor lands in the slot's own buffer first, then the buffer becomes the
# reconstruction against the already-gathered `:u` buffer.
function load_slot!(buf, src::AffineRate, ws)
    dofs = item_dofs(ws)
    resize!(buf, length(dofs))
    buf .= @view src.anchor[dofs]
    buf .= src.slope .* (ws.slot_buffers.u .- buf)
    return buf
end
function load_slot!(buf, src::InternalSource, ws)
    range = internal_variable_range(ws.ivh, cellid(ws.cell))
    resize!(buf, length(range))
    buf .= @view src.u[range]
    return buf
end

execute_kind!(kind::PrimalKind, task, ws) = primal_cell_sweep!(kind, task, ws)

"""
    primal_cell_sweep!(kind, task, ws)

The built-in primal driver body, reusable by a downstream kind's own
`execute_kind!`. Zeroes the buffers [`assembles_matrix`](@ref) /
[`assembles_vector`](@ref) name, reinitializes the element's values, queries
the cell parameters, runs the cell and facet kernels — gathering the state
slots iff [`depends_on_unknowns`](@ref) — and scatters through
[`scatter_local!`](@ref).

The kernel it calls is `cell_kernel!(kind, …)`, whose generic method issues the
kind's request analytically; the built-in kinds with an AD fallback specialize
it. It writes nothing back: [`condense_internal!`](@ref) is the only writer of
`q`, so a primal sweep is a pure evaluation at whatever `q` is stored.
"""
function primal_cell_sweep!(kind, task, ws)
    assembles_matrix(kind) && fill!(ws.Ke, zero(eltype(ws.Ke)))
    assembles_vector(kind) && fill!(ws.re, zero(eltype(ws.re)))
    reinit_values!(ws.element, ws.cell, kind)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    if depends_on_unknowns(kind)
        statesₑ = load_slots!(ws, task.states)
        @timeit_debug "assemble element" cell_kernel!(kind, ws.element, ws, statesₑ, pₑ, task.ctx)
        @timeit_debug "assemble boundary" boundary_kernel!(kind, ws.boundary_element, ws, statesₑ, task)
    else
        @timeit_debug "assemble element" cell_kernel!(kind, ws.element, ws, (;), pₑ, task.ctx)
        @timeit_debug "assemble boundary" boundary_kernel!(kind, ws.boundary_element, ws, (;), task)
    end
    scatter_local!(kind, task.inner_assembler, ws)
end

# The single CellArgs/FacetArgs construction seams.
_cell_args(ws, statesₑ, pₑ, ctx) = CellArgs(statesₑ, ws.cell, pₑ, ctx)
_facet_args(ws, statesₑ, pₑ, ctx) = FacetArgs(statesₑ, ws.cell, pₑ, ctx)

# The framework-owned facet driver: walk the cell's facets, gate on
# is_facet_in_cache, query facet parameters SEPARATELY per facet, and hand the
# kind's request over the shared local buffers to the facet kernel.
boundary_kernel!(kind, ::EmptySurfaceElementCache, ws, statesₑ, task) = nothing
function boundary_kernel!(kind, cache::AbstractSurfaceElementCache, ws, statesₑ, task)
    for lfi in 1:nfacets(ws.cell)
        if is_facet_in_cache(FacetIndex(cellid(ws.cell), lfi), ws.cell, cache)
            pᵦ = query_facet_parameters(cache, ws.cell, lfi, task.p)
            facet_kernel!(kind, cache, ws, _facet_args(ws, statesₑ, pᵦ, task.ctx), lfi)
        end
    end
end

"""
    facet_kernel!(kind, cache, ws, args, lfi)

One facet's contribution to a sweep of `kind`, over the workspace buffers.
Facet contributions have no AD fallback in any sweep — a surface cache serves
the sweep's request analytically or not at all — so the generic method simply
issues the kind's request.
"""
facet_kernel!(kind, cache, ws, args, lfi::Int) =
    assemble_facet!(materialize_request(kind, ws), cache, args, lfi)

# A weighted sweep takes the cache's FUSED weighted kernel where it declares
# one, and otherwise composes `Σₛ wₛ·(∂F/∂s facet kernel)` from the per-slot
# Jacobians the cache DOES declare. A slot it does not claim contributes
# nothing — the statement a spring makes about ∂F/∂v under `(u, v)` weights.
# Analytic wins: a cache claiming the weighted kind keeps its fused kernel,
# which is the only route that can carry a combination no single-slot Jacobian
# computes.
function facet_kernel!(kind::WeightedJacobianKind{slots, C}, cache, ws, args, lfi::Int) where {slots, C}
    T = typeof(cache)
    provides_analytic(T, kind) &&
        return assemble_facet!(materialize_request(kind, ws), cache, args, lfi)
    any(_claimed_facet_slots(T, kind)) || _throw_no_weighted_facet_route(T, kind)
    return _fold_weighted_facet!(C, slots, values(kind.weights), cache, ws, args, lfi)
end

_claimed_facet_slots(::Type{T}, ::WeightedJacobianKind{slots, C}) where {T, slots, C} =
    ntuple(i -> provides_analytic(T, JacobianKind{slots[i], C}()), Val(length(slots)))

# Unrolled by tuple recursion: an `ntuple`/`map` closure over the workspace and
# the args record is materialized once per facet, which the alloc gate forbids.
@inline _fold_weighted_facet!(::Type{C}, ::Tuple{}, ::Tuple{}, cache, ws, args, lfi) where {C} = nothing
@inline function _fold_weighted_facet!(::Type{C}, slots::Tuple, weights::Tuple, cache, ws, args, lfi) where {C}
    _add_weighted_facet_slot!(ws, JacobianKind{first(slots), C}(), first(weights), cache, args, lfi)
    return _fold_weighted_facet!(C, Base.tail(slots), Base.tail(weights), cache, ws, args, lfi)
end

# The slot lands in the per-worker scratch first: a facet kernel ACCUMULATES
# into the request's matrix, so its contribution has to be separable before the
# weight can be applied to it.
@inline function _add_weighted_facet_slot!(ws, ::JacobianKind{slot, C}, w, cache, args, lfi) where {slot, C}
    provides_analytic(typeof(cache), JacobianKind{slot, C}()) || return nothing
    scratch = ws.facet_Ke
    fill!(scratch, zero(eltype(scratch)))
    assemble_facet!(JacobianRequest{slot, C}(scratch), cache, args, lfi)
    ws.Ke .+= w .* scratch
    return nothing
end

@noinline _throw_no_weighted_facet_route(T::Type, kind::WeightedJacobianKind{slots, C}) where {slots, C} =
    throw(ArgumentError(
        "$(T) declares neither route a weighted Jacobian sweep can take on a facet: no fused " *
        "`assemble_facet!(::WeightedJacobianRequest, …)` kernel (declared through " *
        "`provides_analytic(::Type{<:$(nameof(T))}, ::WeightedJacobianKind)`), and no per-slot " *
        "`assemble_facet!(::JacobianRequest{slot}, …)` kernel for any of $(slots) either. Facet " *
        "kernels have no automatic-differentiation fallback, so declare one of the two — the " *
        "fused kernel for a hand-derived combination, per-slot kernels for terms the sweep's " *
        "weights fold."))

# ONE generic method for every primal kind: `cache` (`ws.element`) is EITHER
# genuinely analytic for `kind` or a decorator ([`ADElementCache`](@ref)) that
# resolves it — the engine never forks on `provides_analytic` itself, and a
# downstream kind riding [`primal_cell_sweep!`](@ref) gets this without
# writing a kernel dispatch of its own.
cell_kernel!(kind, cache, ws, statesₑ, pₑ, ctx) =
    assemble_cell!(materialize_request(kind, ws), cache, _cell_args(ws, statesₑ, pₑ, ctx))

# Sensitivity sweeps: gather the trial state, never write anything back into
# `u`. `materialize_request`/`scatter_request!` bind and scatter the local
# output — `cache` is resolved exactly like a primal sweep's.
execute_kind!(kind::SensitivityKind, task, ws) = sensitivity_cell_sweep!(kind, task, ws)

"""
    sensitivity_cell_sweep!(kind, task, ws)

The built-in sensitivity driver body, reusable by a downstream kind's own
`execute_kind!`. Gathers the trial state, queries the cell parameters, binds
the request through `materialize_request(kind, ws, task)`, issues it against
the resolved element cache, and scatters through [`scatter_request!`](@ref).
Nothing is written back into `u`.
"""
function sensitivity_cell_sweep!(kind, task, ws)
    reinit_values!(ws.element, ws.cell, kind)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    args = _cell_args(ws, statesₑ, pₑ, task.ctx)
    @timeit_debug "assemble sensitivity" sensitivity_kernel!(kind, task, ws, args)
end

"""
    sensitivity_kernel!(kind, task, ws, args)

Bind `kind`'s request over `ws.sensitivity`, issue it against the resolved
element cache, and scatter the result — the entry point a downstream
sensitivity-family kind reaches (via [`sensitivity_cell_sweep!`](@ref)) with
nothing beyond a `materialize_request(::MyKind, ws, task)` / `scatter_request!`
pair.
"""
function sensitivity_kernel!(kind, task, ws, args)
    req = materialize_request(kind, ws, task)
    assemble_cell!(req, ws.element, args)
    scatter_request!(req, task.inner_assembler, scatter_address(ws))
end

# Residual-shaped gather (plain [`item_dofs`](@ref) slice — adjoint vectors
# carry no condensed tail, unlike the slot gathers).
_gather_residual_dofs!(dest, src, dofs) = dest .= @view src[dofs]

execute_kind!(kind::FunctionalKind, task, ws) = functional_cell_sweep(kind, task, ws)

"""
    functional_cell_sweep(kind, task, ws) -> value

The built-in VALUE-RETURNING driver body, reusable by a downstream kind's own
`execute_kind!`. Reinitializes the element's values, gathers the state slots
without writing anything back, and returns what
[`evaluate_cell_functional`](@ref) gives for the cell — `nothing` for a cell
that contributes nothing.

Unlike [`primal_cell_sweep!`](@ref) and [`sensitivity_cell_sweep!`](@ref) it
writes no result into the workspace and scatters nothing; the sweep reduces
what it returns ([`run_reduction`](@ref)).
"""
function functional_cell_sweep(kind, task, ws)
    reinit_values!(ws.element, ws.cell, kind)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    return evaluate_cell_functional(kind, ws.element, _cell_args(ws, statesₑ, pₑ, task.ctx))
end


"""
    scatter_local!(kind, assembler, ws)

Hand the local buffers a sweep of `kind` filled to the assembler. Which
buffers those are is [`assembles_matrix`](@ref)/[`assembles_vector`](@ref), so
the three routes are selected at compile time and a downstream kind is
scattered by the same body. The item is addressed through
[`scatter_address`](@ref).
"""
function scatter_local!(kind, assembler, ws)
    address = scatter_address(ws)
    if assembles_matrix(kind) && assembles_vector(kind)
        assemble!(assembler, address, ws.Ke, ws.re)
    elseif assembles_matrix(kind)
        assemble!(assembler, address, ws.Ke)
    elseif assembles_vector(kind)
        assemble!(assembler, address, ws.re)
    end
    return nothing
end

# A Jacobian sweep differentiates the buffer of ONE slot, so that slot must be
# present and must carry a plain vector source. `AffineRate` slots are formed
# at gather time and stay frozen under AD; the chain rule through the
# reconstruction is the solver's per-slot weight, not an assemblable quantity.
_check_differentiated_slot(kind, engine, states) = nothing
function _check_differentiated_slot(kind::JacobianKind{slot}, engine, states::NamedTuple{names}) where {slot, names}
    slot in names || throw(ArgumentError(
        "A `JacobianKind{:$slot}` sweep differentiates the `:$slot` slot, which the states " *
        "$names do not carry."))
    states[slot] isa AffineRate && throw(ArgumentError(
        "Slot `:$slot` carries an `AffineRate` source, which is reconstructed at gather time " *
        "and frozen under AD — ∂F/∂:$slot cannot be assembled against it. Assemble the " *
        "components against plain vector sources and combine them with the reconstruction " *
        "slope solver-side."))
    # `provides_analytic` is declared against the `JacobianKind` UnionAll, so a
    # cache that only implements the `:u` kernel claims every slot. Catch the
    # missing kernel here instead of as a per-cell MethodError; `:u` itself is
    # already covered by the setup-time validation.
    if slot !== :u
        for sc in engine.subdomain_caches
            _assert_domain_trait_backed(sc.domain, kind)
        end
    end
    return nothing
end

# A weighted sweep differentiates SEVERAL slots at once, so every participating
# slot must be present. Under the AD route the same `AffineRate` rejection as
# for `JacobianKind` applies; an analytic weighted kernel is exempt because it
# forms the combination itself. The exemption is engine-wide: one cache falling
# back to AD makes the whole sweep AD-seeded.
function _check_differentiated_slot(kind::WeightedJacobianKind{slots}, engine, states::NamedTuple{names}) where {slots, names}
    all(w -> w isa Real, values(kind.weights)) || throw(ArgumentError(
        "A weighted Jacobian sweep accumulates into the operator's real element matrix and " *
        "seeds real ForwardDiff Duals, so its weights must be real, got $(kind.weights). " *
        "Assemble the per-slot components and fold them with a complex `combine!` instead — " *
        "`assemble_weighted_jacobian!` routes complex weights there automatically."))
    fused_analytic = all(sc -> provides_analytic(typeof(sc.domain.element), kind), engine.subdomain_caches)
    for slot in slots
        slot in names || throw(ArgumentError(
            "A `WeightedJacobianKind{$slots}` sweep weights ∂F/∂:$slot, which the states " *
            "$names do not carry."))
        (!fused_analytic && states[slot] isa AffineRate) && throw(ArgumentError(
            "Slot `:$slot` carries an `AffineRate` source, which is reconstructed at gather " *
            "time and frozen under AD, so it cannot participate in an AD-seeded weighted " *
            "sweep. Provide an analytic `WeightedJacobianRequest` kernel (which is exempt, " *
            "since it forms the combination itself), or pass plain vector sources."))
    end
    for sc in engine.subdomain_caches
        _assert_domain_trait_backed(sc.domain, kind)
    end
    return nothing
end

# The one assembly driver shared by every operator entry point. Entry points
# with custom scatter targets (parameter space, quadrature storage) pass their
# own pre-built assembler to `run_sweep!`; everything dof-scattered goes
# through `assemble_into!`, which builds the assembler from the global targets.
function run_sweep!(kind, assembler, op, states::NamedTuple, p, ctx)
    _check_declared_slots(op.engine, states)
    _check_rate_slots(states)
    _check_differentiated_slot(kind, op.engine, states)
    task = AssemblyTask(kind, assembler, states, p, ctx)
    execute_on_subdomains!(task, op.engine)
    finalize_assembly!(assembler)
end
assemble_into!(kind, out::Tuple, op, states::NamedTuple, p, ctx) =
    run_sweep!(kind, start_assemble(op.engine.strategy, out...), op, states, p, ctx)

"""
    run_reduction(kind, op, states::NamedTuple, p, ctx) -> value

The value-returning counterpart of `run_sweep!`: the same per-sweep validation,
but the sweep hands its result back as a value instead of scattering it, and no
workspace state is read or written. `nothing` means no item contributed.

Which shape a kind runs in is [`sweep_family`](@ref): only a `FunctionalFamily`
kind is value-returning, so this is the entry point a downstream scalar or
tensor kind reaches once it declares that family and gives `execute_kind!` a
body returning the item's contribution ([`functional_cell_sweep`](@ref) is the
built-in one).
"""
run_reduction(kind, op, states::NamedTuple, p, ctx) =
    run_reduction(sweep_family(typeof(kind)), kind, op, states, p, ctx)

function run_reduction(::FunctionalFamily, kind, op, states::NamedTuple, p, ctx)
    _check_declared_slots(op.engine, states)
    _check_rate_slots(states)
    _check_reduction_domain(kind, op.engine)
    return reduce_on_subdomains(AssemblyTask(kind, nothing, states, p, ctx), op.engine)
end

# A reduction has two STRUCTURAL preconditions, both decidable before any item
# runs: there have to be items to reduce over, and some subdomain has to be
# able to contribute at all. A sweep whose kernels all answer `nothing` over a
# non-empty, non-empty-cached domain is a legitimate empty sum, not a
# misconfiguration. Both checks are O(subdomains): partition lengths and the
# cache TYPE, never cell data.
function _check_reduction_domain(kind, engine)
    caches = engine.subdomain_caches
    sum(sc -> sum(length, sc.partition; init = 0), caches; init = 0) == 0 && throw(ArgumentError(
        "$(nameof(typeof(kind))) reduces over an empty item set: the operator's " *
        "$(length(caches)) subdomain partition(s) carry no items between them, so there is " *
        "nothing to integrate over."))
    all(sc -> !_may_contribute(sc.domain, kind), caches) && throw(ArgumentError(
        "No subdomain can contribute to $(nameof(typeof(kind))): every subdomain either " *
        "carries an `EmptyVolumetricElementCache`, which returns no contribution by " *
        "construction, or belongs to an item family whose traversal has no body for this kind " *
        "(a facet item contributes to no functional). Set the operator up with an integrator " *
        "whose caches implement `evaluate_cell_functional`."))
    return nothing
end

run_reduction(family, kind, op, states::NamedTuple, p, ctx) = throw(ArgumentError(
    "$(nameof(typeof(kind))) declares $(nameof(typeof(family))), whose sweeps fill the " *
    "workspace buffers and scatter through an assembler — there is no value to return. " *
    "A value-returning kind declares " *
    "`sweep_family(::Type{<:$(nameof(typeof(kind)))}) = FunctionalFamily()`."))

"""
    evaluate_functional(op, kind::FunctionalKind, states, p, ctx = nothing)
    evaluate_functional(op, kind::FunctionalKind, u::AbstractVector, p)

Evaluate the functional named by `kind` over the operator's domain: the sum of
the per-cell contributions returned by [`evaluate_cell_functional`](@ref) (a
`Number` or a Tensors tensor). Contributions fold per worker in item order and
the per-worker partials reduce in a fixed order — results are deterministic for
a fixed worker count. Nothing is written into the operator, so an operator
declaring no functional kind serves one just as well.

Cell items contribute through [`evaluate_cell_functional`](@ref) and algebraic
items through [`evaluate_algebraic_functional`](@ref); the facet item family
contributes nothing, a surface functional being a hook this package does not
have.

Two failure modes are kept apart. STRUCTURAL emptiness — no items in the
operator's partitions, or no subdomain whose element cache can contribute — is
a misconfiguration, raised as an `ArgumentError` before any cell runs. A sweep
that runs and whose kernels all answer `nothing` is the DATA-DEPENDENT case: an
empty sum, which a kind declaring [`functional_value_type`](@ref) answers with
`zero(T)` and an undeclared kind cannot answer at all (there is no `T` to take
the identity of) and reports as an `ArgumentError`.
"""
function evaluate_functional(op, kind::FunctionalKind, states::NamedTuple, p, ctx = nothing)
    total = run_reduction(kind, op, states, p, ctx)
    total === nothing && throw(ArgumentError(
        "No element contributed to $(typeof(kind)). Implement " *
        "`evaluate_cell_functional` for the operator's element caches."))
    return total
end
evaluate_functional(op, kind::FunctionalKind, u::AbstractVector, p) =
    evaluate_functional(op, kind, (u = u,), p, nothing)
