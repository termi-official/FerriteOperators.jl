# What a single assembly sweep computes. Kinds select which request is
# materialized over the workspace buffers and which kernels run.
struct JacobianResidualKind end     # state-dependent J(u) and r(u), fused
struct ResidualKind end             # state-dependent r(u)
struct BilinearKind end             # u-independent matrix
struct LinearKind end               # u-independent vector
struct ParameterJacobianKind end                # ∂F/∂θ, materialized
struct ParameterVJPKind{L}; λ::L; end          # (∂F/∂θ)ᵀλ, adjoint pullback
struct TimeSensitivityKind end                 # ∂F/∂t, seeded through the sweep's ctx
struct StateJVPKind{V}; v::V; end              # (∂F/∂u)·v, matrix-free J action
struct StateVJPKind{L}; λ::L; end              # (∂F/∂u)ᵀλ, matrix-free Jᵀ action

"""
    JacobianKind{slot}
    JacobianKind()

Assembly of the Jacobian ∂F/∂slot, materialized into the operator's matrix.
`slot` names the state slot differentiated against; `JacobianKind()` is
`JacobianKind{:u}()`, the Newton path. Every other slot is a *component* of a
multi-slot linearization (`JacobianKind{:du}()` for the DAE mass block,
`JacobianKind{:v}()`/`JacobianKind{:a}()` for structural dynamics); the
chain-rule weights that fold components into a Newton matrix are the solver's,
not the framework's.

The differentiated slot must carry a plain vector source: [`AffineRate`](@ref)
slots are reconstructed at gather time and frozen under AD, so a Jacobian
w.r.t. them is not what the sweep computes and the entry point rejects it.

Elements serve the kind either analytically — `assemble_cell!` on
`JacobianRequest{slot}`, declared through [`provides_analytic`](@ref) — or
through ForwardDiff seeding of the named slot buffer.
"""
struct JacobianKind{slot} end
JacobianKind() = JacobianKind{:u}()

"""
    WeightedJacobianKind(weights::NamedTuple)
    WeightedJacobianKind{slots}(weights)

Assembly of the weighted Jacobian `W = Σₛ wₛ ∂F/∂s` in ONE sweep, over the
slots `weights` names and at frozen values of every other slot — the matrix a
scheme actually solves with (`W = M/(γΔt) + K` for SDIRK/backward Euler,
`M/(βΔt²) + γ/(βΔt) C + K` for Newmark). The slot set is the type parameter
`slots`; the weights themselves are runtime payload and eltype-generic.

Weights are REQUEST payload: the kernel reads them from
[`WeightedJacobianRequest`](@ref), and the composed fallback folds the very
same NamedTuple with [`combine!`](@ref), so the two routes cannot disagree
about the scheme's scalars. Solvers issue the kind through
[`assemble_weighted_jacobian!`](@ref), which selects the route.

An element opts into the fused route with

    provides_analytic(::Type{<:MyCache}, ::WeightedJacobianKind) = true
    assemble_cell!(req::WeightedJacobianRequest, cache::MyCache, args) = # reads req.weights

Without it the sweep derives `W` from the residual kernel by seeding every
participating slot with its weight-scaled Duals — real weights only, since the
element matrix and the Dual machinery are real. Complex weights (transformed
Radau) go through the composed route.

!!! note "AffineRate participation"
    The AD route freezes [`AffineRate`](@ref) slots at gather time, so a
    reconstructed slot cannot participate in it and the sweep rejects one — the
    same rule as [`JacobianKind`](@ref). An ANALYTIC weighted kernel is exempt:
    it forms the combination internally, which is what a multilevel-Newton
    element with a rate-coupled local problem needs (its condensed tangent
    `−(∂L/∂q)⁻¹(∂L/∂ε + slope·∂L/∂ε̇)` carries the slope inside the local
    inverse and cannot be recovered by weighting separated partials).
"""
struct WeightedJacobianKind{slots, W <: NamedTuple}
    weights::W
end
function WeightedJacobianKind(weights::NamedTuple{slots}) where {slots}
    isempty(slots) && throw(ArgumentError(
        "A `WeightedJacobianKind` needs at least one weighted slot, got empty weights."))
    return WeightedJacobianKind{slots, typeof(weights)}(weights)
end
WeightedJacobianKind{slots}(weights::NamedTuple{slots}) where {slots} = WeightedJacobianKind(weights)

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

Declaring the type is what makes a worker's fold typed from the start: the
accumulator is seeded with `zero(T)` instead of with the first item that
contributes, so the partials are a concretely typed array and a kernel
returning something other than `T` fails loudly instead of widening the
reduction. It is REQUIRED under a parallel device, whose per-worker partials
are allocated before the batch runs; the sequential fold can still infer the
type from the first contribution.

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
    DerivativeSweepKind

The kinds whose sweeps can seed ForwardDiff Duals through the residual kernel,
i.e. the ones served by the per-worker derivative family
([`ADWorkspace`](@ref)). Jacobian kinds belong here through their AD fallback,
even when an element serves them analytically.
"""
const DerivativeSweepKind = Union{JacobianKind, JacobianResidualKind, WeightedJacobianKind, SensitivityKind}

"""
    assembles_matrix(kind) -> Bool
    assembles_vector(kind) -> Bool
    depends_on_unknowns(kind) -> Bool

What a sweep of `kind` does with the workspace: which of `ws.Ke`/`ws.re` it
zeroes and scatters, and whether it gathers the state slots (and therefore
runs the condensed write-back). The driver bodies consult these instead of
testing kind membership, so a downstream kind joins the built-in driver by
declaring them.

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
    DerivativeFamily
    FunctionalFamily

The per-worker sweep-state families, which double as the driver-shape routing.
`DerivativeFamily` is the [`ADWorkspace`](@ref) and `NoFamily` marks a sweep
that reads only the workspace core; both scatter their result through an
assembler. `FunctionalFamily` selects the VALUE-RETURNING driver instead: its
kernel returns the item's contribution and the sweep reduces the returned
values, so it needs — and builds — no per-worker state at all.
"""
struct NoFamily end
@doc (@doc NoFamily) struct DerivativeFamily end
@doc (@doc NoFamily) struct FunctionalFamily end

"""
    sweep_family(::Type{K}) -> family singleton

The per-worker family a sweep of kind `K` reads. Queried on the TYPE, because
declarations carry kind types (normalized to their `UnionAll` base) while
sweeps carry instances — one overload point serves both.

Declaring a family is what makes it exist: `setup_operator(...; requests =
(MyKind,))` builds the family this returns, so

    FerriteOperators.sweep_family(::Type{<:MyKind}) = FerriteOperators.DerivativeFamily()

gives `MyKind` an [`ADWorkspace`](@ref) at setup and lets [`sweep_state`](@ref)
resolve. `FunctionalFamily()` is the one answer that builds NOTHING: it routes
the kind to the value-returning driver ([`run_reduction`](@ref)), which reads
no per-worker state. The default derives the answer from the built-in unions
and folds to a constant.
"""
sweep_family(::Type{K}) where {K} =
    K <: DerivativeSweepKind ? DerivativeFamily() :
    K <: FunctionalKind ? FunctionalFamily() : NoFamily()

"""
    sweep_state(ws, kind)

The per-worker member serving `kind`'s sweep family. The workspace is a fixed
CORE (geometry cache, element caches, slot buffers, scratch, `Ke`/`re`) plus
kind-family members that exist only when the operator's declarations call for
them: derivative kinds read the [`ADWorkspace`](@ref). The other two families
have no member to read — a `NoFamily` kind works out of the core, and a
`FunctionalFamily` kind returns its value rather than parking it anywhere.
"""
sweep_state(ws, kind) = sweep_state(ws, kind, sweep_family(typeof(kind)))
function sweep_state(ws, kind, ::DerivativeFamily)
    ad = ws.ad
    ad === nothing && throw(ArgumentError(
        "This operator carries no derivative sweep family: neither its protocol's declared " *
        "kinds nor its integrator family's mandatory kinds differentiate. Declare the kind at " *
        "setup — `setup_operator(...; requests = ($(nameof(typeof(kind))),))` or a protocol " *
        "whose `get_declared_kinds` names it."))
    return ad
end
sweep_state(ws, kind, ::FunctionalFamily) = throw(ArgumentError(
    "$(nameof(typeof(kind))) rides the value-returning driver: its kernel RETURNS the item's " *
    "contribution and the sweep reduces the returned values, so there is no per-worker state " *
    "to read."))
sweep_state(ws, kind, ::NoFamily) = throw(ArgumentError(
    "$(nameof(typeof(kind))) declares no sweep-state family, so it has no per-worker state " *
    "to read. Declare one with `sweep_family(::Type{<:$(nameof(typeof(kind)))})`."))

# Which per-worker families a declared kind set calls for. Kinds arrive as
# their UnionAll bases, which [`sweep_family`](@ref) is queried on directly.
needs_derivative_family(kinds) = any(K -> sweep_family(K) isa DerivativeFamily, kinds)

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
request_type(::JacobianKind{slot}) where {slot}  = JacobianRequest{slot}
request_type(::BilinearKind)                     = JacobianRequest{:u}
request_type(::JacobianResidualKind)             = JacobianResidualRequest
request_type(::WeightedJacobianKind)             = WeightedJacobianRequest
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

# The kinds whose AD fallback differentiates through an element's local solve.
requires_admissibility_check(::Union{JacobianKind, JacobianResidualKind, ParameterJacobianKind, ParameterVJPKind, StateJVPKind, StateVJPKind}) = true

# Functional kernels return their contribution through `evaluate_cell_functional`
# rather than filling a request, so there is no cell request to validate.
has_cell_request(::Type{<:FunctionalKind}) = false

materialize_request(::ResidualKind, ws)                    = ResidualRequest(ws.re)
materialize_request(::LinearKind, ws)                      = ResidualRequest(ws.re)
materialize_request(::JacobianKind{slot}, ws) where {slot} = JacobianRequest{slot}(ws.Ke)
materialize_request(::BilinearKind, ws)                    = JacobianRequest{:u}(ws.Ke)
materialize_request(::JacobianResidualKind, ws)            = JacobianResidualRequest(ws.Ke, ws.re)
materialize_request(kind::WeightedJacobianKind, ws)        = WeightedJacobianRequest(ws.Ke, kind.weights)

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

# Gather every task slot into the workspace's slot buffers, returning the
# element-local states NamedTuple. Gathering goes through
# `load_element_unknowns!` per slot so condensed elements keep their
# element-overridable [ū; q] layout for every slot.
function load_slots!(ws, states::NamedTuple{names}) where {names}
    return map(NamedTuple{names}(ws.slot_buffers), states) do buf, src
        load_slot!(buf, src, ws)
        buf
    end
end
load_slot!(buf, src, ws) = load_element_unknowns!(buf, src, ws.cell, ws.ivh, ws.element)
# The anchor lands in the slot's own buffer first, then the buffer becomes the
# reconstruction against the already-gathered `:u` buffer.
function load_slot!(buf, src::AffineRate, ws)
    load_element_unknowns!(buf, src.anchor, ws.cell, ws.ivh, ws.element)
    buf .= src.slope .* (ws.slot_buffers.u .- buf)
    return buf
end

execute_kind!(kind::PrimalKind, task, ws) = primal_cell_sweep!(kind, task, ws)

"""
    primal_cell_sweep!(kind, task, ws)

The built-in primal driver body, reusable by a downstream kind's own
`execute_kind!`. Zeroes the buffers [`assembles_matrix`](@ref) /
[`assembles_vector`](@ref) name, reinitializes the element's values, queries
the cell parameters, runs the cell and facet kernels — gathering the state
slots and running the condensed write-back iff [`depends_on_unknowns`](@ref) —
and scatters through [`scatter_local!`](@ref).

The kernel it calls is `cell_kernel!(kind, …)`, whose generic method issues
the kind's request analytically; the built-in kinds with an AD fallback
specialize it.
"""
function primal_cell_sweep!(kind, task, ws)
    assembles_matrix(kind) && fill!(ws.Ke, 0.0)
    assembles_vector(kind) && fill!(ws.re, 0.0)
    reinit_values!(ws.element, ws.cell, kind)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    if depends_on_unknowns(kind)
        statesₑ = load_slots!(ws, task.states)
        @timeit_debug "assemble element" cell_kernel!(kind, ws.element, ws, statesₑ, pₑ, task.ctx)
        @timeit_debug "assemble boundary" boundary_kernel!(kind, ws.boundary_element, ws, statesₑ, task)
        store_condensed_element_unknowns!(statesₑ.u, task.states.u, ws.cell, ws.ivh, ws.element)
    else
        @timeit_debug "assemble element" cell_kernel!(kind, ws.element, ws, (;), pₑ, task.ctx)
        @timeit_debug "assemble boundary" boundary_kernel!(kind, ws.boundary_element, ws, (;), task)
    end
    scatter_local!(kind, task.inner_assembler, ws)
end

# The single KernelArgs construction seam.
_kernel_args(ws, statesₑ, pₑ, ctx) = KernelArgs(statesₑ, ws.cell, pₑ, ws.scratch, ctx)

# The framework-owned facet driver: walk the cell's facets, gate on
# is_facet_in_cache, query facet parameters SEPARATELY per facet, and hand the
# kind's request over the shared local buffers to the facet kernel.
boundary_kernel!(kind, ::EmptySurfaceElementCache, ws, statesₑ, task) = nothing
function boundary_kernel!(kind, cache::AbstractSurfaceElementCache, ws, statesₑ, task)
    for lfi in 1:nfacets(ws.cell)
        if is_facet_in_cache(FacetIndex(cellid(ws.cell), lfi), ws.cell, cache)
            pᵦ = query_facet_parameters(cache, ws.cell, lfi, task.p)
            assemble_facet!(materialize_request(kind, ws), cache,
                            _kernel_args(ws, statesₑ, pᵦ, task.ctx), lfi)
        end
    end
end

# Facet contributions have no AD fallback in any sweep — a surface cache serves
# the sweep's request analytically or not at all. For a weighted sweep that
# means the WEIGHTED facet kernel: per-slot facet kernels are not composed
# behind the driver's back.
function cell_kernel!(kind::ResidualKind, cache, ws, statesₑ, pₑ, ctx)
    assemble_cell!(materialize_request(kind, ws), cache, _kernel_args(ws, statesₑ, pₑ, ctx))
end
function cell_kernel!(kind::JacobianKind{slot}, cache, ws, statesₑ, pₑ, ctx) where {slot}
    args = _kernel_args(ws, statesₑ, pₑ, ctx)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(materialize_request(kind, ws), cache, args)
    else
        ad_state_jacobian!(ws.Ke, ws, args, Val(slot))
    end
end
function cell_kernel!(kind::WeightedJacobianKind, cache, ws, statesₑ, pₑ, ctx)
    args = _kernel_args(ws, statesₑ, pₑ, ctx)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(materialize_request(kind, ws), cache, args)
    else
        ad_weighted_jacobian!(ws.Ke, ws, args, kind.weights)
    end
end
function cell_kernel!(kind::JacobianResidualKind, cache, ws, statesₑ, pₑ, ctx)
    args = _kernel_args(ws, statesₑ, pₑ, ctx)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(materialize_request(kind, ws), cache, args)
    elseif provides_analytic(typeof(cache), JacobianKind())
        # No fused kernel, but both split kernels exist: two of the table's
        # other kinds, issued back to back over the same buffers.
        assemble_cell!(materialize_request(JacobianKind(), ws), cache, args)
        assemble_cell!(materialize_request(ResidualKind(), ws), cache, args)
    else
        ad_state_jacobian!(ws.Ke, ws, args)   # also leaves the primal residual in ws.re
    end
end
# The plain analytic route: issue the kind's request and let the element serve
# it. This is what the state-independent built-in forms use — they have no
# residual kernel to differentiate, so the analytic kernel is mandatory for the
# elements used in these operators — and what a downstream kind riding
# [`primal_cell_sweep!`](@ref) gets without writing a kernel dispatch of its own.
cell_kernel!(kind, cache, ws, statesₑ, pₑ, ctx) =
    assemble_cell!(materialize_request(kind, ws), cache, _kernel_args(ws, statesₑ, pₑ, ctx))

# Sensitivity sweeps: gather the trial state, never write anything back into
# `u`, and route through analytic kernels or AD-from-residual. Local outputs
# live in the per-worker `ws.ad` buffers; analytic kernels ACCUMULATE into
# their target (zeroed in that branch), the ForwardDiff fallbacks overwrite
# it fully.
execute_kind!(kind::SensitivityKind, task, ws) = sensitivity_cell_sweep!(kind, task, ws)

"""
    sensitivity_cell_sweep!(kind, task, ws)

The built-in sensitivity driver body, reusable by a downstream kind's own
`execute_kind!`. Gathers the trial state, queries the cell parameters and
hands the kernel args to `sensitivity_kernel!(kind, task, ws, args)`, which
the kind implements. Nothing is written back into `u`, and the local output
lives in the kind's own sweep-state buffers.
"""
function sensitivity_cell_sweep!(kind, task, ws)
    reinit_values!(ws.element, ws.cell, kind)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    args = _kernel_args(ws, statesₑ, pₑ, task.ctx)
    @timeit_debug "assemble sensitivity" sensitivity_kernel!(kind, task, ws, args)
end

# Residual-shaped gather (plain celldofs slice — adjoint vectors carry no
# condensed tail, unlike the slot gathers).
_gather_residual_dofs!(dest, src, cell) = dest .= @view src[celldofs(cell)]

function sensitivity_kernel!(kind::ParameterJacobianKind, task, ws, args)
    cache = ws.element
    Bₑ = parameter_sweep_buffers!(sweep_state(ws, kind), length(parameter_vector(task.p))).Bₑ
    if provides_analytic(typeof(cache), kind)
        fill!(Bₑ, zero(eltype(Bₑ)))
        assemble_cell!(ParameterJacobianRequest(Bₑ), cache, args)
    else
        ad_parameter_jacobian!(Bₑ, ws, args, task.p)
    end
    assemble!(task.inner_assembler, ws.cell, Bₑ)
end

function sensitivity_kernel!(kind::ParameterVJPKind, task, ws, args)
    cache = ws.element
    ad = sweep_state(ws, kind)
    λₑ = _gather_residual_dofs!(ad.λₑ, kind.λ, ws.cell)
    gₑ = parameter_sweep_buffers!(ad, length(parameter_vector(task.p))).gθ
    if provides_analytic(typeof(cache), kind)
        fill!(gₑ, zero(eltype(gₑ)))
        assemble_cell!(ParameterVJPRequest(gₑ, λₑ), cache, args)
    else
        ad_parameter_vjp!(gₑ, λₑ, ws, args, task.p)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end

function sensitivity_kernel!(kind::StateJVPKind, task, ws, args)
    cache = ws.element
    ad = sweep_state(ws, kind)
    vₑ = ad.vₑ
    load_element_unknowns!(vₑ, kind.v, ws.cell, ws.ivh, cache)
    Jvₑ = ad.Jvₑ
    if provides_analytic(typeof(cache), kind)
        fill!(Jvₑ, zero(eltype(Jvₑ)))
        assemble_cell!(StateJVPRequest(Jvₑ, vₑ), cache, args)
    else
        ad_state_jvp!(Jvₑ, ws, args, vₑ)
    end
    assemble!(task.inner_assembler, ws.cell, Jvₑ)
end

function sensitivity_kernel!(kind::StateVJPKind, task, ws, args)
    cache = ws.element
    ad = sweep_state(ws, kind)
    λₑ = _gather_residual_dofs!(ad.λₑ, kind.λ, ws.cell)
    gₑ = ad.gu
    if provides_analytic(typeof(cache), kind)
        fill!(gₑ, zero(eltype(gₑ)))
        assemble_cell!(StateVJPRequest(gₑ, λₑ), cache, args)
    else
        ad_state_vjp!(gₑ, λₑ, ws, args)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end

execute_kind!(kind::FunctionalKind, task, ws) = functional_cell_sweep(kind, task, ws)

"""
    functional_cell_sweep(kind, task, ws) -> value

The built-in VALUE-RETURNING driver body, reusable by a downstream kind's own
`execute_kind!`. Reinitializes the element's values, gathers the state slots
without writing anything back, and returns what
[`evaluate_cell_functional`](@ref) gives for the cell — `nothing` for a cell
that contributes nothing.

Unlike [`primal_cell_sweep!`](@ref) and [`sensitivity_cell_sweep!`](@ref) this
body writes no result into the workspace and scatters nothing; the sweep
reduces what it returns ([`run_reduction`](@ref)).
"""
function functional_cell_sweep(kind, task, ws)
    reinit_values!(ws.element, ws.cell, kind)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    return evaluate_cell_functional(kind, ws.element, _kernel_args(ws, statesₑ, pₑ, task.ctx))
end

function sensitivity_kernel!(kind::TimeSensitivityKind, task, ws, args)
    cache = ws.element
    gₑ = sweep_state(ws, kind).gₜ
    if provides_analytic(typeof(cache), kind)
        fill!(gₑ, zero(eltype(gₑ)))
        assemble_cell!(TimeSensitivityRequest(gₑ), cache, args)
    else
        ad_time_sensitivity!(gₑ, ws, args)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end


"""
    scatter_local!(kind, assembler, ws)

Hand the local buffers a sweep of `kind` filled to the assembler. Which
buffers those are is [`assembles_matrix`](@ref)/[`assembles_vector`](@ref), so
the three routes are selected at compile time and a downstream kind is
scattered by the same body.
"""
function scatter_local!(kind, assembler, ws)
    if assembles_matrix(kind) && assembles_vector(kind)
        assemble!(assembler, ws.cell, ws.Ke, ws.re)
    elseif assembles_matrix(kind)
        assemble!(assembler, ws.cell, ws.Ke)
    elseif assembles_vector(kind)
        assemble!(assembler, ws.cell, ws.re)
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
            _assert_trait_backed(typeof(sc.domain.element), kind, kernel_args_type(engine))
        end
    end
    return nothing
end

# A weighted sweep differentiates SEVERAL slots at once, so every participating
# slot must be present. Under the AD route the same `AffineRate` rejection as
# for `JacobianKind` applies; an analytic weighted kernel is exempt because it
# forms the combination itself (the MLN/Newmark design intent). The exemption
# is engine-wide: one cache falling back to AD makes the whole sweep AD-seeded.
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
        _assert_trait_backed(typeof(sc.domain.element), kind, kernel_args_type(engine))
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

The value-returning counterpart of `run_sweep!`: the same per-sweep
validation, but the sweep hands its result back as a value instead of
scattering it, and no workspace state is read or written. `nothing` means no
item contributed.

Which shape a kind runs in is [`sweep_family`](@ref): only a
`FunctionalFamily` kind is value-returning, so this is the entry point a
downstream scalar or tensor kind reaches once it declares that family and
gives `execute_kind!` a body returning the item's contribution
([`functional_cell_sweep`](@ref) is the built-in one).
"""
run_reduction(kind, op, states::NamedTuple, p, ctx) =
    run_reduction(sweep_family(typeof(kind)), kind, op, states, p, ctx)

function run_reduction(::FunctionalFamily, kind, op, states::NamedTuple, p, ctx)
    _check_declared_slots(op.engine, states)
    _check_rate_slots(states)
    _check_reduction_domain(kind, op.engine)
    return reduce_on_subdomains(AssemblyTask(kind, nothing, states, p, ctx), op.engine)
end

# A reduction has two STRUCTURAL preconditions, both misconfiguration and both
# decidable before any item runs: there have to be items to reduce over, and
# some subdomain has to be able to contribute at all. Deciding them here is
# what separates them from the data-dependent case — a sweep whose kernels all
# answer `nothing` over a non-empty, non-empty-cached domain is a legitimate
# empty sum, not a misconfiguration. Both checks are O(subdomains): partition
# lengths and the cache TYPE, never cell data.
function _check_reduction_domain(kind, engine)
    caches = engine.subdomain_caches
    sum(sc -> sum(length, sc.partition; init = 0), caches; init = 0) == 0 && throw(ArgumentError(
        "$(nameof(typeof(kind))) reduces over an empty item set: the operator's " *
        "$(length(caches)) subdomain partition(s) carry no items between them, so there is " *
        "nothing to integrate over."))
    all(sc -> sc.domain.element isa EmptyVolumetricElementCache, caches) && throw(ArgumentError(
        "No subdomain can contribute to $(nameof(typeof(kind))): every subdomain's element " *
        "cache is an `EmptyVolumetricElementCache`, which returns no contribution by " *
        "construction. Set the operator up with an integrator whose caches implement " *
        "`evaluate_cell_functional`."))
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

Evaluate the functional named by `kind` over the operator's domain: the sum
of the per-cell contributions returned by
[`evaluate_cell_functional`](@ref) (a `Number` or a Tensors tensor).
Contributions fold per worker in item order and the per-worker partials reduce
in a fixed order — results are deterministic for a fixed worker count. Nothing
is written into the operator, so an operator declaring no functional kind
serves one just as well. Volumetric contributions only.

Two failure modes are kept apart. STRUCTURAL emptiness — no items in the
operator's partitions, or no subdomain whose element cache can contribute — is
a misconfiguration and is an `ArgumentError` raised before any cell runs,
whatever the kind declares. A sweep that runs and whose kernels all answer
`nothing` is the DATA-DEPENDENT case: an empty sum, which a kind declaring
[`functional_value_type`](@ref) answers with `zero(T)` and an undeclared kind
cannot answer at all (there is no `T` to take the identity of) and reports as
an `ArgumentError`. Declaring the value type is what makes an all-quiet sweep
well-defined.
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
