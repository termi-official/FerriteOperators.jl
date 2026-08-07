# What a single assembly sweep computes. Kinds select which request is
# materialized over the workspace buffers and which kernels run.
struct JacobianResidualKind end     # nonlinear J(u) and r(u), fused
struct ResidualKind end             # nonlinear r(u)
struct BilinearKind end             # u-independent matrix
struct LinearKind end               # u-independent vector
struct ParameterJacobianKind end                # ∂F/∂θ, materialized
struct ParameterVJPKind{L}; λ::L; end          # (∂F/∂θ)ᵀλ, adjoint pullback
struct TimeSensitivityKind{T}; t::T; end       # ∂F/∂t, explicit dependence
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
    FunctionalKind{tag}
    FunctionalKind(tag::Symbol)

Names a functional (reduction) query: a global scalar or Tensors tensor
integrated from per-cell contributions (energy, dissipation, …). Elements
implement [`evaluate_cell_functional`](@ref) dispatching on the tag; solvers
evaluate through [`evaluate_functional`](@ref).
"""
struct FunctionalKind{tag} end
FunctionalKind(tag::Symbol) = FunctionalKind{tag}()

const MatrixAssemblyKind  = Union{JacobianResidualKind, JacobianKind, BilinearKind}
const VectorAssemblyKind  = Union{JacobianResidualKind, ResidualKind, LinearKind}
const UnknownDependentKind = Union{JacobianResidualKind, JacobianKind, ResidualKind}
const PrimalKind = Union{JacobianResidualKind, JacobianKind, ResidualKind, BilinearKind, LinearKind}
const SensitivityKind = Union{ParameterJacobianKind, ParameterVJPKind, TimeSensitivityKind, StateJVPKind, StateVJPKind}

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
    issubset(names, engine.slots) || throw(ArgumentError(
        "States pass slots $names but the operator declared slots $(engine.slots). " *
        "Declare every slot at setup: `setup_operator(...; slots = $(Tuple(union(engine.slots, names))))`."))
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

# Per-slot reconstruction metadata reaching kernels through `slot_slope`.
slot_metadata(states::NamedTuple{names}) where {names} =
    NamedTuple{names}(map(_reconstruction_slope, values(states)))
_reconstruction_slope(src::AffineRate) = src.slope
_reconstruction_slope(src) = nothing

function execute_kind!(kind::PrimalKind, task, ws)
    kind isa MatrixAssemblyKind && fill!(ws.Ke, 0.0)
    kind isa VectorAssemblyKind && fill!(ws.re, 0.0)
    reinit_values!(ws.element, ws.cell, kind)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    if kind isa UnknownDependentKind
        statesₑ = load_slots!(ws, task.states)
        metaₑ = slot_metadata(task.states)
        @timeit_debug "assemble element" v2_cell_kernel!(kind, ws.element, ws, statesₑ, pₑ, task.ctx, metaₑ)
        @timeit_debug "assemble boundary" boundary_kernel!(kind, ws.boundary_element, ws, statesₑ, metaₑ, task)
        store_condensed_element_unknowns!(statesₑ.u, task.states.u, ws.cell, ws.ivh, ws.element)
    else
        @timeit_debug "assemble element" v2_cell_kernel!(kind, ws.element, ws, (;), pₑ, task.ctx)
        @timeit_debug "assemble boundary" boundary_kernel!(kind, ws.boundary_element, ws, (;), (;), task)
    end
    scatter_local!(kind, task.inner_assembler, ws)
end

# The single KernelArgs construction seam.
_v2_args(ws, statesₑ, pₑ, ctx, metaₑ = (;)) = KernelArgs(statesₑ, ws.cell, pₑ, ws.scratch, ctx, metaₑ)

# The framework-owned facet driver: walk the cell's facets, gate on
# is_facet_in_cache, query facet parameters SEPARATELY per facet, and hand the
# kind's request over the shared local buffers to the facet kernel.
boundary_kernel!(kind::PrimalKind, ::EmptySurfaceElementCache, ws, statesₑ, metaₑ, task) = nothing
function boundary_kernel!(kind::PrimalKind, cache::AbstractSurfaceElementCache, ws, statesₑ, metaₑ, task)
    for lfi in 1:nfacets(ws.cell)
        if is_facet_in_cache(FacetIndex(cellid(ws.cell), lfi), ws.cell, cache)
            pᵦ = query_facet_parameters(cache, ws.cell, lfi, task.p)
            facet_request!(kind, cache, ws, _v2_args(ws, statesₑ, pᵦ, task.ctx, metaₑ), lfi)
        end
    end
end
facet_request!(::ResidualKind,         cache, ws, args, lfi) = assemble_facet!(ResidualRequest(ws.re), cache, args, lfi)
facet_request!(::JacobianKind{slot},   cache, ws, args, lfi) where {slot} = assemble_facet!(JacobianRequest{slot}(ws.Ke), cache, args, lfi)
facet_request!(::JacobianResidualKind, cache, ws, args, lfi) = assemble_facet!(JacobianResidualRequest(ws.Ke, ws.re), cache, args, lfi)
facet_request!(::BilinearKind,         cache, ws, args, lfi) = assemble_facet!(JacobianRequest{:u}(ws.Ke), cache, args, lfi)
facet_request!(::LinearKind,           cache, ws, args, lfi) = assemble_facet!(ResidualRequest(ws.re), cache, args, lfi)

function v2_cell_kernel!(::ResidualKind, cache, ws, statesₑ, pₑ, ctx, metaₑ = (;))
    assemble_cell!(ResidualRequest(ws.re), cache, _v2_args(ws, statesₑ, pₑ, ctx, metaₑ))
end
function v2_cell_kernel!(kind::JacobianKind{slot}, cache, ws, statesₑ, pₑ, ctx, metaₑ = (;)) where {slot}
    args = _v2_args(ws, statesₑ, pₑ, ctx, metaₑ)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(JacobianRequest{slot}(ws.Ke), cache, args)
    else
        ad_state_jacobian!(ws.Ke, ws, args, Val(slot))
    end
end
function v2_cell_kernel!(kind::JacobianResidualKind, cache, ws, statesₑ, pₑ, ctx, metaₑ = (;))
    args = _v2_args(ws, statesₑ, pₑ, ctx, metaₑ)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(JacobianResidualRequest(ws.Ke, ws.re), cache, args)
    elseif provides_analytic(typeof(cache), JacobianKind())
        assemble_cell!(JacobianRequest{:u}(ws.Ke), cache, args)
        assemble_cell!(ResidualRequest(ws.re), cache, args)
    else
        ad_state_jacobian!(ws.Ke, ws, args)   # also leaves the primal residual in ws.re
    end
end
# State-independent forms have no residual kernel to differentiate; the
# analytic kernel is mandatory for v2 elements used in these operators.
v2_cell_kernel!(::BilinearKind, cache, ws, statesₑ, pₑ, ctx, metaₑ = (;)) =
    assemble_cell!(JacobianRequest{:u}(ws.Ke), cache, _v2_args(ws, statesₑ, pₑ, ctx, metaₑ))
v2_cell_kernel!(::LinearKind, cache, ws, statesₑ, pₑ, ctx, metaₑ = (;)) =
    assemble_cell!(ResidualRequest(ws.re), cache, _v2_args(ws, statesₑ, pₑ, ctx, metaₑ))

# Sensitivity sweeps: gather the trial state, never write anything back into
# `u`, and route through analytic kernels or AD-from-residual. Local outputs
# live in the per-worker `ws.ad` buffers; analytic kernels ACCUMULATE into
# their target (zeroed in that branch), the ForwardDiff fallbacks overwrite
# it fully.
function execute_kind!(kind::SensitivityKind, task, ws)
    reinit_values!(ws.element, ws.cell, kind)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    args = _v2_args(ws, statesₑ, pₑ, task.ctx, slot_metadata(task.states))
    @timeit_debug "assemble sensitivity" sensitivity_kernel!(kind, task, ws, args)
end

# Residual-shaped gather (plain celldofs slice — adjoint vectors carry no
# condensed tail, unlike the slot gathers).
_gather_residual_dofs!(dest, src, cell) = dest .= @view src[celldofs(cell)]

function sensitivity_kernel!(kind::ParameterJacobianKind, task, ws, args)
    cache = ws.element
    Bₑ = parameter_sweep_buffers!(ws.ad, length(parameter_vector(task.p))).Bₑ
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
    λₑ = _gather_residual_dofs!(ws.ad.λₑ, kind.λ, ws.cell)
    gₑ = parameter_sweep_buffers!(ws.ad, length(parameter_vector(task.p))).gθ
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
    vₑ = ws.ad.vₑ
    load_element_unknowns!(vₑ, kind.v, ws.cell, ws.ivh, cache)
    Jvₑ = ws.ad.Jvₑ
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
    λₑ = _gather_residual_dofs!(ws.ad.λₑ, kind.λ, ws.cell)
    gₑ = ws.ad.gu
    if provides_analytic(typeof(cache), kind)
        fill!(gₑ, zero(eltype(gₑ)))
        assemble_cell!(StateVJPRequest(gₑ, λₑ), cache, args)
    else
        ad_state_vjp!(gₑ, λₑ, ws, args)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end

# Functional sweeps: the kernel RETURNS the cell contribution; it accumulates
# into the per-worker slot and reduces at the entry point. `nothing` means "no
# contribution" (empty caches).
function execute_kind!(kind::FunctionalKind, task, ws)
    reinit_values!(ws.element, ws.cell, kind)
    statesₑ = load_slots!(ws, task.states)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    val = evaluate_cell_functional(kind, ws.element, _v2_args(ws, statesₑ, pₑ, task.ctx, slot_metadata(task.states)))
    _accumulate_functional!(ws.functional, val)
end
_accumulate_functional!(acc, ::Nothing) = nothing
function _accumulate_functional!(acc, val)
    acc.value = acc.value === nothing ? val : acc.value + val
    return nothing
end

function sensitivity_kernel!(kind::TimeSensitivityKind, task, ws, args)
    cache = ws.element
    gₑ = ws.ad.gₜ
    if provides_analytic(typeof(cache), kind)
        fill!(gₑ, zero(eltype(gₑ)))
        assemble_cell!(TimeSensitivityRequest(gₑ), cache, args)
    else
        ad_time_sensitivity!(gₑ, ws, args, kind.t)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end


scatter_local!(::JacobianResidualKind, assembler, ws)              = assemble!(assembler, ws.cell, ws.Ke, ws.re)
scatter_local!(::Union{JacobianKind, BilinearKind}, assembler, ws) = assemble!(assembler, ws.cell, ws.Ke)
scatter_local!(::Union{ResidualKind, LinearKind}, assembler, ws)   = assemble!(assembler, ws.cell, ws.re)

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
            _assert_trait_backed(typeof(sc.domain.element), kind, JacobianRequest{slot})
        end
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
    evaluate_functional(op, kind::FunctionalKind, states, p, ctx = nothing)
    evaluate_functional(op, kind::FunctionalKind, u::AbstractVector, p)

Evaluate the functional named by `kind` over the operator's domain: the sum
of the per-cell contributions returned by
[`evaluate_cell_functional`](@ref) (a `Number` or a Tensors tensor).
Contributions accumulate per worker and reduce in a fixed order — results
are deterministic for a fixed worker count. Volumetric contributions only.
"""
function evaluate_functional(op, kind::FunctionalKind, states::NamedTuple, p, ctx = nothing)
    for sc in op.engine.subdomain_caches, ws in sc.device_cache
        ws.functional.value = nothing
    end
    run_sweep!(kind, nothing, op, states, p, ctx)
    total = nothing
    for sc in op.engine.subdomain_caches, ws in sc.device_cache
        v = ws.functional.value
        v === nothing || (total = total === nothing ? v : total + v)
    end
    total === nothing && throw(ArgumentError(
        "No element contributed to $(typeof(kind)). Implement " *
        "`evaluate_cell_functional` for the operator's element caches."))
    return total
end
evaluate_functional(op, kind::FunctionalKind, u::AbstractVector, p) =
    evaluate_functional(op, kind, (u = u,), p, nothing)
