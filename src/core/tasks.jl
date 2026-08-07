# What a single assembly sweep computes. Kinds select which request is
# materialized over the workspace buffers and which kernels run.
struct JacobianResidualKind end     # nonlinear J(u) and r(u), fused
struct JacobianKind end             # nonlinear J(u)
struct ResidualKind end             # nonlinear r(u)
struct BilinearKind end             # u-independent matrix
struct LinearKind end               # u-independent vector
struct ParameterJacobianKind end                # ∂F/∂θ, materialized
struct ParameterVJPKind{L}; λ::L; end          # (∂F/∂θ)ᵀλ, adjoint pullback
struct TimeSensitivityKind{T}; t::T; end       # ∂F/∂t, explicit dependence

const MatrixAssemblyKind  = Union{JacobianResidualKind, JacobianKind, BilinearKind}
const VectorAssemblyKind  = Union{JacobianResidualKind, ResidualKind, LinearKind}
const UnknownDependentKind = Union{JacobianResidualKind, JacobianKind, ResidualKind}
const PrimalKind = Union{JacobianResidualKind, JacobianKind, ResidualKind, BilinearKind, LinearKind}
const SensitivityKind = Union{ParameterJacobianKind, ParameterVJPKind, TimeSensitivityKind}

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

# Gather every task slot into the workspace's slot buffers, returning the
# element-local states NamedTuple. Gathering goes through
# `load_element_unknowns!` per slot so condensed elements keep their
# element-overridable [ū; q] layout for every slot.
function load_slots!(ws, states::NamedTuple{names}, cell, ivh, element) where {names}
    return map(NamedTuple{names}(ws.slot_buffers), states) do buf, src
        load_element_unknowns!(buf, src, cell, ivh, element)
        buf
    end
end

function execute_kind!(kind::PrimalKind, task, ws)
    kind isa MatrixAssemblyKind && fill!(ws.Ke, 0.0)
    kind isa VectorAssemblyKind && fill!(ws.re, 0.0)
    pₑ = query_cell_parameters(ws.element, ws.cell, task.p)
    if kind isa UnknownDependentKind
        statesₑ = load_slots!(ws, task.states, ws.cell, ws.ivh, ws.element)
        @timeit_debug "assemble element" v2_cell_kernel!(kind, ws.element, ws, statesₑ, pₑ, task.ctx)
        @timeit_debug "assemble boundary" boundary_kernel!(kind, ws.boundary_element, ws, statesₑ, task)
        store_condensed_element_unknowns!(statesₑ.u, task.states.u, ws.cell, ws.ivh, ws.element)
    else
        @timeit_debug "assemble element" v2_cell_kernel!(kind, ws.element, ws, (;), pₑ, task.ctx)
        @timeit_debug "assemble boundary" boundary_kernel!(kind, ws.boundary_element, ws, (;), task)
    end
    scatter_local!(kind, task.inner_assembler, ws)
end

_v2_args(ws, statesₑ, pₑ, ctx) = KernelArgs(statesₑ, ws.cell, pₑ, ws.scratch, ctx)

# The framework-owned facet driver: walk the cell's facets, gate on
# is_facet_in_cache, query facet parameters SEPARATELY per facet, and hand the
# kind's request over the shared local buffers to the facet kernel.
boundary_kernel!(kind::PrimalKind, ::EmptySurfaceElementCache, ws, statesₑ, task) = nothing
function boundary_kernel!(kind::PrimalKind, cache::AbstractSurfaceElementCache, ws, statesₑ, task)
    for lfi in 1:nfacets(ws.cell)
        if is_facet_in_cache(FacetIndex(cellid(ws.cell), lfi), ws.cell, cache)
            pᵦ = query_facet_parameters(cache, ws.cell, lfi, task.p)
            facet_request!(kind, cache, ws, _v2_args(ws, statesₑ, pᵦ, task.ctx), lfi)
        end
    end
end
facet_request!(::ResidualKind,         cache, ws, args, lfi) = assemble_facet!(ResidualRequest(ws.re), cache, args, lfi)
facet_request!(::JacobianKind,         cache, ws, args, lfi) = assemble_facet!(JacobianRequest{:u}(ws.Ke), cache, args, lfi)
facet_request!(::JacobianResidualKind, cache, ws, args, lfi) = assemble_facet!(JacobianResidualRequest(ws.Ke, ws.re), cache, args, lfi)
facet_request!(::BilinearKind,         cache, ws, args, lfi) = assemble_facet!(JacobianRequest{:u}(ws.Ke), cache, args, lfi)
facet_request!(::LinearKind,           cache, ws, args, lfi) = assemble_facet!(ResidualRequest(ws.re), cache, args, lfi)

function v2_cell_kernel!(::ResidualKind, cache, ws, statesₑ, pₑ, ctx)
    assemble_cell!(ResidualRequest(ws.re), cache, _v2_args(ws, statesₑ, pₑ, ctx))
end
function v2_cell_kernel!(kind::JacobianKind, cache, ws, statesₑ, pₑ, ctx)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(JacobianRequest{:u}(ws.Ke), cache, _v2_args(ws, statesₑ, pₑ, ctx))
    else
        ad_state_jacobian!(ws.Ke, ws, statesₑ, pₑ, ctx)
    end
end
function v2_cell_kernel!(kind::JacobianResidualKind, cache, ws, statesₑ, pₑ, ctx)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(JacobianResidualRequest(ws.Ke, ws.re), cache, _v2_args(ws, statesₑ, pₑ, ctx))
    elseif provides_analytic(typeof(cache), JacobianKind())
        assemble_cell!(JacobianRequest{:u}(ws.Ke), cache, _v2_args(ws, statesₑ, pₑ, ctx))
        assemble_cell!(ResidualRequest(ws.re), cache, _v2_args(ws, statesₑ, pₑ, ctx))
    else
        ad_state_jacobian!(ws.Ke, ws, statesₑ, pₑ, ctx)   # also leaves the primal residual in ws.re
    end
end
function v2_cell_kernel!(kind::Union{BilinearKind, LinearKind}, cache, ws, statesₑ, pₑ, ctx)
    # State-independent forms have no residual kernel to differentiate; the
    # analytic kernel is mandatory for v2 elements used in these operators.
    kind isa BilinearKind ?
        assemble_cell!(JacobianRequest{:u}(ws.Ke), cache, _v2_args(ws, statesₑ, pₑ, ctx)) :
        assemble_cell!(ResidualRequest(ws.re), cache, _v2_args(ws, statesₑ, pₑ, ctx))
end

# Sensitivity sweeps: gather the trial state, never write anything back into
# `u`, and route through analytic kernels or AD-from-residual.
function execute_kind!(kind::SensitivityKind, task, ws)
    statesₑ = load_slots!(ws, task.states, ws.cell, ws.ivh, ws.element)
    @timeit_debug "assemble sensitivity" sensitivity_kernel!(kind, task, ws, statesₑ)
end

function sensitivity_kernel!(kind::ParameterJacobianKind, task, ws, statesₑ)
    cache = ws.element
    nθ = length(parameter_vector(task.p))
    Bₑ = zeros(length(ws.re), nθ)   # FIXME per-worker buffer once request kinds are declared at setup
    if provides_analytic(typeof(cache), kind)
        pₑ = query_cell_parameters(cache, ws.cell, task.p)
        assemble_cell!(ParameterJacobianRequest(Bₑ), cache, _v2_args(ws, statesₑ, pₑ, task.ctx))
    else
        ad_parameter_jacobian!(Bₑ, ws, statesₑ, task.p)
    end
    assemble!(task.inner_assembler, ws.cell, Bₑ)
end

function sensitivity_kernel!(kind::ParameterVJPKind, task, ws, statesₑ)
    cache = ws.element
    λₑ = kind.λ[celldofs(ws.cell)]   # FIXME per-worker buffer
    gₑ = zeros(length(parameter_vector(task.p)))
    if provides_analytic(typeof(cache), kind)
        pₑ = query_cell_parameters(cache, ws.cell, task.p)
        assemble_cell!(ParameterVJPRequest(gₑ, λₑ), cache, _v2_args(ws, statesₑ, pₑ, task.ctx))
    else
        ad_parameter_vjp!(gₑ, λₑ, ws, statesₑ, task.p)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end

function sensitivity_kernel!(kind::TimeSensitivityKind, task, ws, statesₑ)
    cache = ws.element
    gₑ = zeros(length(ws.re))   # FIXME per-worker buffer
    if provides_analytic(typeof(cache), kind)
        pₑ = query_cell_parameters(cache, ws.cell, task.p)
        assemble_cell!(TimeSensitivityRequest(gₑ), cache, _v2_args(ws, statesₑ, pₑ, task.ctx))
    else
        ad_time_sensitivity!(gₑ, ws, statesₑ, kind.t)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end


scatter_local!(::JacobianResidualKind, assembler, ws)              = assemble!(assembler, ws.cell, ws.Ke, ws.re)
scatter_local!(::Union{JacobianKind, BilinearKind}, assembler, ws) = assemble!(assembler, ws.cell, ws.Ke)
scatter_local!(::Union{ResidualKind, LinearKind}, assembler, ws)   = assemble!(assembler, ws.cell, ws.re)

# The one assembly driver shared by every operator entry point. `out` is the
# tuple of global targets handed to `start_assemble`.
function assemble_into!(kind, out::Tuple, op, states::NamedTuple, p, ctx)
    assembler = start_assemble(op.engine.strategy, out...)
    task = AssemblyTask(kind, assembler, states, p, ctx)
    execute_on_subdomains!(task, op.engine)
    finalize_assembly!(assembler)
end
