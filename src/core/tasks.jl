# Element-level query functions (overridable by element caches)
query_element_parameters(element, geometry_cache, ivh, p) = p
query_element_unknown_buffer(element, ue) = ue

# What a single assembly sweep computes. The primal kinds select the legacy
# element kernel arity below or the v2 request path; the sensitivity kinds
# always go through the request path (analytic or AD).
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
    AssemblyTask(kind, inner_assembler, u, p)

The single per-cell assembly sweep shared by all operators. `kind` selects what
is computed (and thereby which element kernel is called), `u` is the global
unknown vector (`nothing` for u-independent kinds), `p` the user parameters.
"""
@concrete struct AssemblyTask
    kind
    inner_assembler
    u
    p
end
duplicate_for_device(device, task::AssemblyTask) =
    AssemblyTask(task.kind, duplicate_for_device(device, task.inner_assembler), task.u, task.p)

execute_single_task!(task::AssemblyTask, ws::AssemblyWorkspace) = execute_kind!(task.kind, task, ws)

function execute_kind!(kind::PrimalKind, task, ws)
    kind isa MatrixAssemblyKind && fill!(ws.Ke, 0.0)
    kind isa VectorAssemblyKind && fill!(ws.re, 0.0)
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)
    if kind isa UnknownDependentKind
        uₑ = query_element_unknown_buffer(ws.element, ws.ue)
        load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
        @timeit_debug "assemble element" cell_kernel!(kind, ws, uₑ, pₑ)
        @timeit_debug "assemble boundary" element_kernel!(kind, ws.boundary_element, ws, uₑ, pₑ)
        store_condensed_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    else
        @timeit_debug "assemble element" cell_kernel!(kind, ws, nothing, pₑ)
        @timeit_debug "assemble boundary" element_kernel!(kind, ws.boundary_element, ws, nothing, pₑ)
    end
    scatter_local!(kind, task.inner_assembler, ws)
end

# Volumetric primal kernel: v2 request path when the element opts in, legacy
# arity shim otherwise.
function cell_kernel!(kind, ws, uₑ, pₑ)
    cache = ws.element
    if implements_v2_kernels(typeof(cache))
        v2_cell_kernel!(kind, cache, ws, uₑ, pₑ)
    else
        element_kernel!(kind, cache, ws, uₑ, pₑ)
    end
end

_v2_args(ws, uₑ, pₑ) = KernelArgs((u = uₑ,), ws.cell, pₑ, nothing, nothing)

function v2_cell_kernel!(::ResidualKind, cache, ws, uₑ, pₑ)
    assemble_cell!(ResidualRequest(ws.re), cache, _v2_args(ws, uₑ, pₑ))
end
function v2_cell_kernel!(kind::JacobianKind, cache, ws, uₑ, pₑ)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(JacobianRequest{:u}(ws.Ke), cache, _v2_args(ws, uₑ, pₑ))
    else
        ad_state_jacobian!(ws.Ke, ws, uₑ, pₑ)
    end
end
function v2_cell_kernel!(kind::JacobianResidualKind, cache, ws, uₑ, pₑ)
    if provides_analytic(typeof(cache), kind)
        assemble_cell!(JacobianResidualRequest(ws.Ke, ws.re), cache, _v2_args(ws, uₑ, pₑ))
    elseif provides_analytic(typeof(cache), JacobianKind())
        assemble_cell!(JacobianRequest{:u}(ws.Ke), cache, _v2_args(ws, uₑ, pₑ))
        assemble_cell!(ResidualRequest(ws.re), cache, _v2_args(ws, uₑ, pₑ))
    else
        ad_state_jacobian!(ws.Ke, ws, uₑ, pₑ)   # also leaves the primal residual in ws.re
    end
end
function v2_cell_kernel!(kind::Union{BilinearKind, LinearKind}, cache, ws, uₑ, pₑ)
    # u-independent forms have no residual kernel to differentiate; the
    # analytic kernel is mandatory for v2 elements used in these operators.
    kind isa BilinearKind ?
        assemble_cell!(JacobianRequest{:u}(ws.Ke), cache, _v2_args(ws, uₑ, pₑ)) :
        assemble_cell!(ResidualRequest(ws.re), cache, _v2_args(ws, uₑ, pₑ))
end

# Sensitivity sweeps: gather the trial state, never write anything back into
# `u`, and route through analytic kernels or AD-from-residual.
function execute_kind!(kind::SensitivityKind, task, ws)
    uₑ = query_element_unknown_buffer(ws.element, ws.ue)
    load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    @timeit_debug "assemble sensitivity" sensitivity_kernel!(kind, task, ws, uₑ)
end

function sensitivity_kernel!(kind::ParameterJacobianKind, task, ws, uₑ)
    cache = ws.element
    nθ = length(parameter_vector(task.p))
    Bₑ = zeros(length(ws.re), nθ)   # FIXME per-worker buffer once request kinds are declared at setup
    if implements_v2_kernels(typeof(cache)) && provides_analytic(typeof(cache), kind)
        pₑ = query_element_parameters(cache, ws.cell, ws.ivh, task.p)
        assemble_cell!(ParameterJacobianRequest(Bₑ), cache, _v2_args(ws, uₑ, pₑ))
    else
        ad_parameter_jacobian!(Bₑ, ws, uₑ, task.p)
    end
    assemble!(task.inner_assembler, ws.cell, Bₑ)
end

function sensitivity_kernel!(kind::ParameterVJPKind, task, ws, uₑ)
    cache = ws.element
    λₑ = kind.λ[celldofs(ws.cell)]   # FIXME per-worker buffer
    gₑ = zeros(length(parameter_vector(task.p)))
    if implements_v2_kernels(typeof(cache)) && provides_analytic(typeof(cache), kind)
        pₑ = query_element_parameters(cache, ws.cell, ws.ivh, task.p)
        assemble_cell!(ParameterVJPRequest(gₑ, λₑ), cache, _v2_args(ws, uₑ, pₑ))
    else
        ad_parameter_vjp!(gₑ, λₑ, ws, uₑ, task.p)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end

function sensitivity_kernel!(kind::TimeSensitivityKind, task, ws, uₑ)
    cache = ws.element
    gₑ = zeros(length(ws.re))   # FIXME per-worker buffer
    if implements_v2_kernels(typeof(cache)) && provides_analytic(typeof(cache), kind)
        pₑ = query_element_parameters(cache, ws.cell, ws.ivh, task.p)
        assemble_cell!(TimeSensitivityRequest(gₑ), cache, _v2_args(ws, uₑ, pₑ))
    else
        ad_time_sensitivity!(gₑ, ws, uₑ, kind.t)
    end
    assemble!(task.inner_assembler, ws.cell, gₑ)
end

# Shims onto the arity-dispatched legacy element interface; the v2 request
# protocol replaces them.
element_kernel!(::JacobianResidualKind, cache, ws, uₑ, pₑ) = assemble_element!(ws.Ke, ws.re, uₑ, ws.cell, cache, pₑ)
element_kernel!(::JacobianKind,         cache, ws, uₑ, pₑ) = assemble_element!(ws.Ke, uₑ, ws.cell, cache, pₑ)
element_kernel!(::ResidualKind,         cache, ws, uₑ, pₑ) = assemble_element!(ws.re, uₑ, ws.cell, cache, pₑ)
element_kernel!(::BilinearKind,         cache, ws, uₑ, pₑ) = assemble_element!(ws.Ke, ws.cell, cache, pₑ)
element_kernel!(::LinearKind,           cache, ws, uₑ, pₑ) = assemble_element!(ws.re, ws.cell, cache, pₑ)

scatter_local!(::JacobianResidualKind, assembler, ws)              = assemble!(assembler, ws.cell, ws.Ke, ws.re)
scatter_local!(::Union{JacobianKind, BilinearKind}, assembler, ws) = assemble!(assembler, ws.cell, ws.Ke)
scatter_local!(::Union{ResidualKind, LinearKind}, assembler, ws)   = assemble!(assembler, ws.cell, ws.re)

# The one assembly driver shared by every operator entry point. `out` is the
# tuple of global targets handed to `start_assemble`.
function assemble_into!(kind, out::Tuple, op, u, p)
    assembler = start_assemble(op.engine.strategy, out...)
    task = AssemblyTask(kind, assembler, u, p)
    execute_on_subdomains!(task, op.engine)
    finalize_assembly!(assembler)
end
