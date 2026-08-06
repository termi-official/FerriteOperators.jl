# Element-level query functions (overridable by element caches)
query_element_parameters(element, geometry_cache, ivh, p) = p
query_element_unknown_buffer(element, ue) = ue

# What a single assembly sweep computes. These select the legacy element kernel
# arity below; the v2 element interface will grow them into buffer-carrying
# requests.
struct JacobianResidualKind end     # nonlinear J(u) and r(u), fused
struct JacobianKind end             # nonlinear J(u)
struct ResidualKind end             # nonlinear r(u)
struct BilinearKind end             # u-independent matrix
struct LinearKind end               # u-independent vector

const MatrixAssemblyKind  = Union{JacobianResidualKind, JacobianKind, BilinearKind}
const VectorAssemblyKind  = Union{JacobianResidualKind, ResidualKind, LinearKind}
const UnknownDependentKind = Union{JacobianResidualKind, JacobianKind, ResidualKind}

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

function execute_single_task!(task::AssemblyTask, ws::AssemblyWorkspace)
    kind = task.kind
    kind isa MatrixAssemblyKind && fill!(ws.Ke, 0.0)
    kind isa VectorAssemblyKind && fill!(ws.re, 0.0)
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)
    if kind isa UnknownDependentKind
        uₑ = query_element_unknown_buffer(ws.element, ws.ue)
        load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
        @timeit_debug "assemble element" element_kernel!(kind, ws.element, ws, uₑ, pₑ)
        @timeit_debug "assemble boundary" element_kernel!(kind, ws.boundary_element, ws, uₑ, pₑ)
        store_condensed_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    else
        @timeit_debug "assemble element" element_kernel!(kind, ws.element, ws, nothing, pₑ)
        @timeit_debug "assemble boundary" element_kernel!(kind, ws.boundary_element, ws, nothing, pₑ)
    end
    scatter_local!(kind, task.inner_assembler, ws)
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
    assembler = start_assemble(op.strategy, out...)
    task = AssemblyTask(kind, assembler, u, p)
    execute_on_subdomains!(task, op.strategy, op.subdomain_caches)
    finalize_assembly!(assembler)
end
