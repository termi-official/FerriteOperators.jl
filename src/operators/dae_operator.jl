@concrete struct AssembleDAELinearizationJR
    inner_assembler_u
    inner_assembler_du
    du
    u
    p
end
duplicate_for_device(device, task::AssembleDAELinearizationJR) = AssembleDAELinearizationJR(
    duplicate_for_device(device, task.inner_assembler_u),
    duplicate_for_device(device, task.inner_assembler_du),
    task.u,
    task.p,
)

function execute_single_task!(task::AssembleDAELinearizationJR, ws::AssemblyWorkspace)
    Juₑ = ws.Ke
    Jduₑ = copy(ws.Ke) # FIXME
    rₑ = ws.re
    uₑ = query_element_unknown_buffer(ws.element, ws.ue)
    duₑ= copy(uₑ) # FIXME
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)

    fill!(Juₑ, 0.0)
    fill!(Jduₑ, 0.0)
    fill!(rₑ, 0.0)

    load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    load_element_unknowns!(duₑ, task.du, ws.cell, ws.ivh, ws.element)
    @timeit_debug "assemble element" assemble_dae_element!(Jduₑ, Juₑ, rₑ, duₑ uₑ, ws.cell, ws.element, pₑ)
    @timeit_debug "assemble boundary" assemble_dae_element!(Jduₑ, Juₑ, rₑ, duₑ, uₑ, ws.cell, ws.boundary_element, pₑ)
    store_condensed_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    store_condensed_element_unknowns!(duₑ, task.du, ws.cell, ws.ivh, ws.element)

    assemble!(task.inner_assembler_u, ws.cell, Juₑ, rₑ)
    assemble!(task.inner_assembler_du, ws.cell, Jduₑ)
end

@concrete struct AssembleDAELinearizationJ
    inner_assembler_u
    inner_assembler_du
    du
    u
    p
end
duplicate_for_device(device, task::AssembleDAELinearizationJ) = AssembleDAELinearizationJ(duplicate_for_device(device, task.inner_assembler), task.u, task.p)

function execute_single_task!(task::AssembleDAELinearizationJ, ws::AssemblyWorkspace)
    Juₑ = ws.Ke
    Jduₑ = copy(ws.Ke) # FIXME
    uₑ = query_element_unknown_buffer(ws.element, ws.ue)
    duₑ= copy(uₑ) # FIXME
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)

    fill!(Juₑ, 0.0)
    fill!(Jduₑ, 0.0)

    load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    load_element_unknowns!(duₑ, task.du, ws.cell, ws.ivh, ws.element)
    @timeit_debug "assemble element" assemble_dae_element!(Jduₑ, Juₑ, duₑ, uₑ, ws.cell, ws.element, pₑ)
    @timeit_debug "assemble boundary" assemble_dae_element!(Jduₑ, Juₑ, duₑ, uₑ, ws.cell, ws.boundary_element, pₑ)
    store_condensed_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    store_condensed_element_unknowns!(duₑ, task.du, ws.cell, ws.ivh, ws.element)

    assemble!(task.inner_assembler_u, ws.cell, Juₑ)
    assemble!(task.inner_assembler_du, ws.cell, Jduₑ)
end

@concrete struct AssembleDAELinearizationR
    inner_assembler
    du
    u
    p
end
duplicate_for_device(device, task::AssembleDAELinearizationR{<:AbstractVector}) = task
duplicate_for_device(device, task::AssembleDAELinearizationR) = AssembleDAELinearizationR(duplicate_for_device(device, task.inner_assembler), task.u, task.p)

function execute_single_task!(task::AssembleDAELinearizationR, ws::AssemblyWorkspace)
    rₑ = ws.re
    uₑ = query_element_unknown_buffer(ws.element, ws.ue)
    duₑ= copy(uₑ) # FIXME
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)

    fill!(rₑ, 0.0)

    load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    load_element_unknowns!(duₑ, task.du, ws.cell, ws.ivh, ws.element)
    @timeit_debug "assemble element" assemble_dae_element!(rₑ, duₑ, uₑ, ws.cell, ws.element, pₑ)
    @timeit_debug "assemble boundary" assemble_dae_element!(rₑ, duₑ, uₑ, ws.cell, ws.boundary_element, pₑ)
    store_condensed_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    store_condensed_element_unknowns!(duₑ, task.du, ws.cell, ws.ivh, ws.element)

    assemble!(task.inner_assembler, ws.cell, rₑ)
end

"""
    LinearizedFerriteDAEOperator(J, caches)

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_dae_element! -> update jacobian/residual contribution with internal state variables
"""
@concrete struct LinearizedFerriteDAEOperator <: AbstractNonlinearOperator
    Ju
    Jdu
    strategy
    subdomain_caches
    dh
    integrator
end

# Interface
function update_linearization!(op::LinearizedFerriteDAEOperator, du::AbstractVector, u::AbstractVector, p)
    (; Ju, Jdu, strategy, subdomain_caches) = op

    assembler1 = start_assemble(strategy, Ju)
    assembler2 = start_assemble(strategy, Jdu)
    task = AssembleDAELinearizationJ(assembler1, assembler2, du, u, p)

    execute_on_subdomains!(task, strategy, subdomain_caches)

    finalize_assembly!(assembler)
end
function update_linearization!(op::LinearizedFerriteDAEOperator, residual::AbstractVector, du::AbstractVector, u::AbstractVector, p)
    (; Ju, Jdu, strategy, subdomain_caches) = op

    assembler1 = start_assemble(strategy, Ju, residual)
    assembler2 = start_assemble(strategy, Jdu)
    task = AssembleDAELinearizationJR(assembler1, assembler2, du, u, p)

    execute_on_subdomains!(task, strategy, subdomain_caches)

    finalize_assembly!(assembler)
end
function residual!(op::LinearizedFerriteDAEOperator, residual::AbstractVector, du::AbstractVector, u::AbstractVector, p)
    (; strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, residual)
    task = AssembleDAELinearizationR(assembler, du, u, p)

    execute_on_subdomains!(task, strategy, subdomain_caches)

    finalize_assembly!(assembler)
end

Base.eltype(op::LinearizedFerriteDAEOperator) = eltype(op.J)
Base.size(op::LinearizedFerriteDAEOperator) = size(op.J)
Base.size(op::LinearizedFerriteDAEOperator, axis) = size(op.J, axis)

residual_size(op::LinearizedFerriteDAEOperator) = ndofs(op.subdomain_caches[1].domain.sdh.dh)
unknown_size(op::LinearizedFerriteDAEOperator)  = ndofs(op.subdomain_caches[1].domain.sdh.dh) + ndofs(op.subdomain_caches[1].domain.ivh)
