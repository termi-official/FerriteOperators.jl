struct AssembleLinearizationJR{A}
    inner_assembler::A
    u
    p
end
duplicate_for_device(device, task::AssembleLinearizationJR) = AssembleLinearizationJR(duplicate_for_device(device, task.inner_assembler), task.u, task.p)

function execute_single_task!(task::AssembleLinearizationJR, ws::AssemblyWorkspace)
    Jₑ = ws.Ke
    rₑ = ws.re
    uₑ = query_element_unknown_buffer(ws.element, ws.ue)
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)

    fill!(Jₑ, 0.0)
    fill!(rₑ, 0.0)

    load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, ws.cell, ws.element, pₑ)
    store_condensed_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)

    assemble!(task.inner_assembler, ws.cell, Jₑ, rₑ)
end

struct AssembleLinearizationJ{A}
    inner_assembler::A
    u
    p
end
duplicate_for_device(device, task::AssembleLinearizationJ) = AssembleLinearizationJ(duplicate_for_device(device, task.inner_assembler), task.u, task.p)

function execute_single_task!(task::AssembleLinearizationJ, ws::AssemblyWorkspace)
    Jₑ = ws.Ke
    uₑ = query_element_unknown_buffer(ws.element, ws.ue)
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)

    fill!(Jₑ, 0.0)

    load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, ws.cell, ws.element, pₑ)
    store_condensed_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)

    assemble!(task.inner_assembler, ws.cell, Jₑ)
end

struct AssembleLinearizationR{A}
    inner_assembler::A
    u
    p
end
duplicate_for_device(device, task::AssembleLinearizationR{<:AbstractVector}) = task
duplicate_for_device(device, task::AssembleLinearizationR) = AssembleLinearizationR(duplicate_for_device(device, task.inner_assembler), task.u, task.p)

function execute_single_task!(task::AssembleLinearizationR, ws::AssemblyWorkspace)
    rₑ = ws.re
    uₑ = query_element_unknown_buffer(ws.element, ws.ue)
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)

    fill!(rₑ, 0.0)

    load_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)
    @timeit_debug "assemble element" assemble_element!(rₑ, uₑ, ws.cell, ws.element, pₑ)
    store_condensed_element_unknowns!(uₑ, task.u, ws.cell, ws.ivh, ws.element)

    assemble!(task.inner_assembler, ws.cell, rₑ)
end

"""
    LinearizedFerriteOperator(J, caches)

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_element! -> update jacobian/residual contribution with internal state variables
"""
struct LinearizedFerriteOperator{MatrixType} <: AbstractNonlinearOperator
    J::MatrixType
    strategy
    subdomain_caches::Vector{SubdomainCache}
end

# Interface
function update_linearization!(op::LinearizedFerriteOperator, u::AbstractVector, p)
    (; J, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, J)
    task = AssembleLinearizationJ(assembler, u, p)

    for (subdomain_id, sc) in enumerate(subdomain_caches)
        @timeit_debug "assemble subdomain $subdomain_id" execute_on_device!(task, strategy.device, sc.device_cache, sc.partition)
    end

    finalize_assembly!(assembler)
end
function update_linearization!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p)
    (; J, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, J, residual)
    task = AssembleLinearizationJR(assembler, u, p)

    for (subdomain_id, sc) in enumerate(subdomain_caches)
        @timeit_debug "assemble subdomain $subdomain_id" execute_on_device!(task, strategy.device, sc.device_cache, sc.partition)
    end

    finalize_assembly!(assembler)
end
function residual!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p)
    (; strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, residual)
    task = AssembleLinearizationR(assembler, u, p)

    for (subdomain_id, sc) in enumerate(subdomain_caches)
        @timeit_debug "assemble subdomain $subdomain_id" execute_on_device!(task, strategy.device, sc.device_cache, sc.partition)
    end

    finalize_assembly!(assembler)
end

"""
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear form to the vector `in`.
"""
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)
(op::LinearizedFerriteOperator)(residual, u, p) = residual!(op, residual, u, p)
Base.eltype(op::LinearizedFerriteOperator) = eltype(op.J)
Base.size(op::LinearizedFerriteOperator, axis) = size(op.J, axis)

residual_size(op::LinearizedFerriteOperator) = ndofs(op.subdomain_caches[1].sdh.dh)
unknown_size(op::LinearizedFerriteOperator)  = ndofs(op.subdomain_caches[1].sdh.dh) + ndofs(op.subdomain_caches[1].ivh)
