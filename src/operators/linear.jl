struct LinearFerriteOperator{VectorType} <: AbstractLinearOperator
    b::VectorType
    strategy
    subdomain_caches::Vector{SubdomainCache}
end

struct AssembleLinearTerm{A}
    inner_assembler::A
    p
end
duplicate_for_device(device, task::AssembleLinearTerm{<:AbstractVector}) = task
duplicate_for_device(device, task::AssembleLinearTerm) = AssembleLinearTerm(duplicate_for_device(device, task.inner_assembler), task.p)

function execute_single_task!(task::AssembleLinearTerm, ws::AssemblyWorkspace)
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)
    rₑ = ws.re

    fill!(rₑ, 0.0)

    @timeit_debug "assemble element" assemble_element!(rₑ, ws.cell, ws.element, pₑ)

    assemble!(task.inner_assembler, ws.cell, rₑ)
end

function update_operator!(op::LinearFerriteOperator, p)
    (; b, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, b)
    task = AssembleLinearTerm(assembler, p)

    for (subdomain_id, sc) in enumerate(subdomain_caches)
        @timeit_debug "assemble subdomain $subdomain_id" execute_on_device!(task, strategy.device, sc.device_cache, sc.partition)
    end

    finalize_assembly!(assembler)
end
