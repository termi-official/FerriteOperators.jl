struct LinearFerriteOperator{VectorType} <: AbstractLinearOperator
    b::VectorType
    strategy
    subdomain_caches::Vector{<:SubdomainCache}
end

struct AssembleLinearTerm{A}
    inner_assembler::A
end
duplicate_for_device(device, task::AssembleLinearTerm{<:AbstractVector}) = task
duplicate_for_device(device, task::AssembleLinearTerm) = AssembleLinearTerm(duplicate_for_device(device, task.inner_assembler))
function Ferrite.assemble!(task::AssembleLinearTerm, task_buffer::GenericTaskBuffer)
    assemble!(task.inner_assembler, task_buffer.geometry_cache, query_element_residual_buffer(task_buffer))
end
function execute_task_on_single_cell!(task::AssembleLinearTerm, task_buffer)
    pₑ = query_element_parameters(task_buffer)
    rₑ = query_element_residual_buffer(task_buffer)

    fill!(rₑ, 0.0)

    cell    = query_geometry_cache(task_buffer)
    element = query_element(task_buffer)

    # TODO: re-enable when TimerOutputs is GPU-compatible
    # @timeit_debug "assemble element" assemble_element!(bₑ, cell, element, pₑ)
    assemble_element!(bₑ, cell, element, pₑ)

    assemble!(task, task_buffer)
end

function update_operator!(op::LinearFerriteOperator, p)
    (; b, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, b)
    task = AssembleLinearTerm(assembler)

    for (subdomain_id, subdomain_cache) in enumerate(subdomain_caches)
        # Function barrier
        task_cache = SubdomainAssemblyTaskBuffer(nothing, p, subdomain_cache)
        @timeit_debug "assemble subdomain $subdomain_id" execute_task_on_device!(task, subdomain_cache.strategy_cache.device, task_cache)
    end

    finalize_assembly!(assembler)
end
