struct BilinearFerriteOperator{MatrixType} <: AbstractBilinearOperator
    A::MatrixType
    strategy
    subdomain_caches::Vector{SubdomainCache}
end

struct AssembleBilinearTerm{A}
    inner_assembler::A
end
duplicate_for_device(device, task::AssembleBilinearTerm) = AssembleBilinearTerm(duplicate_for_device(device, task.inner_assembler))
function Ferrite.assemble!(task::AssembleBilinearTerm, task_buffer::GenericTaskBuffer)
    assemble!(task.inner_assembler, task_buffer.geometry_cache, task_buffer.Ke)
end
function execute_task_on_single_cell!(task::AssembleBilinearTerm, task_buffer)
    pₑ = query_element_parameters(task_buffer)
    Aₑ = query_element_matrix(task_buffer)

    fill!(Aₑ, 0.0)

    cell    = query_geometry_cache(task_buffer)
    element = query_element(task_buffer)

    @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element, pₑ)

    assemble!(task, task_buffer)
end

# GPU element assembly: launch kernel when GPUElementAssemblyStrategyCache is available
function execute_task_on_device!(task::AssembleBilinearTerm, device::AbstractGPUDevice, cache)
    strategy_cache = cache.subdomain.strategy_cache
    if strategy_cache isa GPUElementAssemblyStrategyCache
        ea_operator = task.inner_assembler.K_element
        element_cache = cache.subdomain.element
        _launch_gpu_bilinear_assembly!(device, ea_operator, strategy_cache, element_cache)
    else
        # Fallback: assemble on CPU using sequential cell loop
        task_buffer = get_task_buffer(task, cache, 1)
        for chunk in get_items(task, cache)
            for taskid in chunk
                reinit!(task_buffer, taskid)
                execute_task_on_single_cell!(task, task_buffer)
            end
        end
    end
end

function update_operator!(op::BilinearFerriteOperator, p)
    (; A, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, A)
    task = AssembleBilinearTerm(assembler)

    for (subdomain_id, subdomain_cache) in enumerate(subdomain_caches)
        # Function barrier
        task_cache = SubdomainAssemblyTaskBuffer(nothing, p, subdomain_cache)
        @timeit_debug "assemble subdomain $subdomain_id" execute_task_on_device!(task, strategy.device, task_cache)
    end

    finalize_assembly!(assembler)
end

mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::BilinearFerriteOperator) = eltype(op.A)
Base.size(op::BilinearFerriteOperator, axis) = sisze(op.A, axis)
