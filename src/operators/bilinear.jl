struct BilinearFerriteOperator{MatrixType} <: AbstractBilinearOperator
    A::MatrixType
    strategy
    subdomain_caches::Vector{SubdomainCache}
end

struct AssembleBilinearTerm{A}
    inner_assembler::A
    p
end
duplicate_for_device(device, task::AssembleBilinearTerm) = AssembleBilinearTerm(duplicate_for_device(device, task.inner_assembler), task.p)

function execute_single_task!(task::AssembleBilinearTerm, ws::AssemblyWorkspace)
    Kₑ = ws.Ke
    pₑ = query_element_parameters(ws.element, ws.cell, ws.ivh, task.p)

    fill!(Kₑ, 0.0)

    @timeit_debug "assemble element" assemble_element!(Kₑ, ws.cell, ws.element, pₑ)

    assemble!(task.inner_assembler, ws.cell, Kₑ)
end

function update_operator!(op::BilinearFerriteOperator, p)
    (; A, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, A)
    task = AssembleBilinearTerm(assembler, p)

    for (subdomain_id, sc) in enumerate(subdomain_caches)
        @timeit_debug "assemble subdomain $subdomain_id" execute_on_device!(task, strategy.device, sc.device_cache, sc.partition)
    end

    finalize_assembly!(assembler)
end

mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::BilinearFerriteOperator) = eltype(op.A)
Base.size(op::BilinearFerriteOperator, axis) = size(op.A, axis)
