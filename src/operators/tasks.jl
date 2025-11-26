struct AssembleLinearizationJR{A}
    inner_assembler::A
end
duplicate_for_device(device, task::AssembleLinearizationJR) = AssembleLinearizationJR(duplicate_for_device(device, task.inner_assembler))
function Ferrite.assemble!(task::AssembleLinearizationJR, task_buffer::GenericTaskBuffer)
    assemble!(task.inner_assembler, task_buffer.geometry_cache, task_buffer.Ke, task_buffer.re)
end
function execute_task_on_single_cell!(task::AssembleLinearizationJR, task_buffer)
    Jₑ = query_element_matrix(task_buffer)
    rₑ = query_element_residual_buffer(task_buffer)
    uₑ = query_element_unknown_buffer(task_buffer)
    pₑ = query_element_parameters(task_buffer)

    fill!(Jₑ, 0.0)
    fill!(rₑ, 0.0)

    cell_cache = query_geometry_cache(task_buffer)
    element    = query_element(task_buffer)

    load_element_unknowns!(uₑ, task_buffer)
    @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell_cache, element, pₑ)
    store_condensed_element_unknowns!(uₑ, task_buffer)

    assemble!(task, task_buffer)
end

struct AssembleLinearizationJ{A}
    inner_assembler::A
end
duplicate_for_device(device, task::AssembleLinearizationJ) = AssembleLinearizationJ(duplicate_for_device(device, task.inner_assembler))
function Ferrite.assemble!(task::AssembleLinearizationJ, task_buffer::GenericTaskBuffer)
    assemble!(task.inner_assembler, task_buffer.geometry_cache, task_buffer.Ke)
end
function execute_task_on_single_cell!(task::AssembleLinearizationJ, task_buffer)
    Jₑ = query_element_matrix(task_buffer)
    uₑ = query_element_unknown_buffer(task_buffer)
    pₑ = query_element_parameters(task_buffer)

    fill!(Jₑ, 0.0)

    cell_cache = query_geometry_cache(task_buffer)
    element    = query_element(task_buffer)

    load_element_unknowns!(uₑ, task_buffer)
    @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell_cache, element, pₑ)
    store_condensed_element_unknowns!(uₑ, task_buffer)

    assemble!(task, task_buffer)
end

struct AssembleLinearizationR{A}
    inner_assembler::A
end
duplicate_for_device(device, task::AssembleLinearizationR{<:AbstractVector}) = task
duplicate_for_device(device, task::AssembleLinearizationR) = AssembleLinearizationR(duplicate_for_device(device, task.inner_assembler))
function Ferrite.assemble!(task::AssembleLinearizationR, task_buffer::GenericTaskBuffer)
    assemble!(task.inner_assembler, task_buffer.geometry_cache, task_buffer.re)
end
function execute_task_on_single_cell!(task::AssembleLinearizationR, task_buffer)
    rₑ = query_element_residual_buffer(task_buffer)
    uₑ = query_element_unknown_buffer(task_buffer)
    pₑ = query_element_parameters(task_buffer)

    fill!(rₑ, 0.0)

    cell_cache = query_geometry_cache(task_buffer)
    element    = query_element(task_buffer)

    load_element_unknowns!(uₑ, task_buffer)
    @timeit_debug "assemble element" assemble_element!(rₑ, uₑ, cell_cache, element, pₑ)
    store_condensed_element_unknowns!(uₑ, task_buffer)

    assemble!(task, task_buffer)
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
    task = AssembleLinearizationJ(assembler)

    for (subdomain_id, subdomain_cache) in enumerate(subdomain_caches)
        # Function barrier
        task_cache = SubdomainAssemblyTaskBuffer(u, p, subdomain_cache)
        @timeit_debug "assemble subdomain $subdomain_id" execute_task_on_device!(task, strategy.device, task_cache)
    end

    finalize_assembly!(assembler)
end
function update_linearization!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p)
    (; J, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, J, residual)
    task = AssembleLinearizationJR(assembler)

    for (subdomain_id, subdomain_cache) in enumerate(subdomain_caches)
        # Function barrier
        task_cache = SubdomainAssemblyTaskBuffer(u, p, subdomain_cache)
        @timeit_debug "assemble subdomain $subdomain_id" execute_task_on_device!(task, strategy.device, task_cache)
    end

    finalize_assembly!(assembler)
end
function residual!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p)
    (; strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, residual)
    task = AssembleLinearizationR(assembler)

    for (subdomain_id, subdomain_cache) in enumerate(subdomain_caches)
        # Function barrier
        task_cache = SubdomainAssemblyTaskBuffer(u, p, subdomain_cache)
        @timeit_debug "assemble subdomain $subdomain_id" execute_task_on_device!(task, strategy.device, task_cache)
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

#################################################################################################

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


###############################################################################

struct LinearFerriteOperator{VectorType} <: AbstractLinearOperator
    b::VectorType
    strategy
    subdomain_caches::Vector{SubdomainCache}
end

struct AssembleLinearTerm{A}
    inner_assembler::A
end
duplicate_for_device(device, task::AssembleLinearTerm{<:AbstractVector}) = task
duplicate_for_device(device, task::AssembleLinearTerm) = AssembleLinearTerm(duplicate_for_device(device, task.inner_assembler))
function Ferrite.assemble!(task::AssembleLinearTerm, task_buffer::GenericTaskBuffer)
    assemble!(task.inner_assembler, task_buffer.geometry_cache, task_buffer.re)
end
function execute_task_on_single_cell!(task::AssembleLinearTerm, task_buffer)
    pₑ = query_element_parameters(task_buffer)
    rₑ = query_element_residual_buffer(task_buffer)

    fill!(rₑ, 0.0)

    cell    = query_geometry_cache(task_buffer)
    element = query_element(task_buffer)

    @timeit_debug "assemble element" assemble_element!(bₑ, cell, element, pₑ)

    assemble!(task, task_buffer)
end

function update_operator!(op::LinearFerriteOperator, p)
    (; b, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, b)
    task = AssembleLinearTerm(assembler)

    for (subdomain_id, subdomain_cache) in enumerate(subdomain_caches)
        # Function barrier
        task_cache = SubdomainAssemblyTaskBuffer(nothing, p, subdomain_cache)
        @timeit_debug "assemble subdomain $subdomain_id" execute_task_on_device!(task, strategy.device, task_cache)
    end

    finalize_assembly!(assembler)
end
