"""
    MatrixFreeFerriteOperator <: AbstractBilinearOperator

The operator [`setup_operator`](@ref) returns for an
[`AbstractBilinearIntegrator`](@ref) under [`MatrixFreeAction`](@ref): the
[`AssemblyEngine`](@ref) and integrator alone. There is no global matrix and no
stored element matrix — every `mul!` re-evaluates the action from the element
kernels ([`apply_element_action!`](@ref)) and scatters it, which is what the
MFEM PARTIAL/NONE assembly level means.

Surface: `mul!(y, op, u)` and the five-argument form, [`evaluate!`](@ref)
(the same action with parameters and a context), `size`, `eltype`. There is no
`get_matrix` and no [`operator_payload`](@ref): an operator that stores nothing
has neither, and asking for them says so.

!!! warning "Experimental surface"
    This operator, its entry points and the element hooks it calls may change
    in a minor release.
"""
@concrete struct MatrixFreeFerriteOperator <: AbstractBilinearOperator
    engine
    integrator
    ndofs::Int
end

Base.size(op::MatrixFreeFerriteOperator) = (op.ndofs, op.ndofs)
Base.size(op::MatrixFreeFerriteOperator, axis) = axis <= 2 ? op.ndofs : 1
Base.eltype(op::MatrixFreeFerriteOperator) = value_type(op.engine.strategy.device)

"""
    evaluate!(op::MatrixFreeFerriteOperator, y, states, p, ctx)
    evaluate!(op::MatrixFreeFerriteOperator, y, u::AbstractVector, p)

The operator's action `y = A·u`, evaluated from the element action kernels and
scattered through this package's own `VectorAssembler` — the entry point that
carries the parameters and the sweep context an element kernel may read. `y` is
zeroed first, so this OVERWRITES rather than accumulates.
"""
evaluate!(op::MatrixFreeFerriteOperator, y::AbstractVector, states::NamedTuple, p, ctx) =
    assemble_into!(MatrixFreeActionKind(), (y,), op, states, p, ctx)
evaluate!(op::MatrixFreeFerriteOperator, y::AbstractVector, u::AbstractVector, p) =
    evaluate!(op, y, (u = u,), p, nothing)

mul!(y::AbstractVector, op::MatrixFreeFerriteOperator, u::AbstractVector) =
    evaluate!(op, y, (u = u,), nothing, nothing)

# The action is linear in `u`, so `α` rides the accumulator instead of the
# element kernels: scale the incoming `y` by `β/α`, accumulate the unscaled
# action into it, and scale the sum by `α`. No temporary, and the common
# solver spellings (α = ±1) are exact.
function mul!(y::AbstractVector, op::MatrixFreeFerriteOperator, u::AbstractVector, α, β)
    iszero(α) && return rmul!(y, β)
    rmul!(y, β / α)
    run_sweep!(MatrixFreeActionKind(), start_assemble(op.engine.strategy, y; fillzero = false),
               op, (u = u,), nothing, nothing)
    return rmul!(y, α)
end

"""
    update_operator!(op::MatrixFreeFerriteOperator, p, ctx = nothing)

Nothing to do: a matrix-free operator holds no assembled array, and its action
reads `p`/`ctx` where it is evaluated ([`evaluate!`](@ref)). Present so the
operator composes into the [`AbstractBilinearOperator`](@ref) entry points.
"""
update_operator!(::MatrixFreeFerriteOperator, p, ctx = nothing) = nothing

# The three-argument form is `update_operator!` and inherited; this one carries
# `p` into the action instead of dropping it.
update_linearization!(op::MatrixFreeFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    evaluate!(op, residual, (u = u,), p, nothing)

####################################
## Setup
####################################

"""
    setup_operator(strategy::AssemblyStrategy{<:MatrixFreeAction}, integrator, dh; …)

Build the [`MatrixFreeFerriteOperator`](@ref) for a bilinear `integrator`: the
same [`AssemblyEngine`](@ref) every other form builds, with no global matrix
allocated.

The form's [`AbstractElementMapping`](@ref) is resolved onto the device here
([`with_element_mapping`](@ref)) — the engine's strategy therefore carries the
mapping on BOTH axes, the form's election and the device's realization of it,
and the per-worker scratch the engine allocates is the one that mapping needs.
Only the bilinear family takes this form: a nonlinear residual is not the
action of a stored operator, and a linear form has no `u` to act on.
"""
function setup_operator(strategy::AssemblyStrategy{<:MatrixFreeAction},
        integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler;
        slots = (:u,), requests::Tuple = (), ad_backend = ForwardDiffAD())
    :u in slots || throw(ArgumentError(
        "A `MatrixFreeAction` operator acts on the `:u` slot, which the declared slots " *
        "$(Tuple(slots)) do not carry."))
    execution = AssemblyStrategy(strategy.form, strategy.scheduling,
                                 with_element_mapping(strategy.device, strategy.form.element_mapping))
    engine = setup_engine(execution, integrator, dh; slots, requests, ad_backend)
    assert_matrix_free_supported(execution.form.element_mapping, engine)
    return MatrixFreeFerriteOperator(engine, integrator, ndofs(dh))
end

setup_operator(::AssemblyStrategy{<:MatrixFreeAction}, integrator::AbstractNonlinearIntegrator,
        ::AbstractDofHandler; kwargs...) = throw(ArgumentError(
    "`MatrixFreeAction` is the action of the operator a BILINEAR form induces (got " *
    "$(nameof(typeof(integrator)))). A nonlinear integrator's residual is not that action, and " *
    "its Jacobian action is `state_jvp!` on a `LinearizedFerriteOperator`."))

setup_operator(::AssemblyStrategy{<:MatrixFreeAction}, integrator::AbstractLinearIntegrator,
        ::AbstractDofHandler; kwargs...) = throw(ArgumentError(
    "`MatrixFreeAction` is the action of the operator a BILINEAR form induces (got " *
    "$(nameof(typeof(integrator)))). A linear form has no argument to act on; assemble its " *
    "vector under `FullAssembly`."))

"""
    assert_matrix_free_supported(mapping, engine)

Reject at setup an element cache that cannot serve the elected mapping,
naming the method it does not implement — the
[`provides_analytic`](@ref)/[`serves_kind`](@ref) rule applied to the
matrix-free entry points, which have no fallback to degrade to.
"""
function assert_matrix_free_supported(mapping::AbstractElementMapping, engine::AssemblyEngine)
    for sc in engine.subdomain_caches
        sc.contributes || continue
        sc.domain isa AssemblyDomain || throw(ArgumentError(
            "`MatrixFreeAction` covers the CELL item family only; this operator carries a " *
            "$(nameof(typeof(sc.domain))). Assemble the operator under `FullAssembly`."))
        _assert_element_action(typeof(unwrap(sc.domain.element)))
        _assert_cooperative_element(mapping, typeof(unwrap(sc.domain.element)))
    end
    return nothing
end

function _assert_element_action(::Type{C}) where {C}
    hasmethod(apply_element_action!, Tuple{AbstractVector, C, AbstractVector, CellArgs}) || throw(ArgumentError(
        "$(C) implements no `apply_element_action!(yₑ::AbstractVector, ::$(nameof(C)), " *
        "uₑ::AbstractVector, ::CellArgs)` method, so it cannot serve a `MatrixFreeAction` " *
        "operator. The action is a separate entry point from the mandatory residual kernel " *
        "because declaring it is the element's promise that `Kₑ·uₑ` is evaluated without " *
        "forming `Kₑ`. Assemble this integrator under `FullAssembly` instead."))
    return nothing
end

_assert_cooperative_element(::WorkerPerElement, ::Type) = nothing

function _assert_cooperative_element(::CooperativeElement, ::Type{C}) where {C}
    entries = ((cooperative_lattice_dim,   Tuple{C},                          "(::$(nameof(C)))"),
               (cooperative_group_size,    Tuple{C},                          "(::$(nameof(C)))"),
               (cooperative_scratch_shape, Tuple{C},                          "(::$(nameof(C)))"),
               (cooperative_load!,         Tuple{Any, C, Any, Int, Int},      "(scratch, ::$(nameof(C)), uₑ, lane::Int, nlanes::Int)"),
               (cooperative_stage!,        Tuple{Any, C, Any, Int, Int, Int}, "(scratch, ::$(nameof(C)), args, stage::Int, lane::Int, nlanes::Int)"),
               (cooperative_store!,        Tuple{Any, Any, C, Int, Int},      "(yₑ, scratch, ::$(nameof(C)), lane::Int, nlanes::Int)"))
    for (entry, signature, spelling) in entries
        hasmethod(entry, signature) || throw(ArgumentError(
            "$(C) implements no `$(nameof(entry))$spelling` method, so it cannot serve " *
            "`CooperativeElement`: mapping one element onto a cooperating workgroup needs the " *
            "element's own lattice split, which no generic route can derive from the " *
            "whole-element kernel. Elect `element_mapping = WorkerPerElement()`, which this " *
            "cache does serve."))
    end
    return nothing
end
