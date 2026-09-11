"""
    MatrixFreeFerriteOperator <: AbstractBilinearOperator

The operator [`setup_operator`](@ref) returns for an
[`AbstractBilinearIntegrator`](@ref) under [`MatrixFreeAction`](@ref): the
[`AssemblyEngine`](@ref) and integrator alone. There is no global matrix —
every `mul!` re-evaluates the action from the element kernels
([`apply_element_action!`](@ref)) and scatters it. Which of the MFEM
ELEMENT/PARTIAL/NONE levels it runs at is the form's `storage` election.

Surface: `mul!(y, op, u)` and the five-argument form, [`evaluate!`](@ref)
(the same action with parameters and a context), `size`, `eltype`. There is no
`get_matrix` and no [`operator_payload`](@ref): an operator that stores nothing
has neither.

`y` and `u` must NOT alias. Both `mul!` forms sweep items in an arbitrary order,
writing `y` as they go while still reading `u`, so `mul!(y, op, y)` silently
returns a partly-updated mix rather than `A·y`. Pass a separate destination.

NOT BITWISE REPRODUCIBLE where the scatter is atomic. Under
[`SequentialScheduling`](@ref) the items of one chunk accumulate into `y` in
whatever order the hardware runs them, so two evaluations of the same action can
differ in the last bits. The sequential CPU device (one worker) and
[`ColoredScheduling`](@ref) (no atomics) do repeat bit for bit. A Krylov method
over this operator therefore sees an operator that is not a function of `u` to
the last bit; treat run-to-run iteration counts accordingly, or elect a
colouring.

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

The operator's action `y = A·u`, carrying the parameters and the sweep context
an element kernel may read. `y` is zeroed first, so this OVERWRITES.
"""
evaluate!(op::MatrixFreeFerriteOperator, y::AbstractVector, states::NamedTuple, p, ctx) =
    assemble_into!(MatrixFreeActionKind(), (y,), op, states, p, ctx)
evaluate!(op::MatrixFreeFerriteOperator, y::AbstractVector, u::AbstractVector, p) =
    evaluate!(op, y, (u = u,), p, nothing)

mul!(y::AbstractVector, op::MatrixFreeFerriteOperator, u::AbstractVector) =
    evaluate!(op, y, (u = u,), nothing, nothing)

# The action is linear in `u`, so `α` rides the accumulator: scale the incoming
# `y` by `β/α`, accumulate the unscaled action into it, and scale the sum by
# `α`. No temporary, and the common solver spellings (α = ±1) are exact.
# `β = 0` ASSIGNS rather than scales (`rmul!(y, 0)` would propagate a NaN/Inf
# already in `y`), matching the LinearAlgebra 5-arg `mul!` convention.
function mul!(y::AbstractVector, op::MatrixFreeFerriteOperator, u::AbstractVector, α, β)
    if iszero(β)
        fill!(y, zero(eltype(y)))
        iszero(α) && return y
    else
        iszero(α) && return rmul!(y, β)
        rmul!(y, β / α)
    end
    run_sweep!(MatrixFreeActionKind(), start_assemble(op.engine.strategy, y; fillzero = false),
               op, (u = u,), nothing, nothing)
    return rmul!(y, α)
end

"""
    update_operator!(op::MatrixFreeFerriteOperator, p, ctx = nothing)

Refill what the operator's `storage` election keeps
([`MatrixFreeAction`](@ref)) — the elements' per-quadrature-point factors under
[`Stored`](@ref), the dense element matrices under [`ElementAssembly`](@ref) —
with one [`QuadratureDataKind`](@ref) sweep carrying `p` and `ctx` to
[`fill_quadrature_data!`](@ref). Under [`Recompute`](@ref) nothing is kept and
this does nothing.

FRESHNESS IS THE CALLER'S, exactly as for an assembled operator: the store holds
what the last such call put there (`setup_operator` makes one with
`p = nothing`), and an action evaluated after `p` or the context time changed
reads stale factors until this is called again.
"""
function update_operator!(op::MatrixFreeFerriteOperator, p, ctx = nothing)
    _keeps_storage(op.engine.strategy.form.storage) || return nothing
    run_sweep!(QuadratureDataKind(), nothing, op, (;), p, ctx)
    return nothing
end

_keeps_storage(::Recompute) = false
_keeps_storage(::StorageElection) = true

update_linearization!(op::MatrixFreeFerriteOperator, residual::AbstractVector, u::AbstractVector, p) =
    evaluate!(op, residual, (u = u,), p, nothing)

####################################
## Setup
####################################

"""
    setup_operator(strategy::AssemblyStrategy{<:MatrixFreeAction}, integrator, dh; …)

Build the [`MatrixFreeFerriteOperator`](@ref) for a bilinear `integrator`: the
same [`AssemblyEngine`](@ref) every other form builds, with no global matrix.

The form's [`AbstractElementMapping`](@ref) is resolved onto the device here
([`with_element_mapping`](@ref)); the `storage` election reaches the element
caches through [`with_assembly_form`](@ref), and what it keeps is FILLED here
with `p = nothing`, so an operator is usable the moment it is set up.

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
    assert_matrix_free_supported(execution.form, engine)
    op = MatrixFreeFerriteOperator(engine, integrator, ndofs(dh))
    update_operator!(op, nothing, nothing)
    return op
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

function assert_matrix_free_supported(form::MatrixFreeAction, engine::AssemblyEngine)
    for sc in engine.subdomain_caches
        sc.contributes || continue
        sc.domain isa AssemblyDomain || throw(ArgumentError(
            "`MatrixFreeAction` covers the CELL item family only; this operator carries a " *
            "$(nameof(typeof(sc.domain))). Assemble the operator under `FullAssembly`."))
        _assert_action_capability(form.storage, typeof(unwrap(sc.domain.element)))
        _assert_mapping_capability(form.element_mapping, form.storage, sc.domain.element)
    end
    return nothing
end

# The ELEMENT level needs no action entry point: `element_matrix_fill_route`
# already refused a cache serving neither fill route.
_assert_action_capability(::ElementAssembly, ::Type) = nothing
_assert_action_capability(::StorageElection, ::Type{C}) where {C} = _assert_element_action(C)

function _assert_element_action(::Type{C}) where {C}
    hasmethod(apply_element_action!, Tuple{AbstractVector, C, AbstractVector, CellArgs}) || throw(ArgumentError(
        "$(C) implements no `apply_element_action!(yₑ::AbstractVector, ::$(nameof(C)), " *
        "uₑ::AbstractVector, ::CellArgs)` method, so it cannot serve a `MatrixFreeAction` " *
        "operator. The action is a separate entry point from the mandatory residual kernel " *
        "because declaring it is the element's promise that `Kₑ·uₑ` is evaluated without " *
        "forming `Kₑ`. Assemble this integrator under `FullAssembly` instead."))
    return nothing
end

_assert_mapping_capability(::WorkerPerElement, ::StorageElection, cache) = nothing

# The rejection is the storage level's, not the cache's — the wrapped element
# may well implement the cooperative pipeline.
_assert_mapping_capability(::CooperativeElement, ::ElementAssembly, cache) = throw(ArgumentError(
    "`CooperativeElement` cannot execute the `ElementAssembly` storage level: one workgroup per " *
    "element exists to split the element's lattice between lanes, and the ELEMENT level replaces " *
    "that lattice with one dense `Kₑ·uₑ` per cell. Elect `element_mapping = WorkerPerElement()` " *
    "or `element_mapping = LanesPerElement()` for `storage = ElementAssembly()`, or keep the " *
    "cooperative mapping with `storage = Stored()`/`Recompute()`."))

_assert_mapping_capability(::LanesPerElement, storage::StorageElection, cache) = throw(ArgumentError(
    "`LanesPerElement` cannot execute the `$(nameof(typeof(storage)))` storage level: a lane owns " *
    "one ROW of a stored `Kₑ`, and a level that visits quadrature points instead re-derives one " *
    "element's values objects per cell — per-worker state the lanes of one element would race " *
    "on. Elect `storage = ElementAssembly()` for the lane mapping, or " *
    "`element_mapping = WorkerPerElement()`/`CooperativeElement()` for " *
    "`storage = Stored()`/`Recompute()`."))

# Walked down the decorator chain: the blanket forward (ad_element.jl) is a
# method too, so `hasmethod` on a decorated cache would answer `true` for every
# inner. `ElementAssemblyCache` answers for itself.
_declares_element_action_row(d::AbstractElementCacheDecorator) = _declares_element_action_row(d.inner)
_declares_element_action_row(::ElementAssemblyCache) = true
_declares_element_action_row(cache) =
    hasmethod(element_action_row, Tuple{typeof(cache), Any, CellArgs, Int})

function _assert_mapping_capability(::LanesPerElement, ::ElementAssembly, cache)
    C = typeof(cache)
    _declares_element_action_row(cache) || throw(ArgumentError(
        "$(C) implements no `element_action_row(::$(nameof(C)), uₑ, ::CellArgs, i::Int)` method, " *
        "so it cannot serve `LanesPerElement`: the mapping gives one lane one ROW of the " *
        "element's action, which no generic route can derive from the whole-element kernel. " *
        "Elect `element_mapping = WorkerPerElement()`, which this cache does serve."))
    element_local_length(cache) isa Val || throw(ArgumentError(
        "$(C) names no compile-time `element_local_length`, so it cannot serve " *
        "`LanesPerElement`: the lane count is a launch geometry derived from the element's " *
        "extent, and a `Val` is what makes it a constant of the kernel rather than a runtime " *
        "trip count. Elect `element_mapping = WorkerPerElement()`."))
    return nothing
end

function _assert_mapping_capability(::CooperativeElement, ::StorageElection, cache)
    C = typeof(unwrap(cache))
    return _assert_cooperative_entries(C)
end

function _assert_cooperative_entries(::Type{C}) where {C}
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
