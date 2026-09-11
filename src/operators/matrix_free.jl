"""
    MatrixFreeFerriteOperator <: AbstractBilinearOperator

The operator [`setup_operator`](@ref) returns for an
[`AbstractBilinearIntegrator`](@ref) under [`MatrixFreeAction`](@ref): the
[`AssemblyEngine`](@ref) and integrator alone. There is no global matrix and no
stored element matrix — every `mul!` re-evaluates the action from the element
kernels ([`apply_element_action!`](@ref)) and scatters it, which is what the
MFEM ELEMENT/PARTIAL/NONE assembly levels mean. Which of the three the operator
runs at is the form's `storage` election: nothing kept, the element's own
per-quadrature-point factors, or its dense element matrices
([`ElementAssemblyCache`](@ref)) — never a global matrix.

Surface: `mul!(y, op, u)` and the five-argument form, [`evaluate!`](@ref)
(the same action with parameters and a context), `size`, `eltype`. There is no
`get_matrix` and no [`operator_payload`](@ref): an operator that stores nothing
has neither, and asking for them says so.

`y` and `u` must NOT alias. Both `mul!` forms sweep items in an arbitrary order,
writing `y` as they go while still reading `u`, so `mul!(y, op, y)` returns a
partly-updated mix rather than `A·y` — silently, there being nothing to detect
it with. Pass a separate destination.

NOT BITWISE REPRODUCIBLE where the scatter is atomic. Under
[`SequentialScheduling`](@ref) the items of one chunk accumulate into `y`
concurrently and in whatever order the hardware runs them, so two evaluations of
the same action can differ in the last bits. The sequential CPU device (one
worker) and [`ColoredScheduling`](@ref) (no atomics) do repeat bit for bit. A
Krylov method over this operator therefore sees an operator that is not a
function of `u` to the last bit; treat run-to-run iteration counts accordingly,
or elect a colouring.

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
# solver spellings (α = ±1) are exact. `β = 0` ASSIGNS rather than scales
# (`rmul!(y, 0)` would propagate a NaN/Inf already in `y`), matching the
# LinearAlgebra 5-arg `mul!` convention.
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
what the last such call put there, `setup_operator` makes that call with
`p = nothing`, and an action evaluated after `p` or the context time changed
reads stale factors until this is called again. An element whose factors depend
on neither is fresh from setup on.
"""
function update_operator!(op::MatrixFreeFerriteOperator, p, ctx = nothing)
    _keeps_storage(op.engine.strategy.form.storage) || return nothing
    run_sweep!(QuadratureDataKind(), nothing, op, (;), p, ctx)
    return nothing
end

# `Recompute()` is the one member with nothing to refill.
_keeps_storage(::Recompute) = false
_keeps_storage(::StorageElection) = true

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
Its `storage` election reaches the element caches through
[`with_assembly_form`](@ref) inside [`setup_engine`](@ref), and what it elects
to keep is FILLED here, with `p = nothing`: an operator is usable the moment it
is set up, whichever election it carries, and a `p`-dependent factor is
refreshed by [`update_operator!`](@ref).

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

"""
    assert_matrix_free_supported(form, engine)

Reject at setup an element cache that cannot serve the elected mapping and
storage level, naming the method it does not implement — the
[`provides_analytic`](@ref)/[`serves_kind`](@ref) rule applied to the
matrix-free entry points, which have no fallback to degrade to.

Which capability is required is the STORAGE election's: the two
per-quadrature-point levels reach [`apply_element_action!`](@ref) on every
action, while [`ElementAssembly`](@ref) reaches it (or the element-matrix
kernel) only at fill time and has already resolved that route when the cache was
built ([`element_matrix_fill_route`](@ref)).
"""
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

# The ELEMENT level consumes the element's kernels once per fill and its own
# store thereafter, so the action entry point is not what it needs;
# `element_matrix_fill_route` already refused a cache serving neither route.
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

"""
    _assert_mapping_capability(mapping, storage, cache)

Reject at setup an element cache that cannot serve the elected
[`AbstractElementMapping`](@ref) at the elected storage level, naming the entry
it does not implement and the mapping that would take it.

[`WorkerPerElement`](@ref) is what every cache serving the level already serves.
The other two members are each admissible at ONE end of the storage ladder and
say so here: the cooperative mapping splits an element's lattice, which the
ELEMENT level does not have, and the lane mapping splits an ELEMENT-level dense
product's rows, which the per-quadrature-point levels do not have.
"""
_assert_mapping_capability(::WorkerPerElement, ::StorageElection, cache) = nothing

# One workgroup per element splits the element's LATTICE; a dense `Kₑ·uₑ` has no
# lattice, and the store the group would read is one matrix per cell rather than
# per-lane slabs. The rejection is the storage level's, not the cache's — the
# wrapped element may well implement the cooperative pipeline.
_assert_mapping_capability(::CooperativeElement, ::ElementAssembly, cache) = throw(ArgumentError(
    "`CooperativeElement` cannot execute the `ElementAssembly` storage level: one workgroup per " *
    "element exists to split the element's lattice between lanes, and the ELEMENT level replaces " *
    "that lattice with one dense `Kₑ·uₑ` per cell. Elect `element_mapping = WorkerPerElement()` " *
    "or `element_mapping = LanesPerElement()` for `storage = ElementAssembly()`, or keep the " *
    "cooperative mapping with `storage = Stored()`/`Recompute()`."))

# The lane mapping is the ELEMENT level's, and only that level's: a lane owns one
# ROW of a dense product, while the two per-quadrature-point levels reinitialize
# the element's values objects per cell into per-worker state that the lanes of
# one element would race on.
_assert_mapping_capability(::LanesPerElement, storage::StorageElection, cache) = throw(ArgumentError(
    "`LanesPerElement` cannot execute the `$(nameof(typeof(storage)))` storage level: a lane owns " *
    "one ROW of a stored `Kₑ`, and a level that visits quadrature points instead re-derives one " *
    "element's values objects per cell — per-worker state the lanes of one element would race " *
    "on. Elect `storage = ElementAssembly()` for the lane mapping, or " *
    "`element_mapping = WorkerPerElement()`/`CooperativeElement()` for " *
    "`storage = Stored()`/`Recompute()`."))

# The row entry's subject, walked down the decorator chain: `hasmethod` on a
# decorated cache answers `true` for every inner, the blanket forward
# (ad_element.jl) being a method too — so the question passes through a forwarding
# decorator to what it wraps. `ElementAssemblyCache` is the exception and answers
# for itself: the row it serves is the STORED matrix's, not the wrapped element's.
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
