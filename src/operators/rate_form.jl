####################################
## The rate form: `M⁻¹ · rhs`
####################################

"""
    BilinearRateFormIntegrator(rhs, mass)

The rate form of a BILINEAR right-hand side — `RateFormIntegrator(rhs, mass)`
elects it. See [`RateFormIntegrator`](@ref).
"""
struct BilinearRateFormIntegrator{R <: AbstractBilinearIntegrator, M} <: AbstractBilinearIntegrator
    rhs::R
    mass::M
end

"""
    LinearRateFormIntegrator(rhs, mass)

The rate form of a LINEAR right-hand side — `RateFormIntegrator(rhs, mass)`
elects it. See [`RateFormIntegrator`](@ref).
"""
struct LinearRateFormIntegrator{R <: AbstractLinearIntegrator, M} <: AbstractLinearIntegrator
    rhs::R
    mass::M
end

const AnyRateFormIntegrator = Union{BilinearRateFormIntegrator, LinearRateFormIntegrator}

"""
    RateFormIntegrator(rhs, mass)

The INVERSE-MASS-WEIGHTED (rate) form of `rhs`: the term whose operator is

    action(u) = M⁻¹ · (A · u)      # `rhs` a bilinear form, `A` the operator it induces
    payload   = M⁻¹ · b            # `rhs` a linear form, `b` its load vector

One spelling; which type it builds follows `rhs`'s own family
([`BilinearRateFormIntegrator`](@ref), [`LinearRateFormIntegrator`](@ref)),
since a bilinear right-hand side induces an operator and a linear one a vector.
`mass` is an ordinary bilinear MASS integrator — the model's own, lumped
([`RowSumLumped`](@ref)) or not. The PAIR is the whole declaration: how `M⁻¹` is
realized is decided here and never by the caller.

**The mass is checked STRUCTURALLY at setup, never approximated silently.**
[`element_matrix_structure`](@ref) of the mass's element cache and the space's
dof sharing decide what `M⁻¹` is:

| mass structure | space | `M⁻¹` is |
|---|---|---|
| [`DiagonalElementMatrix`](@ref) | any | the reciprocal of the ASSEMBLED global diagonal |
| [`DenseElementMatrix`](@ref) | cell-disjoint dofs (discontinuous) | the per-cell inverse blocks, [`ElementInverse`](@ref)`(mass)` under [`ElementAssembly`](@ref) |
| [`DenseElementMatrix`](@ref) | continuous | REFUSED at setup |

A dense mass over a continuous space has no cell-local inverse, and this
package will not substitute one: lump the mass (`RowSumLumped(mass)`), or give
the model a mass that is diagonal by construction — a collocated (spectral)
element declaring [`DiagonalElementMatrix`](@ref). Both are DISCRETIZATION
decisions and both are spelled by the caller.

**The operator is a COMPOSITION.** The rhs is assembled exactly as it would be
alone, under the caller's strategy, and `M⁻¹` is applied to its result:

| form / storage | realization |
|---|---|
| [`FullAssembly`](@ref), [`MatrixFreeAction`](@ref) + [`Stored`](@ref)/[`Recompute`](@ref)/[`ElementAssembly`](@ref) | `A·u`, then the reciprocal scaling (diagonal mass) or the block-inverse action (dense mass) |
| [`MatrixFreeAction`](@ref) + [`BlockRowAssembly`](@ref) | `M⁻¹` fused into the block-row store at fill ([`finalize_action_storage!`](@ref)); nothing is applied at action time |
| linear `rhs` | `b ← M⁻¹b` at every fill — under an ASSEMBLING form, a load vector having no action to evaluate |

No fused matrix is formed: `get_matrix` of the rate operator is an error, the
rhs's own operator is [`rate_form_rhs`](@ref) and `M⁻¹` is
[`rate_form_inverse_mass`](@ref).

`M⁻¹` is as fresh as the operator's last fill: [`update_operator!`](@ref)
re-derives it with that call's `(p, ctx)`, and [`setup_operator`](@ref) with the
`initial_parameters`/`initial_context` pair, so a mass reading
[`evaluation_time`](@ref) is evaluated at the same point the rhs is.

!!! warning "Experimental surface"
    The rate form, its operator types and the realizations above may change in a
    minor release.
"""
RateFormIntegrator(rhs::AbstractBilinearIntegrator, mass) = BilinearRateFormIntegrator(rhs, mass)
RateFormIntegrator(rhs::AbstractLinearIntegrator, mass) = LinearRateFormIntegrator(rhs, mass)
RateFormIntegrator(rhs, mass) = throw(ArgumentError(
    "A rate form weights a BILINEAR or a LINEAR right-hand side by `M⁻¹` (got " *
    "$(nameof(typeof(rhs)))). A nonlinear residual is not the action of an operator, so `M⁻¹F(u)` " *
    "is not a term this package assembles; scale the residual solver-side."))

# A rate form never reaches an element cache: `setup_operator` takes the pair
# apart and builds the engine over the rhs alone. Reaching here means it was
# composed into someone else's local system — which `M⁻¹` does not distribute
# over.
setup_element_cache(i::AnyRateFormIntegrator, sdh::SubDofHandler) = throw(ArgumentError(
    "A rate form is an OPERATOR-level term: `M⁻¹` weights the assembled right-hand side, not one " *
    "element's local system, and `M⁻¹(A₁ + A₂) ≠ M⁻¹A₁ + A₂` — so it composes with no other term " *
    "into a shared element cache. Build it on its own — `setup_operator(strategy, " *
    "RateFormIntegrator(rhs, mass), dh)` — and add the terms at the operator level."))

####################################
## The term the ENGINE assembles
####################################

# The rhs as the engine sees it, carrying the mass ONLY where the storage fuses
# `M⁻¹` into its own store (`BlockRowAssembly`). It exists for two reasons: the
# inner operator is then built through the ordinary `setup_operator` methods —
# a rate-form integrator would dispatch back into this file — and the mass
# reaches `with_action_storage`, which is keyed on the element cache and sees no
# integrator at all.
struct _RateFormRHS{R, M} <: AbstractBilinearIntegrator
    rhs::R
    mass::M
end

setup_element_cache(i::_RateFormRHS, sdh::SubDofHandler) =
    _rate_form_cache(setup_element_cache(i.rhs, sdh), i.mass)
_rate_form_cache(cache, ::Nothing) = cache
_rate_form_cache(cache, mass) = RateFormElementCache(cache, mass)

global_dofs(i::_RateFormRHS, sdh::SubDofHandler) = global_dofs(i.rhs, sdh)
facet_items(i::_RateFormRHS, sdh::SubDofHandler) = facet_items(i.rhs, sdh)
facet_item_global_dofs(i::_RateFormRHS, sdh::SubDofHandler) = facet_item_global_dofs(i.rhs, sdh)
setup_facet_item_cache(i::_RateFormRHS, sdh::SubDofHandler) = setup_facet_item_cache(i.rhs, sdh)
algebraic_items(i::_RateFormRHS, dh::AbstractDofHandler) = algebraic_items(i.rhs, dh)
setup_algebraic_cache(i::_RateFormRHS, dh::AbstractDofHandler) = setup_algebraic_cache(i.rhs, dh)
function _declaration_subjects!(subjects, integrator::_RateFormRHS)
    push!(subjects, integrator)
    _declaration_subjects!(subjects, integrator.rhs)
    return subjects
end

# Carries the mass to the one storage that fuses `M⁻¹` into its own store; the
# store's cache takes the mass and wraps the rhs cache directly.
struct RateFormElementCache{Inner, M} <: AbstractElementCacheDecorator{Inner}
    inner::Inner
    mass::M
end
rewrap(d::RateFormElementCache, inner) = RateFormElementCache(inner, d.mass)

with_action_storage(d::RateFormElementCache, storage::BlockRowAssembly, sdh::SubDofHandler) =
    _block_row_storage(d.inner, sdh, d.mass)

####################################
## The mass sweep
####################################

# Hands every cell's element matrix of `integrator` to `f(slot, cellid, cell,
# Mₑ)`, `Mₑ` in the shape the cache's `element_matrix_structure` declares. A
# host cell sweep of its own: a mass is a cell term and the rhs need not be one.
function foreach_element_mass(f, integrator, sdh::SubDofHandler, p, ctx)
    cache = setup_element_cache(integrator, sdh)
    _assert_mass_matrix_kernel(cache)
    Mₑ    = allocate_element_matrix(cache, sdh)
    it    = assembly_iterator(nothing, cache, sdh)
    flags = item_update_flags(nothing, cache)
    for (slot, cellid) in enumerate(sdh.cellset)
        cell = position_iterator(it, cellid, flags)
        reinit_values!(cache, cell)
        fill!(Mₑ, zero(eltype(Mₑ)))
        assemble_cell!(JacobianRequest{:u}(Mₑ), cache,
                       CellArgs((;), cell, query_cell_parameters(cache, cell, p), ctx))
        f(slot, cellid, cell, Mₑ)
    end
    return nothing
end

function _assert_mass_matrix_kernel(cache)
    provides_analytic(typeof(cache), JacobianKind{:u}()) || throw(ArgumentError(
        "A rate form reads its mass's ELEMENT MATRIX, and $(nameof(typeof(cache))) declares no " *
        "analytic `JacobianKind{:u}` kernel (`provides_analytic`). A mass whose element matrix is " *
        "only reachable by differentiating its residual kernel cannot be the `M` of `M⁻¹A`."))
    return nothing
end

####################################
## The inverse mass
####################################

# `M⁻¹` of a diagonal mass: the reciprocal of the ASSEMBLED global diagonal, so
# one datum serves a continuous space (several cells per entry) as well as a
# discontinuous one. The sweep accumulates in `host`; `minv` is the same array
# on a host device, its device mirror otherwise.
struct DiagonalInverseMass{HV, DV}
    host::HV
    minv::DV
end

# The structural election itself: ONE treatment for the whole handler. It runs
# whether or not the storage goes on to keep `M⁻¹` as data, so a mass this
# package cannot invert cell-locally is refused with the rate form's own
# message and not with some store's.
function _resolve_mass_structure(mass, dh::AbstractDofHandler)
    isempty(dh.subdofhandlers) && throw(ArgumentError(
        "A rate form needs a mass to invert and this `DofHandler` carries no subdomain."))
    structures = map(dh.subdofhandlers) do sdh
        _assert_cell_only_term(mass, sdh, dh, "RateFormIntegrator")
        cache = setup_element_cache(mass, sdh)
        _assert_mass_matrix_kernel(cache)
        element_matrix_structure(cache)
    end
    allequal(map(typeof, structures)) || throw(ArgumentError(
        "The mass of a rate form declares " *
        "$(join(map(s -> string(nameof(typeof(s)), "()"), structures), ", ")) across this " *
        "handler's subdomains. `M⁻¹` is ONE treatment for the whole operator, so the mass must " *
        "have the same `element_matrix_structure` on every subdomain."))
    structure = first(structures)
    structure isa DenseElementMatrix && _assert_cell_local_mass(dh)
    return structure
end

function _assert_cell_local_mass(dh::AbstractDofHandler)
    d = _first_shared_cell_dof(dh)
    d == 0 && return nothing
    throw(ArgumentError(
        "The mass of a rate form has a DENSE element matrix over a CONTINUOUS space: dof $(d) " *
        "is shared by two cells of this handler, so `M` is not block diagonal by cell and " *
        "`M⁻¹` is a global solve this package will not hide inside an operator. Two escapes, both a " *
        "DISCRETIZATION decision and neither of them silent: wrap the mass in " *
        "`RowSumLumped(mass)`, whose element matrix IS its diagonal, or give the model a " *
        "collocated (spectral) mass element declaring " *
        "`element_matrix_structure(cache) = DiagonalElementMatrix()`."))
end

function _setup_inverse_mass(::DiagonalElementMatrix, strategy, mass, dh, p, ctx)
    host = zeros(value_type(strategy.device), ndofs(dh))
    data = DiagonalInverseMass(host, adapt_shared(strategy.device, host))
    _refresh_inverse_mass!(data, mass, dh, p, ctx)
    return data
end

# The per-cell inverse blocks are an operator of their own: `ElementInverse`
# assembled at the ELEMENT level, on the caller's device and scheduling.
_setup_inverse_mass(::DenseElementMatrix, strategy, mass, dh, p, ctx) =
    setup_operator(AssemblyStrategy(MatrixFreeAction(; storage = ElementAssembly()),
                                    strategy.scheduling, strategy.device),
                   ElementInverse(mass), dh; initial_parameters = p, initial_context = ctx)

function _refresh_inverse_mass!(data::DiagonalInverseMass, mass, dh, p, ctx)
    m = data.host
    fill!(m, zero(eltype(m)))
    for sdh in dh.subdofhandlers
        buf = Vector{Int}(undef, ndofs_per_cell(sdh))
        foreach_element_mass(mass, sdh, p, ctx) do slot, cellid, cell, mₑ
            celldofs!(buf, sdh.dh, cellid)
            for i in eachindex(buf)
                @inbounds m[buf[i]] += mₑ[i]
            end
        end
    end
    for d in eachindex(m)
        v = @inbounds m[d]
        iszero(v) && throw(ArgumentError(
            "The assembled lumped mass has a zero diagonal entry at dof $(d), so `M⁻¹` does not " *
            "exist. A rate form checks its mass STRUCTURALLY at setup; a vanishing entry is a " *
            "property of its VALUES — a dof no cell of this handler carries mass for, or a " *
            "density that integrates to zero."))
        @inbounds m[d] = inv(v)
    end
    data.minv === m || copyto!(data.minv, m)
    return data
end
_refresh_inverse_mass!(op::MatrixFreeFerriteOperator, mass, dh, p, ctx) = update_operator!(op, p, ctx)
_refresh_inverse_mass!(::Nothing, mass, dh, p, ctx) = nothing

# `y ← M⁻¹y`; `scratch` is a vector of the dof space the block route needs.
_apply_inverse_mass!(y, ::Nothing, scratch) = y
_apply_inverse_mass!(y, data::DiagonalInverseMass, scratch) = (y .*= data.minv; y)
_apply_inverse_mass!(y, op::MatrixFreeFerriteOperator, scratch) = (copyto!(scratch, y); mul!(y, op, scratch))

_inverse_mass_operator(::Nothing) = nothing
_inverse_mass_operator(data::DiagonalInverseMass) = Diagonal(data.minv)
_inverse_mass_operator(op::MatrixFreeFerriteOperator) = op

####################################
## The operators
####################################

"""
    RateFormFerriteOperator <: AbstractBilinearOperator

The operator [`setup_operator`](@ref) returns for a
[`BilinearRateFormIntegrator`](@ref): the rhs's own operator and `M⁻¹`,
composed ([`RateFormIntegrator`](@ref) tabulates the realizations).

Its surface is the inner operator's — `mul!` in both forms,
[`evaluate!`](@ref), [`update_operator!`](@ref), `size`, `eltype`. It holds no
fused matrix, so `get_matrix` is an error: [`rate_form_rhs`](@ref) is the rhs's
operator and [`rate_form_inverse_mass`](@ref) is `M⁻¹`. `y` and `u` must not
alias, the matrix-free inner operator's rule.

!!! warning "Experimental surface"
    This operator may change in a minor release.
"""
@concrete struct RateFormFerriteOperator <: AbstractBilinearOperator
    inner
    integrator
    minv           # `nothing` where the store fuses `M⁻¹`, else `DiagonalInverseMass` or the `ElementInverse` operator
    scratch        # a dof-space vector; `nothing` where `M⁻¹` is fused
end

Base.size(op::RateFormFerriteOperator) = size(op.inner)
Base.size(op::RateFormFerriteOperator, axis) = size(op.inner, axis)
Base.eltype(op::RateFormFerriteOperator) = eltype(op.inner)
get_dof_handler(op::RateFormFerriteOperator) = get_dof_handler(op.inner)
get_strategy(op::RateFormFerriteOperator) = get_strategy(op.inner)
get_subdomain_caches(op::RateFormFerriteOperator) = get_subdomain_caches(op.inner)
get_matrix(op::RateFormFerriteOperator) = throw(ArgumentError(
    "A rate form holds no fused `M⁻¹A` matrix: it composes the rhs's operator with `M⁻¹`. " *
    "`rate_form_rhs(op)` is the rhs's operator and `rate_form_inverse_mass(op)` is `M⁻¹`."))

"""
    rate_form_rhs(op) -> operator

The operator of the rate form's right-hand side, `A`. Under
[`BlockRowAssembly`](@ref) alone its store already carries `M⁻¹A`, the fusion
living nowhere else; every other realization keeps `A` unweighted here.
"""
rate_form_rhs(op::RateFormFerriteOperator) = op.inner

"""
    rate_form_inverse_mass(op) -> Diagonal, operator, or nothing

`M⁻¹` as the rate form applies it: a `Diagonal` for a diagonal mass, the
[`ElementInverse`](@ref) operator for a dense one, `nothing` where
[`BlockRowAssembly`](@ref) fused it into the rhs's store.
"""
rate_form_inverse_mass(op::RateFormFerriteOperator) = _inverse_mass_operator(op.minv)

mul!(y::AbstractVector, op::RateFormFerriteOperator, u::AbstractVector) = _rate_action!(y, op, u, op.minv)

_rate_action!(y, op, u, ::Nothing) = mul!(y, op.inner, u)
_rate_action!(y, op, u, data::DiagonalInverseMass) = (mul!(y, op.inner, u); y .*= data.minv; y)
_rate_action!(y, op, u, minv::MatrixFreeFerriteOperator) = (mul!(op.scratch, op.inner, u); mul!(y, minv, op.scratch))

mul!(y::AbstractVector, op::RateFormFerriteOperator, u::AbstractVector, α, β) =
    _rate_action!(y, op, u, α, β, op.minv)

_rate_action!(y, op, u, α, β, ::Nothing) = mul!(y, op.inner, u, α, β)
# `β = 0` ASSIGNS rather than scales, the LinearAlgebra convention a `NaN`
# already in `y` would otherwise propagate through.
function _rate_action!(y, op, u, α, β, data::DiagonalInverseMass)
    w = op.scratch
    mul!(w, op.inner, u)
    if iszero(β)
        y .= α .* data.minv .* w
    else
        y .= α .* data.minv .* w .+ β .* y
    end
    return y
end
_rate_action!(y, op, u, α, β, minv::MatrixFreeFerriteOperator) =
    (mul!(op.scratch, op.inner, u); mul!(y, minv, op.scratch, α, β))

evaluate!(op::RateFormFerriteOperator, y::AbstractVector, states::NamedTuple, p, ctx) =
    (evaluate!(op.inner, y, states, p, ctx); _apply_inverse_mass!(y, op.minv, op.scratch); y)
evaluate!(op::RateFormFerriteOperator, y::AbstractVector, u::AbstractVector, p) =
    evaluate!(op, y, (u = u,), p, nothing)

"""
    update_operator!(op::RateFormFerriteOperator, p, ctx = nothing)

Refill the rhs and re-derive `M⁻¹` at `(p, ctx)`, so the two halves of `M⁻¹A`
are never evaluated at different points.
"""
function update_operator!(op::RateFormFerriteOperator, p, ctx = nothing)
    update_operator!(op.inner, p, ctx)
    _refresh_inverse_mass!(op.minv, op.integrator.mass, get_dof_handler(op), p, ctx)
    return nothing
end

"""
    LinearRateFormFerriteOperator <: AbstractLinearOperator

The operator [`setup_operator`](@ref) returns for a
[`LinearRateFormIntegrator`](@ref): the rhs's load vector, weighted — `op.b` IS
`M⁻¹b`, which is what a rate-form source contributes to `du/dt`.

!!! warning "Experimental surface"
    This operator may change in a minor release.
"""
@concrete struct LinearRateFormFerriteOperator <: AbstractLinearOperator
    b
    engine
    integrator
    minv
    scratch
end

"""
    update_operator!(op::LinearRateFormFerriteOperator, p, ctx = nothing)

Assemble the rhs into `op.b`, re-derive `M⁻¹` at the same `(p, ctx)` and weight
the vector — the entry point a time-varying source is refilled through.
"""
function update_operator!(op::LinearRateFormFerriteOperator, p, ctx = nothing)
    assemble_into!(LinearKind(), (op.b,), op, (;), p, ctx)
    _refresh_inverse_mass!(op.minv, op.integrator.mass, op.engine.dh, p, ctx)
    _apply_inverse_mass!(op.b, op.minv, op.scratch)
    return nothing
end

####################################
## Setup
####################################

"""
    setup_operator(strategy, integrator::BilinearRateFormIntegrator, dh; …)
    setup_operator(strategy, integrator::LinearRateFormIntegrator, dh; …)

Build the rate form's operator: the rhs's own operator under `strategy`, and
`M⁻¹` as the mass's structure and the storage decide
([`RateFormIntegrator`](@ref) tabulates both, and states which pairs are
refused). `initial_parameters`/`initial_context` are the pair the mass — and,
under [`MatrixFreeAction`](@ref), the rhs store — are first evaluated with.

An ASSEMBLED rate form holds an unfilled matrix until the first
[`update_operator!`](@ref), exactly as any other [`FullAssembly`](@ref) operator
does; what setup does evaluate is the mass, so a singular one is an error here
and not at the first refill.
"""
setup_operator(strategy::AbstractAssemblyStrategy, integrator::BilinearRateFormIntegrator,
        dh::AbstractDofHandler; kwargs...) = _setup_rate_form(strategy, integrator, dh; kwargs...)
setup_operator(strategy::AbstractAssemblyStrategy, integrator::LinearRateFormIntegrator,
        dh::AbstractDofHandler; kwargs...) = _setup_rate_form(strategy, integrator, dh; kwargs...)

# The matrix-free form has `setup_operator` methods of its OWN for the bilinear
# and linear families, each more specific in the STRATEGY than the two above and
# less specific in the integrator: without these two the pair would be
# ambiguous, and a rate form under `MatrixFreeAction` — the shipped GPU route —
# would not resolve at all.
setup_operator(strategy::AssemblyStrategy{<:MatrixFreeAction}, integrator::BilinearRateFormIntegrator,
        dh::AbstractDofHandler; kwargs...) = _setup_rate_form(strategy, integrator, dh; kwargs...)
setup_operator(strategy::AssemblyStrategy{<:MatrixFreeAction}, integrator::LinearRateFormIntegrator,
        dh::AbstractDofHandler; kwargs...) = _setup_rate_form(strategy, integrator, dh; kwargs...)

function _setup_rate_form(strategy, integrator::BilinearRateFormIntegrator, dh::AbstractDofHandler;
        initial_parameters = nothing, initial_context = nothing, kwargs...)
    structure = _resolve_mass_structure(integrator.mass, dh)
    fused = _fuses_in_store(strategy.form)
    inner = _setup_rate_form_rhs(strategy, _RateFormRHS(integrator.rhs, fused ? integrator.mass : nothing),
                                 dh, initial_parameters, initial_context; kwargs...)
    minv = fused ? nothing :
        _setup_inverse_mass(structure, strategy, integrator.mass, dh, initial_parameters, initial_context)
    return RateFormFerriteOperator(inner, integrator, minv, _rate_scratch(minv, strategy, dh))
end

function _setup_rate_form(strategy, integrator::LinearRateFormIntegrator, dh::AbstractDofHandler;
        initial_parameters = nothing, initial_context = nothing, kwargs...)
    _assert_assembled_linear_rate_form(strategy.form)
    structure = _resolve_mass_structure(integrator.mass, dh)
    inner = setup_operator(strategy, integrator.rhs, dh; kwargs...)
    minv  = _setup_inverse_mass(structure, strategy, integrator.mass, dh, initial_parameters, initial_context)
    return LinearRateFormFerriteOperator(inner.b, inner.engine, integrator, minv, _rate_scratch(minv, strategy, dh))
end

_rate_scratch(::Nothing, strategy, dh) = nothing
_rate_scratch(minv, strategy, dh) = allocate_vector(strategy.device, dh)

# A linear form has no action for a matrix-free operator to evaluate, so its
# rate form has none either. The refusal is spelled here rather than left to the
# inner `setup_operator`'s, which names the wrapped integrator and says nothing
# about the pair the caller actually wrote.
_assert_assembled_linear_rate_form(form) = nothing
_assert_assembled_linear_rate_form(form::MatrixFreeAction) = throw(ArgumentError(
    "A rate form over a LINEAR right-hand side carries the load vector `M⁻¹b` and has no action " *
    "to evaluate, so it has no `MatrixFreeAction` realization. Assemble the source under " *
    "`FullAssembly` — the vector it holds is what a matrix-free rate operator's own consumer adds " *
    "to `du/dt` — and keep the matrix-free strategy for the bilinear term beside it."))

# The initial evaluation pair is the MATRIX-FREE setup's, that form being the
# one that fills a store at setup; an assembling form fills at
# `update_operator!` and takes no such pair.
_setup_rate_form_rhs(strategy, term, dh, p, ctx; kwargs...) = setup_operator(strategy, term, dh; kwargs...)
_setup_rate_form_rhs(strategy::AssemblyStrategy{<:MatrixFreeAction}, term, dh, p, ctx; kwargs...) =
    setup_operator(strategy, term, dh; initial_parameters = p, initial_context = ctx, kwargs...)

# Only the block-row store fuses `M⁻¹` into what it keeps.
_fuses_in_store(form) = false
_fuses_in_store(form::MatrixFreeAction) = form.storage isa BlockRowAssembly
