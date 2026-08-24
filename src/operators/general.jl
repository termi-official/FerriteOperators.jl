@concrete struct AssemblyDomain
    sdh
    ivh
    element
    boundary_element
end

@concrete struct TransferDomain
    sdh_row
    sdh_col
end

"""
    AlgebraicDomain(element, items)

Domain descriptor of an algebraic subdomain cache: the cache
[`setup_algebraic_cache`](@ref) built (field named after `AssemblyDomain`'s so
the per-sweep validation helpers read both domains) and the resolved dof
vectors.
"""
@concrete struct AlgebraicDomain
    element
    items
end

"""
    FacetItemDomain(sdh, element, items)

Domain descriptor of a facet item subdomain cache: the `SubDofHandler` whose
cells own the facets, the surface cache [`setup_facet_item_cache`](@ref) built
(field named after `AssemblyDomain`'s so the per-sweep validation helpers read
every domain), and the resolved [`FacetItem`](@ref)s.
"""
@concrete struct FacetItemDomain
    sdh
    element
    items
end

# (kernel function, args record, trailing argument types) of a domain's item
# family; a new family adds one method and both checks below follow.
kernel_entry(domain) = (assemble_cell!, CellArgs, ())
kernel_entry(::AlgebraicDomain) = (assemble_algebraic!, AlgebraicArgs, ())
kernel_entry(::FacetItemDomain) = (assemble_facet!, FacetArgs, (Int,))

# Trait ↔ kernel and call-time admissibility checks over the family's entry,
# so errors name the family's own kernel and args record.
_assert_domain_trait_backed(domain, kind) =
    _assert_trait_backed(typeof(domain.element), kind, kernel_entry(domain)...)
_assert_domain_sensitivity_admissible(domain, kind) =
    assert_sensitivity_admissible(typeof(domain.element), kind, kernel_entry(domain)...)

# Structural half of the reduction precondition, per kind. The facet family's
# answer lives in facet-task.jl beside its `execute_kind!` bodies; algebraic
# domains answer `true` (`hasmethod` cannot see an `evaluate_algebraic_functional`
# override past the untyped default).
_may_contribute(domain, kind) = true
_may_contribute(domain::AssemblyDomain, kind) = !(domain.element isa EmptyVolumetricElementCache)

@concrete struct SubdomainCache
    domain
    device_cache
    partition
end

"""
    AssemblyEngine

The assembly machinery shared by all operators: the execution strategy, the
per-subdomain caches (workspaces + partitions), the dof handler the operator
assembles against, the engine-scoped internal-variable handler, and the scheme
protocol carrying the setup-time declarations ([`AbstractSchemeProtocol`](@ref)).
Operators are payload (matrices/vectors) plus an engine plus their integrator.
"""
@concrete struct AssemblyEngine
    strategy
    subdomain_caches
    dh
    ivh         # shared by all subdomains
    protocol    # the setup-time declarations
end

function execute_on_subdomains!(task, strategy, subdomain_caches)
    for (subdomain_id, sc) in enumerate(subdomain_caches)
        @timeit_debug "assemble subdomain $subdomain_id" execute_on_device!(task, strategy.device, sc.device_cache, sc.partition)
    end
end
execute_on_subdomains!(task, engine::AssemblyEngine) =
    execute_on_subdomains!(task, engine.strategy, engine.subdomain_caches)

"""
    reduce_on_subdomains(task, engine) -> value

The value-returning counterpart of `execute_on_subdomains!`: run `task` over
every subdomain and reduce the per-subdomain partials in subdomain order.
Within a subdomain [`reduce_on_device`](@ref) reduces the per-worker partials
in worker order, so the reduction order is fixed by the partition and the
result is deterministic for a fixed worker count.
"""
function reduce_on_subdomains(task, strategy, subdomain_caches)
    total = initial_partial(task.kind)
    for (subdomain_id, sc) in enumerate(subdomain_caches)
        partial = @timeit_debug "reduce subdomain $subdomain_id" reduce_on_device(task, strategy.device, sc.device_cache, sc.partition)
        total = _reduce_partials(total, partial)
    end
    return total
end
reduce_on_subdomains(task, engine::AssemblyEngine) =
    reduce_on_subdomains(task, engine.strategy, engine.subdomain_caches)

"""
    AbstractNonlinearOperator

Models of a nonlinear function `F(u)v`, where `v` is a test function.

Interface:

    evaluate!(op, residual::AbstractVector, states::NamedTuple, p, ctx)
    evaluate!(op, residual::AbstractVector, u::AbstractVector, p)
    Base.eltype(op)
    Base.size(op[, axis])

    # linearization
    mul!(out::AbstractVector, op, in::AbstractVector)
    mul!(out::AbstractVector, op, in::AbstractVector, α, β)
    update_linearization!(op, states::NamedTuple, p, ctx)
    update_linearization!(op, residual::AbstractVector, states::NamedTuple, p, ctx)

The `(states, p, ctx)` forms are canonical: `states` carries one entry per
declared slot, `p` the parameter object element kernels query, `ctx` the
per-sweep evaluation context. The `(u, p)` forms are stationary conveniences
evaluating at `states = (u = u,)` with no context.
"""
abstract type AbstractNonlinearOperator end

"""
    update_linearization!(op, residual, u, p)

Setup the linearized operator `Jᵤ(u) := dᵤF(u)` in op and its residual `F(u)`, in
preparation to solve `J(u) Δu = F(u)` for the increment `Δu`.
"""
update_linearization!(Jᵤ::AbstractNonlinearOperator, residual::AbstractVector, u::AbstractVector, p)

"""
    update_linearization!(op, u, p)

Setup the linearized operator `Jᵤ(u)` in op.
"""
update_linearization!(Jᵤ::AbstractNonlinearOperator, u::AbstractVector, p)

"""
    evaluate!(op, residual, u, p)

Evaluate the residual `F(u)` into `residual` without updating the Jacobian.
"""
function evaluate! end


"""
Supertype for block-structured operators. Carries no subtypes in this package;
it exists for downstream operators assembled into hand-built block systems.

!!! warning "Slated for removal"
    Do not subtype this in new code. Block-structured targets are served by
    [`BlockedOperatorSpecification`](@ref); this supertype is removed once its
    remaining downstream consumers migrate.
"""
abstract type AbstractBlockOperator <: AbstractNonlinearOperator end

get_matrix(op) = error("Operator matrix is not explicitly accessible for given operator")

function *(op::AbstractNonlinearOperator, x::AbstractVector)
    y = similar(x)
    mul!(y, op, x)
    return y
end

#########################################################################################################################

"""
    AbstractBilinearOperator <: AbstractNonlinearOperator

The operator a bilinear form induces: a state-independent matrix `A` with
action `F(u) = A·u`. The matrix IS the Jacobian, so
[`update_linearization!`](@ref) is `update_operator!` plus that action, and
the family carries no differentiation machinery (see
[`AbstractBilinearIntegrator`](@ref)).
"""
abstract type AbstractBilinearOperator <: AbstractNonlinearOperator end

update_linearization!(op::AbstractBilinearOperator, u::AbstractVector, p) = update_operator!(op, p)
function update_linearization!(op::AbstractBilinearOperator, residual::AbstractVector, u::AbstractVector, p)
    update_operator!(op, p)
    mul!(residual, op, u)
    return residual
end


"""
    DiagonalOperator <: AbstractBilinearOperator

Literally a "diagonal matrix".
"""
struct DiagonalOperator{TV <: AbstractVector} <: AbstractBilinearOperator
    values::TV
end

mul!(out::AbstractVector, op::DiagonalOperator, in::AbstractVector) = out .= op.values .* in
mul!(out::AbstractVector, op::DiagonalOperator, in::AbstractVector, α, β) = out .= α * op.values .* in + β * out
Base.eltype(op::DiagonalOperator) = eltype(op.values)
Base.size(op::DiagonalOperator, axis) = length(op.values)

get_matrix(op::DiagonalOperator) = spdiagm(op.values)

update_operator!(::DiagonalOperator, p) = nothing

"""
    NullOperator <: AbstractBilinearOperator

Literally a "null matrix".
"""
struct NullOperator{T, SIN, SOUT} <: AbstractBilinearOperator
end

mul!(out::AbstractVector, op::NullOperator, in::AbstractVector) = out .= 0.0
mul!(out::AbstractVector, op::NullOperator, in::AbstractVector, α, β) = out .= β*out
Base.eltype(op::NullOperator{T}) where {T} = T
Base.size(op::NullOperator{T,S1,S2}, axis) where {T,S1,S2} = axis == 1 ? S1 : (axis == 2 ? S2 : error("faulty axis!"))

get_matrix(op::NullOperator{T, SIN, SOUT}) where {T, SIN, SOUT} = spzeros(T,SIN,SOUT)

update_operator!(::NullOperator, p) = nothing

#########################################################################################################################

"""
    AbstractLinearOperator

Supertype for operators which only depend on the test space.
"""
abstract type AbstractLinearOperator end

"""
    LinearNullOperator <: AbstractLinearOperator

Literally the null vector.
"""
struct LinearNullOperator{T,S} <: AbstractLinearOperator
end
Ferrite.add!(b::AbstractVector, op::LinearNullOperator) = b
Base.eltype(op::LinearNullOperator{T,S}) where {T,S} = T
Base.size(op::LinearNullOperator{T,S}) where {T,S} = S

update_operator!(op::LinearNullOperator, p) = nothing


Ferrite.add!(b::AbstractVector, op::AbstractLinearOperator) = __add_to_vector!(b, op.b)
__add_to_vector!(b::AbstractVector, a::AbstractVector) = b .+= a
Base.eltype(op::AbstractLinearOperator) = eltype(op.b)
Base.size(op::AbstractLinearOperator) = size(op.b)

