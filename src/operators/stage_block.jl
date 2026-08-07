####################################
## Stage-block operator (fully implicit Runge-Kutta)
####################################

"""
    StageBlockOperator(op, A, c, Δt)

The Newton block operator of an s-stage fully implicit Runge-Kutta scheme:
stage `i` contributes the residual row `F(t + cᵢΔt, zᵢ, kᵢ)` with
`zᵢ = uₙ + Δt Σⱼ aᵢⱼ kⱼ`, so the block against the stage derivatives `k` is

    block(i, j) = δᵢⱼ · Jdu⁽ⁱ⁾ + Δt · aᵢⱼ · Ju⁽ⁱ⁾ ,

with `Ju⁽ⁱ⁾ = ∂F/∂u` and `Jdu⁽ⁱ⁾ = ∂F/∂du` assembled at stage `i`'s state and
time. All s² blocks come from those 2s component matrices — the tableau enters
as coefficients, never as extra assembly. Every component shares one sparsity
pattern (see [`allocate_components`](@ref)).

Fill the components with [`assemble_stages!`](@ref), apply the block matrix
with `mul!` over stage-stacked vectors of length `s·n`.

!!! note "Transformed (simplified-Newton) Radau needs no dedicated machinery"
    The Hairer-Wanner transformation replaces the stage-dependent Jacobians by
    ONE stage-independent pair `(Ju, Jdu)` evaluated at `uₙ` and diagonalizes
    `A⁻¹`, leaving one decoupled matrix per eigenvalue `λ`:
    `W_λ = Jdu + Δt·λ·Ju`, complex for the complex conjugate pairs. That is a
    single component bag plus a complex [`combine!`](@ref) — build it with
    `share_pattern(Ju, ComplexF64)` and
    `combine!(W_λ, (Ju = Ju, Jdu = Jdu), (Jdu = 1.0, Ju = Δt*λ))`, not with
    this operator.
"""
mutable struct StageBlockOperator{M <: AbstractMatrix, T}
    const Ju::Vector{M}
    const Jdu::Vector{M}
    const A::Matrix{T}
    const c::Vector{T}
    Δt::T
end

function StageBlockOperator(op, A::AbstractMatrix, c::AbstractVector, Δt)
    s = length(c)
    size(A) == (s, s) || throw(DimensionMismatch(
        "tableau A is $(size(A)) but c has $s entries."))
    T   = float(promote_type(eltype(A), eltype(c), typeof(Δt)))
    K   = create_system_matrix(op.engine.strategy, op.engine.dh)
    Ju  = [i == 1 ? K : share_pattern(K) for i in 1:s]
    Jdu = [share_pattern(K) for _ in 1:s]
    return StageBlockOperator(Ju, Jdu, Matrix{T}(A), Vector{T}(c), convert(T, Δt))
end

"""
    assemble_stages!(sbop, op, stage_states, p, ctxs) -> sbop

Assemble the stage components of `sbop`: for every stage `i`, one sweep of
`JacobianKind{:u}()` into `Ju[i]` and one of `JacobianKind{:du}()` into
`Jdu[i]`, both at `stage_states[i]` and `ctxs[i]`. The caller owns the stage
arithmetic — `stage_states[i]` is `(u = zᵢ, du = kᵢ, …)` with both slots plain
vectors, and `ctxs[i]` carries `t + cᵢΔt` and the stage's local interval.
"""
function assemble_stages!(sbop::StageBlockOperator, op, stage_states::AbstractVector{<:NamedTuple}, p, ctxs::AbstractVector)
    s = nstages(sbop)
    (length(stage_states) == s && length(ctxs) == s) || throw(DimensionMismatch(
        "expected $s stage states and contexts, got $(length(stage_states)) and $(length(ctxs))."))
    for i in 1:s
        assemble_slot_jacobian!(sbop.Ju[i],  op, JacobianKind{:u}(),  stage_states[i], p, ctxs[i])
        assemble_slot_jacobian!(sbop.Jdu[i], op, JacobianKind{:du}(), stage_states[i], p, ctxs[i])
    end
    return sbop
end

"Number of stages of the tableau."
nstages(sbop::StageBlockOperator) = length(sbop.c)

Base.eltype(sbop::StageBlockOperator) = eltype(sbop.Ju[1])
function Base.size(sbop::StageBlockOperator)
    n = size(sbop.Ju[1], 1)
    return (nstages(sbop) * n, nstages(sbop) * n)
end
Base.size(sbop::StageBlockOperator, axis) = size(sbop)[axis]

"""
    mul!(y, sbop::StageBlockOperator, x)
    mul!(y, sbop::StageBlockOperator, x, α, β)

Action of the stage-block matrix on the stage-stacked vector `x` (`s` blocks
of length `n`): `yᵢ = Jdu⁽ⁱ⁾ xᵢ + Δt · Ju⁽ⁱ⁾ Σⱼ aᵢⱼ xⱼ`. The tableau-combined
vector is formed once per stage row.
"""
function mul!(y::AbstractVector, sbop::StageBlockOperator, x::AbstractVector)
    s = nstages(sbop)
    n = size(sbop.Ju[1], 2)
    (length(x) == s * n && length(y) == s * n) || throw(DimensionMismatch(
        "stage vectors must have length $(s * n), got $(length(x)) and $(length(y))."))
    xblock(j) = view(x, ((j - 1) * n + 1):(j * n))
    xa = similar(x, n)
    for i in 1:s
        fill!(xa, zero(eltype(xa)))
        for j in 1:s
            xa .+= sbop.A[i, j] .* xblock(j)
        end
        yᵢ = view(y, ((i - 1) * n + 1):(i * n))
        mul!(yᵢ, sbop.Jdu[i], xblock(i))
        mul!(yᵢ, sbop.Ju[i], xa, sbop.Δt, true)
    end
    return y
end

function mul!(y::AbstractVector, sbop::StageBlockOperator, x::AbstractVector, α, β)
    z = mul!(similar(y), sbop, x)
    @. y = α * z + β * y
    return y
end
