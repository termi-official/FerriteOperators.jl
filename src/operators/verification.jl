####################################
## Derivative checker
####################################

# Deterministic, dependency-free probe directions.
function _probe_vector(n::Int, k::Int)
    v = [sin(0.7k * i + 0.3k) for i in 1:n]
    return v ./ norm(v)
end

_relerr(a, b) = norm(a .- b) / max(norm(b), eps(Float64))

# One check body: ArgumentError (inadmissible kind) and MethodError (entry
# point not implemented by this operator type) are recorded as skips.
function _run_check(f)
    try
        return f()
    catch e
        e isa Union{ArgumentError, MethodError} || rethrow()
        return (passed = true, err = NaN, skipped = sprint(showerror, e))
    end
end
_check_entry(passed, err) = (passed = passed, err = err, skipped = nothing)

"""
    check_derivatives(op, states, p, ctx = nothing; h = cbrt(eps(Float64)),
                      rtol = 1e-5, atol = 1e-8, nprobes = 3,
                      weights = nothing) -> (passed, checks)

Cross-check the operator's derivative paths against central finite differences
of its own residual, through the same entry points solvers use — a wrong
analytic kernel ([`provides_analytic`](@ref)) fails its check against the FD
referee. Checks: assembled Jacobian action along `nprobes` deterministic
directions, fused-vs-split residual consistency, parameter Jacobian (per
flat-θ column) and VJP, matrix-free state JVP/VJP actions, and — when `ctx` is
given — the time sensitivity (default method against
[`FiniteDifferenceSensitivity`](@ref); the evaluation time comes from the
context, which is where ∂F/∂t seeds).

Passing `weights` (a per-slot NamedTuple, e.g. `(u = 1.0, du = 1/(γ*Δt))`)
adds the two weighted-Jacobian checks: `weighted_jacobian` probes the fused
`Σₛ wₛ ∂F/∂s` against per-slot finite differences of the residual, and
`weighted_jacobian_routes` requires the fused and composed routes of
[`assemble_weighted_jacobian!`](@ref) to agree — the caller cannot tell which
route a given operator takes, so the two must be interchangeable. Both are
skipped with the reason recorded when no weights are given, when the weights
are complex (the FD referee is real), or when a participating slot carries an
[`AffineRate`](@ref) source (nothing to difference against).

The FD referee evaluates the operator's FULL residual, boundary terms
included, while the sensitivity sweeps it checks skip a boundary term that
rides the cell sweep (a [`facet_items`](@ref) term is its own traversal and
does enter them). On an operator carrying a fused-route boundary cache, a
failing parameter, time, or state-product check is therefore the diagnostic
for a boundary term that depends on the seeded quantity.

`checks` holds one `(passed, err, skipped)` entry per check; inadmissible or
unsupported checks are skipped with the reason recorded, and `passed` is the
conjunction of all non-skipped checks. The caller's vectors are never
mutated. Condensed operators (`unknown_size(op) > residual_size(op)`) are
probed along the field dofs only; the FD evaluations exercise the full local
solves, so the check validates the consistent condensed tangent.

Supports [`LinearizedFerriteOperator`](@ref) only — the family with a
Jacobian, a residual, and the sensitivity entry points this checks against.
"""
function check_derivatives(op, states::NamedTuple, p, ctx = nothing;
        h::Float64 = cbrt(eps(Float64)),
        rtol::Float64 = 1e-5, atol::Float64 = 1e-8, nprobes::Int = 3,
        weights::Union{Nothing, NamedTuple} = nothing,
        correction::Type{<:CorrectionMode} = Consistent)
    op isa LinearizedFerriteOperator || throw(ArgumentError(
        "check_derivatives supports LinearizedFerriteOperator only (got $(typeof(op))): " *
        "a bilinear or linear operator has no Jacobian/residual pair to cross-check against " *
        "finite differences."))
    nres  = residual_size(op)
    ubase = copy(states.u)
    uw    = copy(states.u)
    statesw = merge(states, (u = uw,))
    # `uw` is mutated in place (never reassigned) below, so a `:q` slot
    # sourced by `InternalSource` — which wraps `uw` once, here — stays valid
    # across every perturbation without being rebuilt per trial point.
    condensed = unknown_size(op) > residual_size(op)
    if condensed && haskey(states, :q) && states.q isa InternalSource
        statesw = merge(statesw, (q = InternalSource(uw),))
    end
    # The FD referee must re-solve `q` at every trial point to be a total,
    # matching what a `Consistent` analytic kernel computes — condensation is
    # the evaluation that makes that solve happen post-phase (see
    # `condense_internal!`). A `FrozenQ` election is a deliberately partial
    # kernel with no total to compare against, so its checks are skipped
    # rather than run against a doomed reference — an elected mismatch is
    # as-elected, not a failure.
    _condense!(s, pc, cc) = condensed && correction === Consistent && condense_internal!(op, s, pc, cc)
    hs = h * max(1.0, maximum(abs, view(ubase, 1:nres)))

    rp = zeros(nres); rm = zeros(nres)
    # Central FD of the residual along the field-dof direction v.
    function fd_dir!(out, v, pfd)
        uw .= ubase; view(uw, 1:nres) .+= hs .* v
        _condense!(statesw, pfd, ctx)
        evaluate!(op, rp, statesw, pfd, ctx)
        uw .= ubase; view(uw, 1:nres) .-= hs .* v
        _condense!(statesw, pfd, ctx)
        evaluate!(op, rm, statesw, pfd, ctx)
        out .= (rp .- rm) ./ 2hs
        return out
    end

    r_fused = zeros(nres)
    jacobian = _run_check() do
        if condensed && correction !== Consistent
            return (passed = true, err = NaN, skipped =
                "correction = $(correction) is an elected partial; the FD referee validates " *
                "the total and is not a reference for it")
        end
        uw .= ubase
        _condense!(statesw, p, ctx)
        update_linearization!(op, r_fused, statesw, p, ctx)
        Jv = zeros(nres); fd = zeros(nres)
        err = 0.0; ok = true
        for k in 1:nprobes
            v = _probe_vector(nres, k)
            mul!(Jv, op, v)
            fd_dir!(fd, v, p)
            ok &= isapprox(Jv, fd; rtol, atol)
            err = max(err, _relerr(Jv, fd))
        end
        _check_entry(ok, err)
    end

    fused_residual = _run_check() do
        uw .= ubase
        _condense!(statesw, p, ctx)
        r_split = zeros(nres)
        evaluate!(op, r_split, statesw, p, ctx)
        _check_entry(isapprox(r_fused, r_split; rtol, atol), _relerr(r_fused, r_split))
    end

    Bref = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
    parameter_jacobian = _run_check() do
        θ = Vector(parameter_vector(p))
        nθ = length(θ)
        B = zeros(nres, nθ)
        uw .= ubase
        _condense!(statesw, p, ctx)
        update_parameter_jacobian!(B, op, statesw, p, ctx)
        Bref[] = B
        Bfd = zeros(nres, nθ)
        hθ = h * max(1.0, maximum(abs, θ; init = 0.0))
        for j in 1:nθ
            θj = copy(θ); θj[j] += hθ
            pj = rebuild_parameters(p, θj)
            uw .= ubase; _condense!(statesw, pj, ctx); evaluate!(op, rp, statesw, pj, ctx)
            θj[j] -= 2hθ
            pj = rebuild_parameters(p, θj)
            uw .= ubase; _condense!(statesw, pj, ctx); evaluate!(op, rm, statesw, pj, ctx)
            Bfd[:, j] .= (rp .- rm) ./ 2hθ
        end
        _check_entry(isapprox(B, Bfd; rtol, atol), _relerr(B, Bfd))
    end

    parameter_vjp = _run_check() do
        B = Bref[]
        B === nothing && return (passed = true, err = NaN, skipped = "parameter Jacobian unavailable as referee")
        uw .= ubase
        _condense!(statesw, p, ctx)
        λ = _probe_vector(nres, 7)
        g = zeros(size(B, 2))
        parameter_vjp!(g, op, λ, statesw, p, ctx)
        ref = B' * λ
        _check_entry(isapprox(g, ref; rtol, atol), _relerr(g, ref))
    end

    state_jvp = _run_check() do
        v = _probe_vector(nres, 11)
        Jv = zeros(nres); fd = zeros(nres)
        uw .= ubase
        state_jvp!(Jv, op, v, statesw, p, ctx)
        fd_dir!(fd, v, p)
        _check_entry(isapprox(Jv, fd; rtol, atol), _relerr(Jv, fd))
    end

    # (∂F/∂u)ᵀλ is checked through the probe identity ⟨g, v⟩ = ⟨λ, J v⟩, so no
    # transposed operator action is required of `op`.
    state_vjp = _run_check() do
        λ = _probe_vector(nres, 13)
        g = zeros(nres); fd = zeros(nres)
        uw .= ubase
        state_vjp!(g, op, λ, statesw, p, ctx)
        err = 0.0; ok = true
        for k in 1:nprobes
            v = _probe_vector(nres, 17 + k)
            fd_dir!(fd, v, p)
            lhs = dot(g, v); rhs = dot(λ, fd)
            ok &= isapprox(lhs, rhs; rtol, atol)
            err = max(err, abs(lhs - rhs) / max(abs(rhs), eps(Float64)))
        end
        _check_entry(ok, err)
    end

    time_sensitivity = _run_check() do
        ctx === nothing && return (passed = true, err = NaN, skipped =
            "no context given — time sensitivities seed through ctx; pass a TimeIntegrationContext")
        g = zeros(nres); gfd = zeros(nres)
        uw .= ubase
        time_sensitivity!(g, op, statesw, p, ctx)
        uw .= ubase
        time_sensitivity!(gfd, op, statesw, p, ctx; method = FiniteDifferenceSensitivity())
        _check_entry(isapprox(g, gfd; rtol, atol), _relerr(g, gfd))
    end

    # The weighted routes assemble into two members of one pattern group, so
    # the fused-vs-composed comparison is a plain `nzval` comparison.
    Wf = Ref{Any}(nothing)
    weighted_jacobian = _run_check() do
        skip = _weighted_check_skip(weights, states)
        skip === nothing || return (passed = true, err = NaN, skipped = skip)
        kind = WeightedJacobianKind(weights)
        _fused_weighted_route(op, kind) || return (passed = true, err = NaN, skipped =
            "this operator's element caches route the weighted Jacobian to the composed path only")
        slots = keys(weights)
        # Writable copies of every participating slot; `:u` reuses the copy the
        # checker already protects the caller's state with.
        wbase = NamedTuple{slots}(map(s -> s === :u ? ubase : copy(states[s]), slots))
        wwork = NamedTuple{slots}(map(s -> s === :u ? uw : copy(states[s]), slots))
        statesq = merge(statesw, wwork)
        reset!() = foreach(i -> wwork[i] .= wbase[i], 1:length(slots))

        W = allocate_components(op, (:fused,)).fused
        reset!()
        _condense!(statesq, p, ctx)
        _weighted_jacobian_fused!(W, op, kind, statesq, p, ctx)
        Wf[] = W

        Wv = zeros(nres); fd = zeros(nres); fdw = zeros(nres)
        err = 0.0; ok = true
        for k in 1:nprobes
            v = _probe_vector(nres, 23 + k)
            fill!(fdw, 0.0)
            for (i, s) in enumerate(slots)
                hi = h * max(1.0, maximum(abs, view(wbase[i], 1:nres)))
                reset!(); view(wwork[i], 1:nres) .+= hi .* v
                _condense!(statesq, p, ctx)
                evaluate!(op, rp, statesq, p, ctx)
                reset!(); view(wwork[i], 1:nres) .-= hi .* v
                _condense!(statesq, p, ctx)
                evaluate!(op, rm, statesq, p, ctx)
                @. fd = (rp - rm) / 2hi
                @. fdw += weights[i] * fd
            end
            reset!()
            mul!(Wv, W, v)
            ok &= isapprox(Wv, fdw; rtol, atol)
            err = max(err, _relerr(Wv, fdw))
        end
        _check_entry(ok, err)
    end

    weighted_jacobian_routes = _run_check() do
        Wfused = Wf[]
        Wfused === nothing && return (passed = true, err = NaN,
            skipped = "fused weighted Jacobian unavailable as reference")
        Wc = share_pattern(Wfused)
        uw .= ubase
        _condense!(statesw, p, ctx)
        _weighted_jacobian_composed!(Wc, op, WeightedJacobianKind(weights), statesw, p, ctx)
        _check_entry(isapprox(Wfused.nzval, Wc.nzval; rtol, atol), _relerr(Wfused.nzval, Wc.nzval))
    end

    checks = (; jacobian, fused_residual, parameter_jacobian, parameter_vjp,
                state_jvp, state_vjp, time_sensitivity,
                weighted_jacobian, weighted_jacobian_routes)
    return (passed = all(c.skipped !== nothing || c.passed for c in values(checks)), checks = checks)
end

# Reasons the weighted checks have nothing to compare against.
function _weighted_check_skip(weights, states)
    weights === nothing && return "no `weights` given — pass e.g. `weights = (u = 1.0, du = 1/(γ*Δt))` " *
        "to check the weighted Jacobian"
    all(w -> w isa Real, values(weights)) || return "complex weights assemble through the composed " *
        "route only, and the finite-difference referee is real"
    for s in keys(weights)
        haskey(states, s) || return "states carry no `:$s` slot"
        states[s] isa AbstractVector || return "slot `:$s` carries a reconstructed source, " *
            "which the finite-difference referee cannot perturb independently"
    end
    return nothing
end

check_derivatives(op, u::AbstractVector, p; kwargs...) =
    check_derivatives(op, (u = u,), p, nothing; kwargs...)
