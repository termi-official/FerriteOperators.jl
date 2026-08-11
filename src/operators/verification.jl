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
                      rtol = 1e-5, atol = 1e-8, nprobes = 3) -> (passed, checks)

Cross-check the operator's derivative paths against central finite differences
of its own residual, through the same entry points solvers use — a wrong
analytic kernel ([`provides_analytic`](@ref)) fails its check against the FD
referee. Checks: assembled Jacobian action along `nprobes` deterministic
directions, fused-vs-split residual consistency, parameter Jacobian (per
flat-θ column) and VJP, matrix-free state JVP/VJP actions, and — when `ctx` is
given — the time sensitivity (default method against
[`FiniteDifferenceSensitivity`](@ref); the evaluation time comes from the
context, which is where ∂F/∂t seeds).

`checks` holds one `(passed, err, skipped)` entry per check; inadmissible or
unsupported checks are skipped with the reason recorded, and `passed` is the
conjunction of all non-skipped checks. The caller's vectors are never
mutated. Condensed operators (`unknown_size(op) > residual_size(op)`) are
probed along the field dofs only; the FD evaluations exercise the full local
solves, so the check validates the consistent condensed tangent.
"""
function check_derivatives(op, states::NamedTuple, p, ctx = nothing;
        h::Float64 = cbrt(eps(Float64)),
        rtol::Float64 = 1e-5, atol::Float64 = 1e-8, nprobes::Int = 3)
    nres  = residual_size(op)
    ubase = copy(states.u)
    uw    = copy(states.u)
    statesw = merge(states, (u = uw,))
    hs = h * max(1.0, maximum(abs, view(ubase, 1:nres)))

    rp = zeros(nres); rm = zeros(nres)
    # Central FD of the residual along the field-dof direction v.
    function fd_dir!(out, v, pfd)
        uw .= ubase; view(uw, 1:nres) .+= hs .* v
        evaluate!(op, rp, statesw, pfd, ctx)
        uw .= ubase; view(uw, 1:nres) .-= hs .* v
        evaluate!(op, rm, statesw, pfd, ctx)
        out .= (rp .- rm) ./ 2hs
        return out
    end

    r_fused = zeros(nres)
    jacobian = _run_check() do
        uw .= ubase
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
        update_parameter_jacobian!(B, op, statesw, p, ctx)
        Bref[] = B
        Bfd = zeros(nres, nθ)
        hθ = h * max(1.0, maximum(abs, θ; init = 0.0))
        for j in 1:nθ
            θj = copy(θ); θj[j] += hθ
            uw .= ubase; evaluate!(op, rp, statesw, rebuild_parameters(p, θj), ctx)
            θj[j] -= 2hθ
            uw .= ubase; evaluate!(op, rm, statesw, rebuild_parameters(p, θj), ctx)
            Bfd[:, j] .= (rp .- rm) ./ 2hθ
        end
        _check_entry(isapprox(B, Bfd; rtol, atol), _relerr(B, Bfd))
    end

    parameter_vjp = _run_check() do
        B = Bref[]
        B === nothing && return (passed = true, err = NaN, skipped = "parameter Jacobian unavailable as referee")
        uw .= ubase
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

    checks = (; jacobian, fused_residual, parameter_jacobian, parameter_vjp,
                state_jvp, state_vjp, time_sensitivity)
    return (passed = all(c.skipped !== nothing || c.passed for c in values(checks)), checks = checks)
end

check_derivatives(op, u::AbstractVector, p; kwargs...) =
    check_derivatives(op, (u = u,), p, nothing; kwargs...)
