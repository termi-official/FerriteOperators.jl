# One cold call per driver entry point the element-agnostic engine dispatches
# through — a nonlinear assembly (residual, then the fused Jacobian+residual),
# a bilinear operator update, and a functional reduction — on the smallest
# mesh that reaches every stage. Precompiling them moves the first-call
# inference cost out of every user session and into the package image.
#
# Core ships no element implementations (those live in the separate,
# unregistered lib/FerriteOperatorsExampleElements, which depends on core and
# so cannot be depended on back), so the workload below defines its own
# throwaway diffusion double. It runs `execute_on_device!`/`reduce_on_device`
# only for (SequentialAssemblyStrategy, SequentialCPUDevice): PolyesterDevice's
# methods live in the weak-dependency extension FerriteOperatorsPolyesterExt,
# which core cannot load, so that device has no route to precompile from here.
#
# Set the `precompile_workload` preference to `false` to skip the workload,
# e.g. for a fast development rebuild:
#
#     using Preferences, UUIDs
#     set_preferences!(UUID("27d9367a-5072-424e-9c5f-fe582399bac3"), "precompile_workload" => false)
#
# The workload writes no files, prints nothing and leaves no global state behind.
#
# The element double is defined in its OWN top-level `if`, separate from the
# `@setup_workload`/`@compile_workload` pair below: nesting both under one
# block runs them as a single compiled chunk at one world age, so
# `setup_operator` would look up `setup_element_cache` before this file's own
# method for it exists and fall through to the generic "no method" error.

using PrecompileTools: @setup_workload, @compile_workload

if Preferences.@load_preference("precompile_workload", true)
    # Nonlinear diffusion r(u) = ∫ ∇v⋅∇u dΩ. No analytic Jacobian kernel, so
    # setup wraps it in the ADElementCache decoration — the residual and the
    # fused JacobianResidual both run through ForwardDiff.
    struct PrecompileDiffusionIntegrator <: AbstractNonlinearIntegrator
        qrc::QuadratureRuleCollection
        field_name::Symbol
    end
    struct PrecompileDiffusionCache{CV <: CellValues} <: AbstractVolumetricElementCache
        cv::CV
    end
    setup_element_cache(m::PrecompileDiffusionIntegrator, sdh::SubDofHandler) =
        PrecompileDiffusionCache(CellValues(getquadraturerule(m.qrc, sdh),
            Ferrite.getfieldinterpolation(sdh, m.field_name),
            geometric_subdomain_interpolation(sdh)))
    duplicate_for_device(device, c::PrecompileDiffusionCache) =
        PrecompileDiffusionCache(duplicate_for_device(device, c.cv))
    reinit_values!(c::PrecompileDiffusionCache, cell) = Ferrite.reinit!(c.cv, cell)
    function assemble_cell!(req::ResidualRequest, c::PrecompileDiffusionCache, args)
        (; cv) = c
        uₑ = args.states.u
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            ∇u = function_gradient(cv, qp, uₑ)
            for i in 1:getnbasefunctions(cv)
                req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u) * dΩ
            end
        end
    end
    functional_value_type(::FunctionalKind{:precompile_energy}) = Float64
    function evaluate_cell_functional(::FunctionalKind{:precompile_energy}, c::PrecompileDiffusionCache, args)
        (; cv) = c
        uₑ = args.states.u
        Φ = 0.0
        for qp in 1:getnquadpoints(cv)
            ∇u = function_gradient(cv, qp, uₑ)
            Φ += (∇u ⋅ ∇u) / 2 * getdetJdV(cv, qp)
        end
        return Φ
    end

    # Bilinear diffusion a(u,v) = ∫ ∇v⋅∇u dΩ, the analytic-Jacobian
    # counterpart the AD-fallback double above has none of.
    struct PrecompileBilinearIntegrator <: AbstractBilinearIntegrator
        qrc::QuadratureRuleCollection
        field_name::Symbol
    end
    struct PrecompileBilinearCache{CV <: CellValues} <: AbstractVolumetricElementCache
        cv::CV
    end
    setup_element_cache(m::PrecompileBilinearIntegrator, sdh::SubDofHandler) =
        PrecompileBilinearCache(CellValues(getquadraturerule(m.qrc, sdh),
            Ferrite.getfieldinterpolation(sdh, m.field_name),
            geometric_subdomain_interpolation(sdh)))
    duplicate_for_device(device, c::PrecompileBilinearCache) =
        PrecompileBilinearCache(duplicate_for_device(device, c.cv))
    reinit_values!(c::PrecompileBilinearCache, cell) = Ferrite.reinit!(c.cv, cell)
    provides_analytic(::Type{<:PrecompileBilinearCache}, ::JacobianKind{:u}) = true
    function assemble_cell!(req::JacobianRequest{:u}, c::PrecompileBilinearCache, args)
        (; cv) = c
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:getnbasefunctions(cv), j in 1:getnbasefunctions(cv)
                req.K[i, j] += (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) * dΩ
            end
        end
    end
    # Mandatory: the bilinear form's residual is the element matrix acting on
    # the element vector, so the element composes into the same engine.
    function assemble_cell!(req::ResidualRequest, c::PrecompileBilinearCache, args)
        (; cv) = c
        uₑ = args.states.u
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            ∇u = function_gradient(cv, qp, uₑ)
            for i in 1:getnbasefunctions(cv)
                req.r[i] += (shape_gradient(cv, qp, i) ⋅ ∇u) * dΩ
            end
        end
    end
end

if Preferences.@load_preference("precompile_workload", true)
    @setup_workload begin
        grid = generate_grid(Quadrilateral, (2, 2))
        dh   = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        qrc      = QuadratureRuleCollection(2)
        strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
        u = zeros(ndofs(dh))
        r = zeros(ndofs(dh))

        @compile_workload begin
            nlop = setup_operator(strategy, PrecompileDiffusionIntegrator(qrc, :u), dh)
            update_linearization!(nlop, r, u, 0.0)   # fused JacobianResidual
            evaluate!(nlop, r, u, 0.0)                # residual alone
            evaluate_functional(nlop, FunctionalKind(:precompile_energy), u, nothing)

            bilop = setup_operator(strategy, PrecompileBilinearIntegrator(qrc, :u), dh)
            update_operator!(bilop, 0.0)
        end
    end
end
