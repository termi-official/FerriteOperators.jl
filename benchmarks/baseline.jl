using FerriteOperators, FerriteOperatorsExampleElements, Tensors, TimerOutputs, BenchmarkTools  # Ferrite via FerriteOperators reexport
TimerOutputs.enable_debug_timings(FerriteOperators)

struct NeoHookean
    E::Float64
    ν::Float64
end
function (p::NeoHookean)(F)
    (; E, ν) = p
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    C = tdot(F)
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end
function ψ(F, p::NeoHookean)
    (; E, ν) = p
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    C = tdot(F)
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

function constitutive_driver(F, mp)
    ## Compute all derivatives in one function call
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(y -> ψ(y, mp), F, :all)
    return ∂Ψ∂F, ∂²Ψ∂F²
end;

function assemble_element!(ke, ge, cell, cv, mp, ue)
    ## Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    b = Vec{3}((0.0, -0.5, 0.0)) # Body force
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ## Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        ## Compute stress and tangent
        P, ∂P∂F = constitutive_driver(F, mp)

        ## Loop over test functions
        for i in 1:ndofs
            ## Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            ## Add contribution to the residual from this test function
            ge[i] += (∇δui ⊡ P - δui ⋅ b) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                ## Add contribution to the tangent
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end

    return
end

function assemble_global!(K, g, dh, cv, mp, u)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)
    ue = zeros(n)

    ## start_assemble resets K and g
    assembler = start_assemble(K, g)

    ## Loop over all cells in the grid
    @timeit "assemble" for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue .= view(u, global_dofs) # element dofs
        assemble_element!(ke, ge, cell, cv, mp, ue)
        assemble!(assembler, global_dofs, ke, ge)
    end
    return
end

function benchmark_assembly_plain(N = 20)
    ## Generate a grid
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(Hexahedron, (N, N, N), left, right)

    ## Material parameters
    mp = NeoHookean(10.0, 0.3)

    ## Finite element base
    ip = Lagrange{RefHexahedron, 1}()^3
    qr = QuadratureRule{RefHexahedron}(2)
    cv = CellValues(qr, ip)

    ## DofHandler
    dh = DofHandler(grid)
    add!(dh, :u, ip) # Add a displacement field
    close!(dh)

    ## Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    u = zeros(_ndofs)

    ## Create sparse matrix and residual vector
    K = allocate_matrix(dh)
    g = zeros(_ndofs)

    assemble_global!(K, g, dh, cv, mp, u)

    @btime assemble_global!($K, $g, $dh, $cv, $mp, $u)

    reset_timer!()
    assemble_global!(K, g, dh, cv, mp, u)
    print_timer()
end

benchmark_assembly_plain()

function benchmark_assembly_ferriteoperators(N = 20)
    ## Generate a grid
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(Hexahedron, (N, N, N), left, right)

    ## Material
    integrator = SimpleHyperelasticityIntegrator(
        NeoHookean(10.0, 0.3),
        QuadratureRuleCollection(2),
        :u
    )

    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}()^3) # Add a displacement field
    close!(dh)

    ## Setup the operator
    op = setup_operator(AssemblyStrategy(SequentialCPUDevice()), integrator, dh)
    u = zeros(ndofs(op.engine.dh))
    g = zeros(ndofs(op.engine.dh))
    update_linearization!(op, g, u, 0.5)

    reset_timer!()
    update_linearization!(op, g, u, 0.5)
    print_timer()

    @btime update_linearization!($op, $g, $u, 0.5)
end

benchmark_assembly_ferriteoperators()
