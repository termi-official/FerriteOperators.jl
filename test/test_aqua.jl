using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        FerriteDiffEq;
        ambiguities = false, # Tested below for now
        # unbound_args=true,
        # undefined_exports=true,
        # project_extras=true,
        deps_compat = true,
        piracies = true, # Comment out after https://github.com/Ferrite-FEM/Ferrite.jl/pull/864 is merged
        persistent_tasks = false,
    )
    Aqua.test_ambiguities(FerriteDiffEq) # Must be separate for now, see https://github.com/JuliaTesting/Aqua.jl/issues/77
end
