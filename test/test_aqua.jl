using Aqua, FerriteOperators

@testset "Aqua.jl" begin
    Aqua.test_all(
        FerriteOperators;
        ambiguities = false, # Tested below for now
        # unbound_args=true,
        # undefined_exports=true,
        # project_extras=true,
        deps_compat = true,
        piracies = false, 
        persistent_tasks = false,
    )
    Aqua.test_ambiguities(FerriteOperators) # Must be separate for now, see https://github.com/JuliaTesting/Aqua.jl/issues/77
end
