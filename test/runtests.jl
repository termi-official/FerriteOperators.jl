# The example-elements subpackage is unregistered and lives in this repo.
# A [sources] section would wire it in by path, but Pkg only understands
# [sources] from Julia 1.11 on while the compat floor is 1.10 — so
# unresolvable path deps are dev'ed into the active test environment before
# anything loads. The develop lands in the on-disk environment, which
# ParallelTestRunner's worker processes share. Under Pkg.test only the
# subpackage is missing; in a direct `--project=test` run FerriteOperators
# itself is too, and the develop then records itself in test/Project.toml
# (keep that out of commits) and writes test/Manifest.toml — a leftover
# test/Manifest.toml breaks `Pkg.test` on Julia 1.10 ("can not merge
# projects"), so delete it after direct runs.
import Pkg
let specs = Pkg.PackageSpec[]
    resolvable(name, uuid) = Base.locate_package(Base.PkgId(Base.UUID(uuid), name)) !== nothing
    resolvable("FerriteOperators", "27d9367a-5072-424e-9c5f-fe582399bac3") ||
        push!(specs, Pkg.PackageSpec(path = joinpath(@__DIR__, "..")))
    resolvable("FerriteOperatorsExampleElements", "465fd1ee-fdf1-4c5c-a097-38ab1ffcf927") ||
        push!(specs, Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "lib", "FerriteOperatorsExampleElements")))
    isempty(specs) || Pkg.develop(specs)
end

using FerriteOperators
using ParallelTestRunner

args = parse_args(ARGS)
testsuite = find_tests(@__DIR__)
# Shared element doubles and testbeds, `include`d by the files that need them.
delete!(testsuite, "fixture_elements")
runtests(FerriteOperators, args; testsuite)
