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

# Every worker gets real threads: `addworker` otherwise pins JULIA_NUM_THREADS=1,
# which leaves the parallel assembly paths (atomic scatter, per-worker caches)
# covered in name only. `--threads` on the worker command line overrides that env
# var, and passing `exeflags` globally applies it to the pooled workers without
# creating any process a per-file `test_worker` hook would add on top of them.
const WORKER_THREADS = 2

# The runner's own default budgets *every* free byte at 2 GiB per worker and
# counts one core per worker; a worker here holds `WORKER_THREADS` cores, and the
# box has to keep running. Halving covers both: cores ÷ threads-per-worker, and
# 4 GiB of free memory per worker. That default is spelled out here rather than
# called through the unexported `ParallelTestRunner.default_njobs()`, so a runner
# minor release cannot break this entry point. Explicit `--jobs=N` still wins.
default_jobs() = max(1, min(Sys.CPU_THREADS, Int64(Sys.free_memory()) ÷ (2 * Int64(2)^30)) ÷ WORKER_THREADS)

argv = copy(ARGS)
any(startswith("--jobs"), argv) || push!(argv, "--jobs=$(default_jobs())")

args = parse_args(argv)
testsuite = find_tests(@__DIR__)
# Shared element doubles and testbeds, `include`d by the files that need them.
delete!(testsuite, "fixture_elements")
runtests(FerriteOperators, args; testsuite, exeflags = ["--threads=$(WORKER_THREADS)"])
