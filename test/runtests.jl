using FerriteOperators
using ParallelTestRunner

args = parse_args(ARGS)
testsuite = find_tests(@__DIR__)
runtests(FerriteOperators, args; testsuite)
