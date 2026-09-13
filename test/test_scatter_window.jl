# G2 — the two-window scatter (`element_scatter_length`): a test-local family
# whose matrix-free action GATHERS a wider window than it SCATTERS, exercising
# the seam `src/core/tasks.jl` and `ext/FerriteOperatorsKernelAbstractionsExt.jl`
# consult it through, and the setup wall that rejects the mismatch where the
# declaration is absent.
#
# `WindowOverrideCursor` wraps whatever the package's own default iterator
# builds for a kind/subdomain (a host `Ferrite.CellCache` or a device
# `DeviceCellCursor`, picked by `default_assembly_iterator`) and overrides only
# the two dof-window accessors through a pair of plain functions — so ONE
# iterator type serves both the host and `KernelAbstractionsDevice(KA.CPU())`.
#
# Two shapes, for a structural reason spelled out at each cache:
# - `PrefixCache` gathers a cell's OWN dofs twice (`2n`) and scatters only the
#   first `n` — a genuine size-changing prefix, the DG shape `element_local_length`/
#   `element_scatter_length` target. It runs under `WorkerPerElement()`
#   (`Stored()`), which places no restriction on the cache's storage.
# - `ReorderedCache` gathers a cell's own `n` dofs UNCHANGED and scatters them
#   in REVERSED order — same length, different identity. It runs under
#   `LanesPerElement()`, which today accepts ONLY `storage = ElementAssembly()`
#   (`_assert_mapping_capability(::LanesPerElement, ::StorageElection, cache)`
#   throws for every other storage); `ElementAssemblyCache` fixes its own
#   `element_local_length` at `ndofs_per_cell(sdh)` for gather AND scatter, so a
#   genuine size mismatch cannot reach the lane kernel until a storage election
#   shaped for it exists (the BlockRowAssembly work). The same-length,
#   different-identity shape still exercises exactly what changed in
#   `_lane_action!`: the scatter address, not the gather.

using FerriteOperators
using Test
using LinearAlgebra
import FerriteOperators: iterator_dofs, iterator_scatter_address, iterator_handler,
    position_iterator, device_worker_view, setup_device_instances, default_assembly_iterator,
    element_scatter_length, element_local_length, reinit_values!, assemble_cell!,
    apply_element_action!, allocate_element_residual_vector, duplicate_for_device,
    setup_element_cache, assembly_iterator, provides_analytic
# FerriteKAExt — the device handler, `distribute_to_workers` and every `Adapt`
# rule the device kernel builds on — is triggered by these four together.
import Adapt, GPUArrays, GPUArraysCore
import KernelAbstractions as KA

####################################
## The shared iterator wrapper
####################################

struct WindowOverrideCursor{IT, G, S}
    inner::IT
    gather::G    # inner -> gather dof AbstractVector
    scatter::S   # inner -> scatter dof AbstractVector
end
Adapt.@adapt_structure WindowOverrideCursor

Ferrite.cellid(w::WindowOverrideCursor) = Ferrite.cellid(w.inner)
iterator_dofs(w::WindowOverrideCursor) = w.gather(w.inner)
iterator_scatter_address(w::WindowOverrideCursor) = w.scatter(w.inner)
iterator_handler(w::WindowOverrideCursor) = iterator_handler(w.inner)
position_iterator(w::WindowOverrideCursor, item, flags) =
    WindowOverrideCursor(position_iterator(w.inner, item, flags), w.gather, w.scatter)
device_worker_view(w::WindowOverrideCursor, worker) =
    WindowOverrideCursor(device_worker_view(w.inner, worker), w.gather, w.scatter)
setup_device_instances(device::KernelAbstractionsDevice, w::WindowOverrideCursor, n) =
    WindowOverrideCursor(setup_device_instances(device, w.inner, n), w.gather, w.scatter)

# Zero-copy window views, so neither shape allocates per item.
struct DoubledDofs{V <: AbstractVector{Int}} <: AbstractVector{Int}
    single::V
end
Base.size(d::DoubledDofs) = (2 * length(d.single),)
Base.IndexStyle(::Type{<:DoubledDofs}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(d::DoubledDofs, i::Int) =
    d.single[i <= length(d.single) ? i : i - length(d.single)]

struct ReversedDofs{V <: AbstractVector{Int}} <: AbstractVector{Int}
    fwd::V
end
Base.size(d::ReversedDofs) = size(d.fwd)
Base.IndexStyle(::Type{<:ReversedDofs}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(d::ReversedDofs, i::Int) = d.fwd[length(d.fwd) + 1 - i]

_doubled_gather(inner) = DoubledDofs(iterator_dofs(inner))
_prefix_scatter(inner) = iterator_dofs(inner)
_reversed_scatter(inner) = ReversedDofs(iterator_dofs(inner))

# Deterministic, dependency-free entries (no RNG).
_entry(i, j) = sin(0.37i + 0.53j) + 0.1cos(0.19i * j)

####################################
## PrefixCache: gather 2n, scatter the first n — genuine size mismatch
####################################

struct PrefixIntegrator <: AbstractBilinearIntegrator end

# `Declared` and `N` are TYPE parameters, not fields — `n` a plain field would
# make `element_local_length`/`element_scatter_length` return a `Val` whose
# type depends on a RUNTIME value, exactly the instability `ElementAssemblyCache`
# avoids by keeping its own `local_size::Val{ND}` a type parameter (`n` folds
# into the type as `N`, so `Val(2N)`/`Val(N)` are inferred, not boxed).
# `PrefixCache{true}` names `element_scatter_length`, `PrefixCache{false}` does
# not (the undeclared arm the setup wall must reject).
struct PrefixCache{Declared, N} <: FerriteOperators.AbstractVolumetricElementCache
    W::Matrix{Float64}   # n × 2n
end
PrefixCache{Declared}(n::Int) where {Declared} = PrefixCache{Declared, n}([_entry(i, j) for i in 1:n, j in 1:(2n)])

setup_element_cache(::PrefixIntegrator, sdh::SubDofHandler) = PrefixCache{true}(ndofs_per_cell(sdh))

duplicate_for_device(device, c::PrefixCache) = c
setup_device_instances(device::KernelAbstractionsDevice, c::PrefixCache{D, N}, n) where {D, N} =
    PrefixCache{D, N}(Adapt.adapt(device.backend, c.W))
device_worker_view(c::PrefixCache, worker) = c

element_local_length(::PrefixCache{Declared, N}) where {Declared, N} = Val(2N)
element_scatter_length(::PrefixCache{true, N}) where {N} = Val(N)   # PrefixCache{false,_} keeps the `nothing` default

allocate_element_residual_vector(::PrefixCache{Declared, N}, sdh) where {Declared, N} = zeros(N)

assemble_cell!(::ResidualRequest, ::PrefixCache, ::CellArgs) = nothing
reinit_values!(::PrefixCache, cell) = nothing

function apply_element_action!(yₑ, c::PrefixCache{Declared, N}, uₑ, args::CellArgs) where {Declared, N}
    for i in 1:N
        acc = zero(eltype(yₑ))
        for j in 1:(2N)
            acc += c.W[i, j] * uₑ[j]
        end
        yₑ[i] += acc
    end
    return nothing
end

assembly_iterator(kind, ::PrefixCache, sdh) =
    WindowOverrideCursor(default_assembly_iterator(kind, sdh), _doubled_gather, _prefix_scatter)

# `PrefixIntegrator` always answers the DECLARED cache; the undeclared arm the
# setup wall must reject needs its own integrator, since a cache type — not an
# instance — is what `element_scatter_length` dispatches on.
struct UndeclaredPrefixIntegrator <: AbstractBilinearIntegrator end
setup_element_cache(::UndeclaredPrefixIntegrator, sdh::SubDofHandler) = PrefixCache{false}(ndofs_per_cell(sdh))

prefix_reference(dh, W, n) = function (u)
    y = zeros(ndofs(dh))
    for cell in 1:getncells(Ferrite.get_grid(dh))
        d = celldofs(dh, cell)
        uₑ = vcat(u[d], u[d])
        y[d] .+= W * uₑ
    end
    return y
end

####################################
## AgreeingCache: the wall's negative control — undeclared, agreeing seams
####################################
# An ORDINARY cache: `element_local_length` is a `Val`, `element_scatter_length`
# is undeclared, and NO custom iterator is named, so `assembly_iterator` falls
# through to the package default (`Ferrite.CellCache` / `DeviceCellCursor`),
# whose `iterator_scatter_address` is the untouched catch-all. The wall must NOT
# fire here — this is every shipped `element_local_length`-declaring cache's
# shape.

struct AgreeingIntegrator <: AbstractBilinearIntegrator end
struct AgreeingCache{N} <: FerriteOperators.AbstractVolumetricElementCache end
AgreeingCache(n::Int) = AgreeingCache{n}()
setup_element_cache(::AgreeingIntegrator, sdh::SubDofHandler) = AgreeingCache(ndofs_per_cell(sdh))
duplicate_for_device(device, c::AgreeingCache) = c
element_local_length(::AgreeingCache{N}) where {N} = Val(N)
allocate_element_residual_vector(::AgreeingCache{N}, sdh) where {N} = zeros(N)
assemble_cell!(::ResidualRequest, ::AgreeingCache, ::CellArgs) = nothing
reinit_values!(::AgreeingCache, cell) = nothing
apply_element_action!(yₑ, ::AgreeingCache, uₑ, args::CellArgs) = nothing

####################################
## ReorderedCache: gather n, scatter n in REVERSED order — same length,
## different identity, the shape `ElementAssembly()`/`LanesPerElement` can host
####################################

struct ReorderedIntegrator <: AbstractBilinearIntegrator end
struct ReorderedCache{N} <: FerriteOperators.AbstractVolumetricElementCache
    K::Matrix{Float64}   # n × n
end
ReorderedCache(n::Int) = ReorderedCache{n}([_entry(i, j) + (i == j ? 2.0 : 0.0) for i in 1:n, j in 1:n])

setup_element_cache(::ReorderedIntegrator, sdh::SubDofHandler) = ReorderedCache(ndofs_per_cell(sdh))
duplicate_for_device(device, c::ReorderedCache) = c
setup_device_instances(device::KernelAbstractionsDevice, c::ReorderedCache{N}, n) where {N} =
    ReorderedCache{N}(Adapt.adapt(device.backend, c.K))
device_worker_view(c::ReorderedCache, worker) = c

provides_analytic(::Type{<:ReorderedCache}, ::JacobianKind{:u}) = true
assemble_cell!(req::JacobianRequest{:u}, c::ReorderedCache, ::CellArgs) = (req.K .+= c.K; nothing)
assemble_cell!(::ResidualRequest, ::ReorderedCache, ::CellArgs) = nothing
reinit_values!(::ReorderedCache, cell) = nothing

element_scatter_length(::ReorderedCache{N}) where {N} = Val(N)   # SAME length as `ndofs_per_cell` — see the header note

assembly_iterator(kind, ::ReorderedCache, sdh) =
    WindowOverrideCursor(default_assembly_iterator(kind, sdh), iterator_dofs, _reversed_scatter)

reordered_reference(dh, K, n) = function (u)
    y = zeros(ndofs(dh))
    for cell in 1:getncells(Ferrite.get_grid(dh))
        d = celldofs(dh, cell)
        yₑ = K * u[d]
        for i in 1:n
            y[d[n + 1 - i]] += yₑ[i]
        end
    end
    return y
end

####################################
## Testbed
####################################

function scatter_window_testbed()
    grid = generate_grid(Quadrilateral, (3, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    return dh
end

ka_cpu(mapping; storage = Stored()) = AssemblyStrategy(
    MatrixFreeAction(; element_mapping = mapping, storage), SequentialScheduling(),
    KernelAbstractionsDevice(KA.CPU(); value_type = Float64, index_type = Int,
                             items_per_worker = 2, max_workgroup_size = 8))

@testset "G2: the two-window scatter (element_scatter_length)" begin
    dh = scatter_window_testbed()
    n = ndofs_per_cell(first(dh.subdofhandlers))
    u = [sin(0.7i) + 0.2cos(1.3i) for i in 1:ndofs(dh)]

    @testset "PrefixCache — WorkerPerElement, genuine gather ≠ scatter (prefix)" begin
        cache = PrefixCache{true}(n)
        reference = prefix_reference(dh, cache.W, n)(u)

        @testset "SequentialCPUDevice" begin
            op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                                PrefixIntegrator(), dh)
            y = zeros(ndofs(dh))
            mul!(y, op, u)
            @test y ≈ reference rtol = 1.0e-12

            # Allocation check on the NEW scatter path (item 6d): warm up, then
            # measure — a testset-scope closure would box and count itself.
            z = zeros(ndofs(dh))
            measure(op, z, u) = (mul!(z, op, u); mul!(z, op, u); @allocated mul!(z, op, u))
            @test measure(op, z, u) == 0
        end

        @testset "KernelAbstractionsDevice(KA.CPU())" begin
            op = setup_operator(ka_cpu(WorkerPerElement()), PrefixIntegrator(), dh)
            y = zeros(ndofs(dh))
            mul!(y, op, u)
            @test y ≈ reference rtol = 1.0e-12
        end
    end

    @testset "ReorderedCache — LanesPerElement, same length, different identity" begin
        cache = ReorderedCache(n)
        reference = reordered_reference(dh, cache.K, n)(u)
        # The permutation is load-bearing: a scatter that used `dofs[i]` (the
        # GATHER window) instead of the declared `iterator_scatter_address`
        # would silently produce THIS vector instead.
        unpermuted = let y = zeros(ndofs(dh))
            for cell in 1:getncells(Ferrite.get_grid(dh))
                d = celldofs(dh, cell)
                y[d] .+= cache.K * u[d]
            end
            y
        end
        @test !(reference ≈ unpermuted)

        for (label, mapping) in (("WorkerPerElement", WorkerPerElement()), ("LanesPerElement", LanesPerElement()))
            @testset "$label" begin
                op = setup_operator(ka_cpu(mapping; storage = ElementAssembly()), ReorderedIntegrator(), dh)
                y = zeros(ndofs(dh))
                mul!(y, op, u)
                @test y ≈ reference rtol = 1.0e-12
                @test !(y ≈ unpermuted)
            end
        end
    end

    @testset "the setup wall" begin
        @testset "undeclared + differing seams → rejected" begin
            for (label, device) in ("SequentialCPUDevice" => SequentialCPUDevice(),
                                    "KA.CPU device" => KernelAbstractionsDevice(KA.CPU()))
                strategy = AssemblyStrategy(MatrixFreeAction(), SequentialScheduling(), device)
                err = @test_throws ArgumentError setup_operator(strategy, UndeclaredPrefixIntegrator(), dh)
                msg = err.value.msg
                @test occursin("element_scatter_length", msg)
                @test occursin("element_local_length", msg)
                @test occursin("iterator_scatter_address", msg)
                @test occursin("PrefixCache", msg)
            end
        end

        @testset "nothing + agreeing seams → not rejected" begin
            op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                                AgreeingIntegrator(), dh)
            @test op isa MatrixFreeFerriteOperator
        end

        @testset "declared + differing seams → not rejected" begin
            op = setup_operator(AssemblyStrategy(SequentialCPUDevice(); form = MatrixFreeAction()),
                                PrefixIntegrator(), dh)
            @test op isa MatrixFreeFerriteOperator
        end
    end
end
