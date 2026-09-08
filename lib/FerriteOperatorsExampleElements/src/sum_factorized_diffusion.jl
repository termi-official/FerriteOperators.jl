@doc raw"""
    SumFactorizedDiffusionIntegrator(D, qrc, field_name)

The same bilinear form as [`SimpleBilinearDiffusionIntegrator`](@ref),
``a(u,v) = \int \nabla v \cdot D \nabla u \,dx``, written for the MATRIX-FREE
level: its cache evaluates the ACTION `yₑ = Kₑ·uₑ` by sum factorization
(Deville–Fischer–Mund) in `O(p^{d+1})` per cell instead of forming `Kₑ`.

`D` is a constant scalar (isotropic) or a `SymmetricTensor{2, dim}`. Set the
operator up with `form = MatrixFreeAction()`; the cache serves both element
mappings, so one definition of the element math runs one worker per element and
one workgroup per element.

Scope of this example: tensor-product Lagrange on `RefQuadrilateral` and
`RefHexahedron` (Ferrite ships orders 1–3 there), a tensor-product Gauss rule
through `qrc` — whose `order` is its POINT COUNT per direction — and a
tensor-product Lagrange geometry. Distorted cells are covered: the Jacobian is
evaluated at every quadrature point. Anything else is rejected at
`setup_element_cache`.

The cache has no element-matrix kernel: `FullAssembly` over this integrator is
refused where the matrix would be formed. Assemble
[`SimpleBilinearDiffusionIntegrator`](@ref) for a matrix of the same form.
"""
struct SumFactorizedDiffusionIntegrator{DT, QRC} <: AbstractBilinearIntegrator
    D::DT
    qrc::QRC
    field_name::Symbol
end

"""
    TensorProductValues

The values object of a tensor-product element: the 1D operators evaluated at
the 1D quadrature points, and the two lattice permutations that turn Ferrite's
local orderings into the lexicographic lattice sum factorization contracts over.

Everything is `isbits` and every extent is a type parameter, which is what lets
the whole object cross a device boundary unchanged and lets a cooperative
kernel take its group-local memory sizes from the element's type:

- `B`, `dB` — `φₐ(ξ_q)` and `φ'ₐ(ξ_q)` of the 1D Lagrange basis, `Nq × Nb`
- `Bg`, `dBg` — the same for the 1D geometric basis, `Nq × Nn`
- `w` — the 1D quadrature weights
- `dof_lattice`, `node_lattice` — lexicographic lattice index → Ferrite's local
  dof / local node

The per-cell storage lives beside it on the cache rather than here, because a
mutable field is what the device layout has to batch and adapt and the ext's
field-wise `Adapt` rule reaches element CACHES — see
[`SumFactorizedDiffusionElementCache`](@ref).
"""
struct TensorProductValues{dim, Nb, Nq, Nn, T, LB, LG, ND, NN}
    B::SMatrix{Nq, Nb, T, LB}
    dB::SMatrix{Nq, Nb, T, LB}
    Bg::SMatrix{Nq, Nn, T, LG}
    dBg::SMatrix{Nq, Nn, T, LG}
    w::SVector{Nq, T}
    dof_lattice::SVector{ND, Int32}
    node_lattice::SVector{NN, Int32}
end

"""
    SumFactorizedDiffusionElementCache

The cache [`SumFactorizedDiffusionIntegrator`](@ref) sets up: the diffusion
tensor and one [`TensorProductValues`](@ref). It carries BOTH matrix-free
kernel entries over one definition of the element math —
`apply_element_action!` for `WorkerPerElement` and the `cooperative_stage!`
pipeline for `CooperativeElement` — and the whole-element entry IS that
pipeline, run by a single lane.

`scratch` is the per-cell storage both mappings contract through: `1 + 2·dim`
lattice boxes of `max(Nb, Nq)^dim` entries — the element state, and a working
pair per gradient component. `WorkerPerElement` gives every worker its own,
batched with the worker as the leading index on a device;
[`CooperativeElement`](@ref) stages the same boxes in group-local memory
instead, so its device layout drops the batch. Nothing survives an item either
way: the pointwise geometry is formed at the quadrature point that consumes it.
"""
struct SumFactorizedDiffusionElementCache{dim, Nb, Nq, Nn, T, VT, DT, ST} <: AbstractVolumetricElementCache
    D::DT
    values::VT
    scratch::ST
end
SumFactorizedDiffusionElementCache(D, values::TensorProductValues{dim, Nb, Nq, Nn, T}, scratch) where {dim, Nb, Nq, Nn, T} =
    SumFactorizedDiffusionElementCache{dim, Nb, Nq, Nn, T, typeof(values), typeof(D), typeof(scratch)}(D, values, scratch)

element_value_type(::SumFactorizedDiffusionElementCache{dim, Nb, Nq, Nn, T}) where {dim, Nb, Nq, Nn, T} = T
# The values object holds no per-cell state, so there is nothing to position.
reinit_values!(::SumFactorizedDiffusionElementCache, cell) = nothing
# A matrix-free element forms no element matrix, so the engine's per-worker `Ke`
# stays empty instead of costing `ndofs_per_cell^2` per worker — which is the
# difference between a few megabytes and a few hundred on a device.
FerriteOperators.allocate_element_matrix(cache::SumFactorizedDiffusionElementCache, sdh) =
    zeros(element_value_type(cache), 0, 0)

# The reference operators are `isbits` and shared; the scratch is the one
# per-worker field, so the device pair batches and slices exactly that.
# `CooperativeElement` reads its boxes from group-local memory and never touches
# the batch, so its layout carries an empty one instead of one array per item.
duplicate_for_device(device, cache::SumFactorizedDiffusionElementCache) =
    SumFactorizedDiffusionElementCache(cache.D, cache.values, copy(cache.scratch))
setup_device_instances(device::AbstractGPUDevice, cache::SumFactorizedDiffusionElementCache, n) =
    SumFactorizedDiffusionElementCache(cache.D, cache.values,
        setup_device_instances(device, _scratch_prototype(device, cache.scratch), n))
device_worker_view(cache::SumFactorizedDiffusionElementCache, worker) =
    SumFactorizedDiffusionElementCache(cache.D, cache.values, device_worker_view(cache.scratch, worker))

_scratch_prototype(device, scratch) =
    FerriteOperators.element_mapping(device) isa CooperativeElement ? similar(scratch, 0, 0) : scratch

####################################
## Setup
####################################

_collection_order(::QuadratureRuleCollection{order}) where {order} = order

_lagrange_order(::Lagrange{<:Any, order}) where {order} = order
_lagrange_order(ip) = throw(ArgumentError(
    "sum factorization needs a tensor-product Lagrange basis, got $(typeof(ip))."))

# The 1D nodal basis and its derivative at `ξ`, on the equispaced nodes Ferrite's
# Lagrange interpolations place their dofs at.
function _lagrange_1d(::Type{T}, p::Int, ξ::AbstractVector) where {T}
    nodes = collect(T, range(-one(T), one(T); length = p + 1))
    nb, nq = p + 1, length(ξ)
    B, dB = zeros(T, nq, nb), zeros(T, nq, nb)
    for q in 1:nq, a in 1:nb
        value = one(T)
        for b in 1:nb
            b == a || (value *= (ξ[q] - nodes[b]) / (nodes[a] - nodes[b]))
        end
        derivative = zero(T)
        for c in 1:nb
            c == a && continue
            term = one(T) / (nodes[a] - nodes[c])
            for b in 1:nb
                (b == a || b == c) || (term *= (ξ[q] - nodes[b]) / (nodes[a] - nodes[b]))
            end
            derivative += term
        end
        B[q, a], dB[q, a] = value, derivative
    end
    return SMatrix{nq, nb, T}(B), SMatrix{nq, nb, T}(dB)
end

# Lexicographic lattice index (dimension 1 fastest) → the interpolation's own
# local index. A basis whose nodes are not that lattice is rejected here, which
# is also what certifies that it factorizes into the 1D basis above.
function _lattice_permutation(ip, ::Val{dim}, p::Int) where {dim}
    reference = Ferrite.reference_coordinates(ip)
    nodes = collect(range(-1.0, 1.0; length = p + 1))
    n = p + 1
    perm = Vector{Int32}(undef, n^dim)
    for l in 1:(n^dim)
        r = l - 1
        target = Vec{dim, Float64}(ntuple(c -> nodes[(r ÷ n^(c - 1)) % n + 1], dim))
        match = findfirst(x -> norm(x - target) < 1.0e-10, reference)
        match === nothing && throw(ArgumentError(
            "$(typeof(ip)) has no node at $target, so its basis is not the tensor product of a " *
            "1D Lagrange basis and cannot be contracted dimension by dimension."))
        perm[l] = match
    end
    return SVector{n^dim, Int32}(perm)
end

_diffusion_matrix(D::Number, ::Val{dim}, ::Type{T}) where {dim, T} = SMatrix{dim, dim, T}(
    ntuple(k -> (k - 1) % dim == (k - 1) ÷ dim ? T(D) : zero(T), dim * dim))
_diffusion_matrix(D::SymmetricTensor{2, dim}, ::Val{dim}, ::Type{T}) where {dim, T} = SMatrix{dim, dim, T}(
    ntuple(k -> T(D[(k - 1) % dim + 1, (k - 1) ÷ dim + 1]), dim * dim))

function setup_element_cache(model::SumFactorizedDiffusionIntegrator, sdh::SubDofHandler)
    T = element_value_type(model.qrc)
    ip = Ferrite.getfieldinterpolation(sdh, model.field_name)
    shape = Ferrite.getrefshape(ip)
    shape <: Ferrite.RefHypercube || throw(ArgumentError(
        "SumFactorizedDiffusionIntegrator covers tensor-product reference shapes " *
        "(RefQuadrilateral, RefHexahedron), got $shape."))
    dim = Ferrite.getrefdim(shape)
    dim in (2, 3) || throw(ArgumentError(
        "SumFactorizedDiffusionIntegrator covers 2D and 3D cells, got reference dimension $dim."))

    ip_geo = Ferrite.geometric_interpolation(typeof(get_first_cell(sdh)))
    p, pg = _lagrange_order(ip), _lagrange_order(ip_geo)
    qr = QuadratureRule{RefLine}(T, _collection_order(model.qrc))
    ξ = [x[1] for x in Ferrite.getpoints(qr)]
    nq = length(ξ)
    B, dB = _lagrange_1d(T, p, ξ)
    Bg, dBg = _lagrange_1d(T, pg, ξ)

    values = TensorProductValues{dim, p + 1, nq, pg + 1, T,
                                 nq * (p + 1), nq * (pg + 1), (p + 1)^dim, (pg + 1)^dim}(
        B, dB, Bg, dBg, SVector{nq, T}(Ferrite.getweights(qr)),
        _lattice_permutation(ip, Val(dim), p), _lattice_permutation(ip_geo, Val(dim), pg))
    scratch = zeros(T, max(p + 1, nq)^dim, 1 + 2dim)
    return SumFactorizedDiffusionElementCache(_diffusion_matrix(model.D, Val(dim), T), values, scratch)
end

####################################
## The lattice
####################################

# The scratch lattice is a `max(Nb, Nq)^dim` box, so ONE index arithmetic serves
# every stage whatever the current extent along an axis is. Slots outside the
# current extent are computed and discarded; a valid slot never reads one.

# Box index (1-based) → index into an `n^dim` lattice, or `0` where the box slot
# lies outside it.
@inline function _lattice_index(i::Int, ::Val{n}, ::Val{M}, ::Val{dim}) where {n, M, dim}
    r, index, stride = i - 1, 0, 1
    for _ in 1:dim
        digit = r % M
        digit ≥ n && return 0
        r ÷= M
        index += digit * stride
        stride *= n
    end
    return index + 1
end

"""
One contraction, `dst[…, iₖ, …] = Σₐ Op[iₖ, a] · src[…, a, …]` along axis `k` of
the scratch box, over the spectator slabs `lane` owns.

This is the ONE loop both element mappings run: `lane = nlanes = 1` walks every
slab, and a workgroup of `nlanes` splits them. `dcol`/`scol` are scratch
columns, never the same one.
"""
@inline function _contract!(scratch, dcol::Int, scol::Int, Op, k::Int, nout::Int, nin::Int,
        ::Val{M}, ::Val{dim}, lane::Int, nlanes::Int) where {M, dim}
    stride = M^(k - 1)
    for s in (lane - 1):nlanes:(M^(dim - 1) - 1)
        base = (s % stride) + (s ÷ stride) * stride * M
        for io in 1:nout
            acc = zero(eltype(scratch))
            for ii in 1:nin
                @inbounds acc += Op[io, ii] * scratch[base + (ii - 1) * stride + 1, scol]
            end
            @inbounds scratch[base + (io - 1) * stride + 1, dcol] = acc
        end
    end
    return nothing
end

# Scratch columns: the element state, then a working pair per gradient
# component. Successive contractions ping-pong between the pair, so the stage
# index alone decides which column holds what.
@inline _col_a(dim::Int, d::Int) = 1 + d
@inline _col_b(dim::Int, d::Int) = 1 + dim + d
@inline _col_gradient(dim::Int, d::Int) = isodd(dim) ? _col_a(dim, d) : _col_b(dim, d)
@inline _col_other(dim::Int, d::Int) = isodd(dim) ? _col_b(dim, d) : _col_a(dim, d)

####################################
## The cooperative pipeline
####################################

cooperative_lattice_dim(::SumFactorizedDiffusionElementCache{dim}) where {dim} = dim
cooperative_group_size(::SumFactorizedDiffusionElementCache{dim, Nb, Nq}) where {dim, Nb, Nq} =
    max(Nb, Nq)^(dim - 1)
cooperative_scratch_shape(::SumFactorizedDiffusionElementCache{dim, Nb, Nq}) where {dim, Nb, Nq} =
    (Val(max(Nb, Nq)^dim), Val(1 + 2dim))

@inline function cooperative_load!(scratch, cache::SumFactorizedDiffusionElementCache{dim, Nb, Nq, Nn, T},
        uₑ, lane::Int, nlanes::Int) where {dim, Nb, Nq, Nn, T}
    M = max(Nb, Nq)
    lattice = cache.values.dof_lattice
    # A lane zeroes the same lattice slots it then fills, so the two loops need
    # no barrier between them.
    for column in 1:(1 + 2dim), i in lane:nlanes:(M^dim)
        @inbounds scratch[i, column] = zero(T)
    end
    for i in lane:nlanes:(M^dim)
        l = _lattice_index(i, Val(Nb), Val(M), Val(dim))
        l == 0 && continue
        @inbounds scratch[i, 1] = convert(T, uₑ[lattice[l]])
    end
    return nothing
end

@inline function cooperative_stage!(scratch, cache::SumFactorizedDiffusionElementCache{dim}, args,
        stage::Int, lane::Int, nlanes::Int) where {dim}
    if stage ≤ dim
        _forward_stage!(scratch, cache, stage, lane, nlanes)
        # The map runs on the slab the contraction above just wrote, so no
        # barrier separates them.
        stage == dim && _pointwise_stage!(scratch, cache, args, lane, nlanes)
    else
        _backward_stage!(scratch, cache, stage - dim, lane, nlanes)
    end
    return nothing
end

# Forward stage `k`: contract axis `k` of every gradient component with the
# derivative operator in that component's direction and the value operator
# elsewhere.
@inline function _forward_stage!(scratch, cache::SumFactorizedDiffusionElementCache{dim, Nb, Nq},
        k::Int, lane::Int, nlanes::Int) where {dim, Nb, Nq}
    values = cache.values
    for d in 1:dim
        source = k == 1 ? 1 : (iseven(k) ? _col_a(dim, d) : _col_b(dim, d))
        destination = isodd(k) ? _col_a(dim, d) : _col_b(dim, d)
        _contract!(scratch, destination, source, k == d ? values.dB : values.B,
                   k, Nq, Nb, Val(max(Nb, Nq)), Val(dim), lane, nlanes)
    end
    return nothing
end

# Backward stage `j`: the transposed contraction of axis `j`, back from the
# quadrature lattice towards the dof lattice.
@inline function _backward_stage!(scratch, cache::SumFactorizedDiffusionElementCache{dim, Nb, Nq},
        j::Int, lane::Int, nlanes::Int) where {dim, Nb, Nq}
    values = cache.values
    for d in 1:dim
        source = isodd(j) ? _col_gradient(dim, d) : _col_other(dim, d)
        destination = isodd(j) ? _col_other(dim, d) : _col_gradient(dim, d)
        _contract!(scratch, destination, source, transpose(j == d ? values.dB : values.B),
                   j, Nb, Nq, Val(max(Nb, Nq)), Val(dim), lane, nlanes)
    end
    return nothing
end

# The pointwise map `Wq = w_q det(Jq) Jq⁻¹ D Jq⁻ᵀ`, applied in place to the
# reference gradient. The lane that produced a quadrature point is the one that
# maps it, which is why the geometry needs neither storage nor a barrier.
@inline function _pointwise_stage!(scratch, cache::SumFactorizedDiffusionElementCache{dim, Nb, Nq, Nn, T},
        args, lane::Int, nlanes::Int) where {dim, Nb, Nq, Nn, T}
    M = max(Nb, Nq)
    stride = M^(dim - 1)
    coordinates = getcoordinates(args.cell)
    for s in (lane - 1):nlanes:(stride - 1)
        _lattice_index(s + 1, Val(Nq), Val(M), Val(dim)) == 0 && continue
        for last in 1:Nq
            q = ntuple(c -> c == dim ? last : (s ÷ M^(c - 1)) % M + 1, Val(dim))
            i = s + (last - 1) * stride + 1
            g = SVector{dim, T}(ntuple(d -> (@inbounds scratch[i, _col_gradient(dim, d)]), Val(dim)))
            mapped = _pointwise_map(cache, coordinates, q) * g
            for d in 1:dim
                @inbounds scratch[i, _col_gradient(dim, d)] = mapped[d]
            end
        end
    end
    return nothing
end

@inline function _pointwise_map(cache::SumFactorizedDiffusionElementCache{dim, Nb, Nq, Nn, T},
        coordinates, q::NTuple{dim, Int}) where {dim, Nb, Nq, Nn, T}
    values = cache.values
    J = zero(SMatrix{dim, dim, T})
    for l in 1:(Nn^dim)
        r = l - 1
        n = ntuple(c -> (r ÷ Nn^(c - 1)) % Nn + 1, Val(dim))
        ∇M = SVector{dim, T}(ntuple(Val(dim)) do b
            (@inbounds values.dBg[q[b], n[b]]) *
                prod(ntuple(c -> c == b ? one(T) : (@inbounds values.Bg[q[c], n[c]]), Val(dim)))
        end)
        x = @inbounds coordinates[values.node_lattice[l]]
        J += SVector{dim, T}(ntuple(c -> T(x[c]), Val(dim))) * ∇M'
    end
    Jinv = inv(J)
    weight = prod(ntuple(c -> (@inbounds values.w[q[c]]), Val(dim)))
    return (weight * det(J)) * (Jinv * cache.D * Jinv')
end

# The last backward contraction, summed over the gradient components and
# accumulated into the element vector through the dof permutation. Lanes own
# disjoint dofs, so nothing accumulates across lanes.
@inline function cooperative_store!(yₑ, scratch, cache::SumFactorizedDiffusionElementCache{dim, Nb, Nq, Nn, T},
        lane::Int, nlanes::Int) where {dim, Nb, Nq, Nn, T}
    values = cache.values
    M = max(Nb, Nq)
    stride = M^(dim - 1)
    for s in (lane - 1):nlanes:(stride - 1)
        for io in 1:Nb
            l = _lattice_index(s + (io - 1) * stride + 1, Val(Nb), Val(M), Val(dim))
            l == 0 && continue
            acc = zero(T)
            for d in 1:dim
                Op = dim == d ? values.dB : values.B
                column = iseven(dim - 1) ? _col_gradient(dim, d) : _col_other(dim, d)
                for ii in 1:Nq
                    @inbounds acc += Op[ii, io] * scratch[s + (ii - 1) * stride + 1, column]
                end
            end
            @inbounds yₑ[values.dof_lattice[l]] += acc
        end
    end
    return nothing
end

####################################
## The whole-element entries
####################################

"""
    apply_element_action!(yₑ, cache::SumFactorizedDiffusionElementCache, uₑ, args)

The worker-per-element action: the cooperative pipeline run by a single lane
over the worker's own scratch. One element definition, two execution mappings —
the stage bodies are shared verbatim, and only the slab range a worker walks
differs.
"""
function apply_element_action!(yₑ, cache::SumFactorizedDiffusionElementCache{dim},
        uₑ, args::CellArgs) where {dim}
    scratch = cache.scratch
    cooperative_load!(scratch, cache, uₑ, 1, 1)
    for stage in 1:(2dim - 1)
        cooperative_stage!(scratch, cache, args, stage, 1, 1)
    end
    cooperative_store!(yₑ, scratch, cache, 1, 1)
    return nothing
end

# The bilinear form's residual IS the action, and the residual kernel is
# mandatory for every element cache.
assemble_cell!(req::ResidualRequest, cache::SumFactorizedDiffusionElementCache, args::CellArgs) =
    apply_element_action!(req.r, cache, args.states.u, args)

# `provides_analytic` stays `false` for the Jacobian kinds: this cache has no
# element matrix to declare. The method exists so a `FullAssembly` sweep says
# what is wrong instead of reporting a `MethodError` or an empty buffer.
assemble_cell!(::Union{JacobianRequest, JacobianResidualRequest},
        cache::SumFactorizedDiffusionElementCache, ::CellArgs) = throw(ArgumentError(
    "$(nameof(typeof(cache))) forms no element matrix — it evaluates the operator's ACTION, and " *
    "its element-matrix buffer is empty by construction. Set the operator up with " *
    "`form = MatrixFreeAction()`, or assemble `SimpleBilinearDiffusionIntegrator`, which is the " *
    "same bilinear form with an element-matrix kernel."))
