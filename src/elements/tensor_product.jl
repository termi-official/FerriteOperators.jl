####################################
## The tensor-product sum-factorization core
####################################
#
# The element-agnostic half of a matrix-free tensor-product element: the 1D
# reference operators, the lattice permutations, the contraction loop, and the
# pipeline both element mappings run over them. What an ELEMENT adds is its
# POINTWISE MAP at the quadrature point — the `D` block of the MFEM/libCEED
# decomposition — and the storage that map reads.
#
# Every extent reaches a `Val` through a method's own type parameters rather
# than through a value computed from one, so the lattice arithmetic stays
# constant-folded inside a device kernel.

"""
    TensorProductValues(sdh::SubDofHandler, field_name::Symbol, qrc)

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

Scope: tensor-product Lagrange on `RefQuadrilateral` and `RefHexahedron` (Ferrite
ships orders 1–3 there), a tensor-product Gauss rule through `qrc` — whose
`order` is its POINT COUNT per direction — and a tensor-product Lagrange
geometry. Anything else is rejected here, and rejecting a basis whose nodes are
not that lattice is also what certifies that it factorizes into the 1D basis.

The per-cell storage lives on the element CACHE
([`AbstractTensorProductElementCache`](@ref)) rather than here: a mutable field
is what a device layout has to batch and adapt.

!!! warning "Experimental surface"
    The tensor-product core, its names and the element entry points it calls
    may change in a minor release.
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

element_value_type(::TensorProductValues{dim, Nb, Nq, Nn, T}) where {dim, Nb, Nq, Nn, T} = T
Ferrite.getnquadpoints(::TensorProductValues{dim, Nb, Nq}) where {dim, Nb, Nq} = Nq^dim

_tensor_product_collection_order(::QuadratureRuleCollection{order}) where {order} = order

_tensor_product_order(::Lagrange{<:Any, order}) where {order} = order
_tensor_product_order(ip) = throw(ArgumentError(
    "sum factorization needs a tensor-product Lagrange basis, got $(typeof(ip))."))

# The 1D nodal basis and its derivative at `ξ`, on the equispaced nodes Ferrite's
# Lagrange interpolations place their dofs at.
function _tensor_product_lagrange_1d(::Type{T}, p::Int, ξ::AbstractVector) where {T}
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
# local index.
function _tensor_product_permutation(ip, ::Val{dim}, p::Int) where {dim}
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

function TensorProductValues(sdh::SubDofHandler, field_name::Symbol, qrc)
    T = element_value_type(qrc)
    ip = Ferrite.getfieldinterpolation(sdh, field_name)
    shape = Ferrite.getrefshape(ip)
    shape <: Ferrite.RefHypercube || throw(ArgumentError(
        "Sum factorization covers tensor-product reference shapes (RefQuadrilateral, " *
        "RefHexahedron), got $shape."))
    dim = Ferrite.getrefdim(shape)
    dim in (2, 3) || throw(ArgumentError(
        "Sum factorization covers 2D and 3D cells, got reference dimension $dim."))

    ip_geo = Ferrite.geometric_interpolation(typeof(get_first_cell(sdh)))
    p, pg = _tensor_product_order(ip), _tensor_product_order(ip_geo)
    qr = QuadratureRule{RefLine}(T, _tensor_product_collection_order(qrc))
    ξ = [x[1] for x in Ferrite.getpoints(qr)]
    nq = length(ξ)
    B, dB = _tensor_product_lagrange_1d(T, p, ξ)
    Bg, dBg = _tensor_product_lagrange_1d(T, pg, ξ)

    return TensorProductValues{dim, p + 1, nq, pg + 1, T,
                               nq * (p + 1), nq * (pg + 1), (p + 1)^dim, (pg + 1)^dim}(
        B, dB, Bg, dBg, SVector{nq, T}(Ferrite.getweights(qr)),
        _tensor_product_permutation(ip, Val(dim), p), _tensor_product_permutation(ip_geo, Val(dim), pg))
end

####################################
## What the element contracts to at a quadrature point
####################################

"""
    AbstractQuadratureQuantity
    QuadratureValue()
    QuadratureGradient()

What the forward contraction produces at a quadrature point, and thereby the
components the element's pointwise map ([`tensor_product_pointwise`](@ref))
receives: the single interpolated VALUE, or the `dim` components of the
REFERENCE GRADIENT. It also fixes the 1D operator each component is contracted
with along each lattice axis — `B` everywhere for a value, `dB` along the
component's own axis for a gradient — and thereby the scratch width.
"""
abstract type AbstractQuadratureQuantity end
@doc (@doc AbstractQuadratureQuantity) struct QuadratureValue <: AbstractQuadratureQuantity end
@doc (@doc AbstractQuadratureQuantity) struct QuadratureGradient <: AbstractQuadratureQuantity end

@inline quadrature_components(::QuadratureValue, ::Val{dim}) where {dim} = Val(1)
@inline quadrature_components(::QuadratureGradient, ::Val{dim}) where {dim} = Val(dim)
@inline _val_int(::Val{N}) where {N} = N

@inline _axis_operator(::QuadratureValue, values, d::Int, k::Int) = values.B
@inline _axis_operator(::QuadratureGradient, values, d::Int, k::Int) = k == d ? values.dB : values.B

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
    quadrature_lattice_index(values::TensorProductValues, qp::Int) -> NTuple{dim, Int}

The per-axis 1D quadrature indices of the `qp`-th quadrature point, the inverse
of the lexicographic (dimension 1 fastest) numbering the contraction lattice
uses. That numbering is what a per-quadrature-point store is laid out in, so a
sweep filling one and a kernel reading it agree by construction.
"""
@inline quadrature_lattice_index(::TensorProductValues{dim, Nb, Nq}, qp::Int) where {dim, Nb, Nq} =
    ntuple(c -> ((qp - 1) ÷ Nq^(c - 1)) % Nq + 1, Val(dim))

"""
    tensor_product_contract!(scratch, dcol, scol, Op, k, nout, nin, Val(M), Val(dim), lane, nlanes)

One contraction, `dst[…, iₖ, …] = Σₐ Op[iₖ, a] · src[…, a, …]` along axis `k` of
the scratch box, over the spectator slabs `lane` owns.

This is the ONE loop both element mappings run: `lane = nlanes = 1` walks every
slab, and a workgroup of `nlanes` splits them. `dcol`/`scol` are scratch
columns, never the same one.
"""
@inline function tensor_product_contract!(scratch, dcol::Int, scol::Int, Op, k::Int, nout::Int, nin::Int,
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

# Scratch columns: the element state, then a working pair per contracted
# component. Successive contractions ping-pong between the pair, so the stage
# index alone decides which column holds what, and the LATTICE dimension (the
# number of forward contractions) decides where the pointwise map finds them.
@inline _tp_col_a(nc::Int, d::Int) = 1 + d
@inline _tp_col_b(nc::Int, d::Int) = 1 + nc + d
@inline _tp_col_head(dim::Int, nc::Int, d::Int) = isodd(dim) ? _tp_col_a(nc, d) : _tp_col_b(nc, d)
@inline _tp_col_tail(dim::Int, nc::Int, d::Int) = isodd(dim) ? _tp_col_b(nc, d) : _tp_col_a(nc, d)

####################################
## Geometry at a quadrature point
####################################

"""
    tensor_product_jacobian(values::TensorProductValues, coordinates, q::NTuple{dim, Int}) -> SMatrix{dim, dim}

The Jacobian `∂x/∂ξ` of the isoparametric map at the quadrature point `q` names
(the per-axis 1D indices [`quadrature_lattice_index`](@ref) returns), formed from
`coordinates` — the cell's node coordinates in Ferrite's own order.

Distorted cells are covered: nothing here assumes an affine map.
"""
@inline function tensor_product_jacobian(values::TensorProductValues{dim, Nb, Nq, Nn, T},
        coordinates, q::NTuple{dim, Int}) where {dim, Nb, Nq, Nn, T}
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
    return J
end

"""
    tensor_product_weight(values::TensorProductValues, q::NTuple{dim, Int}) -> Number

The tensor-product quadrature weight at the point `q` names — the companion of
[`tensor_product_jacobian`](@ref); the two together are the geometric factor a
pointwise map scales by.
"""
@inline tensor_product_weight(values::TensorProductValues{dim, Nb, Nq, Nn, T},
        q::NTuple{dim, Int}) where {dim, Nb, Nq, Nn, T} =
    prod(ntuple(c -> (@inbounds values.w[q[c]]), Val(dim)))

####################################
## The element cache contract
####################################

"""
    AbstractTensorProductElementCache <: AbstractVolumetricElementCache

An element evaluating its operator's ACTION by Deville–Fischer–Mund sum
factorization over a tensor-product lattice, in `O(p^{d+1})` per cell instead of
forming `Kₑ`. The core supplies the contractions, the lattice bookkeeping and
BOTH matrix-free entries — [`apply_element_action!`](@ref) for
[`WorkerPerElement`](@ref) and the [`cooperative_stage!`](@ref) pipeline for
[`CooperativeElement`](@ref) — over one definition of the element math.

What a concrete cache brings:

- the fields `values` (a [`TensorProductValues`](@ref)) and `scratch` (the
  contraction box, [`allocate_tensor_product_scratch`](@ref)), or overrides of
  [`tensor_product_values`](@ref)/[`tensor_product_scratch`](@ref)
- [`tensor_product_quantity`](@ref) — what the forward contraction produces
- [`tensor_product_pointwise`](@ref) — THE POINTWISE MAP, the one place the
  form enters
- optionally [`fill_quadrature_data!`](@ref), where that map reads factors a
  [`QuadratureDataKind`](@ref) sweep stored instead of re-deriving them
- the device struct-of-arrays trio ([`setup_device_instances`](@ref),
  [`device_worker_view`](@ref), [`duplicate_for_device`](@ref)), which names the
  concrete type and is therefore not derivable here; `scratch` is the one
  per-worker field and everything else is shared read-only

The cache has NO element-matrix kernel: assembling it under
[`FullAssembly`](@ref) is refused where the matrix would be formed.

!!! warning "Experimental surface"
    This supertype, its accessors and the pipeline it drives may change in a
    minor release.
"""
abstract type AbstractTensorProductElementCache <: AbstractVolumetricElementCache end

"""
    tensor_product_values(cache) -> TensorProductValues
    tensor_product_scratch(cache) -> AbstractMatrix
    tensor_product_quantity(cache) -> AbstractQuadratureQuantity

The three things the core reads off an
[`AbstractTensorProductElementCache`](@ref). The first two default to the
`values` and `scratch` FIELDS; the quantity has no default, a wrong one
evaluating a different form in silence.
"""
tensor_product_values(cache::AbstractTensorProductElementCache) = cache.values
@doc (@doc tensor_product_values) tensor_product_scratch(cache::AbstractTensorProductElementCache) = cache.scratch
@doc (@doc tensor_product_values) function tensor_product_quantity end

"""
    tensor_product_pointwise(cache, args::CellArgs, q::NTuple{dim, Int}, qp::Int, v::SVector) -> SVector

THE POINTWISE MAP: the element's `D` block, applied to what the forward
contraction produced at one quadrature point and returning what the backward
contraction takes back to the dofs. `v` carries the components
[`tensor_product_quantity`](@ref) names — the reference gradient, or the
interpolated value.

`q` is the point's per-axis 1D index, which the geometry helpers
[`tensor_product_jacobian`](@ref)/[`tensor_product_weight`](@ref) take; `qp` is
its linear index in the cell's own quadrature numbering, which addresses a
per-quadrature-point store ([`quadrature_lattice_index`](@ref) relates the two).
A map that RE-DERIVES its geometry reads `q`, one that reads stored factors
reads `qp`.

Called by the lane that produced the point, so nothing here synchronizes and
nothing is written outside the point.
"""
function tensor_product_pointwise end

"""
    allocate_tensor_product_scratch(values::TensorProductValues, quantity) -> Matrix

The per-cell contraction box both element mappings work in: `1 + 2·nc` lattice
boxes of `max(Nb, Nq)^dim` entries, `nc` being the component count `quantity`
names — the element state, and a working pair per component.
[`WorkerPerElement`](@ref) gives every worker its own, batched with the worker
as the leading index on a device; [`CooperativeElement`](@ref) stages the same
boxes in group-local memory instead
([`tensor_product_scratch_prototype`](@ref)).
"""
function allocate_tensor_product_scratch(values::TensorProductValues{dim, Nb, Nq, Nn, T},
        quantity::AbstractQuadratureQuantity) where {dim, Nb, Nq, Nn, T}
    nc = _val_int(quadrature_components(quantity, Val(dim)))
    return zeros(T, max(Nb, Nq)^dim, 1 + 2nc)
end

"""
    tensor_product_scratch_prototype(device, scratch) -> scratch

The scratch a cache's [`setup_device_instances`](@ref) hands the device to
batch: the scratch itself, or an EMPTY one under [`CooperativeElement`](@ref),
whose kernel stages the boxes in group-local memory and never touches the batch.
A cache's device layout passes its scratch through this instead of testing the
mapping itself.
"""
tensor_product_scratch_prototype(device, scratch) =
    element_mapping(device) isa CooperativeElement ? similar(scratch, 0, 0) : scratch

element_value_type(cache::AbstractTensorProductElementCache) =
    element_value_type(tensor_product_values(cache))
Ferrite.getnquadpoints(cache::AbstractTensorProductElementCache) =
    getnquadpoints(tensor_product_values(cache))
# The values object holds no per-cell state, so there is nothing to position.
reinit_values!(::AbstractTensorProductElementCache, cell) = nothing
# A matrix-free element forms no element matrix, so the engine's per-worker `Ke`
# stays empty instead of costing `ndofs_per_cell^2` per worker — which is the
# difference between a few megabytes and a few hundred on a device.
allocate_element_matrix(cache::AbstractTensorProductElementCache, sdh) =
    zeros(element_value_type(cache), 0, 0)

####################################
## The pipeline
####################################

cooperative_lattice_dim(cache::AbstractTensorProductElementCache) = _tp_dim(tensor_product_values(cache))
_tp_dim(::TensorProductValues{dim}) where {dim} = dim

cooperative_group_size(cache::AbstractTensorProductElementCache) = _tp_group_size(tensor_product_values(cache))
_tp_group_size(::TensorProductValues{dim, Nb, Nq}) where {dim, Nb, Nq} = max(Nb, Nq)^(dim - 1)

cooperative_scratch_shape(cache::AbstractTensorProductElementCache) =
    _tp_scratch_shape(tensor_product_values(cache), tensor_product_quantity(cache))
_tp_scratch_shape(values::TensorProductValues{dim, Nb, Nq}, quantity) where {dim, Nb, Nq} =
    (Val(max(Nb, Nq)^dim), Val(1 + 2 * _val_int(quadrature_components(quantity, Val(dim)))))

@inline cooperative_load!(scratch, cache::AbstractTensorProductElementCache, uₑ, lane::Int, nlanes::Int) =
    _tp_load!(scratch, tensor_product_values(cache), tensor_product_quantity(cache), uₑ, lane, nlanes)

@inline function _tp_load!(scratch, values::TensorProductValues{dim, Nb, Nq, Nn, T}, quantity,
        uₑ, lane::Int, nlanes::Int) where {dim, Nb, Nq, Nn, T}
    M = max(Nb, Nq)
    nc = _val_int(quadrature_components(quantity, Val(dim)))
    lattice = values.dof_lattice
    # A lane zeroes the same lattice slots it then fills, so the two loops need
    # no barrier between them.
    for column in 1:(1 + 2nc), i in lane:nlanes:(M^dim)
        @inbounds scratch[i, column] = zero(T)
    end
    for i in lane:nlanes:(M^dim)
        l = _lattice_index(i, Val(Nb), Val(M), Val(dim))
        l == 0 && continue
        @inbounds scratch[i, 1] = convert(T, uₑ[lattice[l]])
    end
    return nothing
end

@inline cooperative_stage!(scratch, cache::AbstractTensorProductElementCache, args,
        stage::Int, lane::Int, nlanes::Int) =
    _tp_stage!(scratch, cache, tensor_product_values(cache), tensor_product_quantity(cache),
               args, stage, lane, nlanes)

@inline function _tp_stage!(scratch, cache, values::TensorProductValues{dim}, quantity, args,
        stage::Int, lane::Int, nlanes::Int) where {dim}
    if stage ≤ dim
        _tp_forward_stage!(scratch, values, quantity, stage, lane, nlanes)
        # The map runs on the slab the contraction above just wrote, so no
        # barrier separates them.
        stage == dim && _tp_pointwise_stage!(scratch, cache, values, args,
                                             quadrature_components(quantity, Val(dim)), lane, nlanes)
    else
        _tp_backward_stage!(scratch, values, quantity, stage - dim, lane, nlanes)
    end
    return nothing
end

# Forward stage `k`: contract axis `k` of every component with the 1D operator
# the quantity names for that (component, axis) pair.
@inline function _tp_forward_stage!(scratch, values::TensorProductValues{dim, Nb, Nq}, quantity,
        k::Int, lane::Int, nlanes::Int) where {dim, Nb, Nq}
    nc = _val_int(quadrature_components(quantity, Val(dim)))
    for d in 1:nc
        source = k == 1 ? 1 : (iseven(k) ? _tp_col_a(nc, d) : _tp_col_b(nc, d))
        destination = isodd(k) ? _tp_col_a(nc, d) : _tp_col_b(nc, d)
        tensor_product_contract!(scratch, destination, source, _axis_operator(quantity, values, d, k),
                                 k, Nq, Nb, Val(max(Nb, Nq)), Val(dim), lane, nlanes)
    end
    return nothing
end

# Backward stage `j`: the transposed contraction of axis `j`, back from the
# quadrature lattice towards the dof lattice.
@inline function _tp_backward_stage!(scratch, values::TensorProductValues{dim, Nb, Nq}, quantity,
        j::Int, lane::Int, nlanes::Int) where {dim, Nb, Nq}
    nc = _val_int(quadrature_components(quantity, Val(dim)))
    for d in 1:nc
        source = isodd(j) ? _tp_col_head(dim, nc, d) : _tp_col_tail(dim, nc, d)
        destination = isodd(j) ? _tp_col_tail(dim, nc, d) : _tp_col_head(dim, nc, d)
        tensor_product_contract!(scratch, destination, source,
                                 transpose(_axis_operator(quantity, values, d, j)),
                                 j, Nb, Nq, Val(max(Nb, Nq)), Val(dim), lane, nlanes)
    end
    return nothing
end

# The element's pointwise map, applied in place to what the forward contractions
# produced. The lane that produced a quadrature point is the one that maps it.
#
# The two scratch accesses around the callback are deliberately BOUNDS-CHECKED
# while every other access in this file is not. The indices are the lattice's own
# and in range by construction, but asserting that ACROSS the element callback
# changes what the NVPTX backend computes: a one-component quantity with the
# geometry inlined (the mass action at Nq = 3) came out wrong by O(1) under
# `@inbounds` and is correct without it, on the same code that the CPU backends
# and the cooperative kernel evaluate correctly either way. The check costs one
# comparison per component per quadrature point, beside a Jacobian evaluation.
@inline function _tp_pointwise_stage!(scratch, cache, values::TensorProductValues{dim, Nb, Nq, Nn, T},
        args, ::Val{NC}, lane::Int, nlanes::Int) where {dim, Nb, Nq, Nn, T, NC}
    M = max(Nb, Nq)
    stride = M^(dim - 1)
    for s in (lane - 1):nlanes:(stride - 1)
        _lattice_index(s + 1, Val(Nq), Val(M), Val(dim)) == 0 && continue
        for last in 1:Nq
            q = ntuple(c -> c == dim ? last : (s ÷ M^(c - 1)) % M + 1, Val(dim))
            i = s + (last - 1) * stride + 1
            v = SVector{NC, T}(ntuple(d -> scratch[i, _tp_col_head(dim, NC, d)], Val(NC)))
            mapped = tensor_product_pointwise(cache, args, q,
                                              _lattice_index(i, Val(Nq), Val(M), Val(dim)), v)
            for d in 1:NC
                scratch[i, _tp_col_head(dim, NC, d)] = mapped[d]
            end
        end
    end
    return nothing
end

@inline cooperative_store!(yₑ, scratch, cache::AbstractTensorProductElementCache, lane::Int, nlanes::Int) =
    _tp_store!(yₑ, scratch, tensor_product_values(cache), tensor_product_quantity(cache), lane, nlanes)

# The last backward contraction, summed over the components and accumulated into
# the element vector through the dof permutation. Lanes own disjoint dofs, so
# nothing accumulates across lanes.
@inline function _tp_store!(yₑ, scratch, values::TensorProductValues{dim, Nb, Nq, Nn, T}, quantity,
        lane::Int, nlanes::Int) where {dim, Nb, Nq, Nn, T}
    M = max(Nb, Nq)
    stride = M^(dim - 1)
    nc = _val_int(quadrature_components(quantity, Val(dim)))
    lattice = values.dof_lattice
    for s in (lane - 1):nlanes:(stride - 1)
        for io in 1:Nb
            l = _lattice_index(s + (io - 1) * stride + 1, Val(Nb), Val(M), Val(dim))
            l == 0 && continue
            acc = zero(T)
            for d in 1:nc
                Op = _axis_operator(quantity, values, d, dim)
                column = iseven(dim - 1) ? _tp_col_head(dim, nc, d) : _tp_col_tail(dim, nc, d)
                for ii in 1:Nq
                    @inbounds acc += Op[ii, io] * scratch[s + (ii - 1) * stride + 1, column]
                end
            end
            @inbounds yₑ[lattice[l]] += acc
        end
    end
    return nothing
end

"""
    apply_element_action!(yₑ, cache::AbstractTensorProductElementCache, uₑ, args)

The worker-per-element action: the cooperative pipeline run by a single lane
over the worker's own scratch. One element definition, two execution mappings —
the stage bodies are shared verbatim, and only the slab range a worker walks
differs.
"""
apply_element_action!(yₑ, cache::AbstractTensorProductElementCache, uₑ, args::CellArgs) =
    _tp_apply!(yₑ, cache, tensor_product_values(cache), uₑ, args)

@inline function _tp_apply!(yₑ, cache, ::TensorProductValues{dim}, uₑ, args) where {dim}
    scratch = tensor_product_scratch(cache)
    cooperative_load!(scratch, cache, uₑ, 1, 1)
    for stage in 1:(2dim - 1)
        cooperative_stage!(scratch, cache, args, stage, 1, 1)
    end
    cooperative_store!(yₑ, scratch, cache, 1, 1)
    return nothing
end

# The bilinear form's residual IS the action, and the residual kernel is
# mandatory for every element cache.
assemble_cell!(req::ResidualRequest, cache::AbstractTensorProductElementCache, args::CellArgs) =
    apply_element_action!(req.r, cache, args.states.u, args)

# `provides_analytic` stays `false` for the Jacobian kinds: this cache has no
# element matrix to declare. The method exists so a `FullAssembly` sweep says
# what is wrong instead of reporting a `MethodError` or an empty buffer.
assemble_cell!(::Union{JacobianRequest, JacobianResidualRequest},
        cache::AbstractTensorProductElementCache, ::CellArgs) = throw(ArgumentError(
    "$(nameof(typeof(cache))) forms no element matrix — it evaluates the operator's ACTION by " *
    "sum factorization, and its element-matrix buffer is empty by construction. Set the " *
    "operator up with `form = MatrixFreeAction()`, or assemble an integrator for the same form " *
    "that has an element-matrix kernel."))
