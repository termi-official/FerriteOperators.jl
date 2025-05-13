# TODO split nonlinear operator and the linearization concepts
# TODO energy based operator?
# TODO maybe a trait system for operators?

"""
    AbstractNonlinearOperator

Models of a nonlinear function F(u)v, where v is a test function.

Interface:
    (op::AbstractNonlinearOperator)(residual::AbstractVector, in::AbstractNonlinearOperator)
    eltype()
    size()

    # linearization
    mul!(out::AbstractVector, op::AbstractNonlinearOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::AbstractNonlinearOperator, in::AbstractVector, α, β)
    update_linearization!(op::AbstractNonlinearOperator, u::AbstractVector, time)
    update_linearization!(op::AbstractNonlinearOperator, residual::AbstractVector, u::AbstractVector, time)
"""
abstract type AbstractNonlinearOperator end

"""
    update_linearization!(op, residual, u, t)

Setup the linearized operator `Jᵤ(u) := dᵤF(u)` in op and its residual `F(u)` in
preparation to solve for the increment `Δu` with the linear problem `J(u) Δu = F(u)`.
"""
update_linearization!(Jᵤ::AbstractNonlinearOperator, residual::AbstractVector, u::AbstractVector, t)

"""
    update_linearization!(op, u, t)

Setup the linearized operator `Jᵤ(u)` in op.
"""
update_linearization!(Jᵤ::AbstractNonlinearOperator, u::AbstractVector, t)

"""
    update_residual!(op, residual, u, problem, t)

Evaluate the residual `F(u)` of the problem.
"""
update_residual!(op::AbstractNonlinearOperator, residual::AbstractVector, u::AbstractVector, t)


abstract type AbstractBlockOperator <: AbstractNonlinearOperator end

get_matrix(op) = error("Operator matrix is not explicitly accessible for given operator")

function *(op::AbstractNonlinearOperator, x::AbstractVector)
    y = similar(x)
    mul!(y, op, x)
    return y
end

# # TODO constructor which checks for axis compat
# struct BlockOperator{OPS <: Tuple, JT} <: AbstractBlockOperator
#     # TODO custom "square matrix tuple"
#     operators::OPS # stored row by row as in [1 2; 3 4]
#     J::JT
# end

# function BlockOperator(operators::Tuple)
#     nblocks = isqrt(length(operators))
#     mJs = reshape([get_matrix(opi) for opi ∈ operators], (nblocks, nblocks))
#     block_sizes = [size(op,1) for op in mJs[:,1]]
#     total_size = sum(block_sizes)
#     # First we define an empty dummy block array
#     J = BlockArray(spzeros(total_size,total_size), block_sizes, block_sizes)
#     # Then we move the local Js into the dummy to transfer ownership
#     for i in 1:nblocks
#         for j in 1:nblocks
#             J[Block(i,j)] = mJs[i,j]
#         end
#     end

#     return BlockOperator(operators, J)
# end

# function get_matrix(op::BlockOperator, i::Block)
#     @assert length(i.n) == 2
#     return @view op.J[i]
# end

# get_matrix(op::BlockOperator) = op.J

# function *(op::BlockOperator, x::AbstractVector)
#     y = similar(x)
#     mul!(y, op, x)
#     return y
# end

# mul!(y, op::BlockOperator, x) = mul!(y, get_matrix(op), x)

# # TODO can we be clever with broadcasting here?
# function update_linearization!(op::BlockOperator, u::BlockVector, time)
#     for opi ∈ op.operators
#         update_linearization!(opi, u, time)
#     end
# end

# # TODO can we be clever with broadcasting here?
# function update_linearization!(op::BlockOperator, residual::BlockVector, u::BlockVector, time)
#     nops = length(op.operators)
#     nrows = isqrt(nops)
#     for i ∈ 1:nops
#         row, col = divrem(i-1, nrows) .+ 1 # index shift due to 1-based indices
#         i1 = Block(row)
#         row_residual = @view residual[i1]
#         @timeit_debug "update block ($row,$col)" update_linearization!(op.operators[i], row_residual, u, time) # :)
#     end
# end

# # TODO can we be clever with broadcasting here?
# function mul!(out::BlockVector, op::BlockOperator, in::BlockVector)
#     out .= 0.0
#     # 5-arg-mul over 3-ar-gmul because the bocks would overwrite the solution!
#     mul!(out, op, in, 1.0, 1.0)
# end

# # TODO can we be clever with broadcasting here?
# function mul!(out::BlockVector, op::BlockOperator, in::BlockVector, α, β)
#     nops = length(op.operators)
#     nrows = isqrt(nops)
#     for i ∈ 1:nops
#         i1, i2 = Block.(divrem(i-1, nrows) .+1) # index shift due to 1-based indices
#         in_next  = @view in[i1]
#         out_next = @view out[i2]
#         mul!(out_next, op.operators[i], in_next, α, β)
#     end
# end

#########################################################################################################################

abstract type AbstractBilinearOperator <: AbstractNonlinearOperator end

update_linearization!(op::AbstractBilinearOperator, residual::AbstractVector, u::AbstractVector, time) = update_operator!(op, time)
update_linearization!(op::AbstractBilinearOperator, u::AbstractVector, time) = update_operator!(op, time)


"""
    DiagonalOperator <: AbstractBilinearOperator

Literally a "diagonal matrix".
"""
struct DiagonalOperator{TV <: AbstractVector} <: AbstractBilinearOperator
    values::TV
end

mul!(out::AbstractVector, op::DiagonalOperator, in::AbstractVector) = out .= op.values .* out
mul!(out::AbstractVector, op::DiagonalOperator, in::AbstractVector, α, β) = out .= α * op.values .* in + β * out
Base.eltype(op::DiagonalOperator) = eltype(op.values)
Base.size(op::DiagonalOperator, axis) = length(op.values)

get_matrix(op::DiagonalOperator) = spdiagm(op.values)

update_linearization!(::DiagonalOperator, ::AbstractVector, ::AbstractVector, t) = nothing

"""
    NullOperator <: AbstractBilinearOperator

Literally a "null matrix".
"""

struct NullOperator{T, SIN, SOUT} <: AbstractBilinearOperator
end

mul!(out::AbstractVector, op::NullOperator, in::AbstractVector) = out .= 0.0
mul!(out::AbstractVector, op::NullOperator, in::AbstractVector, α, β) = out .= β*out
Base.eltype(op::NullOperator{T}) where {T} = T
Base.size(op::NullOperator{T,S1,S2}, axis) where {T,S1,S2} = axis == 1 ? S1 : (axis == 2 ? S2 : error("faulty axis!"))

get_matrix(op::NullOperator{T, SIN, SOUT}) where {T, SIN, SOUT} = spzeros(T,SIN,SOUT)

update_linearization!(::NullOperator, ::AbstractVector, ::AbstractVector, t) = nothing

#########################################################################################################################

"""
    AbstractLinearOperator

Supertype for operators which only depend on the test space.
"""
abstract type AbstractLinearOperator end

"""
    LinearNullOperator <: AbstractLinearOperator

Literally the null vector.
"""
struct LinearNullOperator{T,S} <: AbstractLinearOperator
end
Ferrite.add!(b::AbstractVector, op::LinearNullOperator) = b
Base.eltype(op::LinearNullOperator{T,S}) where {T,S} = T
Base.size(op::LinearNullOperator{T,S}) where {T,S} = S

update_operator!(op::LinearNullOperator, time) = nothing


Ferrite.add!(b::AbstractVector, op::AbstractLinearOperator) = __add_to_vector!(b, op.b)
__add_to_vector!(b::AbstractVector, a::AbstractVector) = b .+= a
Base.eltype(op::AbstractLinearOperator) = eltype(op.b)
Base.size(op::AbstractLinearOperator) = sisze(op.b)

