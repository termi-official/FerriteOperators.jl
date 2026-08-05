# This is the easiest solution for now.
# It is assumed that the element knows how many dofs per quadrature point are there locally.
@concrete mutable struct InternalVariableHandler <: AbstractDofHandler
    # `ncells+1` offsets relative to the start of the internal variable block, with
    # `internal_variable_offsets[1] == 0`, such that cell `cid` owns the relative entries
    # `internal_variable_offsets[cid]+1 : internal_variable_offsets[cid+1]`.
    internal_variable_offsets
    # Where the block starts in the solution vector, i.e. `ndofs(dh)` of the handler it was built for.
    base_offset <: Integer
    ndofs <: Integer
end
Ferrite.ndofs(lvh::InternalVariableHandler) = lvh.ndofs
# Both are absolute, i.e. they address the solution vector and not the internal variable block.
internal_variable_offset(lvh::InternalVariableHandler, cellid::Int) = lvh.base_offset + lvh.internal_variable_offsets[cellid]
internal_variable_range(lvh::InternalVariableHandler, cellid::Int)  = (internal_variable_offset(lvh, cellid)+1):(lvh.base_offset + lvh.internal_variable_offsets[cellid+1])
Ferrite.close!(lvh::InternalVariableHandler) = nothing

# Offsets are shared read-only data, so duplication just returns the same instance.
duplicate_for_device(device, ivh::InternalVariableHandler) = ivh

# # Utils to distribute and visualize local variables
# struct QuadratureInterpolation{RefShape, QR <: QuadratureRule{RefShape}} <:
#        Ferrite.ScalarInterpolation{RefShape, -1}
#     qr::QR
# end

# Ferrite.getnbasefunctions(ip::QuadratureInterpolation) = getnquadpoints(ip.qr)
# Ferrite.n_components(ip::QuadratureInterpolation) = 1
# Ferrite.n_dbc_components(::QuadratureInterpolation) = 0
# Ferrite.adjust_dofs_during_distribution(::QuadratureInterpolation) = false
# Ferrite.volumedof_interior_indices(ip::QuadratureInterpolation) =
#     ntuple(i->i, getnbasefunctions(ip))
# # conformity is only used for VTK export and updating the constraint handler. This is not needed since the internal variables are not constrained.
# Ferrite.conformity(::QuadratureInterpolation) = Ferrite.L2Conformity()

# function Ferrite.reference_coordinates(ip::QuadratureInterpolation)
#     return [qp for i = 1:ip.num_components for qp in getpoints(ip.qr)]
# end

# function Ferrite.reference_shape_value(ip::QuadratureInterpolation, ::Vec, i::Int)
#     throw(ArgumentError("shape function evaluation for interpolation $ip not implemented yet"))
# end
