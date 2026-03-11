# This is the easiest solution for now.
# It is assumed that the element knows how many dofs per quadrature point are there locally.
@concrete struct InternalVariableHandler <: AbstractDofHandler
    internal_variable_offsets
    ndofs <: Integer
end
Ferrite.ndofs(lvh::InternalVariableHandler) = lvh.ndofs
internal_variable_offset(lvh::InternalVariableHandler, cellid::Int) = lvh.internal_variable_offsets[cellid]

# Read-only lookup table, shared across threads.
# CPU: deep-copy offsets so threads don't alias.
# GPU: adapt offsets to GPU array; nothing offsets stay nothing.
function duplicate_for_device(device, ivh::InternalVariableHandler)
    if ivh.internal_variable_offsets === nothing
        return ivh
    end
    return InternalVariableHandler(
        duplicate_for_device(device, ivh.internal_variable_offsets),
        ivh.ndofs,
    )
end

Adapt.adapt_structure(to, ivh::InternalVariableHandler) =
    InternalVariableHandler(
        Adapt.adapt(to, ivh.internal_variable_offsets),
        ivh.ndofs,
    )
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
