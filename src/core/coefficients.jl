"""
    ConstantCoefficient(value)

Evaluates to the same value in space and time everywhere.
"""
struct ConstantCoefficient{T}
    val::T
end

duplicate_for_device(device, cache::ConstantCoefficient) = cache

function setup_coefficient_cache(coefficient::ConstantCoefficient, qr::QuadratureRule, sdh::SubDofHandler)
    return coefficient
end

evaluate_coefficient(coeff::ConstantCoefficient, ::FerriteUtils.AnyCellCache, qp, t) = coeff.val
