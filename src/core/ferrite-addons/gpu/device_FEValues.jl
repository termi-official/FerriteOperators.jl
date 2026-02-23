
# Device CellValues data (shared immutable, read-only in kernel)
# Built on CPU with CPU arrays, then adapted to device via Adapt.adapt
struct DeviceCellValuesData{T1, T2, T3}
    dNdξ::T1       # Array/GPUArray of Vec{dim,T} — size (nbf, nqp)
    dMdξ::T2       # Array/GPUArray of Vec{dim,T} — size (ngeo, nqp)
    weights::T3    # Array/GPUArray of T          — size (nqp,)
    nqp::Int32
    nbf::Int32
    ngeo::Int32
end

# Extract shared immutable CellValues data into CPU arrays
function build_device_cv_data(cv::CellValues)
    dNdξ = cv.fun_values.dNdξ
    dMdξ = cv.geo_mapping.dMdξ
    weights = cv.qr.weights
    return DeviceCellValuesData(
        collect(dNdξ),
        collect(dMdξ),
        collect(weights),
        Int32(size(dNdξ, 2)),   # nqp
        Int32(size(dNdξ, 1)),   # nbf
        Int32(size(dMdξ, 1)),   # ngeo
    )
end
