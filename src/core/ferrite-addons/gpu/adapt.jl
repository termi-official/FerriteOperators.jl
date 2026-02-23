
# Adapt.adapt_structure methods for converting CPU structs to device

function Adapt.adapt_structure(to, g::DeviceGrid)
    DeviceGrid(
        Adapt.adapt(to, g.cells),
        Adapt.adapt(to, g.nodes),
    )
end

function Adapt.adapt_structure(to, cv::DeviceCellValuesData)
    DeviceCellValuesData(
        Adapt.adapt(to, cv.dNdξ),
        Adapt.adapt(to, cv.dMdξ),
        Adapt.adapt(to, cv.weights),
        cv.nqp,
        cv.nbf,
        cv.ngeo,
    )
end
