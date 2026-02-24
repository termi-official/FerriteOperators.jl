# Adapt.jl integration for Ferrite types

## DofHandler ##
Adapt.adapt_structure(::AbstractCPUDevice, dh::DofHandler) = dh
function Adapt.adapt_structure(::AbstractGPUDevice, dh::DofHandler)
    error("adapt_structure(::AbstractGPUDevice, ::DofHandler) is not implemented yet")
end

## Grid ##
Adapt.adapt_structure(::AbstractCPUDevice, grid::AbstractGrid) = grid
function Adapt.adapt_structure(::AbstractGPUDevice, grid::AbstractGrid)
    error("adapt_structure(::AbstractGPUDevice, ::AbstractGrid) is not implemented yet — use DeviceGrid instead")
end

## CellCache ##
Adapt.adapt_structure(::AbstractCPUDevice, cc::CellCache) = cc
function Adapt.adapt_structure(::AbstractGPUDevice, cc::CellCache)
    error("adapt_structure(::AbstractGPUDevice, ::CellCache) is not implemented yet — use DeviceCellCache instead")
end

## CellValues ##
Adapt.adapt_structure(::AbstractCPUDevice, cv::CellValues) = cv
function Adapt.adapt_structure(::AbstractGPUDevice, cv::CellValues)
    error("adapt_structure(::AbstractGPUDevice, ::CellValues) is not implemented yet — use DeviceCellValues instead")
end
