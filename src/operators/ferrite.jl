"""
    LinearizedFerriteOperator(J, caches)

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_element! -> update jacobian/residual contribution with internal state variables
"""
struct LinearizedFerriteOperator{MatrixType} <: AbstractNonlinearOperator
    J::MatrixType
    strategy
    subdomain_caches::Vector{SubdomainCache}
end

# Interface
function update_linearization!(op::LinearizedFerriteOperator, u::AbstractVector, p)
    (; J, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, J)

    for sudomain_cache in subdomain_caches
        # Function barrier
        _update_linearization_J!(assembler, u, sudomain_cache.sdh, sudomain_cache.element_cache, sudomain_cache.strategy_cache, p)
    end

    finalize_assembly!(assembler)
end
function update_linearization!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p)
    (; J, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, J, residual)

    for sudomain_cache in subdomain_caches
        # Function barrier
        _update_linearization_Jr!(assembler, u, sudomain_cache.sdh, sudomain_cache.element_cache, sudomain_cache.strategy_cache, p)
    end

    finalize_assembly!(assembler)
end
function residual!(op::LinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, p)
    (; strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, residual)

    for sudomain_cache in subdomain_caches
        # Function barrier
        _residual!(assembler, u, sudomain_cache.sdh, sudomain_cache.element_cache, sudomain_cache.strategy_cache, p)
    end

    finalize_assembly!(assembler)
end

"""
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear form to the vector `in`.
"""
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::LinearizedFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)
(op::LinearizedFerriteOperator)(residual, u, p) = residual!(op, residual, u, p)
Base.eltype(op::LinearizedFerriteOperator) = eltype(op.J)
Base.size(op::LinearizedFerriteOperator, axis) = size(op.J, axis)

# -------------------------------------------------- Sequential on CPU --------------------------------------------------

function _update_linearization_J!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::SequentialAssemblyStrategyCache, p)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    Jₑ = allocate_element_matrix(element_cache, sdh)
    uₑ = allocate_element_unknown_vector(element_cache, sdh)
    @inbounds for cell in CellIterator(sdh)
        # Prepare buffers
        fill!(Jₑ, 0.0)
        load_element_unknowns!(uₑ, u, cell, element_cache)

        # Fill buffers
        @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, p)

        assemble!(assembler, cell, Jₑ)
    end
end

function _update_linearization_Jr!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::SequentialAssemblyStrategyCache, p)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    Jₑ = allocate_element_matrix(element_cache, sdh)
    uₑ = allocate_element_unknown_vector(element_cache, sdh)
    rₑ = allocate_element_residual_vector(element_cache, sdh)
    @inbounds for cell in CellIterator(sdh)
        fill!(Jₑ, 0.0)
        fill!(rₑ, 0.0)
        load_element_unknowns!(uₑ, u, cell, element_cache)

        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, p)

        assemble!(assembler, cell, Jₑ, rₑ)
    end
end


# This function is defined to control the dispatch
function _residual!(residual::AbstractVector, u::AbstractVector, sdh, element_cache, strategy_cache::SequentialAssemblyStrategyCache, p)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    Jₑ = allocate_element_matrix(element_cache, sdh)
    uₑ = allocate_element_unknown_vector(element_cache, sdh)
    rₑ = allocate_element_residual_vector(element_cache, sdh)
    @inbounds for cell in CellIterator(sdh)
        fill!(Jₑ, 0.0)
        fill!(rₑ, 0.0)
        load_element_unknowns!(uₑ, u, cell, element_cache)

        @timeit_debug "assemble element" assemble_element!(rₑ, uₑ, cell, element_cache, p)

        assemble!(residual, cell, rₑ)
    end
end

# -------------------------------------------------- Colored on CPU --------------------------------------------------

# This function is defined to control the dispatch
function _update_linearization_J!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    (; colors) = strategy_cache

    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    Jₑ = allocate_element_matrix(element_cache, sdh)
    uₑ = allocate_element_unknown_vector(element_cache, sdh)
    @inbounds for color in colors
        for cell in CellIterator(sdh.dh, color)
            # Prepare buffers
            fill!(Jₑ, 0)
            load_element_unknowns!(uₑ, u, cell, element_cache)

            # Fill buffers
            @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, p)

            assemble!(assembler, cell, Jₑ)
        end
    end
end

function _update_linearization_J!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device, device_cache, colors) = strategy_cache
    (; chunksize) = device

    ndofs = ndofs_per_cell(sdh)
    (; tlds, Aes, ues)  = device_cache

    assemblers = [duplicate_for_device(device, assembler) for tid in 1:length(tlds)]

    @inbounds for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Jₑ        = Aes[chunk]
            uₑ        = ues[chunk]
            tld       = tlds[chunk]
            assembler = assemblers[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                cell = tld.cc
                reinit!(cell, eid)

                # Prepare buffers
                fill!(Jₑ, 0)
                load_element_unknowns!(uₑ, u, cell, tld.ec)

                # Fill buffers
                assemble_element!(Jₑ, uₑ, cell, tld.ec, p)

                assemble!(assembler, cell, Jₑ)
            end
        end
    end
end

function _update_linearization_Jr!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    (; colors)= strategy_cache

    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    Jₑ = allocate_element_matrix(element_cache, sdh)
    uₑ = allocate_element_unknown_vector(element_cache, sdh)
    rₑ = allocate_element_residual_vector(element_cache, sdh)
    @inbounds for color in colors
        for cell in CellIterator(sdh.dh, color)
            # Prepare buffers
            fill!(Jₑ, 0)
            fill!(rₑ, 0)
            load_element_unknowns!(uₑ, u, cell, element_cache)

            # Fill buffers
            @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, p)

            assemble!(assembler, cell, Jₑ, rₑ)
        end
    end
end

function _update_linearization_Jr!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device, device_cache, colors) = strategy_cache
    (; chunksize) = device

    (; tlds, Aes, ues, res)  = device_cache

    assemblers = [duplicate_for_device(device, assembler) for tid in 1:length(tlds)]

    @inbounds for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Jₑ        = Aes[chunk]
            uₑ        = ues[chunk]
            rₑ        = res[chunk]
            tld       = tlds[chunk]
            assembler = assemblers[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                cell = tld.cc
                reinit!(cell, eid)

                # Prepare buffers
                fill!(Jₑ, 0)
                fill!(rₑ, 0)
                load_element_unknowns!(uₑ, u, cell, tld.ec)

                # Fill buffers
                assemble_element!(Jₑ, rₑ, uₑ, cell, tld.ec, p)

                assemble!(assembler, cell, Jₑ, rₑ)
            end
        end
    end
end

function _residual!(residual::AbstractVector, u::AbstractVector, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    (; colors) = strategy_cache

    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    uₑ = allocate_element_unknown_vector(element_cache, sdh)
    rₑ = allocate_element_residual_vector(element_cache, sdh)
    @inbounds for color in colors
        for cell in CellIterator(sdh.dh, color)
            # Prepare buffers
            fill!(rₑ, 0)
            load_element_unknowns!(uₑ, u, cell, element_cache)

            # Fill buffers
            @timeit_debug "assemble element" assemble_element!(rₑ, uₑ, cell, element_cache, p)

            assemble!(residual, cell, rₑ)
        end
    end
end

function _residual!(residual::AbstractVector, u::AbstractVector, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device, device_cache, colors) = strategy_cache
    (; chunksize) = device

    (; tlds, ues, res)  = device_cache

    @inbounds for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            uₑ        = ues[chunk]
            rₑ        = res[chunk]
            tld       = tlds[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                cell = tld.cc
                reinit!(cell, eid)

                # Prepare buffers
                fill!(rₑ, 0)
                load_element_unknowns!(uₑ, u, cell, tld.ec)

                # Fill buffers
                assemble_element!(rₑ, uₑ, cell, tld.ec, p)

                assemble!(residual, cell, rₑ)
            end
        end
    end
end

# -------------------------------------------------- EA on CPU --------------------------------------------------

function _update_linearization_J!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    (; device, device_cache) = strategy_cache

    ndofs = ndofs_per_cell(sdh)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    Jₑ = allocate_element_matrix(element_cache, sdh)
    uₑ = allocate_element_unknown_vector(element_cache, sdh)

    for cell in CellIterator(sdh)
        # Prepare buffers
        fill!(Jₑ, 0)
        load_element_unknowns!(uₑ, u, cell, element_cache)

        # Fill buffers
        assemble_element!(Jₑ, uₑ, cell, element_cache, p)

        assemble!(assembler, cell, Jₑ)
    end
end

function _update_linearization_J!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device, device_cache) = strategy_cache
    (; chunksize) = device

    ndofs = ndofs_per_cell(sdh)
    (; tlds, Aes, ues)  = device_cache

    assemblers = [duplicate_for_device(device, assembler) for tid in 1:length(tlds)]

    ncells  = length(sdh.cellset)
    nchunks = ceil(Int, ncells / chunksize)
    @batch for chunk in 1:nchunks
        chunkbegin = (chunk-1)*chunksize+1
        chunkbound = min(ncells, chunk*chunksize)

        # Unpack chunk scratch
        Jₑ        = Aes[chunk]
        uₑ        = ues[chunk]
        tld       = tlds[chunk]
        assembler = assemblers[chunk]

        for i in chunkbegin:chunkbound
            eid = sdh.cellset[i]
            cell = tld.cc
            reinit!(cell, eid)

            # Prepare buffers
            fill!(Jₑ, 0)
            load_element_unknowns!(uₑ, u, cell, tld.ec)

            # Fill buffers
            assemble_element!(Jₑ, uₑ, cell, tld.ec, p)

            assemble!(assembler, cell, Jₑ)
        end
    end
end

function _residual!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    (; device, device_cache) = strategy_cache

    ndofs = ndofs_per_cell(sdh)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    uₑ = allocate_element_unknown_vector(element_cache, sdh)
    rₑ = allocate_element_residual_vector(element_cache, sdh)

    for cell in CellIterator(sdh)
        # Prepare buffers
        fill!(rₑ, 0)
        load_element_unknowns!(uₑ, u, cell, element_cache)

        # Fill buffers
        assemble_element!(rₑ, uₑ, cell, element_cache, p)

        assemble!(assembler, cell, rₑ)
    end
end

function _residual!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device, device_cache) = strategy_cache
    (; chunksize) = device

    (; tlds, ues, res)  = device_cache

    ncells  = length(sdh.cellset)
    nchunks = ceil(Int, ncells / chunksize)
    @batch for chunk in 1:nchunks
        chunkbegin = (chunk-1)*chunksize+1
        chunkbound = min(ncells, chunk*chunksize)

        # Unpack chunk scratch
        uₑ        = ues[chunk]
        rₑ        = res[chunk]
        tld       = tlds[chunk]

        for i in chunkbegin:chunkbound
            eid = sdh.cellset[i]
            cell = tld.cc
            reinit!(cell, eid)

            # Prepare buffers
            fill!(rₑ, 0)
            load_element_unknowns!(uₑ, u, cell, tld.ec)

            # Fill buffers
            assemble_element!(rₑ, uₑ, cell, tld.ec, p)

            assemble!(assembler, cell, rₑ)
        end
    end
end

function _update_linearization_Jr!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    (; device, device_cache) = strategy_cache

    ndofs = ndofs_per_cell(sdh)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    # TODO query from strategy_cache
    Jₑ = allocate_element_matrix(element_cache, sdh)
    uₑ = allocate_element_unknown_vector(element_cache, sdh)
    rₑ = allocate_element_residual_vector(element_cache, sdh)

    for cell in CellIterator(sdh)
        # Prepare buffers
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        load_element_unknowns!(uₑ, u, cell, element_cache)

        # Fill buffers
        assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, p)

        assemble!(assembler, cell, Jₑ, rₑ)
    end
end

function _update_linearization_Jr!(assembler, u::AbstractVector, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device, device_cache) = strategy_cache
    (; chunksize) = device

    (; tlds, Aes, ues, res)  = device_cache

    assemblers = [duplicate_for_device(device, assembler) for tid in 1:length(tlds)]

    ncells  = length(sdh.cellset)
    nchunks = ceil(Int, ncells / chunksize)
    @batch for chunk in 1:nchunks
        chunkbegin = (chunk-1)*chunksize+1
        chunkbound = min(ncells, chunk*chunksize)

        # Unpack chunk scratch
        Jₑ        = Aes[chunk]
        uₑ        = ues[chunk]
        rₑ        = res[chunk]
        tld       = tlds[chunk]
        assembler = assemblers[chunk]

        for i in chunkbegin:chunkbound
            eid = sdh.cellset[i]
            cell = tld.cc
            reinit!(cell, eid)

            # Prepare buffers
            fill!(Jₑ, 0)
            fill!(rₑ, 0)
            load_element_unknowns!(uₑ, u, cell, tld.ec)

            # Fill buffers
            assemble_element!(Jₑ, rₑ, uₑ, cell, tld.ec, p)

            assemble!(assembler, cell, Jₑ, rₑ)
        end
    end
end


#################################################################################################

struct BilinearFerriteOperator{MatrixType} <: AbstractBilinearOperator
    A::MatrixType
    strategy
    subdomain_caches::Vector{SubdomainCache}
end

function update_operator!(op::BilinearFerriteOperator, p)
    (; A, strategy, subdomain_caches)  = op

    assembler = start_assemble(strategy, A)

    for sudomain_cache in subdomain_caches
        # Function barrier
        _update_bilinear_operator_on_subdomain!(assembler, sudomain_cache.sdh, sudomain_cache.element_cache, sudomain_cache.strategy_cache, p)
    end

    finalize_assembly!(assembler)
end

function _update_bilinear_operator_on_subdomain!(assembler, sdh, element_cache, strategy_cache::SequentialAssemblyStrategyCache, p)
    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    Aₑ = zeros(ndofs, ndofs)
    # (; Aₑ) = strategy_cache.device_cache

    @inbounds for cell in CellIterator(sdh)
        fill!(Aₑ, 0)
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element_cache, p)
        assemble!(assembler, cell, Aₑ)
    end
end

function _update_bilinear_operator_on_subdomain!(assembler, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    (; device, device_cache, colors) = strategy_cache
    chunksize = 1

    (; Aes, tlds)  = device_cache

    for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Aₑ        = Aes[chunk]
            tld       = tlds[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                reinit!(tld.cc, eid)

                fill!(Aₑ, 0)
                assemble_element!(Aₑ, tld.cc, tld.ec, p)
                assemble!(assembler, celldofs(tld.cc), Aₑ)
            end
        end
    end
end

function _update_bilinear_operator_on_subdomain!(assembler, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device, device_cache, colors) = strategy_cache
    (; chunksize) = device

    # TODO this should be in the device cache
    (; Aes, tlds)  = device_cache
    assemblers = [duplicate_for_device(device, assembler) for tid in 1:length(tlds)]

    for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Aₑ        = Aes[chunk]
            tld       = tlds[chunk]
            assembler = assemblers[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                reinit!(tld.cc, eid)

                fill!(Aₑ, 0)
                assemble_element!(Aₑ, tld.cc, tld.ec, p)
                assemble!(assembler, celldofs(tld.cc), Aₑ)
            end
        end
    end
end

function _update_bilinear_operator_on_subdomain!(assembler, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache, p)
    error("Element assembly not implemented yet for bilinear operators.")
end

mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::BilinearFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::BilinearFerriteOperator) = eltype(op.A)
Base.size(op::BilinearFerriteOperator, axis) = sisze(op.A, axis)


###############################################################################

struct LinearFerriteOperator{VectorType} <: AbstractLinearOperator
    b::VectorType
    strategy
    subdomain_caches::Vector{SubdomainCache}
end

# Control dispatch for assembly strategy
function update_operator!(op::LinearFerriteOperator, p)
    (; b, strategy, subdomain_caches) = op

    assembler = start_assemble(strategy, b)

    for (subdomain_id, subdomain_cache) in enumerate(subdomain_caches)
        (; sdh, element_cache, strategy_cache) = subdomain_caches
        # Function barrier
        @timeit_debug "assemble subdomain $subdomain_id" _update_linear_operator!(assembler, sdh, element_cache, strategy_cache, p)
    end

    finalize_assembly!(assembler)
end

function _update_linear_operator!(b, sdh, element_cache, strategy_cache::SequentialAssemblyStrategyCache{<:AbstractCPUDevice}, p)
    ndofs = ndofs_per_cell(sdh)
    bₑ = zeros(ndofs)
    @inbounds for cell in CellIterator(sdh)
        fill!(bₑ, 0)
        assemble_element!(bₑ, cell, element_cache, p)
        assemble!(b, cell, bₑ)
    end
end

function _update_linear_operator!(b, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:AbstractCPUDevice}, p)
    ndofs = ndofs_per_cell(sdh)
    bₑ = zeros(ndofs)
    @inbounds for cell in CellIterator(sdh)
        fill!(bₑ, 0)
        assemble_element!(bₑ, cell, element_cache, p)
        assemble!(b, cell, bₑ)
    end
end

function _update_linear_operator!(b, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    @inbounds for cell in CellIterator(sdh)
        bₑ = get_data_for_index(beas, cellid(cell))
        fill!(bₑ, 0)
        assemble_element!(bₑ, cell, element_cache, p)
    end

    finalize_assembly!(assembler)
end

function _update_linear_operator!(b, sdh, element_cache, strategy_cache::ElementAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device) = strategy_cache
    (; chunksize) = device
    ncells = length(sdh.cellset)
    nchunks = ceil(Int, ncells / chunksize)
    # TODO from device cache
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunks]
    @batch for chunk in 1:nchunks
        chunkbegin = (chunk-1)*chunksize+1
        chunkbound = min(ncells, chunk*chunksize)
        for i in chunkbegin:chunkbound
            eid = sdh.cellset[i]
            tld = tlds[chunk]
            reinit!(tld.cc, eid)
            bₑ = get_data_for_index(beas, eid)
            fill!(bₑ, 0.0)
            assemble_element!(bₑ, tld.cc, tld.ec, p)
        end
    end

    finalize_assembly!(assembler)
end

function _update_linear_operator!(b, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:SequentialCPUDevice}, p)
    (; colors) = strategy_cache
    ndofs = ndofs_per_cell(sdh)
    bₑ = zeros(ndofs)
    for color in colors
        @timeit_debug "assemble subdomain" @inbounds for cell in CellIterator(sdh.dh, color)
            fill!(bₑ, 0)
            assemble_element!(bₑ, cell, element_cache, p)
            assemble!(b, cell, bₑ)
        end
    end
end

function _update_linear_operator!(b, sdh, element_cache, strategy_cache::PerColorAssemblyStrategyCache{<:PolyesterDevice}, p)
    (; device, colors) = strategy_cache
    (; chunksize) = device

    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)
    # TODO this should be in the device cache
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunksmax]

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    bes  = [zeros(ndofs) for tid in 1:nchunksmax]

    for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            bₑ  = bes[chunk]
            tld = tlds[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                reinit!(tld.cc, eid)

                fill!(bₑ, 0)
                assemble_element!(bₑ, tld.cc, tld.ec, p)
                assemble!(b, celldofs(tld.cc), bₑ)
            end
        end
    end
end
