"""
    AssembledLinearizedFerriteOperator(J, integrator, dh)
    AssembledLinearizedFerriteOperator(integrator, dh)


!!! todo
    signatures

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_element! -> update jacobian/residual contribution with internal state variables
    assemble_face! -> update jacobian/residual contribution for boundary

!!! todo
    assemble_interface! -> update jacobian/residual contribution for interface contributions (e.g. DG or FSI)
"""
struct AssembledLinearizedFerriteOperator{MatrixType <: AbstractSparseMatrix, IntegratorType, DHType <: AbstractDofHandler, StrategyCacheType} <: AbstractNonlinearOperator
    J::MatrixType
    integrator::IntegratorType
    dh::DHType
    strategy_cache::StrategyCacheType
end

function Base.show(io::IO, cache::AssembledLinearizedFerriteOperator)
    println(io, "AssembledLinearizedFerriteOperator:")
    Base.show(io, typeof(cache.integrator))
    Base.show(io, MIME"text/plain"(), cache.dh)
end

get_matrix(op::AssembledLinearizedFerriteOperator) = op.J

# Interface
function update_linearization!(op::AssembledLinearizedFerriteOperator, u::AbstractVector, time)
    _update_linearization_J!(op, op.strategy_cache, u, time)
end
function update_linearization!(op::AssembledLinearizedFerriteOperator, residual::AbstractVector, u::AbstractVector, time)
    _update_linearization_Jr!(op, op.strategy_cache, residual, u, time)
end

"""
    mul!(out::AbstractVector, op::AssembledLinearizedFerriteOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::AssembledLinearizedFerriteOperator, in::AbstractVector, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear form to the vector `in`.

!!! TODO
    Revisit this decision. Should mul! be the action of the nonlinear operator (i.e. the residual kernel) or of its linearization (i.e. the MV kernel for iterative solvers)?
"""
mul!(out::AbstractVector, op::AssembledLinearizedFerriteOperator, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::AssembledLinearizedFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)

Base.eltype(op::AssembledLinearizedFerriteOperator) = eltype(op.J)
Base.size(op::AssembledLinearizedFerriteOperator, axis) = size(op.J, axis)

# -------------------------------------------------- Sequential on CPU --------------------------------------------------

# This function is defined to control the dispatch
function _update_linearization_J!(op::AssembledLinearizedFerriteOperator, strategy_cache::SequentialAssemblyStrategyCache, u::AbstractVector, time)
    @unpack J, dh, integrator = op

    assembler = start_assemble(J)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        face_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _sequential_update_linearization_on_subdomain_J!(assembler, sdh, element_cache, face_cache, u, time)
    end

    #finish_assemble(assembler)
end
# This function is defined to make things sufficiently type-stable
function _sequential_update_linearization_on_subdomain_J!(assembler, sdh, element_cache, face_cache, u, time)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    @inbounds for cell in CellIterator(sdh)
        # Prepare buffers
        fill!(Jₑ, 0)
        uₑ .= @view u[celldofs(cell)]

        # Fill buffers
        @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble faces" assemble_element!(Jₑ, uₑ, cell, face_cache, time)
        assemble!(assembler, celldofs(cell), Jₑ)
    end
end

# This function is defined to control the dispatch
function _update_linearization_Jr!(op::AssembledLinearizedFerriteOperator, strategy_cache::SequentialAssemblyStrategyCache, residual::AbstractVector, u::AbstractVector, time)
    @unpack J, dh, integrator, strategy_cache = op

    assembler = start_assemble(J, residual)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        face_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _sequential_update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, face_cache, u, time)
    end

    #finish_assemble(assembler)
end
# This function is defined to make things sufficiently type-stable
function _sequential_update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, face_cache, u, time)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    rₑ = zeros(ndofs)
    @inbounds for cell in CellIterator(sdh)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        dofs = celldofs(cell)

        uₑ .= @view u[dofs]
        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble faces" assemble_element!(Jₑ, rₑ, uₑ, cell, face_cache, time)
        assemble!(assembler, dofs, Jₑ, rₑ)
    end
end

# -------------------------------------------------- Colored on CPU --------------------------------------------------

# This function is defined to control the dispatch
function _update_linearization_J!(op::AssembledLinearizedFerriteOperator, strategy_cache::PerColorAssemblyStrategyCache, u::AbstractVector, time)
    @unpack J, dh, integrator = op

    assembler = start_assemble(J)

    for (sdhidx, sdh) in enumerate(dh.subdofhandlers)
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        face_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _update_colored_linearization_on_subdomain_J!(assembler, strategy_cache.color_cache[sdhidx], sdh, element_cache, face_cache, u, time, strategy_cache.device_cache)
    end

    #finish_assemble(assembler)
end

# This function is defined to make things sufficiently type-stable
function _update_colored_linearization_on_subdomain_J!(assembler, colors, sdh, element_cache, face_cache, u, time, ::SequentialCPUDevice)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    @inbounds for color in colors
        for cell in CellIterator(sdh.dh, color)
            # Prepare buffers
            fill!(Jₑ, 0)
            uₑ .= @view u[celldofs(cell)]

            # Fill buffers
            @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, time)
            # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
            # TODO benchmark against putting this into the FacetIterator
            @timeit_debug "assemble faces" assemble_element!(Jₑ, uₑ, cell, face_cache, time)
            assemble!(assembler, celldofs(cell), Jₑ)
        end
    end
end
function _update_colored_linearization_on_subdomain_J!(assembler, colors, sdh, element_cache, face_cache, u, time, device::PolyesterDevice)
    (; chunksize) = device
    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    Jes  = [zeros(ndofs,ndofs) for tid in 1:nchunksmax]
    ues  = [zeros(ndofs) for tid in 1:nchunksmax]
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), (duplicate_for_device(device, element_cache), duplicate_for_device(device, face_cache))) for tid in 1:nchunksmax]
    assemblers = [duplicate_for_device(device, assembler) for tid in 1:nchunksmax]

    @inbounds for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Jₑ        = Jes[chunk]
            uₑ        = ues[chunk]
            tld       = tlds[chunk]
            assembler = assemblers[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                cell = tld.cc
                reinit!(cell, eid)

                # Prepare buffers
                fill!(Jₑ, 0)
                uₑ .= @view u[celldofs(cell)]

                # Fill buffers
                @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, tld.ec[1], time)
                # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
                # TODO benchmark against putting this into the FacetIterator
                @timeit_debug "assemble faces" assemble_element!(Jₑ, uₑ, cell, tld.ec[2], time)
                assemble!(assembler, celldofs(cell), Jₑ)
            end
        end
    end
end

# This function is defined to control the dispatch
function _update_linearization_Jr!(op::AssembledLinearizedFerriteOperator, strategy_cache::PerColorAssemblyStrategyCache, residual::AbstractVector, u::AbstractVector, time)
    @unpack J, dh, integrator = op

    assembler = start_assemble(J, residual)

    for (sdhidx,sdh) in enumerate(dh.subdofhandlers)
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        face_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _update_colored_linearization_on_subdomain_Jr!(assembler, strategy_cache.color_cache[sdhidx], sdh, element_cache, face_cache, u, time, strategy_cache.device_cache)
    end

    #finish_assemble(assembler)
end

# This function is defined to make things sufficiently type-stable
function _update_colored_linearization_on_subdomain_Jr!(assembler, colors, sdh, element_cache, face_cache, u, time, ::SequentialCPUDevice)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    rₑ = zeros(ndofs)
    @inbounds for color in colors
        for cell in CellIterator(sdh.dh, color)
            # Prepare buffers
            fill!(Jₑ, 0)
            fill!(rₑ, 0)
            uₑ .= @view u[celldofs(cell)]

            # Fill buffers
            @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
            # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
            # TODO benchmark against putting this into the FacetIterator
            @timeit_debug "assemble faces" assemble_element!(Jₑ, rₑ, uₑ, cell, face_cache, time)
            assemble!(assembler, celldofs(cell), Jₑ, rₑ)
        end
    end
end
function _update_colored_linearization_on_subdomain_Jr!(assembler, colors, sdh, element_cache, face_cache, u, time, device::PolyesterDevice)
    (; chunksize) = device
    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    Jes  = [zeros(ndofs,ndofs) for tid in 1:nchunksmax]
    res  = [zeros(ndofs) for tid in 1:nchunksmax]
    ues  = [zeros(ndofs) for tid in 1:nchunksmax]
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), (duplicate_for_device(device, element_cache), duplicate_for_device(device, face_cache))) for tid in 1:nchunksmax]
    assemblers = [duplicate_for_device(device, assembler) for tid in 1:nchunksmax]

    @inbounds for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Jₑ        = Jes[chunk]
            rₑ        = res[chunk]
            uₑ        = ues[chunk]
            tld       = tlds[chunk]
            assembler = assemblers[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                cell = tld.cc
                reinit!(cell, eid)

                # Prepare buffers
                fill!(Jₑ, 0)
                fill!(rₑ, 0)
                uₑ .= @view u[celldofs(cell)]

                # Fill buffers
                @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, tld.ec[1], time)
                # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
                # TODO benchmark against putting this into the FacetIterator
                @timeit_debug "assemble faces" assemble_element!(Jₑ, rₑ, uₑ, cell, tld.ec[2], time)
                assemble!(assembler, celldofs(cell), Jₑ, rₑ)
            end
        end
    end
end

struct AssembledBilinearFerriteOperator{MatrixType, MatrixType2, IntegratorType, DHType <: AbstractDofHandler, StrategyCacheType} <: AbstractBilinearOperator
    A::MatrixType
    A_::MatrixType2 # FIXME we need this if we assemble on a different device type than we solve on (e.g. CPU and GPU)
    integrator::IntegratorType
    dh::DHType
    strategy_cache::StrategyCacheType
end

create_system_matrix(system_matrix_type, dh) = allocate_matrix(dh)

function setup_assembled_operator(strategy::SequentialAssemblyStrategy{<:AbstractCPUDevice}, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler, system_matrix_type::Type = SparseMatrixCSC{Float64, Int64})
    A  = create_system_matrix(system_matrix_type, dh)
    A_ = if strategy.device isa AbstractCPUDevice && system_matrix_type isa SparseMatrixCSC #if "can assemble with system_matrix_type"
        A
    else
        allocate_matrix(dh)
    end

    return AssembledBilinearFerriteOperator(
        A, A_,
        integrator,
        dh,
        SequentialAssemblyStrategyCache(strategy.device),
    )
end

function setup_assembled_operator(strategy::PerColorAssemblyStrategy, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler, system_matrix_type::Type = SparseMatrixCSC{Float64, Int64})
    A  = create_system_matrix(system_matrix_type, dh)
    A_ = if strategy.device isa AbstractCPUDevice && system_matrix_type isa SparseMatrixCSC #if "can assemble with system_matrix_type"
        A
    else
        allocate_matrix(dh)
    end

    return AssembledBilinearFerriteOperator(
        A, A_,
        integrator,
        dh,
        PerColorAssemblyStrategyCache(strategy.device, create_dh_coloring(dh)), # TODO grab coloring algorithm from strategy
    )
end


function update_operator!(op::AssembledBilinearFerriteOperator, time)
    _update_bilinaer_operator!(op, op.strategy_cache, time)
end

function _update_bilinaer_operator!(op::AssembledBilinearFerriteOperator, strategy_cache::SequentialAssemblyStrategyCache, time)
    @unpack A, A_, integrator, dh  = op

    assembler = start_assemble(A_)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_bilinear_operator_on_subdomain_sequential!(assembler, sdh, element_cache, time)
    end

    #finish_assemble(assembler)

    if A !== A_
        copyto!(nonzeros(A), nonzeros(A_))
    end
end

function _update_bilinear_operator_on_subdomain_sequential!(assembler, sdh, element_cache, time)
    ndofs = ndofs_per_cell(sdh)
    Aₑ = zeros(ndofs, ndofs)

    @inbounds for cell in CellIterator(sdh)
        fill!(Aₑ, 0)
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element_cache, time)
        assemble!(assembler, celldofs(cell), Aₑ)
    end
end

function _update_bilinaer_operator!(op::AssembledBilinearFerriteOperator, strategy_cache::PerColorAssemblyStrategyCache{<:AbstractCPUDevice}, time)
    @unpack A, A_, integrator, dh  = op

    assembler = start_assemble(A_)

    for (sdhidx,sdh) in enumerate(dh.subdofhandlers)
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_colored_bilinear_operator_on_subdomain!(assembler, strategy_cache.color_cache[sdhidx], sdh, element_cache, time, strategy_cache.device_cache)
    end

    #finish_assemble(assembler)
    if A !== A_
        copyto!(nonzeros(A), nonzeros(A_))
    end
end

function _update_colored_bilinear_operator_on_subdomain!(assembler, colors, sdh, element_cache, time, device::SequentialCPUDevice)
    # TODO this should be in the device cache
    ndofs     = ndofs_per_cell(sdh)
    Aₑ        = zeros(ndofs,ndofs)

    @timeit_debug "assemble subdomain" for color in colors
        @inbounds for cell in CellIterator(sdh.dh, color)
            fill!(Aₑ, 0)
            @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element_cache, time)
            assemble!(assembler, celldofs(cell), Aₑ)
        end
    end
end

function _update_colored_bilinear_operator_on_subdomain!(assembler, colors, sdh, element_cache, time, device::PolyesterDevice)
    (; chunksize) = device
    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)
    # TODO this should be in the device cache
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunksmax]

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    Aes  = [zeros(ndofs,ndofs) for tid in 1:nchunksmax]
    assemblers = [duplicate_for_device(device, assembler) for tid in 1:nchunksmax]

    @timeit_debug "assemble subdomain" for color in colors
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
                assemble_element!(Aₑ, tld.cc, tld.ec, time)
                assemble!(assembler, celldofs(tld.cc), Aₑ)
            end
        end
    end
end

mul!(out::AbstractVector, op::AssembledBilinearFerriteOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::AssembledBilinearFerriteOperator, in::AbstractVector, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::AssembledBilinearFerriteOperator) = eltype(op.A)
Base.size(op::AssembledBilinearFerriteOperator, axis) = sisze(op.A, axis)


###############################################################################

struct AssembledLinearFerriteOperator{VectorType, IntegratorType, DHType <: AbstractDofHandler, StrategyCacheType} <: AbstractLinearOperator
    b::VectorType
    integrator::IntegratorType
    dh::DHType
    strategy_cache::StrategyCacheType
end

# Control dispatch for assembly strategy
update_operator!(op::AssembledLinearFerriteOperator, time) = _update_linear_operator!(op, op.strategy_cache, time)

function _update_linear_operator!(op::AssembledLinearFerriteOperator, strategy_cache::SequentialAssemblyStrategyCache, time)
    @unpack b, dh, integrator  = op

    fill!(b, 0.0)
    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_linear_operator_on_subdomain_sequential!(b, sdh, element_cache, time)
    end

    #finish_assemble(assembler)
end

function _update_linear_operator_on_subdomain_sequential!(b, sdh, element_cache, time)
    ndofs = ndofs_per_cell(sdh)
    bₑ = zeros(ndofs)
    @inbounds for cell in CellIterator(sdh)
        fill!(bₑ, 0)
        @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
        # assemble!(assembler, celldofs(cell), bₑ)
        b[celldofs(cell)] .+= bₑ
    end
end

# CPU EA dispatch
function _update_linear_operator!(op::AbstractLinearOperator, strategy_cache::ElementAssemblyStrategyCache{<:AbstractCPUDevice}, time)
    @unpack b, dh, integrator  = op

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_pealinear_operator_on_subdomain!(strategy_cache.ea_data, sdh, element_cache, time, strategy_cache.device_cache)
    end

    fill!(b, 0.0)
    ea_collapse!(b, strategy_cache.ea_data)
end

# function _update_pealinear_operator_on_subdomain!(beas::EAVector, sdh, element_cache, time, device::SequentialCPUDevice)
#     @timeit_debug "assemble subdomain" @inbounds for cell in CellIterator(sdh)
#         bₑ = get_data_for_index(beas, cellid(cell))
#         fill!(bₑ, 0)
#         @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
#     end
# end

# function _update_pealinear_operator_on_subdomain!(beas::EAVector, sdh, element_cache, time, device::PolyesterDevice)
#     (; chunksize) = device
#     ncells = length(sdh.cellset)
#     nchunks = ceil(Int, ncells / chunksize)
#     tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunks]
#     @timeit_debug "assemble subdomain" @batch for chunk in 1:nchunks
#         chunkbegin = (chunk-1)*chunksize+1
#         chunkbound = min(ncells, chunk*chunksize)
#         for i in chunkbegin:chunkbound
#             eid = sdh.cellset[i]
#             tld = tlds[chunk]
#             reinit!(tld.cc, eid)
#             bₑ = get_data_for_index(beas, eid)
#             fill!(bₑ, 0.0)
#             # @timeit_debug "assemble element" assemble_element!(bₑ, tld.cc, tld.ec, time)
#             assemble_element!(bₑ, tld.cc, tld.ec, time)
#         end
#     end
# end

function _update_linear_operator!(op::AbstractLinearOperator, strategy_cache::PerColorAssemblyStrategyCache{<:AbstractCPUDevice}, time)
    @unpack b, dh, integrator  = op

    fill!(b, 0.0)

    for (sdhidx, sdh) in enumerate(dh.subdofhandlers)
        # Build evaluation caches
        element_cache = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_colored_linear_operator_on_subdomain!(b, strategy_cache.color_cache[sdhidx], sdh, element_cache, time, strategy_cache.device_cache)
    end
end

function _update_colored_linear_operator_on_subdomain!(b, colors, sdh, element_cache, time, device::SequentialCPUDevice)
    ndofs = ndofs_per_cell(sdh)
    bₑ = zeros(ndofs)
    for color in colors
        @timeit_debug "assemble subdomain" @inbounds for cell in CellIterator(sdh.dh, color)
            fill!(bₑ, 0)
            @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
            b[celldofs(cell)] .+= bₑ
        end
    end
end

function _update_colored_linear_operator_on_subdomain!(b, colors, sdh, element_cache, time, device::PolyesterDevice)
    (; chunksize) = device

    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)
    # TODO this should be in the device cache
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunksmax]

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    bes  = [zeros(ndofs) for tid in 1:nchunksmax]

    @timeit_debug "assemble subdomain" for color in colors
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
                assemble_element!(bₑ, tld.cc, tld.ec, time)
                b[celldofs(tld.cc)] .+= bₑ
            end
        end
    end
end

function needs_update(op::AssembledLinearFerriteOperator, t)
    return _needs_update(op, op.integrator.integrand, t)
end
