"""
    create_system_matrix(strategy, dh)

Allocate the operator's global matrix over the sparsity pattern the strategy's
operator specification declares ([`StandardOperatorSpecification`](@ref) →
`SparsityPattern`, [`BlockedOperatorSpecification`](@ref) →
`BlockSparsityPattern`), with entries added by Ferrite's
`add_sparsity_entries!` from the specification's `algebraic_couplings` and
`constraint_handler`, allocated as `matrix_type(strategy)`.

The constraint handler contributes SPARSITY ENTRIES only; applying the
constraints to the assembled system stays the caller's, through Ferrite's
`apply!`/`apply_assemble!`.
"""
create_system_matrix(strategy, dh) = _create_system_matrix(strategy, strategy.form.operator_specification, dh)
create_system_vector(strategy, dh) = allocate_vector(strategy.device, dh)

function _create_system_matrix(strategy, spec, dh)
    sp = init_operator_sparsity_pattern(spec, dh)
    couplings = spec.algebraic_couplings
    # The `algebraic_couplings` keyword exists only on Ferrite versions with
    # mesh-free algebraic variables, so it is passed only when non-empty.
    if isempty(couplings)
        add_sparsity_entries!(sp, dh, spec.constraint_handler)
    else
        add_sparsity_entries!(sp, dh, spec.constraint_handler; algebraic_couplings = couplings)
    end
    return allocate_operator_matrix(strategy.device, matrix_type(strategy), sp)
end

"""
    allocate_operator_matrix(device, matrix_type, sp)

The operator's global matrix over the sparsity pattern `sp`
[`create_system_matrix`](@ref) built.

Ferrite ships no `allocate_matrix(::Type{<:device matrix}, ::SparsityPattern)`
— only the `DofHandler` form, which rebuilds the pattern from the handler and
would drop this package's coupling declarations. So a GPU device allocates the
host `SparseMatrixCSC` over the SAME pattern every other device uses and hands
it to the device type's constructor, which is the conversion Ferrite's
`DofHandler` form performs internally. The type parameters come from that
constructor (CUSPARSE fixes the index type at `Cint`), not from the requested
spelling.
"""
allocate_operator_matrix(::AbstractDevice, ::Type{MT}, sp) where {MT} = allocate_matrix(MT, sp)
allocate_operator_matrix(device::AbstractGPUDevice, ::Type{MT}, sp) where {MT} =
    Base.typename(MT).wrapper(
        allocate_matrix(SparseMatrixCSC{value_type(device), index_type(device)}, sp))

init_operator_sparsity_pattern(::StandardOperatorSpecification, dh) = Ferrite.init_sparsity_pattern(dh)
init_operator_sparsity_pattern(spec::BlockedOperatorSpecification, dh) = BlockSparsityPattern(spec.block_sizes)

# The form's element-side elections ([`with_assembly_form`](@ref)) are applied to
# the raw cache, before any decoration and before the engine builds the
# workspaces and device layouts from it.
function setup_elements(integrator, dh, form, ad_backend, n_global_dofs)
    needs_ad_decoration(integrator) ||
        return [with_assembly_form(setup_element_cache(integrator, sdh), form, sdh) for sdh in dh.subdofhandlers]
    return [setup_decorated_element_cache(integrator, sdh, form, ad_backend, n)
            for (sdh, n) in zip(dh.subdofhandlers, n_global_dofs)]
end

# One subdomain's element cache, built and decorated. Both counts the decorator
# is sized from are the INTEGRATOR's declarations, resolved here, which is what
# keeps `decorate_element_cache` itself integrator-free.
function setup_decorated_element_cache(integrator, sdh, form, ad_backend, n_global_dofs::Int)
    cache = with_assembly_form(setup_element_cache(integrator, sdh), form, sdh)
    return decorate_element_cache(cache, sdh, ad_backend, n_global_dofs;
                                  n_internal_dofs = resolve_internal_dofs_per_element(integrator, cache, sdh))
end

function _cell_internal_offsets(integrator, element_caches, dh)
    num_dofs_per_element = zeros(Int, getncells(get_grid(dh))+1)
    for (sdh, cache) in zip(dh.subdofhandlers, element_caches)
        for (cellid, nidofs) in zip(sdh.cellset, get_number_of_internal_dofs_per_element(integrator, cache, sdh))
            num_dofs_per_element[1+cellid] = nidofs
        end
    end
    @assert all(num_dofs_per_element .≥ 0) "Number of internal dofs must be non-negative!"
    # The leading zero pad makes the cumulative sum exactly the `ncells+1`
    # block-relative offsets the handler expects.
    return cumsum(num_dofs_per_element)
end

# `has_internal_state` alone is not a safe signal that a cache needs a REAL
# internal-dof block: it may be declared purely to opt into the
# sensitivity-admissibility rules (`internal_state_insensitive`) and never
# condense anything, without implementing the count hook at all.
#
# The probe subject is the `unwrap` fixpoint, so a decorator's forwarding
# method answers for its inner, and the ARGUMENT types must be the concrete
# ones the later call passes — an author-annotated method is not matched by an
# `Any` probe, and missing it hands the operator a placeholder handler that
# fails inside `condense_cell!` instead.
_declares_internal_dofs(hook, integrator, cache, arg) =
    has_internal_state(typeof(cache)) &&
    hasmethod(hook, Tuple{typeof(integrator), typeof(unwrap(cache)), typeof(arg)})

"""
    resolve_internal_dofs_per_element(integrator, cache, sdh) -> Int

The condensed internal-dof count of ONE cell of `sdh`, from the same
[`get_number_of_internal_dofs_per_element`](@ref) declaration the
[`InternalVariableHandler`](@ref) lays the `q` block out from. This is what
sizes [`ADElementCache`](@ref)'s `:q` seeds and configurations, so the length a
generic route seeds is the length the [`InternalSource`](@ref) gather hands the
kernel — a tensor-valued internal variable owns several internal dofs per
quadrature point, which is why the quadrature-point count does not answer this.

`0` where the cache declares no real internal block, matching the placeholder
block `setup_internal_variable_handler` builds for the same caches; the
decorator then builds no `:q` configuration and a generic `:q` route refuses by
name.

One configuration serves every cell of the subdomain, so a count that varies
between them is refused here rather than surfacing as a `DimensionMismatch`
inside the first generic sweep.
"""
function resolve_internal_dofs_per_element(integrator, cache, sdh)
    _declares_internal_dofs(get_number_of_internal_dofs_per_element, integrator, cache, sdh) || return 0
    counts = get_number_of_internal_dofs_per_element(integrator, cache, sdh)
    isempty(counts) && return 0
    allequal(counts) || throw(ArgumentError(
        "$(nameof(typeof(unwrap(cache)))) declares internal-dof counts between " *
        "$(minimum(counts)) and $(maximum(counts)) across this subdomain's cells. The AD " *
        "decorator seeds `:q` through ONE ForwardDiff configuration per subdomain, whose seed " *
        "length is fixed, so a count varying between cells cannot be swept generically. " *
        "Declare a uniform count, or serve every AD-decorator kind analytically " *
        "(`provides_analytic`) so no decorator is built for this subdomain."))
    return Int(first(counts))
end

"""
    setup_internal_variable_handler(integrator, element_caches, algebraic_domain, dh)

Build the [`InternalVariableHandler`](@ref) from what the resolved caches
declare; `algebraic_domain` is `resolve_algebraic_domain`'s result, `nothing`
where the integrator declares no algebraic items.

Whether a block is REAL (an offsets array) or the placeholder (`nothing`) is
decided per block by the cache — [`has_internal_state`](@ref) AND an
implemented dof-count hook
([`get_number_of_internal_dofs_per_element`](@ref) /
[`get_number_of_internal_dofs_per_algebraic_item`](@ref)) — not by the
integrator's type: the cell and item blocks are independent, and either or
both can be real at once (the layout-collision case).
"""
function setup_internal_variable_handler(integrator, element_caches, algebraic_domain, dh)
    needs_cells = any(zip(dh.subdofhandlers, element_caches)) do (sdh, cache)
        _declares_internal_dofs(get_number_of_internal_dofs_per_element, integrator, cache, sdh)
    end
    needs_items = algebraic_domain !== nothing && _declares_internal_dofs(
        get_number_of_internal_dofs_per_algebraic_item, integrator, algebraic_domain[1], algebraic_domain[2])
    (needs_cells || needs_items) && return _build_internal_variable_handler(
        integrator, element_caches, algebraic_domain, dh, needs_cells, needs_items)
    return InternalVariableHandler(nothing, nothing, 0, 0)
end

function _build_internal_variable_handler(integrator, element_caches, algebraic_domain, dh, needs_cells, needs_items)
    cell_offsets = needs_cells ? _cell_internal_offsets(integrator, element_caches, dh) : nothing
    item_offsets = needs_items ? _algebraic_item_offsets(integrator, algebraic_domain[1], algebraic_domain[2]) : nothing
    cell_len = cell_offsets === nothing ? 0 : cell_offsets[end]
    item_len = item_offsets === nothing ? 0 : item_offsets[end]
    return InternalVariableHandler(cell_offsets, item_offsets, ndofs(dh), cell_len + item_len)
end

"""
    iteration_kind(form) -> kind

The sweep kind an operator of `form` resolves its item iterator for
([`assembly_iterator`](@ref)). `nothing` for a form whose sweeps are the primal
family — every one of them positions on the full geometry cache, so there is no
kind to narrow by. A [`MatrixFreeAction`](@ref) operator answers with its ACTION
kind: that is the sweep run per `mul!`, and the quadrature-data fill it also
runs rides the same iterator (which is why what each sweep REFRESHES is
[`item_update_flags`](@ref)'s per-kind declaration and not the iterator's).
"""
iteration_kind(form) = nothing
iteration_kind(::MatrixFreeAction) = MatrixFreeActionKind()

function setup_subdomain_caches(strategy, element_caches, ivh, dh;
        slots::NTuple{<:Any, Symbol}, needs_sensitivity::Bool, global_dof_sets)
    device = strategy.device
    kind = iteration_kind(strategy.form)
    # One device-resident handler for the whole operator, split per subdomain
    # below: it is what a device item iterator must be built from, and
    # rebuilding it per subdomain would upload the cell-id maps of every other
    # subdomain again.
    device_dh = setup_device_handler(device, dh)
    return [begin
        partition = adapt_partition(device, compute_partition(strategy, sdh))
        n = n_workers(device, partition)
        ws = create_assembly_workspace(element_cache, sdh, ivh, slots;
                                       needs_sensitivity, global_dofs = gdofs,
                                       iterator = assembly_iterator(kind, element_cache, sdh))
        dc = setup_device_instances(device, ws, n,
            _device_iterator(kind, element_cache, sdh, device_subdomain_handler(device_dh, index)))
        SubdomainCache(AssemblyDomain(sdh, ivh, element_cache), dc, partition)
    end for (index, (sdh, element_cache, gdofs)) in
        enumerate(zip(dh.subdofhandlers, element_caches, global_dof_sets))]
end

# A CPU device has no device handler and therefore no device iterator; the
# workspace it duplicates already carries the host one. `sdh` is the HOST
# subdomain, passed alongside `device_sdh` since a device iterator may need a
# host-only setup-time fact (`device_assembly_iterator`, C2).
_device_iterator(kind, element_cache, sdh, ::Nothing) = nothing
_device_iterator(kind, element_cache, sdh, device_sdh) =
    device_assembly_iterator(kind, element_cache, sdh, device_sdh)

# Each family's global-dof declaration is resolved once per subdomain, before
# any cache exists, and validated here rather than surfacing later as an
# out-of-bounds scatter or a doubly assembled entry.
resolve_global_dof_sets(strategy, integrator, dh) = _resolve_global_dof_sets(
    strategy, dh, [global_dofs(integrator, sdh) for sdh in dh.subdofhandlers], "global_dofs")

resolve_facet_item_global_dof_sets(strategy, integrator, dh) = _resolve_global_dof_sets(
    strategy, dh, [facet_item_global_dofs(integrator, sdh) for sdh in dh.subdofhandlers],
    "facet_item_global_dofs")

function _resolve_global_dof_sets(strategy, dh, sets, declaration)
    all(isempty, sets) && return sets
    _reject_unsupported_global_dof_strategy(strategy, declaration)
    for (index, (sdh, gdofs)) in enumerate(zip(dh.subdofhandlers, sets))
        _validate_global_dofs(index, sdh, gdofs, ndofs(dh), declaration)
    end
    return sets
end

function _reject_unsupported_global_dof_strategy(strategy::AssemblyStrategy, declaration)
    strategy.form isa MatrixFreeAction && throw(ArgumentError(
        "A subdomain declaring `$declaration` cannot be evaluated under `MatrixFreeAction`: the " *
        "element action is defined on the cell's FIELD space (`apply_element_action!` receives " *
        "`uₑ`/`yₑ` in `celldofs` order and the ELEMENT level's matrices are `ndofs_per_cell` " *
        "square), so the declared tail would be gathered, ignored by the element, and scattered " *
        "back as zero. Assemble this operator under `FullAssembly`."))
    strategy.device isa AbstractGPUDevice && throw(ArgumentError(
        "A subdomain declaring `$declaration` cannot be assembled on " *
        "$(nameof(typeof(strategy.device))): a GPU device assembles under `ColoredScheduling` " *
        "only, and a dof shared by every item of a subdomain admits no coloring. Assemble this " *
        "operator on a CPU device."))
    strategy.scheduling isa ColoredScheduling && throw(ArgumentError(
        "A subdomain declaring `$declaration` cannot be assembled under `ColoredScheduling`: " *
        "coloring makes a scatter race-free by giving no two items of a color a shared dof, " *
        "and a declared global dof is shared by every item that carries it, so no coloring " *
        "isolates it. Use `SequentialScheduling`, whose parallel route is the atomic scatter."))
    return nothing
end

function _validate_global_dofs(index, sdh, gdofs, ndofs_total, declaration)
    for d in gdofs
        1 <= d <= ndofs_total || throw(ArgumentError(
            "Subdomain $index declares the global dof $d through `$declaration`, which is out " *
            "of bounds for a DofHandler with $ndofs_total dofs."))
    end
    allunique(gdofs) || throw(ArgumentError(
        "Subdomain $index declares the global dofs $(collect(gdofs)) through `$declaration`, " *
        "which are not unique. The declaration is the ordered tail of the element-local system, " *
        "so a repeated dof would receive the same contribution twice."))
    # Cheap sample: the first cell witnesses a head/tail overlap for the
    # uniform-field case this covers.
    isempty(sdh.cellset) && return nothing
    cdofs = celldofs(sdh.dh, first(sdh.cellset))
    for d in gdofs
        d in cdofs && throw(ArgumentError(
            "Subdomain $index declares the global dof $d through `$declaration`, which is also " *
            "a cell dof (found on cell $(first(sdh.cellset))). The local system is " *
            "`[celldofs(cell); global dofs]`, so such a dof would receive every contribution " *
            "twice. Only the first cell of the subdomain is sampled."))
    end
    return nothing
end

####################################
## Device support walls
####################################

"""
    assert_device_supported(device, strategy, integrator, dh)
    assert_device_internal_state_supported(device, ivh)

Reject at setup what a device cannot assemble. Both are no-ops for a CPU
device, which serves every item family; the [`AbstractGPUDevice`](@ref) methods
cover the cell-item, coloring-only slice a device kernel supports today, and
each rejection names the limitation rather than surfacing as a `MethodError`
inside the first sweep — or, for the scheduling one, as a silent data race.

The first form runs on the strategy and the integrator's DECLARATIONS, before
any cache is built; the second needs the resolved
[`InternalVariableHandler`](@ref).
"""
assert_device_supported(::AbstractDevice, strategy, integrator, dh) = nothing

function assert_device_supported(device::AbstractGPUDevice, strategy::AssemblyStrategy, integrator, dh)
    dev = nameof(typeof(device))
    # The coloring requirement is the MATRIX assembler's: Ferrite's device one
    # accumulates with a plain `+=` — its `AbstractThreadSafeAssembler` supertype
    # means "safe to alias across workers given a valid coloring", not race-free
    # — so an uncolored device sweep into a matrix is a silent data race. A form
    # that assembles no matrix scatters through this package's own
    # `VectorAssembler`, which IS atomic-capable on device, and takes either
    # scheduling.
    (operator_specification(strategy.form) === nothing || strategy.scheduling isa ColoredScheduling) || throw(ArgumentError(
        "$dev requires `ColoredScheduling` for an assembling form (got " *
        "$(nameof(typeof(strategy.scheduling)))). Ferrite's device matrix assembler accumulates " *
        "with a plain `+=` — its `AbstractThreadSafeAssembler` supertype means \"safe to alias " *
        "across workers given a valid coloring\", not race-free — so an uncolored device sweep " *
        "is a silent data race. Pass `scheduling = ColoredScheduling()`."))
    (operator_specification(strategy.form) === nothing || element_mapping(device) isa WorkerPerElement) ||
        throw(ArgumentError(
            "$dev carries `$(nameof(typeof(element_mapping(device))))`, which executes the " *
            "matrix-free action only. An assembling form is `WorkerPerElement`; build the device " *
            "without `with_element_mapping`, or set the operator up with `form = MatrixFreeAction(; " *
            "element_mapping = CooperativeElement())`, which resolves the mapping itself."))
    _assert_device_specification(device, operator_specification(strategy.form), integrator)

    needs_ad_decoration(integrator) && throw(ArgumentError(
        "$dev assembles bilinear and linear forms only (got $(nameof(typeof(integrator)))). A " *
        "nonlinear integrator carries the `ADElementCache` decoration and the per-worker " *
        "sensitivity buffers, neither of which has a device layout."))
    isempty(algebraic_items(integrator, dh)) || throw(ArgumentError(
        "$dev does not support the algebraic item family (`algebraic_items`): its items are a " *
        "dof set with no cell, and the device geometry cache addresses cells."))
    for sdh in dh.subdofhandlers
        isempty(facet_items(integrator, sdh)) || throw(ArgumentError(
            "$dev does not support the facet item family (`facet_items`): Ferrite 1.7 has no " *
            "device `FacetValues`."))
    end
    return nothing
end

# A form that allocates no global array declares no storage to check.
_assert_device_specification(device, ::Nothing, integrator) = nothing

function _assert_device_specification(device, spec, integrator)
    dev = nameof(typeof(device))
    spec isa BlockedOperatorSpecification && throw(ArgumentError(
        "$dev does not support `BlockedOperatorSpecification`: Ferrite ships no device " *
        "`BlockAssembler`. Use a `StandardOperatorSpecification`, naming the device matrix type."))
    spec.constraint_handler === nothing || throw(ArgumentError(
        "$dev does not support a `constraint_handler` on the operator specification. Allocate " *
        "the operator without one and apply the constraints yourself — Ferrite's `apply!` takes " *
        "a device constraint handler (`adapt(backend, ch)`)."))
    _assert_device_matrix_type(device, spec.matrix_type)
    _assert_no_silent_host_matrix(device, spec, integrator)
    return nothing
end

# `matrix_type(device, spec)` resolves a `StandardOperatorSpecification` with
# no `matrix_type` named — or one explicitly given as a host type — to the
# host `SparseMatrixCSC`, and [`allocate_operator_matrix`](@ref) allocates
# exactly that: the system matrix would silently land on the HOST. Loud here,
# at setup, before any cache is built. Scoped to the integrator families that
# actually allocate the global MATRIX under this form ([`create_system_matrix`](@ref));
# a linear integrator allocates a vector, whose type this spec's `matrix_type`
# has no say over. The `KernelAbstractions.CPU` debug backend is genuinely
# host-resident — there is no device memory to have missed — and is exempt.
_assert_no_silent_host_matrix(device, spec, ::AbstractLinearIntegrator) = nothing
function _assert_no_silent_host_matrix(device, spec, integrator)
    MT = matrix_type(device, spec)
    (MT <: SparseMatrixCSC && !_host_resident_backend(device)) && throw(ArgumentError(
        "$(nameof(typeof(device))) resolves the operator specification's matrix type to the " *
        "host $MT: no device matrix type was named (or a host one was named explicitly), so " *
        "`StandardOperatorSpecification`'s default is what gets allocated on a device whose " *
        "system matrix belongs off the host. Pass `StandardOperatorSpecification(matrix_type = " *
        "<device matrix type>)`."))
    return nothing
end

_host_resident_backend(::AbstractGPUDevice) = false
_host_resident_backend(device::KernelAbstractionsDevice) = nameof(typeof(device.backend)) === :CPU

_assert_device_matrix_type(device, ::Nothing) = nothing

# The GLOBAL side alone: the device's `value_type` and the named matrix type
# describe the same system matrix, so they have to agree. The ELEMENT-local
# scalar is the integrator's own election ([`element_value_type`](@ref)) and is
# deliberately not part of this check — a `Float64` element scattering into a
# `Float32` system converts entry-wise, which is a supported configuration.
function _assert_device_matrix_type(device, ::Type{MT}) where {MT}
    eltype(MT) === value_type(device) || throw(ArgumentError(
        "$(nameof(typeof(device))) assembles in $(value_type(device)) but the operator " *
        "specification names the matrix type $MT, whose element type is $(eltype(MT)). Set the " *
        "device's `value_type` and the matrix type's element type to the same scalar."))
    # A capability check, not a name check: Ferrite 1.7 ships a device assembler
    # for `CuSparseMatrixCSC` and nothing else — `CuSparseMatrixCSR` is
    # allocatable but has no `start_assemble`/`assemble!`, so it would fail on
    # the first sweep instead of here.
    hasmethod(Ferrite.start_assemble, Tuple{MT}) || throw(ArgumentError(
        "No `Ferrite.start_assemble` method accepts $MT, so it cannot be assembled into. " *
        "Ferrite 1.7 ships a device assembler for CSC device matrices only; a CSR device " *
        "matrix is allocatable but not assemblable."))
    return nothing
end

assert_device_internal_state_supported(::AbstractDevice, ivh) = nothing

function assert_device_internal_state_supported(device::AbstractGPUDevice, ivh)
    has_internal_dof_block(ivh) && throw(ArgumentError(
        "$(nameof(typeof(device))) does not support condensed internal state: the " *
        "element-local solves `condense_internal!` runs, and the internal-variable handler " *
        "that lays their block out, have no device path."))
    return nothing
end

"""
    assert_declaration_signatures(integrator, dh)

Reject a declaration hook whose method was written against a signature the
engine does not call. [`global_dofs`](@ref), [`facet_items`](@ref),
[`facet_item_global_dofs`](@ref) and [`algebraic_items`](@ref) all default to an
EMPTY declaration, so a drifted signature never surfaces as a `MethodError`: the
default answers instead, and the operator assembles a subset — a missing
local-system tail, an unvisited boundary, absent algebraic rows — without a word.

The check is type-level and runs once per [`setup_engine`](@ref). For every
integrator that answers these hooks — the outer one and, through the wrappers
that forward them, their sub-integrators — a hook with ANY method specialized on
that integrator's type must have one the engine's own call resolves to. An
integrator declaring nothing has no specialized method and passes; a correct
declarer's method is what the call resolves to and passes.
"""
function assert_declaration_signatures(integrator, dh::AbstractDofHandler)
    subjects = _declaration_subjects!(Any[], integrator)
    for subject in subjects
        _assert_hook_signature(algebraic_items, subject, typeof(dh))
    end
    isempty(dh.subdofhandlers) && return nothing
    # Type-level, and every subdomain of one DofHandler shares `typeof(sdh)`, so
    # one representative answers for all of them.
    SDH = typeof(first(dh.subdofhandlers))
    for subject in subjects
        _assert_hook_signature(global_dofs, subject, SDH)
        _assert_hook_signature(facet_items, subject, SDH)
        _assert_hook_signature(facet_item_global_dofs, subject, SDH)
    end
    return nothing
end

function _assert_hook_signature(hook, subject, argtype::Type)
    IT = typeof(subject)
    _is_empty_declaration_default(which(hook, Tuple{IT, argtype})) || return nothing
    drifted = [m for m in methods(hook, Tuple{IT, Vararg{Any}})
               if !_is_empty_declaration_default(m)]
    isempty(drifted) && return nothing
    expected = "$(nameof(hook))(::$(nameof(IT)), ::$(nameof(argtype)))"
    throw(ArgumentError(
        "$(IT) has a method for the declaration hook `$(nameof(hook))`, but the engine's call " *
        "`$expected` resolves to the empty default, so this integrator declares nothing at " *
        "all. The declaration hooks default to an empty declaration rather than erroring, so " *
        "a drifted signature assembles a silent subset instead of failing.\n" *
        "found:    " * join(drifted, "\n          ") * "\n" *
        "expected: " * expected))
end

# The empty default of a declaration hook is its one method left open in the
# integrator slot; every other method is some integrator type's declaration.
function _is_empty_declaration_default(m::Method)
    params = Base.unwrap_unionall(m.sig).parameters
    return length(params) ≥ 2 && params[2] === Any
end

# Kind types or instances normalize to their UnionAll base, so a declaration
# carrying a payload type parameter (`ParameterVJPKind{Vector{Float64}}`) never
# silently misses its validation entry.
_kind_type(r) = Base.typename(r isa Type ? r : typeof(r)).wrapper

"""
    setup_engine(strategy, integrator, dh; slots = (:u,), requests = (), ad_backend = ForwardDiffAD())

Build the [`AssemblyEngine`](@ref) shared by all operator kinds from the
setup-time declarations: `slots` names the state slots a sweep may carry, one
per-worker buffer each, and `requests` the request kinds whose trait ↔ kernel
and internal-state admissibility checks run here instead of at first use.
Declaring a kind builds no per-worker state; it moves that kind's checks
forward. Kinds are normalized to their UnionAll bases, so an instance or a
payload-parameterized type declares the same kind as its bare name. The engine
carries `slots` — the per-worker buffers are sized for them — and not the kinds,
which are consumed here and restrict nothing afterwards.

Element caches lacking analytic coverage of some AD-decorator kind are wrapped
in [`ADElementCache`](@ref) at construction, for every kind the integrator
might issue and not only the declared ones, decided STRUCTURALLY by
[`needs_ad_decoration`](@ref). `ad_backend = nothing` opts out of wrapping.

The declaration hooks are signature-checked first
([`assert_declaration_signatures`](@ref)), since each defaults to an empty
declaration and a drifted method would otherwise assemble a silent subset.

Each subdomain's [`validate_element_cache`](@ref) call probes `reinit_values!`
against that subdomain's RESOLVED host [`assembly_iterator`](@ref) type rather
than against `CellCache` unconditionally, so an author-annotated
`reinit_values!(c, ::MyIterator)` method is validated on the type it was
written against.

Facet item ([`facet_items`](@ref)) and then algebraic item
([`algebraic_items`](@ref)) caches are appended after the cell subdomains, so
traversal order follows the declarations rather than which families are
present. The algebraic domain is resolved BEFORE the
[`InternalVariableHandler`](@ref) is built, since a condensed algebraic cache's
item block sizes itself from the resolved items and cache, and decorated
afterwards alongside the cell caches.
"""
function setup_engine(strategy::AbstractAssemblyStrategy, integrator, dh::AbstractDofHandler;
        slots = (:u,), requests::Tuple = (), ad_backend = ForwardDiffAD())
    assert_declaration_signatures(integrator, dh)
    assert_device_supported(strategy.device, strategy, integrator, dh)
    declared_slots    = Tuple(slots)
    declared_kinds    = map(_kind_type, requests)
    global_dof_sets   = resolve_global_dof_sets(strategy, integrator, dh)
    facet_item_sets   = resolve_facet_item_global_dof_sets(strategy, integrator, dh)
    element_caches    = setup_elements(integrator, dh, strategy.form, ad_backend, map(length, global_dof_sets))
    kind              = iteration_kind(strategy.form)
    foreach(element_caches, dh.subdofhandlers) do cache, sdh
        validate_element_cache(cache, declared_kinds; iterator_type = typeof(assembly_iterator(kind, cache, sdh)))
    end
    algebraic_domain  = resolve_algebraic_domain(integrator, dh, declared_kinds)
    ivh               = setup_internal_variable_handler(integrator, element_caches, algebraic_domain, dh)
    needs_sensitivity = needs_ad_decoration(integrator)
    assert_device_internal_state_supported(strategy.device, ivh)
    cell_caches       = setup_subdomain_caches(strategy, element_caches, ivh, dh;
                                               slots = declared_slots,
                                               needs_sensitivity,
                                               global_dof_sets)
    facet_caches      = setup_facet_item_caches(strategy, integrator, dh, declared_kinds, ivh;
                                                slots = declared_slots,
                                                needs_sensitivity,
                                                facet_item_global_dof_sets = facet_item_sets)
    algebraic_caches  = setup_algebraic_caches(strategy, algebraic_domain, declared_slots, ad_backend,
                                               needs_sensitivity, ivh)
    # The families carry different domain types; widening only where something
    # is declared keeps a cells-only operator's element type concrete.
    subdomain_caches  = (isempty(facet_caches) && isempty(algebraic_caches)) ? cell_caches :
        vcat(Vector{SubdomainCache}(cell_caches), facet_caches, algebraic_caches)
    return AssemblyEngine(strategy, subdomain_caches, dh, ivh, declared_slots)
end

"""
    setup_operator(strategy, problem, dh; slots = (:u,), requests = (), ad_backend = ForwardDiffAD())

Build the operator for `problem` (an integrator) over `dh` from what the caller
declares: `slots` the state slot names sweeps may carry — the engine sizes one
per-worker buffer per name — and `requests` the request kinds whose setup-time
validation runs here ([`setup_engine`](@ref)). `ad_backend` selects the
[`ADElementCache`](@ref) backend wrapping caches that lack analytic coverage
(`nothing` opts out).

Declaring stays a hint, not a capability restriction: an undeclared kind stays
usable and runs its checks at the call-time entry points instead.

Transfer operators keep their own constructor
([`setup_transfer_operator`](@ref)); patch sweeps run on an ordinary operator
through [`foreach_patch`](@ref).
"""
function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractBilinearIntegrator, dh::AbstractDofHandler;
        slots = (:u,), requests::Tuple = (), ad_backend = ForwardDiffAD())
    engine = setup_engine(strategy, integrator, dh; slots, requests, ad_backend)
    A      = create_system_matrix(engine.strategy, dh)
    return BilinearFerriteOperator(A, engine, integrator)
end

# A matrix specification on an operator that holds no matrix is a
# misconfiguration, not a degraded mode: the layout would be silently dropped.
function _reject_blocked_specification(strategy::AssemblyStrategy{<:FullAssembly})
    strategy.form.operator_specification isa BlockedOperatorSpecification || return nothing
    throw(ArgumentError(
        "A linear operator assembles a vector and holds no matrix, so a " *
        "`BlockedOperatorSpecification` has nothing to lay out. Use a " *
        "`StandardOperatorSpecification`, or build the blocked matrix on the bilinear or " *
        "nonlinear operator it belongs to."))
end
_reject_blocked_specification(strategy) = nothing

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractNonlinearIntegrator, dh::AbstractDofHandler;
        slots = (:u,), requests::Tuple = (), ad_backend = ForwardDiffAD())
    engine = setup_engine(strategy, integrator, dh; slots, requests, ad_backend)
    J      = create_system_matrix(engine.strategy, dh)
    return LinearizedFerriteOperator(J, engine, integrator)
end

function setup_operator(strategy::AbstractAssemblyStrategy, integrator::AbstractLinearIntegrator, dh::AbstractDofHandler;
        slots = (:u,), requests::Tuple = (), ad_backend = ForwardDiffAD())
    _reject_blocked_specification(strategy)
    engine = setup_engine(strategy, integrator, dh; slots, requests, ad_backend)
    b      = create_system_vector(engine.strategy, dh)
    return LinearFerriteOperator(b, engine, integrator)
end

"""
    setup_evaluation_operator(strategy, integrator, dh; slots = (:u,), requests = (), ad_backend = ForwardDiffAD())

Build the [`EvaluationFerriteOperator`](@ref) for `integrator` over `dh`: the
same engine [`setup_operator`](@ref) builds — same element, boundary, facet-item
and algebraic caches through the same family dispatch, same setup-time
validation — and no matrix or vector.

This is the route for a term that is only ever EVALUATED: a functional reduced
over the domain ([`evaluate_functional`](@ref)), a per-quadrature-point
evaluation ([`evaluate_quadrature!`](@ref)). Any integrator family may take it;
what the operator does not do is assemble, and the assembly entry points say so.
"""
function setup_evaluation_operator(strategy::AbstractAssemblyStrategy, integrator, dh::AbstractDofHandler;
        slots = (:u,), requests::Tuple = (), ad_backend = ForwardDiffAD())
    engine = setup_engine(strategy, integrator, dh; slots, requests, ad_backend)
    return EvaluationFerriteOperator(engine, integrator)
end

"""
    init_transfer_sparsity_pattern(dh_row::DofHandler, dh_col::DofHandler)

Build a `Ferrite.SparsityPattern` of size `(ndofs(dh_row) × ndofs(dh_col))` covering all
DoF pairs `(rdof, cdof)` that share a cell. Both DofHandlers must live on the same grid
and have the same number of subdomains.
"""
function init_transfer_sparsity_pattern(dh_row::DofHandler, dh_col::DofHandler)
    nrdofs = ndofs(dh_row)
    ncdofs = ndofs(dh_col)
    nnz_per_row = ndofs_per_cell(dh_col.subdofhandlers[1])
    sp = SparsityPattern(nrdofs, ncdofs; nnz_per_row)
    rdofs_buf = Int[]
    cdofs_buf = Int[]
    for (sdh_row, sdh_col) in zip(dh_row.subdofhandlers, dh_col.subdofhandlers)
        resize!(rdofs_buf, ndofs_per_cell(sdh_row))
        resize!(cdofs_buf, ndofs_per_cell(sdh_col))
        for cellid in sdh_row.cellset
            celldofs!(rdofs_buf, dh_row, cellid)
            celldofs!(cdofs_buf, dh_col, cellid)
            for rdof in rdofs_buf
                for cdof in cdofs_buf
                    Ferrite.add_entry!(sp, rdof, cdof)
                end
            end
        end
    end
    return sp
end

# Shared by `setup_transfer_operator`/`setup_nested_transfer_operator`: both
# restrict to sequential full assembly, only the error label differs.
function _validate_transfer_strategy(strategy, label)
    (strategy isa AssemblyStrategy && strategy.form isa FullAssembly && strategy.scheduling isa SequentialScheduling) ||
        throw(ArgumentError("$label currently only support sequential full-assembly strategies (got $(typeof(strategy)))"))
    strategy.device isa SequentialCPUDevice ||
        throw(ArgumentError("$label currently only support SequentialCPUDevice (got $(typeof(strategy.device)))"))
    return nothing
end

# The whole construction a same-grid and a nested-grid transfer operator share:
# `sp` is the pattern each builds its own way, `tc_builder` the cell cache that
# walks a `(sdh_a, sdh_b)` pair, and the result is one `TransferFerriteOperator`.
function _build_transfer_operator(strategy, integrator, dh_a, dh_b, sp, tc_builder)
    P = allocate_matrix(SparseMatrixCSC{value_type(strategy.device), Int}, sp)
    subdomain_caches = _build_transfer_subdomain_caches(
        strategy, integrator, zip(dh_a.subdofhandlers, dh_b.subdofhandlers), tc_builder)
    return TransferFerriteOperator(P, strategy, subdomain_caches, dh_a, dh_b, integrator)
end

# One `SubdomainCache` per `(sdh_a, sdh_b)` pair.
function _build_transfer_subdomain_caches(strategy, integrator, pairs, tc_builder)
    device = strategy.device
    subdomain_caches = SubdomainCache[]
    for (sdh_a, sdh_b) in pairs
        element = setup_transfer_element_cache(integrator, sdh_a, sdh_b)
        partition = compute_partition(strategy, sdh_a)
        n = n_workers(device, partition)
        tc = tc_builder(sdh_a, sdh_b)
        ws = TransferWorkspace(element, allocate_transfer_element_matrix(element, sdh_a, sdh_b), tc)
        dc = setup_device_instances(device, ws, n)
        push!(subdomain_caches, SubdomainCache(TransferDomain(sdh_a, sdh_b), dc, partition))
    end
    return subdomain_caches
end

"""
    setup_transfer_operator(strategy, integrator, dh_row, dh_col)

Set up a [`TransferFerriteOperator`](@ref) assembling a rectangular sparse matrix of
size `(ndofs(dh_row) × ndofs(dh_col))`. `dh_row` and `dh_col` must live on the **same**
grid with 1-to-1 subdomain lists (same length, same cellsets at each index).

`integrator` must be an [`AbstractTransferIntegrator`](@ref); its
`setup_transfer_element_cache(integrator, sdh_row, sdh_col)` runs once per subdomain
pair.

!!! warning "Experimental surface"
    The transfer constructors and operator types may change in a minor release;
    the assembled matrix and its sparsity are not affected.
"""
function setup_transfer_operator(
        strategy::AbstractAssemblyStrategy,
        integrator::AbstractTransferIntegrator,
        dh_row::DofHandler,
        dh_col::DofHandler,
    )
    _validate_transfer_strategy(strategy, "Transfer operators")
    @assert get_grid(dh_row) === get_grid(dh_col) "Both DofHandlers must share the same grid"
    @assert length(dh_row.subdofhandlers) == length(dh_col.subdofhandlers) "Mismatch in number of subdomains"

    return _build_transfer_operator(strategy, integrator, dh_row, dh_col,
                                    init_transfer_sparsity_pattern(dh_row, dh_col), SameGridCellCache)
end

"""
    init_nested_transfer_sparsity_pattern(dh_fine, dh_coarse, fine2coarse)

Build a `Ferrite.SparsityPattern` of size `(ndofs(dh_fine) × ndofs(dh_coarse))` for a
nested-grid transfer operator: entry `(rdof, cdof)` is added whenever `rdof` belongs to
fine cell `i` and `cdof` to its parent coarse cell `fine2coarse[i]`.
"""
function init_nested_transfer_sparsity_pattern(
        dh_fine::DofHandler,
        dh_coarse::DofHandler,
        fine2coarse::AbstractVector{Int},
    )
    nrdofs   = ndofs(dh_fine)
    ncdofs   = ndofs(dh_coarse)
    nnz_hint = maximum(sdh -> ndofs_per_cell(sdh), dh_coarse.subdofhandlers; init = 1)
    sp       = SparsityPattern(nrdofs, ncdofs; nnz_per_row = nnz_hint)
    rdofs_buf = Int[]
    cdofs_buf = Int[]
    for fine_id in 1:getncells(get_grid(dh_fine))
        coarse_id = fine2coarse[fine_id]
        resize!(rdofs_buf, ndofs_per_cell(dh_fine,   fine_id))
        resize!(cdofs_buf, ndofs_per_cell(dh_coarse, coarse_id))
        celldofs!(rdofs_buf, dh_fine,   fine_id)
        celldofs!(cdofs_buf, dh_coarse, coarse_id)
        for rdof in rdofs_buf, cdof in cdofs_buf
            Ferrite.add_entry!(sp, rdof, cdof)
        end
    end
    return sp
end

"""
    setup_nested_transfer_operator(strategy, integrator, dh_fine, dh_coarse, fine2coarse, child_ref_coords)

Set up a [`TransferFerriteOperator`](@ref) assembling a rectangular sparse matrix
of size `(ndofs(dh_fine) × ndofs(dh_coarse))`. `dh_fine` and `dh_coarse` must live on
**different** grids where every fine cell is a child of exactly one coarse cell, as
encoded by `fine2coarse` and `child_ref_coords`.

!!! warning "Experimental surface"
    The transfer constructors and operator types may change in a minor release;
    the assembled matrix and its sparsity are not affected.
"""
function setup_nested_transfer_operator(
        strategy::AbstractAssemblyStrategy,
        integrator::AbstractTransferIntegrator,
        dh_fine::DofHandler,
        dh_coarse::DofHandler,
        fine2coarse::AbstractVector{Int},
        child_ref_coords::AbstractVector,
    )
    _validate_transfer_strategy(strategy, "Nested transfer operators")

    return _build_transfer_operator(
        strategy, integrator, dh_fine, dh_coarse,
        init_nested_transfer_sparsity_pattern(dh_fine, dh_coarse, fine2coarse),
        (sdh_fine, sdh_coarse) -> NestedGridCellCache(sdh_fine, sdh_coarse, fine2coarse, child_ref_coords))
end
