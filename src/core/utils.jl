"""
    FerriteOperators.DEBUG

Compile-time flag selecting the exhaustive form of setup-time checks whose
production form samples. Loaded from the `use_debug` preference and therefore
constant-folded: production carries neither check nor branch. Mirrors
`Ferrite.DEBUG` but is set independently of it.
"""
const DEBUG = Preferences.@load_preference("use_debug", false)

"""
    FerriteOperators.debug_mode(; enable = true)

Turn the [`FerriteOperators.DEBUG`](@ref) preference on or off. Takes effect
after a Julia session restart, since the flag is baked in at precompilation.
"""
function debug_mode(; enable = true)
    if DEBUG == enable
        @info "Debug mode already $(enable ? "en" : "dis")abled."
    else
        Preferences.@set_preferences!("use_debug" => enable)
        @info "Debug mode $(enable ? "en" : "dis")abled. Restart the Julia session for this change to take effect!"
    end
    return nothing
end

"""
    geometric_subdomain_interpolation(sdh::SubDofHandler) -> VectorizedInterpolation

The geometric interpolation of `sdh`'s cells, vectorized to the grid's spatial
dimension — the `ip_geo` a `CellValues`/`FacetValues` constructor takes. Read
off the subdomain's FIRST cell: a `SubDofHandler` is uniform in cell type by
construction.
"""
function geometric_subdomain_interpolation(sdh::SubDofHandler)
    grid      = get_grid(sdh.dh)
    sdim      = getspatialdim(grid)
    firstcell = getcells(grid, first(sdh.cellset))
    ip_geo    = Ferrite.geometric_interpolation(typeof(firstcell))^sdim
    return ip_geo
end

"""
    get_first_cell(sdh::SubDofHandler) -> AbstractCell

The first cell of `sdh`'s cellset. A `SubDofHandler` is uniform in cell type,
so this is the cell a setup hook reads a subdomain's geometry from.
"""
function get_first_cell(sdh::SubDofHandler)
    grid = get_grid(sdh.dh)
    return getcells(grid, first(sdh.cellset))
end
