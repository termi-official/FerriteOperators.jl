function geometric_subdomain_interpolation(sdh::SubDofHandler)
    grid      = get_grid(sdh.dh)
    sdim      = getspatialdim(grid)
    firstcell = getcells(grid, first(sdh.cellset))
    ip_geo    = Ferrite.geometric_interpolation(typeof(firstcell))^sdim
    return ip_geo
end

function get_first_cell(sdh::SubDofHandler)
    grid = get_grid(sdh.dh)
    return getcells(grid, first(sdh.cellset))
end
