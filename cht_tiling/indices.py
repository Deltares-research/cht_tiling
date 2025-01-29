import math
import os

import numpy as np
from pyproj import CRS, Transformer

from cht_tiling.utils import binary_search, deg2num, int2png, makedir, num2deg
from cht_tiling.webviewer import write_html


def make_index_tiles(grid, path, zoom_range=None, format="png", webviewer=False):
    # The grid must be an XU grid !

    # Test to see whether this grid is:
    # 1. A QuadTree grid
    # 2. A regular grid
    # 3. An irregular grid (a la FM)

    if not zoom_range:
        # Set default zoom range
        zoom_range = [0, 13]

    # Check if grid has a "level" data array
    if "level" in grid:
        # Must be a quadtree grid
        make_index_tiles_quadtree(grid, path, zoom_range, format)

    # elif "n" in grid and "m" in grid:
    #     # Must be a regular grid
    #     make_index_tiles_regular(grid, path, zoom_range, format)

    # elif check for FM:
    #     pass

    else:
        raise ValueError("Grid type not recognized by make_index_tiles")

    if webviewer:
        # Make webviewer
        write_html(os.path.join(path, "index.html"), max_native_zoom=zoom_range[1])


def make_index_tiles_quadtree(grid, path, zoom_range, format):
    npix = 256

    x0 = grid.attrs["x0"]
    y0 = grid.attrs["y0"]
    dx = grid.attrs["dx"]
    dy = grid.attrs["dy"]
    nmax = grid.attrs["nmax"]
    mmax = grid.attrs["mmax"]
    rotation = grid.attrs["rotation"]
    nr_refinement_levels = grid.attrs["nr_levels"]

    nr_cells = len(grid["level"])

    cosrot = math.cos(-rotation * math.pi / 180)
    sinrot = math.sin(-rotation * math.pi / 180)

    # Find index of first cell in each level
    ifirst = np.zeros(nr_refinement_levels, dtype=int)
    for ilev in range(0, nr_refinement_levels):
        # Find index of first cell with this level
        ifirst[ilev] = np.where(grid["level"].to_numpy()[:] == ilev + 1)[0][0]

    # Compute lon/lat range
    bnds = grid.grid.bounds

    xmin = bnds[0] - 2 * dx
    xmax = bnds[2] + 2 * dx
    ymin = bnds[1] - 2 * dy
    ymax = bnds[3] + 2 * dy

    crs = grid.crs.to_numpy()

    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)
    lon_range = [lon_min, lon_max]
    lat_range = [lat_min, lat_max]

    transformer_a = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
    )
    transformer_b = Transformer.from_crs(CRS.from_epsg(3857), crs, always_xy=True)

    i0_lev = []
    i1_lev = []
    nmax_lev = []
    mmax_lev = []
    nm_lev = []
    for level in range(nr_refinement_levels):
        i0 = ifirst[level]
        if level < nr_refinement_levels - 1:
            i1 = ifirst[level + 1]
        else:
            i1 = nr_cells
        i0_lev.append(i0)
        i1_lev.append(i1)
        nmax_lev.append(np.amax(grid["n"].to_numpy()[i0:i1]) + 1)
        mmax_lev.append(np.amax(grid["m"].to_numpy()[i0:i1]) + 1)
        nn = grid["n"].to_numpy()[i0:i1] - 1
        mm = grid["m"].to_numpy()[i0:i1] - 1
        nm_lev.append(mm * nmax_lev[level] + nn)

    for izoom in range(zoom_range[0], zoom_range[1] + 1):
        print("Processing zoom level " + str(izoom))

        zoom_path = os.path.join(path, str(izoom))

        dxy = (40075016.686 / npix) / 2**izoom
        xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
        yy = xx[:]
        xv, yv = np.meshgrid(xx, yy)

        ix0, it0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, it1 = deg2num(lat_range[0], lon_range[1], izoom)
        # ix1 = ix1 + 1
        # it1 = it1 + 1

        for i in range(ix0, ix1 + 1):
            path_okay = False
            zoom_path_i = os.path.join(zoom_path, str(i))

            for j in range(it0, it1 + 1):
                file_name = os.path.join(zoom_path_i, str(j) + ".png")

                # Compute lat/lon at upper left corner of tile
                lat, lon = num2deg(i, j, izoom)

                # Convert to Global Mercator
                xo, yo = transformer_a.transform(lon, lat)

                # Tile grid on local mercator
                x = xo + xv[:] + 0.5 * dxy
                y = yo - yv[:] - 0.5 * dxy

                # Convert tile grid to crs of SFINCS model
                x, y = transformer_b.transform(x, y)

                # Now rotate around origin of SFINCS model
                x00 = x - x0
                y00 = y - y0
                xg = x00 * cosrot - y00 * sinrot
                yg = x00 * sinrot + y00 * cosrot

                indx = np.full((npix, npix), -999, dtype=int)

                for ilev in range(nr_refinement_levels):
                    nmax = nmax_lev[ilev]
                    mmax = mmax_lev[ilev]
                    i0 = i0_lev[ilev]
                    i1 = i1_lev[ilev]
                    dxr = dx / 2**ilev
                    dyr = dy / 2**ilev
                    iind = np.floor(xg / dxr).astype(int)
                    jind = np.floor(yg / dyr).astype(int)
                    # Now check whether this cell exists on this level
                    ind = iind * nmax + jind
                    ind[iind < 0] = -999
                    ind[jind < 0] = -999
                    ind[iind >= mmax] = -999
                    ind[jind >= nmax] = -999

                    ingrid = np.isin(
                        ind, nm_lev[ilev], assume_unique=False
                    )  # return boolean for each pixel that falls inside a grid cell
                    incell = np.where(
                        ingrid
                    )  # tuple of arrays of pixel indices that fall in a cell

                    if incell[0].size > 0:
                        # Now find the cell indices
                        try:
                            cell_indices = (
                                binary_search(nm_lev[ilev], ind[incell[0], incell[1]])
                                + i0_lev[ilev]
                            )
                            indx[incell[0], incell[1]] = cell_indices
                        except Exception:
                            pass

                if np.any(indx >= 0):
                    # Check whether path exists
                    if not path_okay:
                        if not os.path.exists(zoom_path_i):
                            makedir(zoom_path_i)
                            path_okay = True
                    # And write indices to file
                    # print(file_name)
                    int2png(indx, file_name)
