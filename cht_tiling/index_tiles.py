"""Generate slippy-map index tiles that map each pixel to a grid cell index."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from pyproj import CRS, Transformer

from cht_tiling.utils import (
    binary_search,
    deg2num,
    int2png,
    makedir,
    num2deg,
    png2elevation,
)


def make_index_tiles(twm: Any, topo_path: str | None = None) -> None:
    """Create index tiles for a TiledWebMap grid, dispatching by grid type.

    Currently only QuadTree grids (identified by a ``"level"`` data array) are
    supported.

    Parameters
    ----------
    twm : Any
        TiledWebMap instance with ``data``, ``path``, and ``zoom_range`` attributes.
    topo_path : str | None
        Optional path to pre-existing topography tiles used for masking.

    Raises
    ------
    ValueError
        If the grid type cannot be determined.
    """
    if not twm.zoom_range:
        twm.zoom_range = [0, 13]

    if "level" in twm.data:
        make_index_tiles_quadtree(twm.data, twm.path, twm.zoom_range, topo_path)
    else:
        raise ValueError("Grid type not recognized by make_index_tiles")


def make_index_tiles_quadtree(
    grid: Any,
    path: str,
    zoom_range: list[int],
    topo_path: str | None,
) -> None:
    """Generate index tiles for a QuadTree grid at each zoom level.

    For every tile in the zoom range that overlaps the grid extent, each pixel is
    mapped to the index of the finest-level grid cell that contains it. The result
    is written as a 32-bit integer PNG tile.

    Parameters
    ----------
    grid : Any
        xarray-like grid dataset with attributes ``x0``, ``y0``, ``dx``, ``dy``,
        ``nmax``, ``mmax``, ``rotation``, ``nr_levels`` and data variables
        ``level``, ``n``, ``m``.
    path : str
        Output directory for the index tile pyramid.
    zoom_range : list[int]
        Two-element list ``[min_zoom, max_zoom]``.
    topo_path : str | None
        Optional path to topography tiles; pixels where elevation > 0 are masked out.
    """
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

    ifirst = np.zeros(nr_refinement_levels, dtype=int)
    for ilev in range(0, nr_refinement_levels):
        ifirst[ilev] = np.where(grid["level"].to_numpy()[:] == ilev + 1)[0][0]

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

    i0_lev: list[int] = []
    i1_lev: list[int] = []
    nmax_lev: list[int] = []
    mmax_lev: list[int] = []
    nm_lev: list[np.ndarray] = []
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
        print(f"Processing zoom level {izoom}")

        zoom_path = os.path.join(path, str(izoom))

        dxy = (40075016.686 / npix) / 2**izoom
        xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
        yy = xx[:]
        xv, yv = np.meshgrid(xx, yy)

        ix0, iy0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, iy1 = deg2num(lat_range[0], lon_range[1], izoom)

        for i in range(ix0, ix1 + 1):
            path_okay = False
            zoom_path_i = os.path.join(zoom_path, str(i))

            for j in range(iy0, iy1 + 1):
                file_name = os.path.join(zoom_path_i, f"{j}.png")

                zbtile = None
                if topo_path is not None:
                    file_name_topo = os.path.join(
                        topo_path, str(izoom), str(i), f"{j}.png"
                    )
                    if os.path.exists(file_name_topo):
                        zbtile = png2elevation(file_name_topo)

                lat, lon = num2deg(i, j, izoom)

                xo, yo = transformer_a.transform(lon, lat)

                x = xo + xv[:] + 0.5 * dxy
                y = yo - yv[:] - 0.5 * dxy

                x, y = transformer_b.transform(x, y)

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
                    ind = iind * nmax + jind
                    ind[iind < 0] = -999
                    ind[jind < 0] = -999
                    ind[iind >= mmax] = -999
                    ind[jind >= nmax] = -999

                    ingrid = np.isin(ind, nm_lev[ilev], assume_unique=False)
                    incell = np.where(ingrid)

                    if incell[0].size > 0:
                        try:
                            cell_indices = (
                                binary_search(nm_lev[ilev], ind[incell[0], incell[1]])
                                + i0_lev[ilev]
                            )
                            indx[incell[0], incell[1]] = cell_indices
                        except Exception:
                            pass

                if np.any(indx >= 0):
                    if not path_okay:
                        if not os.path.exists(zoom_path_i):
                            makedir(zoom_path_i)
                            path_okay = True

                    if zbtile is not None:
                        ibad = np.where(zbtile > 0.0)
                        indx[ibad] = -999.0

                    int2png(indx, file_name)
