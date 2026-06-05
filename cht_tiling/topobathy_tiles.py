"""Generate topography/bathymetry elevation tiles at the highest zoom level
and downsample them to create lower zoom levels.

Uses terrarium or mapbox PNG encoding for elevation data, with support for
parallel processing via ThreadPool and HydroMT data catalogs.
"""

import logging
import os
import time
from multiprocessing.pool import ThreadPool

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import box

from cht_tiling.utils import (
    deg2num,
    elevation2png,
    makedir,
    num2deg,
    png2elevation,
)

logger = logging.getLogger(__name__)


def make_topobathy_tiles_top_level(twm: object, data_dict: dict) -> None:
    """Generate highest-zoom-level elevation tiles from raster data.

    For each tile at the maximum zoom level, reads elevation data (via a
    HydroMT data catalog or existing tiles), merges with any pre-existing
    tile data, and writes the result as an encoded PNG.

    Parameters
    ----------
    twm : object
        A ``TiledWebMap`` instance providing configuration such as
        ``zoom_range``, ``path``, ``index_path``, ``npix``, ``encoder``,
        ``parallel``, ``interpolation_method``, ``z_range``, and
        ``data_catalog``.
    data_dict : dict
        Dictionary describing the dataset to process. Must contain at
        least a ``"name"`` or ``"elevation"`` key for catalog lookup.
    """
    npix = 256

    transformer_4326_to_3857 = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
    )

    twm.max_zoom = twm.zoom_range[1]

    # Use highest zoom level
    izoom = twm.zoom_range[1]
    zoom_path = os.path.join(twm.path, str(izoom))

    # Determine elapsed time
    t0 = time.time()

    # Create rectangular mesh with origin (0.0) and 256x256 pixels
    dxy = (40075016.686 / npix) / 2**izoom
    xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
    yy = xx[:]
    xv, yv = np.meshgrid(xx, yy)

    # Determine min and max indices for this zoom level
    if twm.index_path:
        # Get ix0, ix1, iy0 and iy1 from the existing index files
        index_zoom_path = os.path.join(twm.index_path, str(izoom))
        # List folders and turn names into integers
        iy0 = 1e15
        iy1 = -1e15
        ix_list = [int(i) for i in os.listdir(index_zoom_path)]
        ix0 = min(ix_list)
        ix1 = max(ix_list)
        # Now loop through the folders to get the min and max y indices
        for i in range(ix0, ix1 + 1):
            it_list = [
                int(j.split(".")[0])
                for j in os.listdir(os.path.join(index_zoom_path, str(i)))
            ]
            iy0 = min(iy0, min(it_list))
            iy1 = max(iy1, max(it_list))
    else:
        if lon_range is None or lat_range is None:
            # Without extent info, use the full world
            lon_range = [-180.0, 180.0]
            lat_range = [-85.0, 85.0]

        ix0, iy0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, iy1 = deg2num(lat_range[0], lon_range[1], izoom)

    # Limit the indices
    ix0 = max(0, ix0)
    iy0 = max(0, iy0)
    ix1 = min(2**izoom - 1, ix1)
    iy1 = min(2**izoom - 1, iy1)

    # Add some stuff to options dict, which is used for parallel processing
    options = {}
    options["index_path"] = twm.index_path
    options["transformer_4326_to_3857"] = transformer_4326_to_3857
    options["xv"] = xv
    options["yv"] = yv
    options["dxy"] = dxy
    options["interpolation_method"] = twm.interpolation_method
    options["z_range"] = twm.z_range
    options["data_catalog"] = twm.data_catalog
    options["skip_existing"] = twm.skip_existing

    # Loop in x direction
    for i in range(ix0, ix1 + 1):
        logger.info(f"Processing column {i - ix0 + 1} of {ix1 - ix0 + 1}")

        zoom_path_i = os.path.join(zoom_path, str(i))

        if not os.path.exists(zoom_path_i):
            makedir(zoom_path_i)

        # Loop in y direction
        if twm.parallel:
            with ThreadPool() as pool:
                pool.starmap(
                    create_highest_zoom_level_tile,
                    [
                        (
                            zoom_path_i,
                            i,
                            j,
                            izoom,
                            twm,
                            data_dict,
                            options,
                        )
                        for j in range(iy0, iy1 + 1)
                    ],
                )
        else:
            for j in range(iy0, iy1 + 1):
                create_highest_zoom_level_tile(
                    zoom_path_i, i, j, izoom, twm, data_dict, options
                )

        # If zoom_path_i is empty, then remove it again
        if not os.listdir(zoom_path_i):
            os.rmdir(zoom_path_i)

    t1 = time.time()

    logger.info(f"Elapsed time for zoom level {izoom}: {t1 - t0}")


def make_topobathy_tiles_lower_levels(twm: object) -> None:
    """Generate lower zoom level tiles by downsampling from the level above.

    Each tile is constructed by averaging 2x2 blocks from four tiles at
    the next-higher zoom level.

    Parameters
    ----------
    twm : object
        A ``TiledWebMap`` instance providing ``zoom_range``, ``path``,
        ``encoder``, ``encoder_vmin``, ``encoder_vmax``, and ``parallel``.
    """
    npix = 256

    for izoom in range(twm.zoom_range[1] - 1, twm.zoom_range[0] - 1, -1):
        logger.info(f"Processing zoom level {izoom}")

        t0 = time.time()

        zoom_path = os.path.join(twm.path, str(izoom))
        zoom_path_higher = os.path.join(twm.path, str(izoom + 1))

        # First determine ix0 and ix1 based on higher zoom level
        ix_list = [int(i) for i in os.listdir(zoom_path_higher)]
        ix0_higher = min(ix_list)
        ix1_higher = max(ix_list)
        ix0 = int(ix0_higher / 2)
        ix1 = int(ix1_higher / 2)

        # Now loop through the folders to get the min and max y indices
        it0_higher = 1e15
        it1_higher = -1e15
        for i in os.listdir(zoom_path_higher):
            it_list = [
                int(j.split(".")[0])
                for j in os.listdir(os.path.join(zoom_path_higher, i))
            ]
            if len(it_list) > 0:
                it0_higher = min(it0_higher, min(it_list))
                it1_higher = max(it1_higher, max(it_list))
        iy0 = int(it0_higher / 2)
        iy1 = int(it1_higher / 2)

        # Loop in x direction
        for i in range(ix0, ix1 + 1):
            path_okay = False
            zoom_path_i = os.path.join(zoom_path, str(i))

            if not path_okay:
                if not os.path.exists(zoom_path_i):
                    makedir(zoom_path_i)
                    path_okay = True

            if twm.parallel:
                with ThreadPool() as pool:
                    pool.starmap(
                        make_lower_level_tile,
                        [
                            (
                                zoom_path_i,
                                zoom_path_higher,
                                i,
                                j,
                                npix,
                                twm,
                            )
                            for j in range(iy0, iy1 + 1)
                        ],
                    )
            else:
                for j in range(iy0, iy1 + 1):
                    make_lower_level_tile(
                        zoom_path_i, zoom_path_higher, i, j, npix, twm
                    )

        t1 = time.time()

        logger.info(f"Elapsed time for zoom level {izoom}: {t1 - t0}")


def bbox_xy2latlon(
    x0: float, x1: float, y0: float, y1: float, crs: CRS
) -> tuple[float, float, float, float]:
    """Convert a bounding box from a projected CRS to WGS 84 lat/lon.

    Parameters
    ----------
    x0 : float
        Minimum x coordinate.
    x1 : float
        Maximum x coordinate.
    y0 : float
        Minimum y coordinate.
    y1 : float
        Maximum y coordinate.
    crs : CRS
        Source coordinate reference system.

    Returns
    -------
    tuple[float, float, float, float]
        ``(lon_min, lon_max, lat_min, lat_max)`` in WGS 84.
    """
    transformer = Transformer.from_crs(crs, crs.from_epsg(4326), always_xy=True)
    lon_min, lat_min = transformer.transform(x0, y0)
    lon_max, lat_min = transformer.transform(x1, y0)
    lon_min, lat_max = transformer.transform(x0, y1)
    lon_max, lat_max = transformer.transform(x1, y1)
    return lon_min, lon_max, lat_min, lat_max


def create_highest_zoom_level_tile(
    zoom_path_i: str,
    i: int,
    j: int,
    izoom: int,
    twm: object,
    data_dict: dict,
    options: dict,
) -> None:
    """Create a single tile at the highest zoom level.

    Reads existing tile data (if present), fetches new elevation data from
    the data catalog, merges them, and writes the result as an encoded PNG.

    Parameters
    ----------
    zoom_path_i : str
        Output directory for this tile column.
    i : int
        Tile column index.
    j : int
        Tile row index.
    izoom : int
        Zoom level.
    twm : object
        The ``TiledWebMap`` instance.
    data_dict : dict
        Dataset descriptor for catalog lookup.
    options : dict
        Processing options including ``transformer_4326_to_3857``, ``xv``,
        ``yv``, ``dxy``, ``z_range``, ``data_catalog``, ``index_path``,
        and ``skip_existing``.
    """
    file_name = os.path.join(zoom_path_i, f"{j}.png")
    transformer_4326_to_3857 = options["transformer_4326_to_3857"]
    xv = options["xv"]
    yv = options["yv"]
    dxy = options["dxy"]
    z_range = options["z_range"]

    skip_existing = options["skip_existing"]

    # Create highest zoom level tile
    if os.path.exists(file_name):
        if skip_existing:
            return
        else:
            zg0 = png2elevation(
                file_name,
                encoder=twm.encoder,
                encoder_vmin=twm.encoder_vmin,
                encoder_vmax=twm.encoder_vmax,
            )
    else:
        zg0 = np.zeros((twm.npix, twm.npix))
        zg0[:] = np.nan

    # If there are no NaNs, we can continue
    if not np.any(np.isnan(zg0)):
        return

    if options["index_path"]:
        # Only make tiles for which there is an index file
        index_file_name = os.path.join(
            options["index_path"], str(izoom), str(i), f"{j}.png"
        )
        if not os.path.exists(index_file_name):
            return

    # Compute lat/lon at upper left corner of tile
    lat, lon = num2deg(i, j, izoom)

    # Convert origin to Global Mercator
    xo, yo = transformer_4326_to_3857.transform(lon, lat)

    # Tile grid on Global mercator
    x3857 = xo + xv[:] + 0.5 * dxy
    y3857 = yo - yv[:] - 0.5 * dxy

    data_catalog = options["data_catalog"]

    if data_catalog is not None:
        # HydroMT data catalog path
        xmin, xmax = float(x3857.min()), float(x3857.max())
        ymin, ymax = float(y3857.min()), float(y3857.max())
        geom = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=3857)
        name = data_dict.get("elevation", data_dict.get("name"))
        try:
            da = data_catalog.get_rasterdataset(name, geom=geom, zoom=(dxy, "metre"))
            zg = da.values.astype(np.float64)
        except Exception:
            zg = np.full((len(y3857), len(x3857)), np.nan)
    else:
        zg = np.full(x3857.shape, np.nan)

    # Any value below zmin is set NaN
    zg[np.where(zg < z_range[0])] = np.nan
    # Any value above zmax is set NaN
    zg[np.where(zg > z_range[1])] = np.nan

    if np.isnan(zg).all():
        return

    # Overwrite zg with zg0 where not zg0 is not nan
    mask = np.isfinite(zg0)
    zg[mask] = zg0[mask]

    # Write to terrarium png format
    elevation2png(
        zg,
        file_name,
        encoder=twm.encoder,
        encoder_vmin=twm.encoder_vmin,
        encoder_vmax=twm.encoder_vmax,
    )


def make_lower_level_tile(
    zoom_path_i: str,
    zoom_path_higher: str,
    i: int,
    j: int,
    npix: int,
    twm: object,
) -> None:
    """Create a single tile by downsampling four tiles from the next-higher zoom level.

    Parameters
    ----------
    zoom_path_i : str
        Output directory for this tile column at the current zoom level.
    zoom_path_higher : str
        Root directory of tiles at the next-higher zoom level.
    i : int
        Tile column index.
    j : int
        Tile row index.
    npix : int
        Number of pixels per tile edge.
    twm : object
        The ``TiledWebMap`` instance providing encoder settings.
    """
    # Get the indices of the tiles in the higher zoom level
    i00, j00 = 2 * i, 2 * j  # upper left
    i10, j10 = 2 * i, 2 * j + 1  # lower left
    i01, j01 = 2 * i + 1, 2 * j  # upper right
    i11, j11 = 2 * i + 1, 2 * j + 1  # lower right

    # Create empty array of NaN to store the elevation data from the higher zoom level
    zg512 = np.zeros((npix * 2, npix * 2))
    zg512[:] = np.nan

    # Create empty array of NaN of 4*npix*npix to store the 2-stride elevation data from higher zoom level
    zg4 = np.zeros((4, npix, npix))
    zg4[:] = np.nan

    okay = False

    # Upper left
    file_name = os.path.join(zoom_path_higher, str(i00), f"{j00}.png")
    if os.path.exists(file_name):
        zgh = png2elevation(
            file_name,
            encoder=twm.encoder,
            encoder_vmin=twm.encoder_vmin,
            encoder_vmax=twm.encoder_vmax,
        )
        zg512[0:npix, 0:npix] = zgh
        okay = True
    # Lower left
    file_name = os.path.join(zoom_path_higher, str(i10), f"{j10}.png")
    if os.path.exists(file_name):
        zgh = png2elevation(
            file_name,
            encoder=twm.encoder,
            encoder_vmin=twm.encoder_vmin,
            encoder_vmax=twm.encoder_vmax,
        )
        zg512[npix:, 0:npix] = zgh
        okay = True
    # Upper right
    file_name = os.path.join(zoom_path_higher, str(i01), f"{j01}.png")
    if os.path.exists(file_name):
        zgh = png2elevation(
            file_name,
            encoder=twm.encoder,
            encoder_vmin=twm.encoder_vmin,
            encoder_vmax=twm.encoder_vmax,
        )
        zg512[0:npix, npix:] = zgh
        okay = True
    # Lower right
    file_name = os.path.join(zoom_path_higher, str(i11), f"{j11}.png")
    if os.path.exists(file_name):
        zgh = png2elevation(
            file_name,
            encoder=twm.encoder,
            encoder_vmin=twm.encoder_vmin,
            encoder_vmax=twm.encoder_vmax,
        )
        zg512[npix:, npix:] = zgh
        okay = True

    if not okay:
        return

    # Compute average of 4 tiles in higher zoom level
    zg4[0, :, :] = zg512[0 : npix * 2 : 2, 0 : npix * 2 : 2]
    zg4[1, :, :] = zg512[1 : npix * 2 : 2, 0 : npix * 2 : 2]
    zg4[2, :, :] = zg512[0 : npix * 2 : 2, 1 : npix * 2 : 2]
    zg4[3, :, :] = zg512[1 : npix * 2 : 2, 1 : npix * 2 : 2]

    zg = np.nanmean(zg4, axis=0)

    file_name = os.path.join(zoom_path_i, f"{j}.png")
    elevation2png(
        zg,
        file_name,
        encoder=twm.encoder,
        encoder_vmin=twm.encoder_vmin,
        encoder_vmax=twm.encoder_vmax,
    )


def read_tfw(tfw_path: str) -> object:
    """Read a TFW (TIFF World File) and return an affine transformation.

    Parameters
    ----------
    tfw_path : str
        Path to the ``.tfw`` file.

    Returns
    -------
    object
        An ``Affine`` transformation from ``rasterio.transform.from_origin``.
    """
    with open(tfw_path, "r") as f:
        lines = f.readlines()

    cell_size_x = float(lines[0].strip())
    rotation_x = float(lines[1].strip())
    rotation_y = float(lines[2].strip())
    cell_size_y = float(lines[3].strip())
    upper_left_x = float(lines[4].strip())
    upper_left_y = float(lines[5].strip())

    return from_origin(upper_left_x, upper_left_y, cell_size_x, abs(cell_size_y))


def list_folders(src: str, basename: bool = False) -> list[str]:
    """List subdirectories matching a glob pattern.

    Parameters
    ----------
    src : str
        Glob pattern to match directories.
    basename : bool
        If True, return only the directory basenames.

    Returns
    -------
    list[str]
        Sorted list of matching directory paths or basenames.
    """
    folder_list = []
    full_list = glob.glob(src)
    for item in full_list:
        if os.path.isdir(item):
            if basename:
                folder_list.append(os.path.basename(item))
            else:
                folder_list.append(item)

    return sorted(folder_list)
