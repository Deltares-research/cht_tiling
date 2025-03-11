import glob
import os
import time
from multiprocessing.pool import ThreadPool

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

from cht_tiling.utils import (
    deg2num,
    elevation2png,
    makedir,
    num2deg,
    png2elevation,
)

# from cht_tiling.webviewer import write_html


def make_topobathy_tiles_top_level(
    twm,
    data_list,
    bathymetry_database=None,
    index_path=None,
    lon_range=None,
    lat_range=None,
    z_range=[-999999.0, 999999.0],
    zoom_range=None,
    skip_existing=False,
    parallel=True,
    interpolation_method="linear",
):
    """
    Generates highest zoom level topo/bathy tiles

    :param path: Path where topo/bathy tiles will be stored.
    :type path: str
    :param dem_name: List of DEM names (dataset names in Bathymetry Database).
    :type dem_name: list
    :param png_path: Output path where the png tiles will be created.
    :type png_path: str
    :param option: Option.
    :type option: str
    :param zoom_range: Zoom range for which the png tiles
    will be created. Defaults to [0, 23].
    :type zoom_range: list of int

    """

    npix = 256

    transformer_4326_to_3857 = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
    )

    # # Set vertical range
    # zmin = -20000.0
    # zmax = 20000.0
    # if "zmin" in data_dict:
    #     zmin = data_dict["zmin"]
    # if "zmax" in data_dict:
    #     zmax = data_dict["zmax"]
    # z_range = [zmin, zmax]

    # First we determine lon_range, lat_range, dx and crs_data
    # lon_range and lat_range are in geographic coordinates and are used to determine the bounding box of the tiles
    # dx is the resolution of the dataset in meters and is used to determine the zoom levels
    # crs_data is the CRS of the dataset and is used for the transformation between this crs and web mercator

    # twm_data = None

    # if "twm" in data_dict:
    #     twm_data = data_dict["twm"]
    #     data_type = "twm"
    #     crs_data = CRS.from_epsg(3857)

    # elif "ncfile" in data_dict:
    #     # Read the netcdf file and get the data array
    #     ds = xr.open_dataset(data_dict["ncfile"])
    #     da = ds[data_dict["dataarray_name"]]
    #     data_type = "xarray"

    # elif "da" in data_dict:
    #     # XArray dataarray as input
    #     da = data_dict["da"]
    #     data_type = "xarray"

    # elif "tiffile" in data_dict:

    #     # if "tfwfile" in data_dict:
    #     #     # Read the transformation from the TFW file
    #     #     tfw_path = data_dict["tfwfile"]
    #     #     transform = read_tfw(tfw_path)
    #     #     da = rioxarray.open_rasterio(data_dict["tiffile"], transform=transform)
    #     # else:

    #     da = rioxarray.open_rasterio(data_dict["tiffile"])
    #     data_type = "xarray"

    # if data_type == "xarray":

    #     # Get the CRS
    #     crs_data = da.rio.crs

    #     if "crs" in data_dict:
    #         # Override what was in the data array
    #         crs_data = data_dict["crs"]

    #     # Get the lon and lat range and dx
    #     if crs_data.is_geographic:
    #         if not lon_range:
    #             lon_range = [da.x.min(), da.x.max()]
    #         if not lat_range:
    #             lat_range = [da.y.min(), da.y.max()]
    #         dx = np.mean(np.diff(da.y)) * 111000.0

    #     else:
    #         if not lon_range and not lat_range:
    #             # Get bounding box
    #             x_range = [da.x.min(), da.x.max()]
    #             y_range = [da.y.min(), da.y.max()]
    #             # Get lon/lat range
    #             transformer = Transformer.from_crs(crs_data, CRS.from_epsg(4326), always_xy=True)
    #             lon_range = [transformer.transform(x_range[0], y_range[0])[0], transformer.transform(x_range[1], y_range[1])[0]]
    #             lat_range = [transformer.transform(x_range[0], y_range[0])[1], transformer.transform(x_range[1], y_range[1])[1]]
    #         dx = np.abs(np.mean(np.diff(da.y)))

    #     transformer_3857_to_crs = Transformer.from_crs(
    #         CRS.from_epsg(3857), crs_data, always_xy=True
    #     )

    #     # Determine zoom range
    #     if zoom_range is None:
    #         zoom_max = get_zoom_level_for_resolution(dx * 0.5)
    #         zoom_range = [0, zoom_max]

    # elif data_type == "twm":
    #     # Max zoom obtained from index path
    #     # List folders in index path and turn names into integers
    #     subfolders = list_folders(os.path.join(index_path, "*" ), basename=True)
    #     # Convert strings to integers
    #     ilevels = [int(item) for item in subfolders]
    #     zoom_max = max(ilevels)
    #     zoom_range = [0, zoom_max]
    #     da = None

    if index_path:
        subfolders = list_folders(os.path.join(index_path, "*"), basename=True)
        # Convert strings to integers
        ilevels = [int(item) for item in subfolders]
        zoom_max = max(ilevels)
        zoom_range = [0, zoom_max]
    else:
        zoom_max = 23
        zoom_range = [0, zoom_max]

    # transformer_3857_to_crs = Transformer.from_crs(
    #     CRS.from_epsg(3857), crs_data, always_xy=True
    # )

    twm.zoom_range = [0, zoom_max]
    twm.max_zoom = zoom_range[1]

    # Use highest zoom level
    izoom = zoom_range[1]
    zoom_path = os.path.join(twm.path, str(izoom))

    # Determine elapsed time
    t0 = time.time()

    # Create rectangular mesh with origin (0.0) and 256x256 pixels
    dxy = (40075016.686 / npix) / 2**izoom
    xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
    yy = xx[:]
    xv, yv = np.meshgrid(xx, yy)

    # Determine min and max indices for this zoom level
    # If the index path is given, then get ix0, ix1, iy0 and iy1 from the existing index files
    # Otherwise, use lon_range and lat_range
    # This option is used when we only want to make tiles that cover a model domain
    if index_path:
        index_zoom_path = os.path.join(index_path, str(izoom))
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
        # Use the lon_range and lat_range from the data array or from user input
        ix0, iy0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, iy1 = deg2num(lat_range[0], lon_range[1], izoom)

    # Limit the indices
    ix0 = max(0, ix0)
    iy0 = max(0, iy0)
    ix1 = min(2**izoom - 1, ix1)
    iy1 = min(2**izoom - 1, iy1)

    # Add some stuff to options dict, which is used for parallel processing
    options = {}
    options["index_path"] = index_path
    # options["twm_data"] = twm_data
    options["transformer_4326_to_3857"] = transformer_4326_to_3857
    # options["transformer_3857_to_crs"] = transformer_3857_to_crs
    options["xv"] = xv
    options["yv"] = yv
    options["dxy"] = dxy
    options["interpolation_method"] = interpolation_method
    options["z_range"] = z_range
    options["bathymetry_database"] = bathymetry_database

    # Loop in x direction
    for i in range(ix0, ix1 + 1):
        print(f"Processing column {i - ix0 + 1} of {ix1 - ix0 + 1}")

        zoom_path_i = os.path.join(zoom_path, str(i))

        if not os.path.exists(zoom_path_i):
            makedir(zoom_path_i)

        # Loop in y direction
        if parallel:
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
                            data_list,
                            options,
                        )
                        for j in range(iy0, iy1 + 1)
                    ],
                )
        else:
            # Loop in y direction
            for j in range(iy0, iy1 + 1):
                # Create highest zoom level tile
                create_highest_zoom_level_tile(
                    zoom_path_i, i, j, izoom, twm, data_list, options
                )

        # If zoom_path_i is empty, then remove it again
        if not os.listdir(zoom_path_i):
            os.rmdir(zoom_path_i)

    t1 = time.time()

    print("Elapsed time for zoom level " + str(izoom) + ": " + str(t1 - t0))

    # Done with highest zoom level


def make_topobathy_tiles_lower_levels(
    twm,
    skip_existing=False,
    parallel=True,
):
    npix = 256

    # Now loop through other zoom levels starting with highest minus 1

    for izoom in range(twm.zoom_range[1] - 1, twm.zoom_range[0] - 1, -1):
        print("Processing zoom level " + str(izoom))

        # Determine elapsed time
        t0 = time.time()

        # Rather than interpolating the data onto tiles, we will take average of 4 tiles in higher zoom level

        zoom_path = os.path.join(twm.path, str(izoom))
        zoom_path_higher = os.path.join(twm.path, str(izoom + 1))

        # #
        # if index_path:
        #     index_zoom_path = os.path.join(index_path, str(izoom))
        #     # List folders and turn names into integers
        #     iy0 = 1e15
        #     iy1 = -1e15
        #     ix_list = [int(i) for i in os.listdir(index_zoom_path)]
        #     ix0 = min(ix_list)
        #     ix1 = max(ix_list)
        #     # Now loop through the folders to get the min and max y indices
        #     for i in range(ix0, ix1 + 1):
        #         it_list = [
        #             int(j.split(".")[0])
        #             for j in os.listdir(os.path.join(index_zoom_path, str(i)))
        #         ]
        #         iy0 = min(iy0, min(it_list))
        #         iy1 = max(iy1, max(it_list))
        # else:
        #     ix0, iy0 = deg2num(lat_range[1], lon_range[0], izoom)
        #     iy1, iy1 = deg2num(lat_range[0], lon_range[1], izoom)

        # ix0 = max(0, ix0)
        # iy0 = max(0, iy0)
        # ix1 = min(2**izoom - 1, ix1)
        # iy1 = min(2**izoom - 1, iy1)

        # First determine ix0 and ix1 based on higher zoom level
        # Get the folders in zoom_path_higher and turn them into integers
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

            if parallel:
                # Loop in y direction
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
                # Loop in y direction
                for j in range(iy0, iy1 + 1):
                    # Create lower level tile
                    make_lower_level_tile(
                        zoom_path_i, zoom_path_higher, i, j, npix, twm
                    )

        t1 = time.time()

        print("Elapsed time for zoom level " + str(izoom) + ": " + str(t1 - t0))


def bbox_xy2latlon(x0, x1, y0, y1, crs):
    # Create a transformer
    transformer = Transformer.from_crs(crs, crs.from_epsg(4326), always_xy=True)
    # Transform the four corners
    lon_min, lat_min = transformer.transform(x0, y0)
    lon_max, lat_min = transformer.transform(x1, y0)
    lon_min, lat_max = transformer.transform(x0, y1)
    lon_max, lat_max = transformer.transform(x1, y1)
    return lon_min, lon_max, lat_min, lat_max


def create_highest_zoom_level_tile(
    zoom_path_i, i, j, izoom, twm, da, data_type, options
):
    file_name = os.path.join(zoom_path_i, str(j) + ".png")
    transformer_4326_to_3857 = options["transformer_4326_to_3857"]
    # transformer_3857_to_crs = options["transformer_3857_to_crs"]
    xv = options["xv"]
    yv = options["yv"]
    dxy = options["dxy"]
    z_range = options["z_range"]

    skip_existing = False

    # Create highest zoom level tile
    if os.path.exists(file_name):
        if skip_existing:
            # Tile already exists
            return
        else:
            # Read the tile
            zg0 = png2elevation(
                file_name,
                encoder=twm.encoder,
                encoder_vmin=twm.encoder_vmin,
                encoder_vmax=twm.encoder_vmax,
            )
    else:
        # Tile does not exist
        zg0 = np.zeros((twm.npix, twm.npix))
        zg0[:] = np.nan

    # If there are no NaNs, we can continue
    if not np.any(np.isnan(zg0)):
        return

    if options["index_path"]:
        # Only make tiles for which there is an index file
        index_file_name = os.path.join(
            options["index_path"], str(izoom), str(i), str(j) + ".png"
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

    bathymetry_database = options["bathymetry_database"]
    zg = bathymetry_database.get_bathymetry_on_grid(
        x3857, y3857, CRS.from_epsg(3857), data_list
    )

    # if data_type == "ddb":
    #     pass
    #     # zg = bathymetry_database.get_bathymetry_on_grid(
    #     #     x3857, y3857, CRS.from_epsg(3857), dem_list
    #     # )
    # elif data_type == "twm":
    #     png_file_name = os.path.join(options["twm_data"].path, str(izoom), str(i), str(j) + ".png")
    #     if os.path.exists(png_file_name):
    #         # Easy, the tile exists
    #         zg = png2elevation(png_file_name, encoder=options["twm_data"].encoder)
    #     else:
    #         xl = [x3857[0,0], x3857[0,-1]]
    #         yl = [y3857[-1,0], y3857[0,1]]
    #         max_pixel_size = dxy
    #         xd, yd, zd = options["twm_data"].get_data(xl, yl, max_pixel_size)
    #         zg = interp2(xd, yd, zd, x3857, y3857, method=options["interpolation_method"])

    # elif data_type == "xarray":
    #     # Make grid of x3857 and y3857, and convert to crs of dataset
    #     # xg, yg = np.meshgrid(x3857, y3857)
    #     xg, yg = transformer_3857_to_crs.transform(x3857, y3857)
    #     # Subtract xytrans
    #     xg = xg - options["xytrans"][0]
    #     yg = yg - options["xytrans"][1]
    #     # Get min and max of xg, yg
    #     xg_min = np.min(xg)
    #     xg_max = np.max(xg)
    #     yg_min = np.min(yg)
    #     yg_max = np.max(yg)
    #     # Add buffer to grid
    #     dbuff = 0.05 * max(xg_max - xg_min, yg_max - yg_min)
    #     xg_min = xg_min - dbuff
    #     xg_max = xg_max + dbuff
    #     yg_min = yg_min - dbuff
    #     yg_max = yg_max + dbuff

    #     # Get the indices of the dataset that are within the xg, yg range
    #     i0 = np.where(da.x <= xg_min)[0]
    #     if len(i0) == 0:
    #         # Take first index
    #         i0 = 0
    #     else:
    #         # Take last index
    #         i0 = i0[-1]
    #     i1 = np.where(da.x >= xg_max)[0]
    #     if len(i1) == 0:
    #         i1 = len(da.x) - 1
    #     else:
    #         i1 = i1[0]
    #     if i1 <= i0 + 1:
    #         # No data for this tile
    #         return

    #     xd = da.x[i0:i1]

    #     if da.y[0] < da.y[-1]:
    #         # South to North
    #         j0 = np.where(da.y <= yg_min)[0]
    #         if len(j0) == 0:
    #             j0 = 0
    #         else:
    #             j0 = j0[-1]
    #         j1 = np.where(da.y >= yg_max)[0]
    #         if len(j1) == 0:
    #             j1 = len(da.y) - 1
    #         else:
    #             j1 = j1[0]
    #         if j1 <= j0 + 1:
    #             # No data for this tile
    #             return
    #         # Get the dataset within the range
    #         yd = da.y[j0:j1]
    #         # Get number of dimensions of dataarray
    #         if len(da.shape) == 2:
    #             zd = da[j0:j1, i0:i1].to_numpy()[:]
    #         else:
    #             zd = np.squeeze(da[0, j0:j1, i0:i1].to_numpy()[:])
    #     else:
    #         # North to South
    #         j0 = np.where(da.y <= yg_min)[0]
    #         if len(j0) == 0:
    #             # Use last index
    #             j0 = len(da.y) - 1
    #         else:
    #             # Use first index
    #             j0 = j0[0]
    #         j1 = np.where(da.y >= yg_max)[0]
    #         if len(j1) == 0:
    #             j1 = 0
    #         else:
    #             j1 = j1[-1]
    #         if j0 <= j1 + 1:
    #             # No data for this tile
    #             return
    #         # Get the dataset within the range
    #         yd = np.flip(da.y[j1:j0])
    #         if len(da.shape) == 2:
    #             zd = np.flip(
    #                 da[j1:j0, i0:i1].to_numpy()[:], axis=0
    #             )
    #         else:
    #             zd = np.squeeze(
    #                 np.flip(
    #                     da[0, j1:j0, i0:i1].to_numpy()[:], axis=0
    #                 )
    #             )

    # zg = interp2(x3857, y3857, zd, xg, yg, method=options["interpolation_method"])

    # Any value below zmin is set NaN
    zg[np.where(zg < z_range[0])] = np.nan
    # Any value above zmax is set NaN
    zg[np.where(zg > z_range[1])] = np.nan

    if np.isnan(zg).all():
        # only nans in this tile
        return

    # Overwrite zg with zg0 where not zg0 is not nan
    mask = np.isfinite(zg0)
    zg[mask] = zg0[mask]

    # Write to terrarium png format
    elevation2png(
        zg,
        file_name,
        # compress_level=twm.compress_level,
        encoder=twm.encoder,
        encoder_vmin=twm.encoder_vmin,
        encoder_vmax=twm.encoder_vmax,
    )


def make_lower_level_tile(
    zoom_path_i,
    zoom_path_higher,
    i,
    j,
    npix,
    twm,
):
    # Get the indices of the tiles in the higher zoom level
    i00, j00 = 2 * i, 2 * j  # upper left
    i10, j10 = 2 * i, 2 * j + 1  # lower left
    i01, j01 = 2 * i + 1, 2 * j  # upper right
    i11, j11 = 2 * i + 1, 2 * j + 1  # lower right

    # Create empty array of NaN to store the elevation data from the higher zoom level
    zg512 = np.zeros((npix * 2, npix * 2))
    zg512[:] = np.nan

    # Create empty array of NaN of 4*npix*npix to store the 2-strid elevation data from higher zoom level
    zg4 = np.zeros((4, npix, npix))
    zg4[:] = np.nan

    okay = False

    # Get the file names of the tiles in the higher zoom level
    # Upper left
    file_name = os.path.join(zoom_path_higher, str(i00), str(j00) + ".png")
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
    file_name = os.path.join(zoom_path_higher, str(i10), str(j10) + ".png")
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
    file_name = os.path.join(zoom_path_higher, str(i01), str(j01) + ".png")
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
    file_name = os.path.join(zoom_path_higher, str(i11), str(j11) + ".png")
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
        # No tiles in higher zoom level, so continue
        return

    # Compute average of 4 tiles in higher zoom level
    # Data from zg512 with stride 2
    zg4[0, :, :] = zg512[0 : npix * 2 : 2, 0 : npix * 2 : 2]
    zg4[1, :, :] = zg512[1 : npix * 2 : 2, 0 : npix * 2 : 2]
    zg4[2, :, :] = zg512[0 : npix * 2 : 2, 1 : npix * 2 : 2]
    zg4[3, :, :] = zg512[1 : npix * 2 : 2, 1 : npix * 2 : 2]

    # Compute average of 4 tiles
    zg = np.nanmean(zg4, axis=0)

    # Write to terrarium png format
    file_name = os.path.join(zoom_path_i, str(j) + ".png")
    elevation2png(
        zg,
        file_name,
        encoder=twm.encoder,
        encoder_vmin=twm.encoder_vmin,
        encoder_vmax=twm.encoder_vmax,
    )


# Function to read the TFW file and return the transformation
def read_tfw(tfw_path):
    with open(tfw_path, "r") as f:
        lines = f.readlines()

    # Extract the values from the TFW
    cell_size_x = float(lines[0].strip())  # Pixel size in the X direction
    rotation_x = float(lines[1].strip())  # Rotation in the X direction (usually 0)
    rotation_y = float(lines[2].strip())  # Rotation in the Y direction (usually 0)
    cell_size_y = float(
        lines[3].strip()
    )  # Pixel size in the Y direction (usually negative)
    upper_left_x = float(lines[4].strip())  # X coordinate of the upper-left corner
    upper_left_y = float(lines[5].strip())  # Y coordinate of the upper-left corner

    # Return as an Affine transformation (for use with Rasterio)
    return from_origin(upper_left_x, upper_left_y, cell_size_x, abs(cell_size_y))


def list_folders(src, basename=False):
    folder_list = []
    full_list = glob.glob(src)
    for item in full_list:
        if os.path.isdir(item):
            if basename:
                folder_list.append(os.path.basename(item))
            else:
                folder_list.append(item)

    return sorted(folder_list)
