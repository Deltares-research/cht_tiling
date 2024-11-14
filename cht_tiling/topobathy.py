import os
import numpy as np
from pyproj import CRS, Transformer
import time
import toml
from multiprocessing.pool import ThreadPool

from cht_utils.misc_tools import interp2

from .webviewer import write_html
from .utils import deg2num
from .utils import num2deg
from .utils import makedir
from .utils import elevation2png
from .utils import png2elevation
from .utils import get_zoom_level_for_resolution

def make_lower_level_tile(zoom_path_i,
                          zoom_path_higher,
                          i,
                          j,
                          npix,
                          encoder,
                          encoder_vmin,
                          encoder_vmax,
                          compress_level):

    # Get the indices of the tiles in the higher zoom level
    i00, j00 = 2 * i, 2 * j         # upper left
    i10, j10 = 2 * i, 2 * j + 1     # lower left
    i01, j01 = 2 * i + 1, 2 * j     # upper right
    i11, j11 = 2 * i + 1, 2 * j + 1 # lower right

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
        zgh = png2elevation(file_name,
                            encoder=encoder,
                            encoder_vmin=encoder_vmin,
                            encoder_vmax=encoder_vmax)
        zg512[0:npix, 0:npix] = zgh
        okay = True
    # Lower left    
    file_name = os.path.join(zoom_path_higher, str(i10), str(j10) + ".png")
    if os.path.exists(file_name):
        zgh = png2elevation(file_name,
                            encoder=encoder,
                            encoder_vmin=encoder_vmin,
                            encoder_vmax=encoder_vmax)
        zg512[npix:, 0:npix] = zgh
        okay = True
    # Upper right    
    file_name = os.path.join(zoom_path_higher, str(i01), str(j01) + ".png")
    if os.path.exists(file_name):
        zgh = png2elevation(file_name,
                            encoder=encoder,
                            encoder_vmin=encoder_vmin,
                            encoder_vmax=encoder_vmax)
        zg512[0:npix, npix:] = zgh
        okay = True
    # Lower right    
    file_name = os.path.join(zoom_path_higher, str(i11), str(j11) + ".png")
    if os.path.exists(file_name):
        zgh = png2elevation(file_name,
                            encoder=encoder,
                            encoder_vmin=encoder_vmin,
                            encoder_vmax=encoder_vmax)
        zg512[npix:, npix:] = zgh
        okay = True

    if not okay:
        # No tiles in higher zoom level, so continue
        return

    # Compute average of 4 tiles in higher zoom level
    # Data from zg512 with stride 2
    zg4[0,:,:] = zg512[0:npix * 2:2, 0:npix * 2:2]
    zg4[1,:,:] = zg512[1:npix * 2:2, 0:npix * 2:2]
    zg4[2,:,:] = zg512[0:npix * 2:2, 1:npix * 2:2]
    zg4[3,:,:] = zg512[1:npix * 2:2, 1:npix * 2:2]

    # Compute average of 4 tiles
    zg = np.nanmean(zg4, axis=0)

    # Write to terrarium png format
    file_name = os.path.join(zoom_path_i, str(j) + ".png")
    elevation2png(zg, file_name,
                    encoder=encoder,
                    encoder_vmin=encoder_vmin,
                    encoder_vmax=encoder_vmax,                                 
                    compress_level=compress_level)


def make_topobathy_tiles(
    path,
    dem_names=None,
    dataset=None, # Must be XArray
    dataarray_name="elevation",
    dataarray_x_name="lon",
    dataarray_y_name="lat",
    lon_range=None,
    lat_range=None,
    index_path=None,
    zoom_range=None,
    dx_max_zoom=None,
    z_range=None,
    bathymetry_database_path="d:\\delftdashboard\\data\\bathymetry",
    quiet=False,
    make_webviewer=True,
    write_metadata=True,    
    metadata=None,
    make_lower_levels=True,
    make_highest_level=True,
    skip_existing=False,
    interpolation_method="linear",
    encoder="terrarium",
    encoder_vmin=None,
    encoder_vmax=None,
    compress_level=6,
    name="unknown",
    long_name="unknown",
    url=None,
    s3_bucket=None,
    s3_key=None,
    s3_region=None,
    source="unknown",
    vertical_reference_level="MSL",
    vertical_units="m",
    difference_with_msl=0.0,
):
    """
    Generates topo/bathy tiles

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

    if dem_names is not None:

        from cht_bathymetry.database import BathymetryDatabase

        dem_type = "ddb"

        # from cht_utils.misc_tools import interp2

        bathymetry_database = BathymetryDatabase(None)
        bathymetry_database.initialize(bathymetry_database_path)

        dem_list = []
        for dem_name in dem_names:
            dem = {}
            dem["name"] = dem_name
            dem["zmin"] = -10000.0
            dem["zmax"] = 10000.0
            dem_list.append(dem)

        if lon_range is None:
            # Try to get lon_range and lat_range from the first dataset
            dataset = bathymetry_database.get_dataset(dem_names[0])
            lon_range, lat_range = dataset.get_bbox(crs=CRS.from_epsg(4326))

    elif dataset is not None:
        dem_type = "xarray"
        ds_lon = dataset[dataarray_x_name].values    
        ds_lat = dataset[dataarray_y_name].values
        ds_z_parameter = dataarray_name
        if "crs" not in dataset:
            dataset_crs_code = 4326
        else:
            dataset_crs_code = dataset.crs.attrs["epsg_code"]
        transformer_3857_to_crs = Transformer.from_crs(CRS.from_epsg(3857), dataset_crs_code, always_xy=True)
        if lon_range is None:
            lon_range = [np.min(ds_lon), np.max(ds_lon)]
        if lat_range is None:
            lat_range = [np.min(ds_lat), np.max(ds_lat)]
        dx = np.mean(np.diff(ds_lat)) * 111000.0    


    if zoom_range is None:
        if dx_max_zoom is None:
            dx_max_zoom = dx * 0.5
        zoom_max = get_zoom_level_for_resolution(dx_max_zoom)
        zoom_range = [0, zoom_max]

    if not z_range:
        z_range = [-20000.0, 20000.0]

    npix = 256

    if make_highest_level:

        transformer_4326_to_3857 = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
        )

        # First do highest zoom level
        izoom = zoom_range[1]

        # Determine elapsed time
        t0 = time.time()

        if not quiet:
            print("Processing zoom level " + str(izoom))

        zoom_path = os.path.join(path, str(izoom))

        dxy = (40075016.686 / npix) / 2**izoom
        xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
        yy = xx[:]
        xv, yv = np.meshgrid(xx, yy)

        ix0, iy0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, iy1 = deg2num(lat_range[0], lon_range[1], izoom)

        ix0 = max(0, ix0)
        iy0 = max(0, iy0)
        ix1 = min(2**izoom - 1, ix1)
        iy1 = min(2**izoom - 1, iy1)

        # Loop in x direction
        for i in range(ix0, ix1 + 1):

            print(f"Processing column {i - ix0 + 1} of {ix1 - ix0 + 1}")

            path_okay = False
            zoom_path_i = os.path.join(zoom_path, str(i))

            if not os.path.exists(zoom_path_i):
                makedir(zoom_path_i)
                path_okay = True

            # Loop in y direction
            for j in range(iy0, iy1 + 1):

                # Create highest zoom level tile
                
                file_name = os.path.join(zoom_path_i, str(j) + ".png")
                if os.path.exists(file_name):
                    if skip_existing:
                        # Tile already exists
                        continue
                    else:
                        # Read the tile
                        zg0 = png2elevation(file_name,
                                            encoder=encoder,
                                            encoder_vmin=encoder_vmin,
                                            encoder_vmax=encoder_vmax)
                        pass
                else:
                    # Tile does not exist
                    zg0 = np.zeros((npix, npix))
                    zg0[:] = np.nan    

                if index_path:
                    # Only make tiles for which there is an index file
                    index_file_name = os.path.join(
                        index_path, str(izoom), str(i), str(j) + ".png"
                    )
                    if not os.path.exists(index_file_name):
                        continue

                # Compute lat/lon at upper left corner of tile
                lat, lon = num2deg(i, j, izoom)

                # Convert origin to Global Mercator
                xo, yo = transformer_4326_to_3857.transform(lon, lat)

                # Tile grid on Global mercator
                x3857 = xo + xv[:] + 0.5 * dxy
                y3857 = yo - yv[:] - 0.5 * dxy

                if dem_type == "ddb":
                    zg = bathymetry_database.get_bathymetry_on_grid(
                        x3857, y3857, CRS.from_epsg(3857), dem_list
                    )
                elif dem_type == "xarray":
                    # Make grid of x3857 and y3857, and convert to crs of dataset
                    # xg, yg = np.meshgrid(x3857, y3857)
                    xg, yg = transformer_3857_to_crs.transform(x3857, y3857)
                    # Get min and max of xg, yg
                    xg_min = np.min(xg)
                    xg_max = np.max(xg)
                    yg_min = np.min(yg)
                    yg_max = np.max(yg)
                    # Add buffer to grid
                    dbuff = 0.05 * max(xg_max - xg_min, yg_max - yg_min)
                    xg_min = xg_min - dbuff
                    xg_max = xg_max + dbuff
                    yg_min = yg_min - dbuff
                    yg_max = yg_max + dbuff

                    # Get the indices of the dataset that are within the xg, yg range
                    i0 = np.where(ds_lon <= xg_min)[0]
                    if len(i0) == 0:
                        # Take first index
                        i0 = 0
                    else:
                        # Take last index
                        i0 = i0[-1]
                    i1 = np.where(ds_lon >= xg_max)[0]
                    if len(i1) == 0:
                        i1 = len(ds_lon) - 1
                    else:
                        i1 = i1[0]
                    if i1 <= i0:
                        # No data for this tile
                        continue
                    j0 = np.where(ds_lat <= yg_min)[0]
                    if len(j0) == 0:
                        j0 = 0
                    else:
                        j0 = j0[-1]
                    j1 = np.where(ds_lat >= yg_max)[0]
                    if len(j1) == 0:
                        j1 = len(ds_lat) - 1
                    else:
                        j1 = j1[0]
                    if j1 <= j0:
                        # No data for this tile
                        continue
                    # i0 = np.where(ds_lon >= xg_min)[0][0]
                    # i1 = np.where(ds_lon <= xg_max)[0][-1]
                    # j0 = np.where(ds_lat >= yg_min)[0][0]
                    # j1 = np.where(ds_lat <= yg_max)[0][-1]
                    # Get the dataset within the range
                    xd = ds_lon[i0:i1]
                    yd = ds_lat[j0:j1]
                    zd = dataset[ds_z_parameter][j0:j1, i0:i1].values[:]
                    zg = interp2(xd, yd, zd, xg, yg, method=interpolation_method)

                if np.isnan(zg).all():
                    # only nans in this tile
                    continue

                if np.nanmax(zg) < z_range[0] or np.nanmin(zg) > z_range[1]:
                    # all values in tile outside z_range
                    continue

                # Overwrite zg with zg0 where not nan
                mask = np.isnan(zg)
                zg[mask] = zg0[mask]

                # Write to terrarium png format
                elevation2png(zg, file_name,
                              compress_level=compress_level,
                              encoder=encoder,
                              encoder_vmin=encoder_vmin,
                              encoder_vmax=encoder_vmax)

        t1 = time.time()

        if not quiet:
            print("Elapsed time for zoom level " + str(izoom) + ": " + str(t1 - t0))

    # Done with highest zoom level

    if make_lower_levels:

        # Now loop through other zoom levels starting with highest minus 1

        for izoom in range(zoom_range[1] - 1, zoom_range[0] - 1, -1):

            if not quiet:
                print("Processing zoom level " + str(izoom))

            # Determine elapsed time
            t0 = time.time()

            # Rather than interpolating the data onto tiles, we will take average of 4 tiles in higher zoom level

            zoom_path = os.path.join(path, str(izoom))
            zoom_path_higher = os.path.join(path, str(izoom + 1))

            ix0, iy0 = deg2num(lat_range[1], lon_range[0], izoom)
            ix1, iy1 = deg2num(lat_range[0], lon_range[1], izoom)
            ix0 = max(0, ix0)
            iy0 = max(0, iy0)
            ix1 = min(2**izoom - 1, ix1)
            iy1 = min(2**izoom - 1, iy1)

            # Loop in x direction
            for i in range(ix0, ix1 + 1):

                path_okay = False
                zoom_path_i = os.path.join(zoom_path, str(i))

                if not path_okay:
                    if not os.path.exists(zoom_path_i):
                        makedir(zoom_path_i)
                        path_okay = True

                # Loop in y direction
                with ThreadPool() as pool:
                    pool.starmap(make_lower_level_tile, [(zoom_path_i,
                          zoom_path_higher,
                          i,
                          j,
                          npix,
                          encoder,
                          encoder_vmin,
                          encoder_vmax,
                          compress_level) for j in range(iy0, iy1 + 1)])

            t1 = time.time()

            if not quiet:
                print("Elapsed time for zoom level " + str(izoom) + ": " + str(t1 - t0))

    if make_webviewer:
        # Make webviewer
        write_html(os.path.join(path, "index.html"), max_native_zoom=zoom_range[1])

    if write_metadata:

        if metadata is None:        

            metadata = {}

            metadata["longname"] = long_name
            metadata["format"] = "tiled_web_map"
            metadata["encoder"] = encoder
            if encoder_vmin is not None:
                metadata["encoder_vmin"] = encoder_vmin
            if encoder_vmax is not None:
                metadata["encoder_vmax"] = encoder_vmax
            metadata["url"] = url
            if s3_bucket is not None:
                metadata["s3_bucket"] = s3_bucket
            if s3_key is not None:
                metadata["s3_key"] = s3_key
            if s3_region is not None:
                metadata["s3_region"] = s3_region
            metadata["max_zoom"] = zoom_range[1]
            metadata["interpolation_method"] = interpolation_method
            metadata["source"] = source
            metadata["coord_ref_sys_name"] = "WGS 84 / Pseudo-Mercator"
            metadata["coord_ref_sys_kind"] = "projected"
            metadata["vertical_reference_level"] = vertical_reference_level
            metadata["vertical_units"] = vertical_units
            metadata["difference_with_msl"] = difference_with_msl
            metadata["available_tiles"] = True

            metadata["description"] = {}
            metadata["description"]["title"] = "LONG_NAME"
            metadata["description"]["institution"] = "INST_NAME"
            metadata["description"]["history"] = "created by : AUTHOR"
            metadata["description"]["references"] = "No reference material available"
            metadata["description"]["comment"] = "none"
            metadata["description"]["email"] = "Your email here"
            metadata["description"]["version"] = "1.0"
            metadata["description"]["terms_for_use"] = "Use as you like"
            metadata["description"]["disclaimer"] = "These data are made available in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."

        # Write to toml file
        # toml_file = os.path.join(path, name + ".tml")
        toml_file = os.path.join(path, "metadata.tml")
        with open(toml_file, "w") as f:
            toml.dump(metadata, f)
