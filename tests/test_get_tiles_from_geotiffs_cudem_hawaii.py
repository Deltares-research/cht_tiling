import os

import cht_utils.fileops as fo

# import xarray as xr
# import numpy as np
# from scipy.interpolate import RegularGridInterpolator
import numpy as np
import rasterio
import xarray as xr

from cht_tiling import TiledWebMap

# def make_topo_tiles(name,
#                     long_name,
#                     source,
#                     dbpath,
#                     ncfile,
#                     vertical_reference_level="unknown",
#                     dataarray_name="elev",
#                     encoder="terrarium",
#                     s3_bucket="deltares-ddb",
#                     s3_key="data/bathymetry",
#                     s3_region="eu-west-1",
#                     available_tiles=True):

#     path = os.path.join(dbpath, name)


#     twm = TiledWebMap(path, name, parameter="elevation")
#     ds = xr.open_dataset(ncfile)
#     twm.generate_topobathy_tiles(dataset=ds,
#                                 dataarray_name=dataarray_name,
#                                 quiet=False,
#                                 make_webviewer=True,
#                                 write_metadata=True,
#                                 skip_existing=True,
#                                 interpolation_method="linear",
#                                 encoder=encoder,
#                                 name=name,
#                                 long_name=long_name,
#                                 source=source,
#                                 vertical_reference_level=vertical_reference_level,
#                                 vertical_units="m",
#                                 difference_with_msl=0.0,
#                                 s3_bucket=s3_bucket,
#                                 s3_key=f"{s3_key}/{name}",
#                                 s3_region=s3_region,
#                                 available_tiles=available_tiles)
#     )
#     ds.close()
def test_get_tiles_from_geotiffs_cudem_hawaii():
    dbpath = r"c:\work\projects\delftdashboard\delftdashboard_python\data\bathymetry"
    s3_bucket = "deltares-ddb"
    s3_key = "data/bathymetry"
    s3_region = "eu-west-1"

    name = "cudem_ninth_hawaii"
    long_name = "CUDEM (9th degree) Hawaii"
    source = "NOAA NCEI"
    vertical_reference_level = "NAVD88"
    encoder = "terrarium"
    dxmax = 10.0

    # Create TiledWebMap object
    path = os.path.join(dbpath, name)
    twm = TiledWebMap(path, name, parameter="elevation")

    # Loop through geotiffs
    datapath = r"c:\work\projects\delftdashboard\bathy_data\cudem_hawaii"
    flist = fo.list_files(os.path.join(datapath, "*.tif"))
    for f in flist:
        print(f)
        with rasterio.open(f) as src:
            print(src.crs)
            # print(src.bounds)
            # print(src.meta)
            # print(src.profile)
            # print(src.read(1))
            # show(src)
            # plt.show()

            # Turn data into xarray dataset
            x = np.linspace(src.bounds.left, src.bounds.right, src.width)
            y = np.linspace(src.bounds.top, src.bounds.bottom, src.height)
            z = src.read(1)
            # flip y axis
            y = np.flip(y)
            z = np.flip(z, axis=0)

        # Create xarray dataset
        ds = xr.Dataset({"elevation": (["y", "x"], z)}, coords={"x": x, "y": y})
        ds["crs"] = src.crs
        ds.crs.attrs["epsg_code"] = src.crs.to_epsg()
        # ds = xr.open_dataset(ncfile)
        twm.generate_topobathy_tiles(
            dataset=ds,
            dataarray_name="elevation",
            dataarray_x_name="x",
            dataarray_y_name="y",
            dx_max_zoom=dxmax,
            quiet=False,
            make_webviewer=True,
            write_metadata=True,
            make_availability_file=True,
            skip_existing=False,
            interpolation_method="linear",
            encoder=encoder,
            name=name,
            long_name=long_name,
            source=source,
            vertical_reference_level=vertical_reference_level,
            vertical_units="m",
            difference_with_msl=0.0,
            s3_bucket=s3_bucket,
            s3_key=f"{s3_key}/{name}",
            # make_availability_file=True,
            s3_region=s3_region,
        )
        # twm.make_availability_file()

        ds.close()

    # TODO asserts


# ncfile = os.path.join(datapath, "usgs_dem_10m_guam.nc")
# dataarray_name = "elev"
# make_topo_tiles(name, long_name, source, dbpath, ncfile,
#                 dataarray_name=dataarray_name,
#                 vertical_reference_level="unknown",
#                 encoder="terrarium",
#                 s3_bucket="deltares-ddb",
#                 s3_key="data/bathymetry",
#                 s3_region="eu-west-1",
#                 available_tiles=True,
#                 upload=False)


# name = "gebco_2024"
# long_name = "GEBCO 2024"
# source = "BODC"
# vertical_reference_level = "unknown"
# ncfile = os.path.join(r"c:\work\data\gebco_2024", "gebco_2024.nc")
# dataarray_name = "elev"
# encoder = "terrarium16"
# make_topo_tiles(name, long_name, source, dbpath, ncfile,
#                 dataarray_name=dataarray_name,
#                 vertical_reference_level="unknown",
#                 encoder="terrarium16",
#                 s3_bucket="deltares-ddb",
#                 s3_key="data/bathymetry",
#                 s3_region="eu-west-1",
#                 available_tiles=False,
#                 upload=False)
