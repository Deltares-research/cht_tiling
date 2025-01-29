import os

# import xarray as xr
# import numpy as np
# from scipy.interpolate import RegularGridInterpolator
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
#                     make_availability_file=True):

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
#                                 make_availability_file=make_availability_file)
#     ds.close()


# name = "usgs_dem_10m_guam"
# long_name = "USGS 10-m Digital Elevation Model (DEM): Guam"
# source = "USGS"
# vertical_reference_level = "unknown"
# datapath = r"c:\work\projects\delftdashboard\bathy_data\guam"
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


def test_get_tiles_from_netcdf():
    dbpath = r"c:\work\projects\delftdashboard\delftdashboard_python\data\bathymetry"
    name = "gebco_2024_tst"
    long_name = "GEBCO 2024"
    source = "BODC"
    vertical_reference_level = "unknown"
    ncfile = os.path.join(r"c:\work\data\gebco_2024", "gebco_2024.nc")
    dataarray_name = "elevation"
    encoder = "terrarium16"
    make_availability_file = False

    s3_bucket = ("deltares-ddb",)
    s3_key = ("data/bathymetry",)
    s3_region = ("eu-west-1",)

    path = os.path.join(dbpath, name)

    twm = TiledWebMap(path, name, parameter="elevation")
    with xr.open_dataset(ncfile) as ds:
        twm.generate_topobathy_tiles(
            dataset=ds,
            dataarray_name=dataarray_name,
            quiet=False,
            make_webviewer=True,
            write_metadata=True,
            skip_existing=True,
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
            s3_region=s3_region,
            make_availability_file=make_availability_file,
        )

    # TODO asserts


# make_topo_tiles(name, long_name, source, dbpath, ncfile,
#                 dataarray_name=dataarray_name,
#                 vertical_reference_level="unknown",
#                 encoder="terrarium16",
#                 s3_bucket="deltares-ddb",
#                 s3_key="data/bathymetry",
#                 s3_region="eu-west-1",
#                 make_availability_file=False)
