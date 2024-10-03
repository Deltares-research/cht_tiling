import os
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cht_sfincs import SFINCS
from cht_tiling import make_index_tiles, make_topobathy_tiles_v2, make_topobathy_overlay
from cht_tiling.utils import get_zoom_level_for_resolution

izoom = get_zoom_level_for_resolution(400.0)
# izoom = 2
# compress_level = 6
izoom = 5

topo_path = "c:\\work\\delftdashboard\\data\\bathymetry\\gebco2024"

topo_path = "c:\\work\\delftdashboard\\data\\bathymetry\\gebco2024_int16"

dem_file = "c:\\work\\data\\gebco_2024\\gebco_2024.nc"

ds = xr.open_dataset(dem_file)

# Make topobathy tiles (should send in a HydroMT function)
make_topobathy_tiles_v2(
    topo_path,
#    dem_names=["gebco19"],
    dataset=ds,
    lon_range=[-180.0, 180.0],
    lat_range=[-90.0, 90.0],
    # lon_range=[-60.0, -60.0],
    # lat_range=[ 30.0,  40.0],
    zoom_range=[0, izoom],
    # z_range=None,
    bathymetry_database_path="c:\\work\\delftdashboard\\data\\bathymetry",
    quiet=False,
    make_webviewer=True,
    make_highest_level=True,
    encoder="terrarium16",
    # decoder="float32",
    # encoder_vmin=-15000.0,
    # encoder_vmax=15000.0,
    # interpolation_method="linear",
    interpolation_method="nearest",
    name="gebco2024_int16",
    long_name="GEBCO 2024 (int16)",
    url="https://www.gebco.net/data_and_products/gridded_bathymetry_data/"
)


# make_topobathy_overlay(
#     topo_path,
#     [-120.0, -100.0],
#     [30.0, 50.0],
#     npixels=800,
#     # color_values=None,
#     color_map="jet",
#     color_range=[-10.0, 10.0],
#     color_scale_auto=False,
#     # color_scale_symmetric=True,
#     color_scale_symmetric_side="min",
#     hillshading=True,
#     hillshading_azimuth=315,
#     hillshading_altitude=30,
#     hillshading_exaggeration=10.0,
#     quiet=False,
#     file_name="overlay.png"
# )
