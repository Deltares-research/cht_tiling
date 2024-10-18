import xarray as xr

from cht_tiling import TiledWebMap
from cht_tiling.utils import get_zoom_level_for_resolution

# izoom = get_zoom_level_for_resolution(400.0)
izoom = 9

topo_path = "c:\\work\\delftdashboard\\data\\bathymetry\\gebco2024"

ds = xr.open_dataset("c:\\work\\data\\gebco_2024\\gebco_2024.nc")

twm = TiledWebMap(topo_path, parameter="elevation")

# twm.generate_topobathy_tiles(dataset=ds,
#                              lon_range=[-180.0, 180.0],
#                              lat_range=[-90.0, 90.0],
#                              zoom_range=[0, izoom],
#                              bathymetry_database_path="c:\\work\\delftdashboard\\data\\bathymetry",
#                              quiet=False,
#                              make_webviewer=True,
#                              make_highest_level=False,
#                              encoder="terrarium16",
#                              interpolation_method="nearest",
#                              name="gebco2024_int16",
#                              long_name="GEBCO 2024 (int16)",
#                              url="https://www.gebco.net/data_and_products/gridded_bathymetry_data/"
#                             )

name = "gebco2024"
bucket_name = "deltares-ddb"
s3_folder = "data/bathymetry/" + name

access_key      = "AKIAQFTTKHPJJ34AN2EH"
secret_key      = "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"
region          = "eu-west-1"

twm.upload(name, bucket_name, s3_folder, access_key, secret_key, region, quiet=False, parallel=True)
