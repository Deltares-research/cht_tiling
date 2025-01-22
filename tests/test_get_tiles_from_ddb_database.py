import os

from cht_tiling import TiledWebMap

# import xarray as xr
# import numpy as np
# from scipy.interpolate import RegularGridInterpolator
# from cht_sfincs import SFINCS
# from cht_tiling import make_index_tiles, make_topobathy_tiles, make_topobathy_overlay
from cht_tiling.utils import get_zoom_level_for_resolution


def test_get_tiles_from_ddb_database():
    s3_bucket = ("deltares-ddb",)
    s3_key = ("data/bathymetry",)
    s3_region = "eu-west-1"

    dbpath = r"c:\work\projects\delftdashboard\delftdashboard_python\data\bathymetry"
    original_db_path = r"c:\work\delftdashboard\data\bathymetry"

    names = ["cudem_ninth_prusvi"]
    long_names = ["CUDEM (9th degree) Puerto Rico and the US Virgin Islands"]
    sources = ["NOAA NCEI"]
    original_db_dataset_names = ["ncei_ninth_prusvi"]
    vertical_reference_levels = ["unknown"]

    for index, name in enumerate(names):
        long_name = long_names[index]
        original_db_dataset_name = original_db_dataset_names[index]
        source = sources[index]
        vertical_reference_level = vertical_reference_levels[index]

        path = os.path.join(dbpath, name)

        twm = TiledWebMap(path, name, parameter="elevation")

        zoom_max = get_zoom_level_for_resolution(10.0)
        zoom_range = [0, zoom_max]

        twm.generate_topobathy_tiles(
            dem_names=[original_db_dataset_name],
            bathymetry_database_path=original_db_path,
            zoom_range=zoom_range,
            quiet=False,
            make_webviewer=True,
            write_metadata=True,
            skip_existing=True,
            interpolation_method="linear",
            encoder="terrarium",
            name=name,
            long_name=long_name,
            url=None,
            source=source,
            vertical_reference_level=vertical_reference_level,
            vertical_units="m",
            difference_with_msl=0.0,
            s3_bucket=s3_bucket,
            s3_key=f"{s3_key}/{name}",
            s3_region=s3_region,
        )

    # TODO asserts
