# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:51:04 2021

@author: ormondt
"""

import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import boto3
import numpy as np
import toml
import xarray as xr
from botocore import UNSIGNED
from botocore.client import Config

from cht_tiling.index_tiles import make_index_tiles
from cht_tiling.data_tiles import make_data_tiles
from cht_tiling.rgba_tiles import make_rgba_tiles
from cht_tiling.topobathy_tiles import (
    make_topobathy_tiles_lower_levels,
    make_topobathy_tiles_top_level,
)
from cht_tiling.utils import (
    get_zoom_level_for_resolution,
    list_files,
    list_folders,
    num2xy,
    png2elevation,
    xy2num,
)
from cht_tiling.webviewer import write_html


class ZoomLevel:
    def __init__(self):
        self.ntiles = 0
        self.ij_available = None


class TiledWebMap:
    def __init__(self,
                 path: str | Path,                 
                 data=None,
                 type="rgba",
                 parameter="other",
                 encoder="terrarium",
                 encoder_vmin=None,
                 encoder_vmax=None,
                 name="unknown",
                 long_name="Unknown dataset",
                 url=None,
                 npix=256,
                 max_zoom=0,
                 s3_client=None,
                 s3_bucket=None,
                 s3_key=None,
                 s3_region=None,
                 source="unknown",
                 vertical_reference_level="MSL",
                 vertical_units="m",
                 difference_with_msl=0.0,
                 index_path=None,
                 topo_path=None,
                 zoom_range=None,
                 color_values=None,
                 caxis=None,
                 zbmax=0.0,
                 minimum_depth=0.05,
                 bathymetry_database=None,
                 lon_range=None,
                 lat_range=None,
                 z_range=[-999999.0, 999999.0],
                 dx_max_zoom=None,
                 write_metadata=False,
                 write_availability=True,
                 make_lower_levels=True,
                 make_highest_level=True,
                 skip_existing=False,
                 make_webviewer=True,
                 merge=True,
                 parallel=True,
                 interpolation_method="linear",
                 quiet=False,
                 ):

        """Tiled Web Map class"""

        # Set all keyword/value pairs as attributes
        for key, value in locals().items():
            if key in ("self", "path"):
                continue
            setattr(self, key, value)

        self.path = str(path)

        # # Parameter may be one of the following: elevation, floodmap, topography, index, data, rgba
        # if self.parameter not in ["elevation", "floodmap", "topography", "index", "data", "rgba"]:
        #     raise ValueError(
        #         "Parameter must be one of the following: elevation, floodmap, topography, index, data, rgba"
        #     )

        self.read_metadata()
        # Check if available_tiles.nc exists. If not, just read the folders to get the zoom range.
        nc_file = os.path.join(self.path, "available_tiles.nc")
        self.availability_loaded = False
        if os.path.exists(nc_file):
            self.availability_exists = True
        else:
            self.availability_exists = False
        if (
            self.s3_bucket is not None
            and self.s3_key is not None
            and self.s3_region is not None
        ):
            self.download = True
        else:
            self.download = False


    def get_data(self, xl, yl, max_pixel_size, crs=None, waitbox=None):
        # xl and yl are in CRS 3857
        # max_pixel_size is in meters
        # returns x, y, and z in CRS 3857
        # Check if availability matrix exists but has not been loaded
        if self.availability_exists and not self.availability_loaded:
            self.read_availability()
            self.availability_loaded = True

        # Determine zoom level
        izoom = get_zoom_level_for_resolution(max_pixel_size)
        izoom = min(izoom, self.max_zoom)

        # Determine the indices of required tiles
        ix0, iy0 = xy2num(xl[0], yl[1], izoom)
        ix1, iy1 = xy2num(xl[1], yl[0], izoom)

        # Make sure indices are within bounds
        ix0 = max(0, ix0)
        iy0 = max(0, iy0)
        # ix1 = min(2**izoom - 1, ix1)
        iy1 = min(2**izoom - 1, iy1)

        # Create empty array
        nx = (ix1 - ix0 + 1) * 256
        ny = (iy1 - iy0 + 1) * 256
        z = np.empty((ny, nx))
        z[:] = np.nan

        # First try to download missing tiles (it's faster if we can do this in parallel)
        download_file_list = []
        download_key_list = []

        if self.download:
            for i in range(ix0, ix1 + 1):
                itile = np.mod(i, 2**izoom)  # wrap around
                ifolder = str(itile)
                for j in range(iy0, iy1 + 1):
                    png_file = os.path.join(
                        self.path, str(izoom), ifolder, str(j) + ".png"
                    )
                    if not os.path.exists(png_file):
                        # File does not yet exist
                        if self.availability_exists:
                            # Check availability of the tile in matrix.
                            if not self.check_availability(i, j, izoom):
                                # Tile is also not available for download
                                continue
                        # Add to download_list
                        download_file_list.append(png_file)
                        download_key_list.append(
                            f"{self.s3_key}/{str(izoom)}/{ifolder}/{str(j)}.png"
                        )
                        # Make sure the folder exists
                        if not Path(png_file).parent.exists():
                            Path(png_file).parent.mkdir(parents=True, exist_ok=True)

            # Now download the missing tiles
            if len(download_file_list) > 0:
                if waitbox is not None:
                    wb = waitbox("Downloading topography tiles ...")
                # make boto s
                if self.s3_client is None:
                    self.s3_client = boto3.client(
                        "s3", config=Config(signature_version=UNSIGNED)
                    )
                # Download missing tiles
                with ThreadPool() as pool:
                    pool.starmap(
                        self.download_tile_parallel,
                        [
                            (self.s3_bucket, key, file)
                            for key, file in zip(download_key_list, download_file_list)
                        ],
                    )
                if waitbox is not None:
                    wb.close()

        # Loop over required tiles
        for i in range(ix0, ix1 + 1):
            itile = np.mod(i, 2**izoom)  # wrap around
            ifolder = str(itile)
            for j in range(iy0, iy1 + 1):
                png_file = os.path.join(self.path, str(izoom), ifolder, str(j) + ".png")

                if not os.path.exists(png_file):
                    continue

                # Read the png file
                valt = png2elevation(
                    png_file,
                    encoder=self.encoder,
                    encoder_vmin=self.encoder_vmin,
                    encoder_vmax=self.encoder_vmax,
                )

                # Fill array
                ii0 = (i - ix0) * 256
                ii1 = ii0 + 256
                jj0 = (j - iy0) * 256
                jj1 = jj0 + 256
                z[jj0:jj1, ii0:ii1] = valt

        # Compute x and y coordinates
        x0, y0 = num2xy(ix0, iy1 + 1, izoom)  # lower left
        x1, y1 = num2xy(ix1 + 1, iy0, izoom)  # upper right
        # Data is stored in centres of pixels so we need to shift the coordinates
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        x = np.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, nx)
        y = np.linspace(y0 + 0.5 * dy, y1 - 0.5 * dy, ny)
        z = np.flipud(z)

        return x, y, z

    def make(self):
        """"Make tiles."""

        if self.zoom_range is None:
            self.set_zoom_range()

        if self.type == "data":  

            if self.parameter == "elevation":
                # Elevation tiles (default terrarium encoding)
                # self.data is a list of dicts (for bathymetry database)
                if self.make_highest_level:
                    # Now loop through datasets in data_list (if zoom range is none and there is an index_path, it is set here)
                    for idata, data_dict in enumerate(self.data):
                        print(f"Processing {data_dict['name']} ... ({idata + 1} of {len(self.data)})")
                        make_topobathy_tiles_top_level(
                            self,
                            data_dict,
                        )
                if self.make_lower_levels:
                    make_topobathy_tiles_lower_levels(self)
                # For anything but global datasets (that have every tile), make an availability file
                if self.write_availability:
                    self.write_availability_file()
                if self.write_metadata:
                    self.write_metadata_file()

            elif self.parameter == "index" or self.parameter == "indices":
                # Index tiles (int32 encoding)
                # self.data is xugrid
                make_index_tiles(self, topo_path=self.topo_path)

            else:
                # Other data tiles (default float32 encoding)
                # self.data is 1D numpy array, index_path must be provided
                make_data_tiles(self)

        else:
            # RGB tiles (self.parameter can be flood map, topography, or any other string)
            # self.data is 1D numpy array, index_path must be provided
            make_rgba_tiles(self)

        if self.make_webviewer:
            write_html(
                os.path.join(self.path, "index.html"), max_native_zoom=self.zoom_range[1]
            )

    def read_metadata(self):
        # Read metadata file
        tml_file = os.path.join(self.path, "metadata.tml")
        if os.path.exists(tml_file):
            tml = toml.load(tml_file)
            for key in tml:
                setattr(self, key, tml[key])

    def read_availability(self):
        # Read netcdf file with dimensions
        nc_file = os.path.join(self.path, "available_tiles.nc")

        with xr.open_dataset(nc_file) as ds:
            self.zoom_levels = []
            # Loop through zoom levels
            for izoom in range(self.max_zoom + 1):
                n = 2**izoom
                iname = f"i_available_{izoom}"
                jname = f"j_available_{izoom}"
                iav = ds[iname].to_numpy()[:]
                jav = ds[jname].to_numpy()[:]
                zoom_level = ZoomLevel()
                zoom_level.ntiles = n
                zoom_level.ij_available = iav * n + jav
                self.zoom_levels.append(zoom_level)

    def set_zoom_range(self):
        if self.index_path is None:
            # No index path either
            if self.dx_max_zoom is None:
                # And no dx_max_zoom!
                # Need to determine dx_max_zoom from all the datasets
                # Loop through datasets in datalist to determine dxmin in metres
                self.dx_max_zoom = 10.0
            # Find appropriate zoom level
            zoom_max = get_zoom_level_for_resolution(self.dx_max_zoom)
        else:
            # Index path is provided, so we can determine the max zoom from that
            zoom_levels = list_folders(os.path.join(self.index_path, "*"), basename=True)
            if zoom_levels:
                # zoom_levels is a list of strings, convert to int and sort
                zoom_levels = [int(z) for z in zoom_levels]
                zoom_levels.sort()
                zoom_max = zoom_levels[-1]
        self.zoom_range = [0, zoom_max]

    def check_availability(self, i, j, izoom):
        # Check if tile exists at all
        zoom_level = self.zoom_levels[izoom]
        ij = i * zoom_level.ntiles + j
        # Use numpy array for fast search
        available = np.isin(ij, zoom_level.ij_available)
        return available

    def download_tile(self, i, j, izoom):
        key = f"{self.s3_key}/{izoom}/{i}/{j}.png"
        filename = os.path.join(self.path, str(izoom), str(i), str(j) + ".png")
        try:
            self.s3_client.download_file(
                Bucket=self.bucket,  # assign bucket name
                Key=key,  # key is the file name
                Filename=filename,
            )  # storage file path
            print(f"Downloaded {key}")
            okay = True
        except Exception:
            # Download failed
            print(f"Failed to download {key}")
            okay = False
        return okay

    def download_tile_parallel(self, bucket, key, file):
        try:
            # Make sure the folder exists
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

            self.s3_client.download_file(
                Bucket=bucket,  # assign bucket name
                Key=key,  # key is the file name
                Filename=file,
            )  # storage file path
            print(f"Downloaded {key}")
            okay = True

        except Exception as e:
            # Download failed
            print(e)
            print(f"Failed to download {key}")
            okay = False

        return okay

    def upload(
        self,
        bucket_name,
        s3_folder,
        access_key,
        secret_key,
        region,
        parallel=True,
        quiet=True,
    ):
        from cht_utils.s3 import S3Session

        # Upload to S3
        try:
            s3 = S3Session(access_key, secret_key, region)
            # Upload entire folder to S3 server
            s3.upload_folder(
                bucket_name,
                self.path,
                f"{s3_folder}/{self.name}",
                parallel=parallel,
                quiet=quiet,
            )
        except BaseException:
            print("An error occurred while uploading !")
        pass

    def write_availability_file(self):
        # Make availability file
        # Loop through zoom levels
        ds = xr.Dataset()
        zoom_level_paths = list_folders(os.path.join(self.path, "*"), basename=True)
        zoom_levels = [int(z) for z in zoom_level_paths]
        zoom_levels.sort()
        for izoom in zoom_levels:
            # Create empty array
            n = 0
            iav = []
            jav = []
            i_paths = list_folders(
                os.path.join(self.path, str(izoom), "*"), basename=True
            )
            for ipath in i_paths:
                i = int(ipath)
                j_paths = list_files(os.path.join(self.path, str(izoom), ipath, "*"))
                for jpath in j_paths:
                    pngfile = os.path.basename(jpath)
                    j = int(pngfile.split(".")[0])
                    iav.append(i)
                    jav.append(j)
                    n += 1
            # Now create dataarrays i_available and j_available
            iav = np.array(iav)
            jav = np.array(jav)
            dimname = f"n_{izoom}"
            iname = f"i_available_{izoom}"
            jname = f"j_available_{izoom}"
            ds[iname] = xr.DataArray(iav, dims=dimname)
            ds[jname] = xr.DataArray(jav, dims=dimname)

        # Save to netcdf file
        nc_file = os.path.join(self.path, "available_tiles.nc")
        ds.to_netcdf(nc_file)

        ds.close()

    def write_metadata_file(self):
        metadata = {}

        metadata["longname"] = self.long_name
        metadata["format"] = "tiled_web_map"
        metadata["encoder"] = self.encoder
        if self.encoder_vmin is not None:
            metadata["encoder_vmin"] = self.encoder_vmin
        if self.encoder_vmax is not None:
            metadata["encoder_vmax"] = self.encoder_vmax
        metadata["url"] = self.url
        if self.s3_bucket is not None:
            metadata["s3_bucket"] = self.s3_bucket
        if self.s3_key is not None:
            metadata["s3_key"] = self.s3_key
        if self.s3_region is not None:
            metadata["s3_region"] = self.s3_region
        metadata["max_zoom"] = self.max_zoom
        metadata["interpolation_method"] = "linear"
        metadata["source"] = self.source
        metadata["coord_ref_sys_name"] = "WGS 84 / Pseudo-Mercator"
        metadata["coord_ref_sys_kind"] = "projected"
        metadata["vertical_reference_level"] = self.vertical_reference_level
        metadata["vertical_units"] = self.vertical_units
        metadata["difference_with_msl"] = self.difference_with_msl
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
        metadata["description"]["disclaimer"] = (
            "These data are made available in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."
        )

        # Write to toml file
        # toml_file = os.path.join(path, name + ".tml")
        toml_file = os.path.join(self.path, "metadata.tml")
        with open(toml_file, "w") as f:
            toml.dump(metadata, f)

