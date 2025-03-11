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

from cht_tiling.indices import make_index_tiles
from cht_tiling.topobathy import (
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
    def __init__(self, path, name, parameter="elevation"):
        # Parameter may be one of the following: elevation, floodmap, index, data, rgb
        if parameter not in ["elevation", "floodmap", "index", "data", "rgb"]:
            raise ValueError(
                "Parameter must be one of the following: elevation, floodmap, index, data, rgb"
            )

        self.name = name
        self.path = path
        self.url = None
        self.npix = 256
        self.parameter = "elevation"
        self.encoder = "terrarium"
        self.encoder_vmin = None
        self.encoder_vmax = None
        self.max_zoom = 0
        self.s3_client = None
        self.s3_bucket = None
        self.s3_key = None
        self.s3_region = None
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
        ix1 = min(2**izoom - 1, ix1)
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
                ifolder = str(i)
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
            ifolder = str(i)
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

    def generate_topobathy_tiles(
        self,
        datalist,
        bathymetry_database=None,
        index_path=None,
        lon_range=None,
        lat_range=None,
        zoom_range=None,
        make_webviewer=True,
        write_metadata=True,
        make_availability_file=True,
        make_lower_levels=True,
        make_highest_level=True,
        skip_existing=False,
        parallel=True,
        interpolation_method="linear",
    ):
        if make_highest_level:
            # for data_dict in datalist:
            # Can loop here around different datasets
            make_topobathy_tiles_top_level(
                self,
                datalist,
                bathymetry_database=bathymetry_database,
                index_path=index_path,
                lon_range=lon_range,
                lat_range=lat_range,
                zoom_range=zoom_range,
                skip_existing=skip_existing,
                parallel=parallel,
                interpolation_method=interpolation_method,
            )

        if make_lower_levels:
            make_topobathy_tiles_lower_levels(
                self, skip_existing=skip_existing, parallel=parallel
            )

        # For anything but global datasets, make an availability file
        if make_availability_file:
            self.make_availability_file()
        if write_metadata:
            self.write_metadata()
        if make_webviewer:
            write_html(
                os.path.join(self.path, "index.html"), max_native_zoom=self.max_zoom
            )

    def generate_flood_map_tiles(self):
        pass

    def generate_index_tiles(self, grid, zoom_range, format="png", webviewer=True):
        make_index_tiles(
            grid, self.path, zoom_range=zoom_range, format=format, webviewer=webviewer
        )

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

    def make_availability_file(self):
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

    def write_metadata(self):
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
