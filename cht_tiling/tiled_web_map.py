# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:51:04 2021

@author: ormondt
"""

import os
import toml
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import numpy as np
import netCDF4 as nc
from multiprocessing.pool import ThreadPool

from .topobathy import make_topobathy_tiles
from .utils import get_zoom_level_for_resolution, png2elevation, xy2num, num2xy, list_folders

class ZoomLevel:
    def __init__(self):        
        self.dx = 0.0
        self.dy = 0.0
        self.i_available = []
        self.j_available = []

class TiledWebMap:
    def __init__(self, path, name, parameter="elevation"):
        # Parameter may be one of the following: elevation, floodmap, index, data, rgb
        if parameter not in ["elevation", "floodmap", "index", "data", "rgb"]:
            raise ValueError("Parameter must be one of the following: elevation, floodmap, index, data, rgb")

        self.name = name        
        self.path = path
        self.url = ""        
        self.parameter = "elevation"
        self.encoder = "terrarium"
        self.encoder_vmin = None
        self.encoder_vmax = None
        self.max_zoom = 0
        self.s3_client = None
        self.read_metadata()

        # Check if available_tiles.nc exists. If not, just read the folders to get the zoom range.
        nc_file = os.path.join(self.path, "available_tiles.nc")
        self.availability_loaded = False
        if os.path.exists(nc_file):
            self.availability_exists = True
        else:
            self.availability_exists = False

            # Check available levels in index tiles
            self.max_zoom = 0
            levs = list_folders(os.path.join(self.path, "*"), basename=True)
            for lev in levs:
                self.max_zoom = max(self.max_zoom, int(lev))

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
        ds = nc.Dataset(nc_file)
        self.max_zoom = ds.dimensions["zoom_levels"].size - 1

    def get_data(self, xl, yl, max_pixel_size, crs=None):
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
        for i in range(ix0, ix1 + 1):
            ifolder = str(i)
            for j in range(iy0, iy1 + 1):
                png_file = os.path.join(
                    self.path, str(izoom), ifolder, str(j) + ".png"
                )                
                if not os.path.exists(png_file):
                    okay = self.check_download(i, j, izoom)
                    if okay:
                        # Add to download_list
                        download_file_list.append(png_file)
                        download_key_list.append("data/bathymetry/" + self.name + "/" + str(izoom) + "/" + ifolder + "/" + str(j) + ".png")
                        # Make sure the folder exists
                        if not os.path.exists(os.path.dirname(png_file)):
                            os.makedirs(os.path.dirname(png_file))

        # Now download the missing tiles    
        if len(download_file_list) > 0:
            # make boto s
            if self.s3_client is None:
                self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            # Download missing tiles
            with ThreadPool() as pool:
                pool.starmap(self.download_tile_parallel, [(self.s3_bucket, key, file) for key, file in zip(download_key_list, download_file_list)])

        # Loop over required tiles
        for i in range(ix0, ix1 + 1):
            ifolder = str(i)
            for j in range(iy0, iy1 + 1):

                png_file = os.path.join(
                    self.path, str(izoom), ifolder, str(j) + ".png"
                )

                if not os.path.exists(png_file):
                    # Fetch the file
                    # okay = self.fetch_tile(i, j, izoom)
                    if not okay:
                        # No bathy for this tile, continue
                        continue

                # Read the png file
                valt = png2elevation(png_file,
                                     encoder=self.encoder,
                                     encoder_vmin=self.encoder_vmin,
                                     encoder_vmax=self.encoder_vmax)
                
                # Fill array
                ii0 = (i - ix0) * 256
                ii1 = ii0 + 256
                jj0 = (j - iy0) * 256
                jj1 = jj0 + 256
                z[jj0:jj1, ii0:ii1] = valt

        # Compute x and y coordinates
        x0, y0 = num2xy(ix0, iy1 + 1, izoom) # lower left
        x1, y1 = num2xy(ix1 + 1, iy0, izoom) # upper right
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
        z = np.flipud(z)

        return x, y, z

    def generate_topobathy_tiles(self, **kwargs):
        make_topobathy_tiles(self.path, **kwargs)

    # def generate_index_tiles(self, kwargs):
    #     make_index_tiles(self.path, **kwargs)

    # def generate_floodmap_tiles(self, kwargs):
    #     make_floodmap_tiles(self.path, **kwargs)

    def fetch_tile(self, i, j, izoom):
        # Checks whether tile should be available, and if so, tries to download it
        okay = False
        if self.check_download(i, j, izoom):
            # It should be on a server somewhere, so download it
            okay = self.download_tile(i, j, izoom)
        return okay            

    def check_download(self, i, j, izoom):
        # Checks whether it's available to download
        okay = False
        # If url is provided, the tile may be on a web server.
        if self.url is not None:
            if not self.availability_exists:
                # There is no availability matrix, so we assume all tiles are available.
                okay = True
            else:
                # Check availability of the tile in matrix.
                okay = self.check_availability(i, j, izoom)
        return okay        

    def check_availability(self, i, j, izoom):
        # Check if tile exists at all
        available = False
        return available

    def download_tile(self, i, j, izoom):

        bucket = "deltares-ddb"
        key = f"data/bathymetry/{self.name}/{izoom}/{i}/{j}.png"

        filename = os.path.join(self.path, str(izoom), str(i), str(j) + ".png")

        # Make sure the folder exists
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
         
        try:


            self.s3_client.download_file(Bucket=bucket,     # assign bucket name
                                         Key=key,           # key is the file name
                                         Filename=filename) # storage file path
            print(f"Downloaded {key}")
            okay = True

        except:
            # Download failed
            print(f"Failed to download {key}")
            okay = False    

        return okay

    def download_tile_parallel(self, bucket, key, file):         
        try:
            self.s3_client.download_file(Bucket=bucket,     # assign bucket name
                                         Key=key,           # key is the file name
                                         Filename=file) # storage file path
            print(f"Downloaded {key}")
            okay = True

        except:
            # Download failed
            print(f"Failed to download {key}")
            okay = False    

        return okay

    def upload(self, name, bucket_name, s3_folder, access_key, secret_key, region, parallel=True, quiet=True):
        from cht_utils.s3 import S3Session
        # Upload to S3
        try:
            s3 = S3Session(access_key, secret_key, region)
            # Upload entire folder to S3 server
            s3.upload_folder(bucket_name, self.path, s3_folder, parallel=parallel, quiet=quiet)
            # pth = os.path.join(self.path, "9") 
            # s3pth = s3_folder + "/9"
            # s3.upload_folder(bucket_name, pth, s3pth, parallel=parallel, quiet=quiet)
        except BaseException as e:
            print("An error occurred while uploading !")
        pass

