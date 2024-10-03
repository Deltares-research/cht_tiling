# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:51:04 2021

@author: ormondt
"""

import glob
import math
import os
import traceback
import toml

import numpy as np
#from matplotlib import cm
#from matplotlib.colors import LightSource
from PIL import Image
#from pyproj import CRS, Transformer
#from scipy.interpolate import RegularGridInterpolator

from .topobathy import make_topobathy_tiles
from .utils import get_zoom_level_for_resolution, png2elevation, xy2num, num2xy, list_folders

class ZoomLevel:
    def __init__(self):        
        self.dx = 0.0
        self.dy = 0.0
        self.i_available = []
        self.j_available = []

class TiledWebMap:
    def __init__(self, path, parameter):
        # Parameter may be one of the following: elevation, floodmap, index, data, rgb
        if parameter not in ["elevation", "floodmap", "index", "data", "rgb"]:
            raise ValueError("Parameter must be one of the following: elevation, floodmap, index, data, rgb")
        
        self.path = path
        self.url = ""        
        self.parameter = "elevation"
        self.encoder = "terrarium"
        self.encoder_vmin = None
        self.encoder_vmax = None
        self.max_zoom = 0
        self.read_metadata()
        self.read_tile_structure()

    def read_metadata(self):
        # Read metadata file
        tml_file = os.path.join(self.path, "metadata.tml")
        if os.path.exists(tml_file):
            tml = toml.load(tml_file)
            for key in tml:
                setattr(self, key, tml[key])

    def read_tile_structure(self):
        # Read netcdf file with dimensions
        # Loop over all zoom levels

        # Check if available_tiles.nc exists. If not, just read the folders to get the zoom range
        nc_file = os.path.join(self.path, "available_tiles.nc")

        if os.path.exists(nc_file):
            pass

            # ds = nc.Dataset(nc_file)
            # self.pixels_in_tile = ds["tile_size_x"][0]
            # self.nr_zoom_levels = ds.dimensions["zoom_levels"].size
            # for izoom in range(self.nr_zoom_levels):
            #     zl = ZoomLevel()
            #     zl.dx = ds["grid_size_x"][izoom]
            #     zl.dy = ds["grid_size_y"][izoom]
            #     zl.nr_tiles_x = ds["nr_tiles_x"][izoom]
            #     zl.nr_tiles_y = ds["nr_tiles_y"][izoom]
            #     self.zoom_level.append(zl)

        else:        

            # Check available levels in index tiles
            self.max_zoom = 0
            levs = list_folders(os.path.join(self.path, "*"), basename=True)
            for lev in levs:
                self.max_zoom = max(self.max_zoom, int(lev))

    def get_data(self, xl, yl, max_pixel_size, crs=None):
        # xl and yl are in CRS 3857
        # max_pixel_size is in meters
        # returns x, y, and z in CRS 3857

        izoom = get_zoom_level_for_resolution(max_pixel_size)

        izoom = min(izoom, self.max_zoom)

        ix0, iy0 = xy2num(xl[0], yl[1], izoom)
        ix1, iy1 = xy2num(xl[1], yl[0], izoom)

        nx = (ix1 - ix0 + 1) * 256
        ny = (iy1 - iy0 + 1) * 256
        z = np.empty((ny, nx))
        z[:] = np.nan

        for i in range(ix0, ix1 + 1):
            ifolder = str(i)
            for j in range(iy0, iy1 + 1):

                # Read bathy
                png_file = os.path.join(
                    self.path, str(izoom), ifolder, str(j) + ".png"
                )
                if not os.path.exists(png_file):
                    # If url is provided, try to download the tile
                    # No bathy for this tile, continue                    
                    continue

                valt = png2elevation(png_file,
                                     encoder=self.encoder,
                                     encoder_vmin=self.encoder_vmin,
                                     encoder_vmax=self.encoder_vmax)

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

    def generate_topobathy_tiles(self, kwargs):
        make_topobathy_tiles(self.path, **kwargs)

    # def generate_index_tiles(self, kwargs):
    #     make_index_tiles(self.path, **kwargs)

    # def generate_floodmap_tiles(self, kwargs):
    #     make_floodmap_tiles(self.path, **kwargs)
