# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:58:08 2021

@author: ormondt
"""
# from . import index_tiles
# from . import data_tiles
# from . import viewer_tiles

__version__ = "0.0.1"

from .tiled_web_map import TiledWebMap
from . import overlays

# from .utils import deg2num
# from .utils import num2deg
# from .utils import get_zoom_level
# from .utils import list_folders
# from .utils import png2int
# from .utils import png2elevation

# class TopoBathyDataset:
#     def __init__(self, path):
#         self.path = path

#     def get_data(self, xl, y, max_pixel_size):
#         # xl and yl are in CRS 3857
#         # max_pixel_size is in meters

#         # Check available levels in index tiles
#         max_zoom = 0
#         levs = list_folders(os.path.join(self.path, "*"), basename=True)
#         for lev in levs:
#             max_zoom = max(max_zoom, int(lev))

#         izoom = get_zoom_level(npixels, lat_range, max_zoom)   

#         ix0, iy0 = deg2num(lat_range[1], lon_range[0], izoom)
#         ix1, iy1 = deg2num(lat_range[0], lon_range[1], izoom)

#         nx = (ix1 - ix0 + 1) * 256
#         ny = (iy1 - iy0 + 1) * 256
#         zz = np.empty((ny, nx))
#         zz[:] = np.nan

#         if not quiet:
#             print("Processing zoom level " + str(izoom))

#         for i in range(ix0, ix1 + 1):
#             ifolder = str(i)
#             for j in range(iy0, iy1 + 1):
#                 # Read bathy
#                 bathy_file = os.path.join(
#                     topo_path, str(izoom), ifolder, str(j) + ".png"
#                 )
#                 if not os.path.exists(bathy_file):
#                     # No bathy for this tile, continue
#                     continue
#                 valt = png2elevation(bathy_file)

#                 ii0 = (i - ix0) * 256
#                 ii1 = ii0 + 256
#                 jj0 = (j - iy0) * 256
#                 jj1 = jj0 + 256
#                 zz[jj0:jj1, ii0:ii1] = valt
#         pass




# def make_index_tiles(grid,
#                      path,
#                      zoom_range=[0, 13],
#                      format="png"):
#     """Make index tiles"""
#     index_tiles.make_index_tiles(grid, path, zoom_range=zoom_range, format=format)

# def make_flood_map_tiles(*args):
#     """Make flood map files"""
#     viewer_tiles.make_flood_map_files(*args)

# def make_flood_map_overlay(*args):
#     """Make flood map overlay"""
#     overlays.make_flood_map_overlay(*args)

# def make_topobathy_overlay(path, lon_range, lat_range,
#     npixels=800,
#     color_map="jet",
#     color_scale_auto=False,
#     color_range=[-100.0, 100.0],
#     quiet=False,
#     file_name=None):
#     """Make data overlay"""
#     overlays.make_topobathy_overlay(path, lon_range, lat_range,
#                                npixels=npixels,
#                                color_map=color_map,
#                                color_range=color_range,
#                                color_scale_auto=color_scale_auto,
#                                quiet=quiet,
#                                file_name=file_name)



def make_data_overlay(val, path, lon_range, lat_range,
    npixels=800,
    color_map="jet",
    color_values=None,
    caxis=None,
    merge=True,
    depth=None,
    quiet=False,
    file_name=None):
    """Make data overlay"""
    overlays.make_data_overlay(val, path, lon_range, lat_range,
                               npixels=npixels,
                               color_map=color_map,
                               color_values=color_values,
                               caxis=caxis,
                               merge=merge,
                               depth=depth,
                               quiet=quiet,
                               file_name=file_name)
    
def make_overlay(lon_range, lat_range,
    option="val",
    val=None,
    topo_path="",
    index_path="",
    npixels=800,
    color_values=None,
    color_map="jet",
    color_range=[-10.0, 10.0],
    color_scale_auto=False,
    color_scale_symmetric=False,
    color_scale_symmetric_side="min",
    hillshading=False,
    hillshading_azimuth=315,
    hillshading_altitude=30,
    hillshading_exaggeration=10.0,
    quiet=False,
    file_name=None,
):
    overlays.make_overlay(lon_range, lat_range,
                            option=option,
                            val=val,
                            topo_path=topo_path,
                            index_path=index_path,
                            npixels=npixels,
                            color_values=color_values,
                            color_map=color_map,
                            color_range=color_range,
                            color_scale_auto=color_scale_auto,
                            color_scale_symmetric=color_scale_symmetric,
                            color_scale_symmetric_side=color_scale_symmetric_side,
                            hillshading=hillshading,
                            hillshading_azimuth=hillshading_azimuth,
                            hillshading_altitude=hillshading_altitude,
                            hillshading_exaggeration=hillshading_exaggeration,
                            quiet=quiet,
                            file_name=file_name)
