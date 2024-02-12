# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:58:08 2021

@author: ormondt
"""
from . import index_tiles
# from . import data_tiles
# from . import viewer_tiles
from . import overlays

__version__ = "0.0.1"

def make_index_tiles(grid,
                     path,
                     zoom_range=[0, 13],
                     format="png"):
    """Make index tiles"""
    index_tiles.make_index_tiles(grid, path, zoom_range=zoom_range, format=format)

# def make_flood_map_tiles(*args):
#     """Make flood map files"""
#     viewer_tiles.make_flood_map_files(*args)

# def make_flood_map_overlay(*args):
#     """Make flood map overlay"""
#     overlays.make_flood_map_overlay(*args)

def make_topobathy_overlay(path, lon_range, lat_range,
    npixels=800,
    color_map="jet",
    color_scale_auto=False,
    color_range=[-100.0, 100.0],
    quiet=False,
    file_name=None):
    """Make data overlay"""
    overlays.make_topobathy_overlay(path, lon_range, lat_range,
                               npixels=npixels,
                               color_map=color_map,
                               color_range=color_range,
                               color_scale_auto=color_scale_auto,
                               quiet=quiet,
                               file_name=file_name)



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
