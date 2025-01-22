# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:51:04 2021

@author: ormondt
"""

import glob
import math
import os
import traceback

import numpy as np
from matplotlib import cm, colormaps
from matplotlib.colors import LightSource
from PIL import Image
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

import cht_tiling.fileops as fo

# class TileLayerTime:
#     def __init__(self):

#         self.time_range  = None
#         self.time        = None
#         self.path        = None
#         self.time_string = None

# class TileLayer:
#     def __init__(self):

#         self.name        = name
#         self.long_name   = None
#         self.path        = None
#         self.zoom_range  = []
#         self.value_range = []
#         self.times       = []

#     def add_time(self, time_range=None, time=None, path=None, time_string=None):

#         t = TileLayerTime()
#         t.time_range     = time_range
#         t.time           = time
#         t.path           = path
#         t.time_string    = time_string
#         self.times.append(t)

# tl = TileLayer()
# tl.path            = "d:\\cosmos\\temp"
# tl.zoom_range      = [0, 14]
# tl.index_tile_path = "d:\\cosmos\\temp\\indices"
# tl.make(z)


def make_png_tiles(
    valg,
    index_path,
    png_path,
    zoom_range=[0, 23],
    option="direct",
    topo_path=None,
    color_values=None,
    caxis=None,
    zbmax=-999.0,
    merge=True,
    depth=None,
    quiet=False,
):
    """
    Generates PNG web tiles

    :param valg: Name of the scenario to be run.
    :type valg: array
    :param index_path: Path where the index tiles are sitting.
    :type index_path: str
    :param png_path: Output path where the png tiles will be created.
    :type png_path: str
    :param option: Option to define the type of tiles to be generated.
    Options are
    'direct', 'floodmap', 'topography'. Defaults to 'direct',
    in which case the values
    in *valg* are used directly.
    :type option: str
    :param zoom_range: Zoom range for
    which the png tiles will be created.
    Defaults to [0, 23].
    :type zoom_range: list of int

    """

    if isinstance(valg, list):
        pass
    else:
        valg = valg.transpose().flatten()

    if not caxis:
        caxis = []
        caxis.append(np.nanmin(valg))
        caxis.append(np.nanmax(valg))

    for izoom in range(zoom_range[0], zoom_range[1] + 1):
        if not quiet:
            print("Processing zoom level " + str(izoom))

        index_zoom_path = os.path.join(index_path, str(izoom))

        if not os.path.exists(index_zoom_path):
            continue

        png_zoom_path = os.path.join(png_path, str(izoom))
        makedir(png_zoom_path)

        for ifolder in list_folders(os.path.join(index_zoom_path, "*")):
            path_okay = False
            ifolder = os.path.basename(ifolder)
            index_zoom_path_i = os.path.join(index_zoom_path, ifolder)
            png_zoom_path_i = os.path.join(png_zoom_path, ifolder)

            for jfile in list_files(os.path.join(index_zoom_path_i, "*.dat")):
                jfile = os.path.basename(jfile)
                j = int(jfile[:-4])

                index_file = os.path.join(index_zoom_path_i, jfile)
                png_file = os.path.join(png_zoom_path_i, str(j) + ".png")

                ind = np.fromfile(index_file, dtype="i4")

                if topo_path and option == "flood_probability_map":
                    # valg is actually CDF interpolator to obtain
                    # probability of water level

                    # Read bathy
                    bathy_file = os.path.join(
                        topo_path, str(izoom), ifolder, str(j) + ".dat"
                    )
                    if not os.path.exists(bathy_file):
                        # No bathy for this tile, continue
                        continue
                    zb = np.fromfile(bathy_file, dtype="f4")
                    zs = zb + depth

                    valt = valg[ind](zs)
                    valt[ind < 0] = np.nan

                elif topo_path and option == "floodmap":
                    # Read bathy
                    bathy_file = os.path.join(
                        topo_path, str(izoom), ifolder, str(j) + ".dat"
                    )
                    if not os.path.exists(bathy_file):
                        # No bathy for this tile, continue
                        continue
                    zb = np.fromfile(bathy_file, dtype="f4")

                    valt = valg[ind]
                    valt = valt - zb
                    valt[valt < 0.05] = np.nan
                    valt[zb < zbmax] = np.nan

                elif topo_path and option == "topography":
                    # Read bathy
                    bathy_file = os.path.join(
                        topo_path, str(izoom), ifolder, str(j) + ".dat"
                    )
                    if not os.path.exists(bathy_file):
                        # No bathy for this tile, continue
                        continue
                    zb = np.fromfile(bathy_file, dtype="f4")

                    valt = zb

                else:
                    valt = valg[ind]
                    valt[ind < 0] = np.nan

                if color_values:
                    rgb = np.zeros((256 * 256, 4), "uint8")

                    # Determine value based on user-defined ranges
                    for color_value in color_values:
                        inr = np.logical_and(
                            valt >= color_value["lower_value"],
                            valt < color_value["upper_value"],
                        )
                        rgb[inr, 0] = color_value["rgb"][0]
                        rgb[inr, 1] = color_value["rgb"][1]
                        rgb[inr, 2] = color_value["rgb"][2]
                        rgb[inr, 3] = 255

                    rgb = rgb.reshape([256, 256, 4])
                    if not np.any(rgb > 0):
                        # Values found, go on to the next tiles
                        continue
                    rgb = np.flip(rgb, axis=0)
                    im = Image.fromarray(rgb)

                else:
                    valt = np.flipud(valt.reshape([256, 256]))
                    valt = (valt - caxis[0]) / (caxis[1] - caxis[0])
                    valt[valt < 0.0] = 0.0
                    valt[valt > 1.0] = 1.0
                    im = Image.fromarray(cm.jet(valt, bytes=True))

                if not path_okay:
                    if not os.path.exists(png_zoom_path_i):
                        makedir(png_zoom_path_i)
                        path_okay = True

                if os.path.exists(png_file):
                    # This tile already exists
                    if merge:
                        im0 = Image.open(png_file)
                        rgb = np.array(im)
                        rgb0 = np.array(im0)
                        isum = np.sum(rgb, axis=2)
                        rgb[isum == 0, :] = rgb0[isum == 0, :]
                        #                        rgb[rgb==0] = rgb0[rgb==0]
                        im = Image.fromarray(rgb)
                #                        im.show()

                im.save(png_file)


def make_floodmap_tiles(
    valg,
    index_path,
    png_path,
    topo_path,
    option="deterministic",
    zoom_range=None,
    color_values=None,
    caxis=None,
    zbmax=-999.0,
    merge=True,
    depth=None,
    quiet=False,
):
    """
    Generates PNG web tiles

    :param valg: Name of the scenario to be run.
    :type valg: array
    :param index_path: Path where the index tiles are sitting.
    :type index_path: str
    :param png_path: Output path where the png tiles will be created.
    :type png_path: str
    :param option: Option to define the type of tiles to be generated.
    Options are 'direct', 'floodmap', 'topography'. Defaults to 'direct',
    in which case the values in *valg* are used directly.
    :type option: str
    :param zoom_range: Zoom range for which
    the png tiles will be created.
    Defaults to [0, 23].
    :type zoom_range: list of int

    """

    if isinstance(valg, list):
        pass
    else:
        valg = valg.transpose().flatten()

    if not caxis:
        caxis = []
        caxis.append(np.nanmin(valg))
        caxis.append(np.nanmax(valg))

    # First do highest zoom level, then derefine from there
    if not zoom_range:
        # Check available levels in index tiles
        levs = fo.list_folders(os.path.join(index_path, "*"), basename=True)
        zoom_range = [999, -999]
        for lev in levs:
            zoom_range[0] = min(zoom_range[0], int(lev))
            zoom_range[1] = max(zoom_range[1], int(lev))

    izoom = zoom_range[1]

    if not quiet:
        print("Processing zoom level " + str(izoom))

    index_zoom_path = os.path.join(index_path, str(izoom))

    png_zoom_path = os.path.join(png_path, str(izoom))
    makedir(png_zoom_path)

    for ifolder in list_folders(os.path.join(index_zoom_path, "*")):
        path_okay = False
        ifolder = os.path.basename(ifolder)
        index_zoom_path_i = os.path.join(index_zoom_path, ifolder)
        png_zoom_path_i = os.path.join(png_zoom_path, ifolder)

        for jfile in list_files(os.path.join(index_zoom_path_i, "*.dat")):
            jfile = os.path.basename(jfile)
            j = int(jfile[:-4])

            index_file = os.path.join(index_zoom_path_i, jfile)
            png_file = os.path.join(png_zoom_path_i, str(j) + ".png")

            ind = np.fromfile(index_file, dtype="i4")

            if option == "probabilistic":
                # valg is actually CDF interpolator to obtain probability of water level

                # Read bathy
                bathy_file = os.path.join(
                    topo_path, str(izoom), ifolder, str(j) + ".dat"
                )
                if not os.path.exists(bathy_file):
                    # No bathy for this tile, continue
                    continue
                zb = np.fromfile(bathy_file, dtype="f4")
                zs = zb + depth

                valt = valg[ind](zs)
                valt[ind < 0] = np.nan

            else:
                # Read bathy
                bathy_file = os.path.join(
                    topo_path, str(izoom), ifolder, str(j) + ".dat"
                )
                if not os.path.exists(bathy_file):
                    # No bathy for this tile, continue
                    continue
                zb = np.fromfile(bathy_file, dtype="f4")

                valt = valg[ind]
                valt = valt - zb
                valt[valt < 0.05] = np.nan
                valt[zb < zbmax] = np.nan

            if color_values:
                rgb = np.zeros((256 * 256, 4), "uint8")

                # Determine value based on user-defined ranges
                for color_value in color_values:
                    inr = np.logical_and(
                        valt >= color_value["lower_value"],
                        valt < color_value["upper_value"],
                    )
                    rgb[inr, 0] = color_value["rgb"][0]
                    rgb[inr, 1] = color_value["rgb"][1]
                    rgb[inr, 2] = color_value["rgb"][2]
                    rgb[inr, 3] = 255

                rgb = rgb.reshape([256, 256, 4])
                if not np.any(rgb > 0):
                    # Values found, go on to the next tiles
                    continue
                rgb = np.flip(rgb, axis=0)
                im = Image.fromarray(rgb)

            else:
                valt = np.flipud(valt.reshape([256, 256]))
                valt = (valt - caxis[0]) / (caxis[1] - caxis[0])
                valt[valt < 0.0] = 0.0
                valt[valt > 1.0] = 1.0
                im = Image.fromarray(cm.jet(valt, bytes=True))

            if not path_okay:
                if not os.path.exists(png_zoom_path_i):
                    makedir(png_zoom_path_i)
                    path_okay = True

            if os.path.exists(png_file):
                # This tile already exists
                if merge:
                    im0 = Image.open(png_file)
                    rgb = np.array(im)
                    rgb0 = np.array(im0)
                    isum = np.sum(rgb, axis=2)
                    rgb[isum == 0, :] = rgb0[isum == 0, :]
                    #                        rgb[rgb==0] = rgb0[rgb==0]
                    im = Image.fromarray(rgb)
            #                        im.show()

            im.save(png_file)

    # Now make tiles for lower level by merging

    for izoom in range(zoom_range[1] - 1, zoom_range[0] - 1, -1):
        if not quiet:
            print("Processing zoom level " + str(izoom))

        index_zoom_path = os.path.join(index_path, str(izoom))

        if not os.path.exists(index_zoom_path):
            continue

        png_zoom_path = os.path.join(png_path, str(izoom))
        png_zoom_path_p1 = os.path.join(png_path, str(izoom + 1))
        makedir(png_zoom_path)

        for ifolder in list_folders(os.path.join(index_zoom_path, "*")):
            path_okay = False
            ifolder = os.path.basename(ifolder)
            i = int(ifolder)
            index_zoom_path_i = os.path.join(index_zoom_path, ifolder)
            png_zoom_path_i = os.path.join(png_zoom_path, ifolder)

            for jfile in list_files(os.path.join(index_zoom_path_i, "*.dat")):
                jfile = os.path.basename(jfile)
                j = int(jfile[:-4])

                png_file = os.path.join(png_zoom_path_i, str(j) + ".png")

                rgb = np.zeros((256, 256, 4), "uint8")

                i0 = i * 2
                i1 = i * 2 + 1
                j0 = j * 2
                j1 = j * 2 + 1

                tile_name_00 = os.path.join(png_zoom_path_p1, str(i0), str(j0) + ".png")
                tile_name_10 = os.path.join(png_zoom_path_p1, str(i0), str(j1) + ".png")
                tile_name_01 = os.path.join(png_zoom_path_p1, str(i1), str(j0) + ".png")
                tile_name_11 = os.path.join(png_zoom_path_p1, str(i1), str(j1) + ".png")

                okay = False

                # Lower-left
                if os.path.exists(tile_name_00):
                    okay = True
                    rgb0 = np.array(Image.open(tile_name_00))
                    rgb[128:256, 0:128, :] = rgb0[0:255:2, 0:255:2, :]
                # Upper-left
                if os.path.exists(tile_name_10):
                    okay = True
                    rgb0 = np.array(Image.open(tile_name_10))
                    rgb[0:128, 0:128, :] = rgb0[0:255:2, 0:255:2, :]
                # Lower-right
                if os.path.exists(tile_name_01):
                    okay = True
                    rgb0 = np.array(Image.open(tile_name_01))
                    rgb[128:256, 128:256, :] = rgb0[0:255:2, 0:255:2, :]
                # Upper-right
                if os.path.exists(tile_name_11):
                    okay = True
                    rgb0 = np.array(Image.open(tile_name_11))
                    rgb[0:128, 128:256, :] = rgb0[0:255:2, 0:255:2, :]

                if okay:
                    im = Image.fromarray(rgb)

                    if not path_okay:
                        if not os.path.exists(png_zoom_path_i):
                            makedir(png_zoom_path_i)
                            path_okay = True

                    if os.path.exists(png_file):
                        # This tile already exists
                        if merge:
                            im0 = Image.open(png_file)
                            rgb = np.array(im)
                            rgb0 = np.array(im0)
                            isum = np.sum(rgb, axis=2)
                            rgb[isum == 0, :] = rgb0[isum == 0, :]
                            #                        rgb[rgb==0] = rgb0[rgb==0]
                            im = Image.fromarray(rgb)
                    #                        im.show()

                    im.save(png_file)


# Following is function with old format. Should be removed asap.
def make_floodmap_overlay(
    valg,
    index_path,
    topo_path,
    npixels=[1200, 800],
    lon_range=None,
    lat_range=None,
    option="deterministic",
    color_values=None,
    caxis=None,
    zbmax=-999.0,
    merge=True,
    depth=None,
    quiet=False,
    file_name=None,
):
    # Used in FloodAdapt
    """
    Generates overlay PNG from tiles

    :param valg: Name of the scenario to be run.
    :type valg: array
    :param index_path: Path where the index tiles are sitting.
    :type index_path: str
    :param png_path: Output path where the png tiles will be created.
    :type png_path: str
    :param option: Option to define the type of tiles to be generated.
    Options are 'direct', 'floodmap', 'topography'. Defaults to 'direct',
    in which case the values in *valg* are used directly.
    :type option: str
    :param zoom_range: Zoom range for which
    the png tiles will be created.
    Defaults to [0, 23].
    :type zoom_range: list of int

    """

    if isinstance(valg, list):
        pass
    else:
        valg = valg.transpose().flatten()

    if not caxis:
        caxis = []
        caxis.append(np.nanmin(valg))
        caxis.append(np.nanmax(valg))

    # Check available levels in index tiles
    max_zoom = 0
    levs = fo.list_folders(os.path.join(index_path, "*"), basename=True)
    for lev in levs:
        max_zoom = max(max_zoom, int(lev))

    # Find zoom level that provides sufficient pixels
    for izoom in range(max_zoom + 1):
        ix0, it0 = deg2num(lat_range[0], lon_range[0], izoom)
        ix1, it1 = deg2num(lat_range[1], lon_range[1], izoom)
        if (ix1 - ix0 + 1) * 256 > npixels[0] and (it1 - it0 + 1) * 256 > npixels[1]:
            # Found sufficient zoom level
            break

    index_zoom_path = os.path.join(index_path, str(izoom))

    #    dxy = (40075016.686/npix) / 2 ** izoom
    #    xx = np.linspace(0.0, (256 - 1)*dxy, num=npix)
    #    yy = xx[:]
    #    xv, yv = np.meshgrid(xx, yy)

    nx = (ix1 - ix0 + 1) * 256
    ny = (it1 - it0 + 1) * 256
    zz = np.empty((ny, nx))
    zz[:] = np.nan

    if not quiet:
        print("Processing zoom level " + str(izoom))

    index_zoom_path = os.path.join(index_path, str(izoom))

    for i in range(ix0, ix1 + 1):
        ifolder = str(i)
        index_zoom_path_i = os.path.join(index_zoom_path, ifolder)

        for j in range(it0, it1 + 1):
            index_file = os.path.join(index_zoom_path_i, str(j) + ".dat")

            if not os.path.exists(index_file):
                continue

            ind = np.fromfile(index_file, dtype="i4")

            if option == "probabilistic":
                # valg is actually CDF interpolator to obtain probability of water level

                # Read bathy
                bathy_file = os.path.join(
                    topo_path, str(izoom), ifolder, str(j) + ".dat"
                )

                if not os.path.exists(bathy_file):
                    # No bathy for this tile, continue
                    continue

                zb = np.fromfile(bathy_file, dtype="f4")
                zs = zb + depth

                valt = valg[ind](zs)
                valt[ind < 0] = np.nan

            else:
                # Read bathy
                bathy_file = os.path.join(
                    topo_path, str(izoom), ifolder, str(j) + ".dat"
                )
                if not os.path.exists(bathy_file):
                    # No bathy for this tile, continue
                    continue
                zb = np.fromfile(bathy_file, dtype="f4")

                valt = valg[ind]
                valt = valt - zb
                valt[valt < 0.05] = np.nan
                valt[zb < zbmax] = np.nan

            ii0 = (i - ix0) * 256
            ii1 = ii0 + 256
            jj0 = (it1 - j) * 256
            jj1 = jj0 + 256
            zz[jj0:jj1, ii0:ii1] = np.flipud(valt.reshape([256, 256]))

    if color_values:
        # Create empty rgb array
        zz = zz.flatten()
        rgb = np.zeros((ny * nx, 4), "uint8")
        # Determine value based on user-defined ranges
        for color_value in color_values:
            inr = np.logical_and(
                zz >= color_value["lower_value"], zz < color_value["upper_value"]
            )
            rgb[inr, 0] = color_value["rgb"][0]
            rgb[inr, 1] = color_value["rgb"][1]
            rgb[inr, 2] = color_value["rgb"][2]
            rgb[inr, 3] = 255
        im = Image.fromarray(rgb.reshape([ny, nx, 4]))

    else:
        zz = (zz - caxis[0]) / (caxis[1] - caxis[0])
        zz[zz < 0.0] = 0.0
        zz[zz > 1.0] = 1.0
        im = Image.fromarray(cm.jet(zz, bytes=True))

    if file_name:
        # Get the directory part of the file_name
        directory = os.path.dirname(file_name)

        # If the directory doesn't exist, create it
        os.makedirs(directory, exist_ok=True)

        # Save the image
        im.save(file_name)

    lat0, lon0 = num2deg_ll(ix0, it0, izoom)  # lat/lon coordinates of lower left cell
    lat1, lon1 = num2deg_ur(ix1, it1, izoom)  # lat/lon coordinates of lower left cell
    return [lon0, lon1], [lat0, lat1]


# Flood map overlay new format
def make_floodmap_overlay_v2(
    valg,
    index_path,
    topo_path,
    npixels=[1200, 800],
    lon_range=None,
    lat_range=None,
    option="deterministic",
    color_values=None,
    caxis=None,
    zbmax=-999.0,
    merge=True,
    depth=None,
    quiet=False,
    file_name=None,
):
    """
    Generates overlay PNG from tiles

    :param valg: Name of the scenario to be run.
    :type valg: array
    :param index_path: Path where the index tiles are sitting.
    :type index_path: str
    :param png_path: Output path where the png tiles will be created.
    :type png_path: str
    :param option: Option to define the type of tiles to be generated.
    Options are 'direct', 'floodmap', 'topography'. Defaults to 'direct',
    in which case the values in *valg* are used directly.
    :type option: str
    :param zoom_range: Zoom range for which
    the png tiles will be created.
    Defaults to [0, 23].
    :type zoom_range: list of int

    """

    try:
        if isinstance(valg, list):
            # Why would this ever be a list ?!
            print("valg is a list!")
            pass
        else:
            valg = valg.transpose().flatten()

        if not caxis:
            caxis = []
            caxis.append(np.nanmin(valg))
            caxis.append(np.nanmax(valg))

        # Check available levels in index tiles
        max_zoom = 0
        levs = fo.list_folders(os.path.join(index_path, "*"), basename=True)
        for lev in levs:
            max_zoom = max(max_zoom, int(lev))

        # Find zoom level that provides sufficient pixels
        for izoom in range(max_zoom + 1):
            # ix0, it0 = deg2num(lat_range[0], lon_range[0], izoom)
            # ix1, it1 = deg2num(lat_range[1], lon_range[1], izoom)
            ix0, it0 = deg2num(lat_range[1], lon_range[0], izoom)
            ix1, it1 = deg2num(lat_range[0], lon_range[1], izoom)
            if (ix1 - ix0 + 1) * 256 > npixels[0] and (it1 - it0 + 1) * 256 > npixels[
                1
            ]:
                # Found sufficient zoom level
                break

        index_zoom_path = os.path.join(index_path, str(izoom))

        nx = (ix1 - ix0 + 1) * 256
        ny = (it1 - it0 + 1) * 256
        zz = np.empty((ny, nx))
        zz[:] = np.nan

        if not quiet:
            print("Processing zoom level " + str(izoom))

        index_zoom_path = os.path.join(index_path, str(izoom))

        for i in range(ix0, ix1 + 1):
            ifolder = str(i)
            index_zoom_path_i = os.path.join(index_zoom_path, ifolder)

            for j in range(it0, it1 + 1):
                index_file = os.path.join(index_zoom_path_i, str(j) + ".png")

                if not os.path.exists(index_file):
                    continue

                ind = png2int(index_file)

                if option == "probabilistic":
                    # This needs to be fixed later on
                    # valg is actually CDF interpolator to obtain probability of water level

                    # Read bathy
                    bathy_file = os.path.join(
                        topo_path, str(izoom), ifolder, str(j) + ".png"
                    )

                    if not os.path.exists(bathy_file):
                        # No bathy for this tile, continue
                        continue

                    zb = np.fromfile(bathy_file, dtype="f4")
                    zs = zb + depth

                    valt = valg[ind](zs)
                    valt[ind < 0] = np.nan

                else:
                    # Read bathy
                    bathy_file = os.path.join(
                        topo_path, str(izoom), ifolder, str(j) + ".png"
                    )
                    if not os.path.exists(bathy_file):
                        # No bathy for this tile, continue
                        continue

                    zb = png2elevation(bathy_file)

                    valt = valg[ind]
                    valt = valt - zb
                    valt[valt < 0.05] = np.nan
                    valt[zb < zbmax] = np.nan

                ii0 = (i - ix0) * 256
                ii1 = ii0 + 256
                jj0 = (j - it0) * 256
                jj1 = jj0 + 256
                zz[jj0:jj1, ii0:ii1] = valt

        if color_values:
            # Create empty rgb array
            zz = zz.flatten()
            rgb = np.zeros((ny * nx, 4), "uint8")
            # Determine value based on user-defined ranges
            for color_value in color_values:
                inr = np.logical_and(
                    zz >= color_value["lower_value"], zz < color_value["upper_value"]
                )
                rgb[inr, 0] = color_value["rgb"][0]
                rgb[inr, 1] = color_value["rgb"][1]
                rgb[inr, 2] = color_value["rgb"][2]
                rgb[inr, 3] = 255
            im = Image.fromarray(rgb.reshape([ny, nx, 4]))

        else:
            zz = (zz - caxis[0]) / (caxis[1] - caxis[0])
            zz[zz < 0.0] = 0.0
            zz[zz > 1.0] = 1.0
            im = Image.fromarray(cm.jet(zz, bytes=True))

        if file_name:
            im.save(file_name)

        lat1, lon0 = num2deg(ix0, it0, izoom)  # lat/lon coordinates of upper left cell
        lat0, lon1 = num2deg(ix1 + 1, it1 + 1, izoom)

        return [lon0, lon1], [lat0, lat1]

    except Exception as e:
        print(e)
        traceback.print_exc()
        return None, None


# Topo overlay old format
def make_topo_overlay(
    topo_path,
    npixels=[1200, 800],
    lon_range=None,
    lat_range=None,
    color_values=None,
    caxis=None,
    #    merge=True,
    #    depth=None,
    quiet=False,
    file_name=None,
):
    """
    Generates overlay PNG from tiles
    :param png_path: Output path where the png tiles will be created.
    :type png_path: str
    :param option: Option to define the type of tiles to be generated.
    Options are 'direct', 'floodmap', 'topography'. Defaults to 'direct',
    in which case the values in *valg* are used directly.
    :type option: str
    :param zoom_range: Zoom range for which
    the png tiles will be created.
    Defaults to [0, 23].
    :type zoom_range: list of int

    """

    # Check available levels in index tiles
    max_zoom = 0
    levs = fo.list_folders(os.path.join(topo_path, "*"), basename=True)
    for lev in levs:
        max_zoom = max(max_zoom, int(lev))

    # Find zoom level that provides sufficient pixels
    for izoom in range(max_zoom + 1):
        ix0, it0 = deg2num(lat_range[0], lon_range[0], izoom)
        ix1, it1 = deg2num(lat_range[1], lon_range[1], izoom)
        if (ix1 - ix0 + 1) * 256 > npixels[0] and (it1 - it0 + 1) * 256 > npixels[1]:
            # Found sufficient zoom level
            break

    nx = (ix1 - ix0 + 1) * 256
    ny = (it1 - it0 + 1) * 256
    zz = np.empty((ny, nx))
    zz[:] = np.nan

    if not quiet:
        print("Processing zoom level " + str(izoom))

    for i in range(ix0, ix1 + 1):
        ifolder = str(i)
        for j in range(it0, it1 + 1):
            # Read bathy
            bathy_file = os.path.join(topo_path, str(izoom), ifolder, str(j) + ".dat")
            if not os.path.exists(bathy_file):
                # No bathy for this tile, continue
                continue
            valt = np.fromfile(bathy_file, dtype="f4")

            ii0 = (i - ix0) * 256
            ii1 = ii0 + 256
            jj0 = (it1 - j) * 256
            jj1 = jj0 + 256
            zz[jj0:jj1, ii0:ii1] = np.flipud(valt.reshape([256, 256]))

    if color_values:
        # Create empty rgb array
        zz = zz.flatten()
        rgb = np.zeros((ny * nx, 4), "uint8")
        # Determine value based on user-defined ranges
        for color_value in color_values:
            inr = np.logical_and(
                zz >= color_value["lower_value"], zz < color_value["upper_value"]
            )
            rgb[inr, 0] = color_value["rgb"][0]
            rgb[inr, 1] = color_value["rgb"][1]
            rgb[inr, 2] = color_value["rgb"][2]
            rgb[inr, 3] = 255
        im = Image.fromarray(rgb.reshape([ny, nx, 4]))

    else:
        zz = (zz - caxis[0]) / (caxis[1] - caxis[0])
        zz[zz < 0.0] = 0.0
        zz[zz > 1.0] = 1.0
        im = Image.fromarray(cm.jet(zz, bytes=True))

    if file_name:
        im.save(file_name)

    lat0, lon0 = num2deg_ll(ix0, it0, izoom)  # lat/lon coordinates of lower left cell
    lat1, lon1 = num2deg_ur(ix1, it1, izoom)  # lat/lon coordinates of lower left cell
    return [lon0, lon1], [lat0, lat1]


# Topo overlay new format
def make_topo_overlay_v2(
    topo_path,
    npixels=[1200, 800],
    lon_range=None,
    lat_range=None,
    color_values=None,
    color_map="jet",
    color_range=None,
    color_scale_auto=False,
    color_scale_symmetric=True,
    color_scale_symmetric_side="min",
    hillshading=True,
    hillshading_azimuth=315,
    hillshading_altitude=30,
    hillshading_exaggeration=10.0,
    quiet=False,
    file_name=None,
):
    """
    Generates overlay PNG from tiles
    :param png_path: Output path where the png tiles will be created.
    :type png_path: str
    :param option: Option to define the type of tiles to be generated.
    Options are 'direct', 'floodmap', 'topography'. Defaults to 'direct',
    in which case the values in *valg* are used directly.
    :type option: str
    :param zoom_range: Zoom range for which
    the png tiles will be created.
    Defaults to [0, 23].
    :type zoom_range: list of int

    """

    try:
        # Check available levels in index tiles
        max_zoom = 0
        levs = fo.list_folders(os.path.join(topo_path, "*"), basename=True)
        for lev in levs:
            max_zoom = max(max_zoom, int(lev))

        izoom = get_zoom_level(npixels, lon_range, lat_range, max_zoom)

        ix0, it0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, it1 = deg2num(lat_range[0], lon_range[1], izoom)

        nx = (ix1 - ix0 + 1) * 256
        ny = (it1 - it0 + 1) * 256
        zz = np.empty((ny, nx))
        zz[:] = np.nan

        if not quiet:
            print("Processing zoom level " + str(izoom))

        for i in range(ix0, ix1 + 1):
            ifolder = str(i)
            for j in range(it0, it1 + 1):
                # Read bathy
                bathy_file = os.path.join(
                    topo_path, str(izoom), ifolder, str(j) + ".png"
                )
                if not os.path.exists(bathy_file):
                    # No bathy for this tile, continue
                    continue
                valt = png2elevation(bathy_file)

                ii0 = (i - ix0) * 256
                ii1 = ii0 + 256
                jj0 = (j - it0) * 256
                jj1 = jj0 + 256
                zz[jj0:jj1, ii0:ii1] = valt

        c0 = None
        c1 = None

        if color_values:
            # Create empty rgb array
            zz = zz.flatten()
            rgb = np.zeros((ny * nx, 4), "uint8")
            # Determine value based on user-defined ranges
            for color_value in color_values:
                inr = np.logical_and(
                    zz >= color_value["lower_value"], zz < color_value["upper_value"]
                )
                rgb[inr, 0] = color_value["rgb"][0]
                rgb[inr, 1] = color_value["rgb"][1]
                rgb[inr, 2] = color_value["rgb"][2]
                rgb[inr, 3] = 255
            im = Image.fromarray(rgb.reshape([ny, nx, 4]))

        else:
            # Two options here:
            # 1. color_scale_auto = True
            #   if color_scale_symmetric = True:
            #       a) color_scale_side = "min": use max(abs(min))
            #       b) color_scale_side = "max": use max(abs(max))
            #       c) color_scale_side = "both": use max(abs(min), abs(max))
            #   else:
            #       use min/max of topo
            # 2. color_range is a list of two values

            if color_scale_auto:
                if color_scale_symmetric:
                    if color_scale_symmetric_side == "min":
                        c0 = np.nanmin(zz)
                        if c0 > 0.0:
                            c0 = -10.0
                        c1 = -1 * c0
                    elif color_scale_symmetric_side == "max":
                        c1 = np.nanmax(zz)
                        if c1 < 0.0:
                            c1 = 10.0
                        c0 = -1 * c1
                    else:
                        c0 = -np.nanmax(np.abs(zz))
                        c1 = np.nanmax(np.abs(zz))

                else:
                    c0 = np.nanmin(zz)
                    c1 = np.nanmax(zz)

            else:
                c0 = color_range[0]
                c1 = color_range[1]

            cmap = colormaps.get_cmap(color_map)

            if hillshading:
                ls = LightSource(azdeg=hillshading_azimuth, altdeg=hillshading_altitude)
                # Compute pixel size in meters
                dxy = 156543.03 / 2**izoom
                rgb = (
                    ls.shade(
                        zz,
                        cmap,
                        vmin=c0,
                        vmax=c1,
                        dx=dxy,
                        dy=dxy,
                        vert_exag=hillshading_exaggeration,
                        blend_mode="soft",
                    )
                    * 255
                )
                # rgb = rgb * 255
                # rgb = rgb.astype(np.uint8)
                im = Image.fromarray(rgb.astype(np.uint8))

            else:
                zz = (zz - c0) / (c1 - c0)
                zz[zz < 0.0] = 0.0
                zz[zz > 1.0] = 1.0
                im = Image.fromarray(cmap(zz, bytes=True))

        if file_name:
            im.save(file_name)

        lat1, lon0 = num2deg(ix0, it0, izoom)  # lat/lon coordinates of upper left cell
        lat0, lon1 = num2deg(
            ix1 + 1, it1 + 1, izoom
        )  # lat/lon coordinates of lower right cell
        return [lon0, lon1], [lat0, lat1], [c0, c1]

    except Exception as e:
        print(e)
        traceback.print_exc()
        return None, None, None


def make_topobathy_tiles(
    path,
    dem_names,
    lon_range,
    lat_range,
    index_path=None,
    zoom_range=None,
    z_range=None,
    bathymetry_database_path="d:\\delftdashboard\\data\\bathymetry",
    quiet=False,
):
    """
    Generates topo/bathy tiles

    :param path: Path where topo/bathy tiles will be stored.
    :type path: str
    :param dem_name: List of DEM names (dataset names in Bathymetry Database).
    :type dem_name: list
    :param png_path: Output path where the png tiles will be created.
    :type png_path: str
    :param option: Option.
    :type option: str
    :param zoom_range: Zoom range for which the png tiles
    will be created. Defaults to [0, 23].
    :type zoom_range: list of int

    """

    from cht_bathymetry.bathymetry_database import BathymetryDatabase

    # from cht_utils.misc_tools import interp2

    bathymetry_database = BathymetryDatabase(None)
    bathymetry_database.initialize(bathymetry_database_path)

    if not zoom_range:
        zoom_range = [0, 13]

    if not z_range:
        z_range = [-20000.0, 20000.0]

    npix = 256

    transformer_4326_to_3857 = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
    )

    dem_list = []
    for dem_name in dem_names:
        dem = {}
        dem["dataset"] = bathymetry_database.get_dataset(dem_name)
        dem["zmin"] = -10000.0
        dem["zmax"] = 10000.0
        dem_list.append(dem)

    # Loop through zoom levels
    for izoom in range(zoom_range[0], zoom_range[1] + 1):
        if not quiet:
            print("Processing zoom level " + str(izoom))

        zoom_path = os.path.join(path, str(izoom))

        dxy = (40075016.686 / npix) / 2**izoom
        xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
        yy = xx[:]
        xv, yv = np.meshgrid(xx, yy)

        ix0, it0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, it1 = deg2num(lat_range[0], lon_range[1], izoom)

        # Loop in x direction
        for i in range(ix0, ix1 + 1):
            path_okay = False
            zoom_path_i = os.path.join(zoom_path, str(i))

            # Loop in y direction
            for j in range(it0, it1 + 1):
                file_name = os.path.join(zoom_path_i, str(j) + ".png")

                if index_path:
                    # Only make tiles for which there is an index file
                    index_file_name = os.path.join(
                        index_path, str(izoom), str(i), str(j) + ".png"
                    )
                    if not os.path.exists(index_file_name):
                        continue

                # Compute lat/lon at upper left corner of tile
                lat, lon = num2deg(i, j, izoom)

                # Convert origin to Global Mercator
                xo, yo = transformer_4326_to_3857.transform(lon, lat)

                # Tile grid on Global mercator
                x3857 = xo + xv[:] + 0.5 * dxy
                y3857 = yo - yv[:] - 0.5 * dxy

                # Get bathymetry on subgrid from bathymetry database
                zg = bathymetry_database.get_bathymetry_on_grid(
                    x3857, y3857, CRS.from_epsg(3857), dem_list
                )

                if np.isnan(zg).all():
                    # only nans in this tile
                    continue

                if np.nanmax(zg) < z_range[0] or np.nanmin(zg) > z_range[1]:
                    # all values in tile outside z_range
                    continue

                if not path_okay:
                    if not os.path.exists(zoom_path_i):
                        makedir(zoom_path_i)
                        path_okay = True

                # Write to terrarium png format
                elevation2png(zg, file_name)


def get_bathy_on_tile(
    x3857, y3857, dem_names, dem_crs, transformers, dxy, bathymetry_database
):
    npix = 256
    zg = np.float32(np.full([npix, npix], np.nan))

    for idem, dem_name in enumerate(dem_names):
        # Convert tile grid to crs of DEM
        xg, yg = transformers[idem].transform(x3857, y3857)

        # Bounding box of tile grid
        if dem_crs[idem].is_geographic:
            xybuf = dxy / 50000.0
        else:
            xybuf = 2 * dxy

        xl = [np.min(np.min(xg)) - xybuf, np.max(np.max(xg)) + xybuf]
        yl = [np.min(np.min(yg)) - xybuf, np.max(np.max(yg)) + xybuf]

        # Get DEM data (ddb format for now)
        x, y, z = bathymetry_database.get_data(dem_name, xl, yl, max_cell_size=dxy)

        if x is np.nan:
            # No data obtained from bathymetry database
            continue

        zg0 = np.float32(interp2(x, y, z, xg, yg))
        zg[np.isnan(zg)] = zg0[np.isnan(zg)]

        if not np.isnan(zg).any():
            # No nans left, so no need to load subsequent DEMs
            break

    return zg


#### Index to degree (and vice versa) functions


def deg2num(lat_deg, lon_deg, zoom):
    """Returns column and row index of slippy tile"""
    lat_rad = math.radians(lat_deg)
    n = 2**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """Returns upper left latitude and longitude of slippy tile"""
    # Return upper left corner of tile
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


### Old index to degrees functions
def num2deg_ll(xtile, ytile, zoom):
    # Return lower left corner of tile (only used in old format)
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(-lat_rad)
    return (lat_deg, lon_deg)


def num2deg_ur(xtile, ytile, zoom):
    # Return upper_right corner of tile (only used in old format)
    n = 2**zoom
    lon_deg = (xtile + 1) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n)))
    lat_deg = math.degrees(-lat_rad)
    return (lat_deg, lon_deg)


# def rgba2int(rgba):
#     """Convert rgba tuple to int"""
#     r, g, b, a = rgba
#     return (r * 256**3) + (g * 256**2) + (b * 256) + a


### Conversion between elevation and png and vice versa
# Note: we only use the RGB channels for this (and not the alpha channel)


def png2elevation(png_file):
    """Convert png to elevation array based on terrarium interpretation"""
    img = Image.open(png_file)
    rgb = np.array(img.convert("RGB"))
    # Convert RGB values to elevation values
    elevation = (rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256) - 32768.0
    # where val is less than -32767, set to NaN
    elevation[elevation < -32767.0] = np.nan
    return elevation


def elevation2png(val, png_file):
    """Convert elevation array to png using terrarium interpretation"""
    rgb = np.zeros((256 * 256, 3), "uint8")
    # r, g, b = elevation2rgb(val)
    val += 32768.0
    rgb[:, 0] = np.floor(val / 256).flatten()
    rgb[:, 1] = np.floor(val % 256).flatten()
    rgb[:, 2] = np.floor((val - np.floor(val)) * 256).flatten()
    rgb = rgb.reshape([256, 256, 3])
    # Create PIL Image from RGB values and save as PNG
    img = Image.fromarray(rgb)
    img.save(png_file)


# def elevation2rgb(val):
#     """Convert elevation to rgb tuple"""
#     val += 32768
#     r = np.floor(val / 256)
#     g = np.floor(val % 256)
#     b = np.floor((val - np.floor(val)) * 256)
#     return (r, g, b)

# def rgb2elevation(r, g, b):
#     """Convert rgb tuple to elevation"""
#     val = (r * 256 + g + b / 256) - 32768
#     return val

# def rgb2elevation(rgb):
#     """Convert rgb tuple to elevation"""
#     val = (rgb[:,:,0] * 256 + rgb[:,:,1] + rgb[:,:,2] / 256) - 32768.0
#     # where val is less than -32767, set to NaN
#     val[val<-32767.0] = np.nan
#     return val


### Conversion between int and png and vice versa


def png2int(png_file):
    """Convert png to int array"""
    # Open the PNG image
    image = Image.open(png_file)
    rgba = np.array(image.convert("RGBA"))
    return (
        (rgba[:, :, 0] * 256**3)
        + (rgba[:, :, 1] * 256**2)
        + (rgba[:, :, 2] * 256)
        + rgba[:, :, 3]
    )


def int2png(val, png_file):
    """Convert int array to png"""
    # Convert index integers to RGBA values
    rgba = np.zeros((256, 256, 4), "uint8")
    r = (val // 256**3) % 256
    g = (val // 256**2) % 256
    b = (val // 256) % 256
    a = val % 256
    rgba[:, :, 0] = r.flatten()
    rgba[:, :, 1] = g.flatten()
    rgba[:, :, 2] = b.flatten()
    rgba[:, :, 3] = a.flatten()
    # Create PIL Image from RGB values and save as PNG
    img = Image.fromarray(rgba)
    img.save(png_file)


# def int2rgba(int_val):
#     """Convert int to rgba tuple"""
#     r = (int_val // 256**3) % 256
#     g = (int_val // 256**2) % 256
#     b = (int_val // 256) % 256
#     a = int_val % 256
#     return (r, g, b, a)

### Util functions


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_files(src):
    file_list = []
    full_list = glob.glob(src)
    for item in full_list:
        if os.path.isfile(item):
            file_list.append(item)
    return file_list


def list_folders(src):
    folder_list = []
    full_list = glob.glob(src)
    for item in full_list:
        if os.path.isdir(item):
            folder_list.append(item)
    return folder_list


def interp2(x0, y0, z0, x1, y1):
    f = RegularGridInterpolator((y0, x0), z0, bounds_error=False, fill_value=np.nan)
    # reshape x1 and y1
    sz = x1.shape
    x1 = x1.reshape(sz[0] * sz[1])
    y1 = y1.reshape(sz[0] * sz[1])
    # interpolate
    z1 = f((y1, x1)).reshape(sz)
    return z1


def get_zoom_level(npixels, lon_range, lat_range, max_zoom):
    # Get required zoom level
    # lat = np.pi * (lat_range[0] + lat_range[1]) / 360
    # dx = (lon_range[1] - lon_range[0]) * 111111 * np.cos(lat) / npixels[0]
    dxr = (lat_range[1] - lat_range[0]) * 111111 / npixels[1]
    # dxr = min(dx, dy)
    # Make numpy array with pixel size in meters for all zoom levels
    dxy = 156543.03 / 2 ** np.arange(max_zoom + 1)
    # Find zoom level that provides sufficient pixels
    izoom = np.where(dxy < dxr)[0]
    if len(izoom) == 0:
        izoom = max_zoom
    else:
        izoom = izoom[0]
    return izoom
