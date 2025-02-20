import os
import glob
import time
from multiprocessing.pool import ThreadPool
import numpy as np
import xarray as xr
import rioxarray
import toml
from rasterio.transform import from_origin


from matplotlib import cm, colormaps
from matplotlib.colors import LightSource
from PIL import Image
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

import cht_tiling.fileops as fo
from cht_tiling.utils import png2elevation, png2int

def make_flood_map_tiles(
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
    fo.mkdir(png_zoom_path)

    for ifolder in fo.list_folders(os.path.join(index_zoom_path, "*")):
        path_okay = False
        ifolder = os.path.basename(ifolder)
        index_zoom_path_i = os.path.join(index_zoom_path, ifolder)
        png_zoom_path_i = os.path.join(png_zoom_path, ifolder)

        for jfile in fo.list_files(os.path.join(index_zoom_path_i, "*.png")):
            jfile = os.path.basename(jfile)
            j = int(jfile[:-4])

            index_file = os.path.join(index_zoom_path_i, jfile)
            png_file = os.path.join(png_zoom_path_i, str(j) + ".png")

            ind = png2int(index_file, -1)
            ind = ind.flatten()

            if option == "probabilistic":
                # valg is actually CDF interpolator to obtain probability of water level

                # Read bathy
                bathy_file = os.path.join(
                    topo_path, str(izoom), ifolder, str(j) + ".png"
                )
                if not os.path.exists(bathy_file):
                    # No bathy for this tile, continue
                    continue
                # zb = np.fromfile(bathy_file, dtype="f4")
                zb = png2elevation(bathy_file).flatten()
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
                # zb = np.fromfile(bathy_file, dtype="f4")
                zb = png2elevation(bathy_file).flatten()

                noval = np.where(ind < 0)
                ind[ind < 0] = 0
                valt = valg[ind]

                # # Get the variance of zb
                # zbvar = np.var(zb)
                # zbmn = np.min(zb)
                # zbmx = np.max(zb)
                # # If there is not a lot of change in bathymetry, set zb to mean of zb
                # # Should try to compute a slope here
                # if zbmx - zbmn < 5.0:
                #     zb = np.full_like(zb, np.mean(zb))

                valt = valt - zb           # depth = water level - topography
                valt[valt < 0.10] = np.nan # 0.10 is the threshold for water level
                valt[zb < zbmax] = np.nan  # don't show flood in water areas
                valt[noval] = np.nan       # don't show flood outside model domain  

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
                # rgb = np.flip(rgb, axis=0)
                im = Image.fromarray(rgb)

            else:
#                valt = np.flipud(valt.reshape([256, 256]))
                valt = valt.reshape([256, 256])
                valt = (valt - caxis[0]) / (caxis[1] - caxis[0])
                valt[valt < 0.0] = 0.0
                valt[valt > 1.0] = 1.0
                im = Image.fromarray(cm.jet(valt, bytes=True))

            if not path_okay:
                if not os.path.exists(png_zoom_path_i):
                    fo.mkdir(png_zoom_path_i)
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
        fo.mkdir(png_zoom_path)

        for ifolder in fo.list_folders(os.path.join(index_zoom_path, "*")):
            path_okay = False
            ifolder = os.path.basename(ifolder)
            i = int(ifolder)
            index_zoom_path_i = os.path.join(index_zoom_path, ifolder)
            png_zoom_path_i = os.path.join(png_zoom_path, ifolder)

            for jfile in fo.list_files(os.path.join(index_zoom_path_i, "*.png")):

                jfile = os.path.basename(jfile)
                j = int(jfile[:-4])

                png_file = os.path.join(png_zoom_path_i, str(j) + ".png")

                rgb = np.zeros((256, 256, 4), "uint8")

                i0 = i * 2
                i1 = i * 2 + 1
                j0 = j * 2 + 1
                j1 = j * 2

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
                            fo.mkdir(png_zoom_path_i)
                            path_okay = True

                    if os.path.exists(png_file):
                        # This tile already exists
                        if merge:
                            im0 = Image.open(png_file)
                            rgb = np.array(im)
                            rgb0 = np.array(im0)
                            isum = np.sum(rgb, axis=2)
                            rgb[isum == 0, :] = rgb0[isum == 0, :]
                            im = Image.fromarray(rgb)
                    #                        im.show()

                    im.save(png_file)
