# Flood map overlay new format
import os

import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
from PIL import Image

from .utils import (
    deg2num,
    get_zoom_level,
    list_folders,
    num2deg,
    png2elevation,
    png2int,
)


def make_floodmap_overlay(
    valg,
    index_path,
    topo_path,
    lon_range,
    lat_range,
    npixels=800,
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
        levs = list_folders(os.path.join(index_path, "*"), basename=True)
        for lev in levs:
            max_zoom = max(max_zoom, int(lev))

        izoom = get_zoom_level(npixels, lat_range, max_zoom)

        ix0, it0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, it1 = deg2num(lat_range[0], lon_range[1], izoom)

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
        return None, None


def make_data_overlay(
    valg,
    index_path,
    lon_range,
    lat_range,
    npixels=800,
    color_values=None,
    color_map="jet",
    caxis=None,
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
    cmap = cm.get_cmap(color_map)

    try:
        if isinstance(valg, list):
            # Why would this ever be a list ?!
            print("valg is a list!")
            pass
        else:
            valg = valg.transpose().flatten()
        # Add dummy nan at the end of valg
        valg = np.append(valg, np.nan)

        if not caxis:
            caxis = []
            caxis.append(np.nanmin(valg))
            caxis.append(np.nanmax(valg))

        # Check available levels in index tiles
        max_zoom = 0
        levs = list_folders(os.path.join(index_path, "*"), basename=True)
        for lev in levs:
            max_zoom = max(max_zoom, int(lev))

        izoom = get_zoom_level(npixels, lat_range, max_zoom)

        ix0, it0 = deg2num(lat_range[1], lon_range[0], izoom)
        ix1, it1 = deg2num(lat_range[0], lon_range[1], izoom)

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

                ind = png2int(index_file, np.size(valg) - 1)
                valt = valg[ind]

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
            im = Image.fromarray(cmap(zz, bytes=True))

        if file_name:
            im.save(file_name)

        lat1, lon0 = num2deg(ix0, it0, izoom)  # lat/lon coordinates of upper left cell
        lat0, lon1 = num2deg(ix1 + 1, it1 + 1, izoom)

        return [lon0, lon1], [lat0, lat1], [caxis[0], caxis[1]]

    except Exception as e:
        print(e)
        return None, None


# Topo overlay new format
def make_topobathy_overlay(
    topo_path,
    lon_range,
    lat_range,
    npixels=800,
    color_values=None,
    color_map="jet",
    color_range=[-10.0, 10.0],
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
        levs = list_folders(os.path.join(topo_path, "*"), basename=True)
        for lev in levs:
            max_zoom = max(max_zoom, int(lev))

        izoom = get_zoom_level(npixels, lat_range, max_zoom)

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

            cmap = cm.get_cmap(color_map)

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
        return None, None, None


# Topo overlay new format
def make_overlay(
    lon_range,
    lat_range,
    option="val",
    val=None,
    topo_path="",
    index_path="",
    npixels=800,
    color_values=None,
    color_map="jet",
    color_range=[-100.0, 100.0],
    color_scale_auto=False,
    color_scale_symmetric=False,
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

    if option == "val":
        # Make sure val is not None
        if val is None:
            raise ValueError("Error! Please provide a value for val.")
        max_zoom = 0
        levs = list_folders(os.path.join(index_path, "*"), basename=True)
        for lev in levs:
            max_zoom = max(max_zoom, int(lev))

    elif option == "topo":
        # Make sure topo_path is not None
        if topo_path is None:
            raise ValueError("Error! Please provide topo_path.")
        max_zoom = 0
        levs = list_folders(os.path.join(topo_path, "*"), basename=True)
        for lev in levs:
            max_zoom = max(max_zoom, int(lev))

    else:
        # Must be floodmap
        if topo_path is None:
            raise ValueError("Error! Please provide topo_path.")
        if val is None:
            raise ValueError("Error! Please provide a value for the water level.")
        max_zoom_1 = 0
        levs = list_folders(os.path.join(index_path, "*"), basename=True)
        for lev in levs:
            max_zoom_1 = max(max_zoom_1, int(lev))
        max_zoom_2 = 0
        levs = list_folders(os.path.join(topo_path, "*"), basename=True)
        for lev in levs:
            max_zoom_2 = max(max_zoom_2, int(lev))
        max_zoom = min(max_zoom_1, max_zoom_2)

    if option != "topo":
        # Flatten matrix
        val = val.transpose().flatten()
        # Add dummy nan at the end of val
        val = np.append(val, np.nan)

    # Get zoom level
    izoom = get_zoom_level(npixels, lat_range, max_zoom)

    # Get tile indices that need to be fetched
    ix0, it0 = deg2num(lat_range[1], lon_range[0], izoom)
    ix1, it1 = deg2num(lat_range[0], lon_range[1], izoom)

    # Number of pixels in x and y direction
    nx = (ix1 - ix0 + 1) * 256
    ny = (it1 - it0 + 1) * 256

    # Make empty array to store the values
    zz = np.empty((ny, nx))
    zz[:] = np.nan

    if not quiet:
        print("Processing zoom level " + str(izoom))

    # Loop over x indices
    for i in range(ix0, ix1 + 1):
        ifolder = str(i)

        # Loop over y indices
        for j in range(it0, it1 + 1):
            ii0 = (i - ix0) * 256
            ii1 = ii0 + 256
            jj0 = (j - it0) * 256
            jj1 = jj0 + 256

            if option == "flood_map" or option == "topo":
                # Get topo bathy for this tile
                bathy_file = os.path.join(
                    topo_path, str(izoom), ifolder, str(j) + ".png"
                )
                if not os.path.exists(bathy_file):
                    # No bathy for this tile, continue
                    continue
                zb = png2elevation(bathy_file)

            if option == "flood_map":
                # Get indices for this tile
                index_file = os.path.join(
                    index_path, str(izoom), ifolder, str(j) + ".png"
                )
                if not os.path.exists(index_file):
                    # No indices for this tile, continue
                    continue
                ind = png2int(index_file, np.size(val) - 1)
                valt = val[ind] - zb  # water depth
                valt[valt < 0.05] = np.nan  # should make this configurable

            elif option == "topo":
                # Get topo for this tile
                valt = zb

            else:
                # Get val for this tile
                index_file = os.path.join(
                    index_path, str(izoom), ifolder, str(j) + ".png"
                )
                if not os.path.exists(index_file):
                    continue
                ind = png2int(index_file, np.size(val) - 1)
                valt = val[ind]

            zz[jj0:jj1, ii0:ii1] = valt

    # Okay, we have the matrix zz with the values. Time to make the bitmap image

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

        cmap = cm.get_cmap(color_map)

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
    return im, [lon0, lon1], [lat0, lat1], [c0, c1]

    # except Exception as e:
    #     print(e)
    #     return None, None, None
