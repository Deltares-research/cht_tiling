import os

import numpy as np
import xarray as xr
from matplotlib import cm
from PIL import Image

import cht_tiling.fileops as fo
from cht_tiling.utils import deg2num, num2deg, png2elevation, png2int


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

                valt = valt - zb  # depth = water level - topography
                valt[valt < 0.10] = np.nan  # 0.10 is the threshold for water level
                valt[zb < zbmax] = np.nan  # don't show flood in water areas
                valt[noval] = np.nan  # don't show flood outside model domain

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


# Flood map overlay new format
def make_flood_map_overlay_v2(
    valg,
    index_path,
    topo_path,
    zmax_minus_zmin=None,
    mean_depth=None,
    npixels=[1200, 800],
    hmin=0.10,
    dzdx_mild=0.01,
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
        elif isinstance(valg, xr.DataArray):
            valg = valg.to_numpy()
            if mean_depth is not None:
                mean_depth = mean_depth.to_numpy()
            if zmax_minus_zmin is not None:
                zmax_minus_zmin = zmax_minus_zmin.to_numpy()
        else:
            # valg is a 2D array
            valg = valg.transpose().flatten()
            if mean_depth is not None:
                mean_depth = mean_depth.transpose().flatten()
            if zmax_minus_zmin is not None:
                zmax_minus_zmin = zmax_minus_zmin.transpose().flatten()

        if mean_depth is not None and zmax_minus_zmin is not None:
            # Mean depth is obtained from SFINCS as volume over cell area
            # zmax_minus_zmin is the difference between zmax and zmin in the cell
            # Set mean_depth to NaN where zmax_minus_zmin is greater than dzmild
            mean_depth[(zmax_minus_zmin > dzdx_mild)] = np.nan

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

                ind = png2int(index_file, -1)

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

                    valt = valg[ind]  # water level in pixels

                    valt = valt - zb  # water depth in pixels

                    # Now we override pixels values in very mild sloping cells with mean depth
                    if mean_depth is not None:
                        # Compute mean depth as volume over cell area
                        mean_depth_p = mean_depth[ind]
                        # Override valt with mean_depth_p where mean_depth_p is not NaN
                        valt[~np.isnan(mean_depth_p)] = mean_depth_p[
                            ~np.isnan(mean_depth_p)
                        ]

                    valt[valt < hmin] = (
                        np.nan
                    )  # set to nan if water depth is less than hmin
                    valt[zb < zbmax] = np.nan  # don't show flood in water areas

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
            if not caxis:
                caxis = []
                caxis.append(np.nanmin(valg))
                caxis.append(np.nanmax(valg))

            zz = (zz - caxis[0]) / (caxis[1] - caxis[0])
            zz[zz < 0.0] = 0.0
            zz[zz > 1.0] = 1.0
            im = Image.fromarray(cm.jet(zz, bytes=True))
            # # For any nan values, set alpha to 0
            # # Get rgb values
            # rgb = np.array(im)
            # im.putalpha(255 * np.isnan(zz))

        if file_name:
            im.save(file_name)

        lat1, lon0 = num2deg(ix0, it0, izoom)  # lat/lon coordinates of upper left cell
        lat0, lon1 = num2deg(ix1 + 1, it1 + 1, izoom)

        return [lon0, lon1], [lat0, lat1], caxis

    except Exception as e:
        print(e)
        traceback.print_exc()
        return None, None


# def deg2num(lat_deg, lon_deg, zoom):
#     """Returns column and row index of slippy tile"""
#     lat_rad = math.radians(lat_deg)
#     n = 2**zoom
#     xtile = int((lon_deg + 180.0) / 360.0 * n)
#     ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
#     return (xtile, ytile)

# def num2deg(xtile, ytile, zoom):
#     """Returns upper left latitude and longitude of slippy tile"""
#     # Return upper left corner of tile
#     n = 2**zoom
#     lon_deg = xtile / n * 360.0 - 180.0
#     lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
#     lat_deg = math.degrees(lat_rad)
#     return (lat_deg, lon_deg)
