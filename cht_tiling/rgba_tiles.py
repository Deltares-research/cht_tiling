import os
from PIL import Image
from matplotlib import cm
import numpy as np

from cht_tiling.utils import makedir, list_folders, list_files, png2int, png2elevation

def make_rgba_tiles(twm):

    """
    Generates RGBA web tiles

    :param valg: Name of the scenario to be run.
    :type valg: array
    :param index_path: Path where the index tiles are sitting.
    :type index_path: str
    :param option: Option to define the type of tiles to be generated.
    # Options are
    # 'direct', 'floodmap', 'topography'. Defaults to 'direct',
    # in which case the values
    # in *valg* are used directly.
    :type option: str
    :param zoom_range: Zoom range for
    which the png tiles will be created.
    Defaults to [0, 23].
    :type zoom_range: list of int

    """

    # index path MUST be provided
    if twm.index_path is None:
        raise ValueError("index_path must be provided for data tiles")

    # There are several options for the type of tiles to be generated
    # "floodmap" - make flood map tiles, requires topo_path to be provided
    # "water level" - make water level tiles, if topo_path is provided, pixels with zb>zbmax are set to nan
    # "flood_probability_map" - make flood probability map tiles, requires topo_path to be provided
    # "topography" - make topography tiles, requires topo_path to be provided
    # other - just make tiles from the data provided

    if twm.parameter == "flood_map" or twm.parameter == "floodmap":
        if twm.topo_path is None:
            raise ValueError("topo_path must be provided for flood map tiles")
        option = "floodmap"
    elif twm.parameter == "water_level" or twm.parameter == "water level":
        if twm.topo_path is None:
            raise ValueError("topo_path must be provided for water level tiles")
        option = "water_level"
    elif twm.parameter == "flood_probability_map" or twm.parameter == "flood probability map":
        if twm.topo_path is None:
            raise ValueError("topo_path must be provided for flood probability map tiles")
        option = "flood_probability_map"
    elif twm.parameter == "topography" or twm.parameter == "topo" or twm.parameter == "elevation":
        if twm.topo_path is None:
            raise ValueError("topo_path must be provided for topography tiles")
        option = "topography"
    else:
        option = "direct"

    valg = twm.data

    if isinstance(valg, list):
        pass
    else:
        valg = valg.transpose().flatten()

    # Determine color axis if not provided
    caxis = twm.caxis
    if not caxis:
        caxis = []
        caxis.append(np.nanmin(valg))
        caxis.append(np.nanmax(valg))

    for izoom in range(twm.zoom_range[0], twm.zoom_range[1] + 1):

        if not twm.quiet:
            print("Processing zoom level " + str(izoom))

        index_zoom_path = os.path.join(twm.index_path, str(izoom))

        if not os.path.exists(index_zoom_path):
            continue

        png_zoom_path = os.path.join(twm.path, str(izoom))
        makedir(png_zoom_path)

        for ifolder in list_folders(os.path.join(index_zoom_path, "*")):
            path_okay = False
            ifolder = os.path.basename(ifolder)
            index_zoom_path_i = os.path.join(index_zoom_path, ifolder)
            png_zoom_path_i = os.path.join(png_zoom_path, ifolder)

            for jfile in list_files(os.path.join(index_zoom_path_i, "*.png")):
                jfile = os.path.basename(jfile)
                j = int(jfile[:-4])

                index_file = os.path.join(index_zoom_path_i, jfile)
                png_file = os.path.join(png_zoom_path_i, str(j) + ".png")

                ind = png2int(index_file, -1)

                if option == "flood_probability_map":
                    # valg is actually CDF interpolator to obtain
                    # probability of water level
                    pass

                    # # Read bathy
                    # bathy_file = os.path.join(
                    #     twm.topo_path, str(izoom), ifolder, str(j) + ".png"
                    # )
                    # if not os.path.exists(bathy_file):
                    #     # No bathy for this tile, continue
                    #     continue
                    # zb = np.fromfile(bathy_file, dtype="f4")
                    # zs = zb + depth

                    # valt = valg[ind](zs)
                    # valt[ind < 0] = np.nan

                elif option == "water_level":
                    bathy_file = os.path.join(
                        twm.topo_path, str(izoom), ifolder, str(j) + ".png"
                    )
                    if not os.path.exists(bathy_file):
                        # No bathy for this tile, continue
                        continue
                    zb = png2elevation(bathy_file)
                    # Create water level map
                    valt = valg[ind]
                    valt[zb > twm.zbmax] = np.nan
                    valt[ind < 0] = np.nan

                elif option == "floodmap":
                    bathy_file = os.path.join(
                        twm.topo_path, str(izoom), ifolder, str(j) + ".png"
                    )
                    if not os.path.exists(bathy_file):
                        # No bathy for this tile, continue
                        continue
                    zb = png2elevation(bathy_file)
                    valt = valg[ind]
                    valt = valt - zb
                    valt[valt < twm.minimum_depth] = np.nan
                    valt[zb < twm.zbmax] = np.nan

                elif option == "topography":
                    bathy_file = os.path.join(
                        twm.topo_path, str(izoom), ifolder, str(j) + ".png"
                    )
                    if not os.path.exists(bathy_file):
                        # No bathy for this tile, continue
                        continue
                    zb = png2elevation(bathy_file)
                    valt = zb

                else: # must be "direct"
                    valt = valg[ind]
                    valt[ind < 0] = np.nan

                if twm.color_values:

                    valt = valt.flatten()

                    rgb = np.zeros((256 * 256, 4), "uint8")

                    # Determine value based on user-defined ranges
                    for color_value in twm.color_values:
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
                    im = Image.fromarray(rgb)

                else:
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
                    if twm.merge:
                        im0 = Image.open(png_file)
                        rgb = np.array(im)
                        rgb0 = np.array(im0)
                        isum = np.sum(rgb, axis=2)
                        rgb[isum == 0, :] = rgb0[isum == 0, :]
                        im = Image.fromarray(rgb)

                im.save(png_file)

    