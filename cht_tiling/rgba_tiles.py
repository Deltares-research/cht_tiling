"""Generate RGBA web tiles from model output data using index tile lookups.

Supports flood maps, water level maps, topography, and direct value rendering
with either discrete color ranges or continuous colormap scaling.
"""

import os

import numpy as np
from matplotlib import cm
from PIL import Image

from cht_tiling.utils import list_files, list_folders, makedir, png2elevation, png2int


def make_rgba_tiles(twm: object) -> None:
    """Generate RGBA PNG tiles for a tiled web map.

    Reads index tiles to map model cell indices onto tile pixels, then
    applies the appropriate coloring based on the parameter type (flood map,
    water level, topography, or direct values).

    Parameters
    ----------
    twm : object
        A ``TiledWebMap`` instance with at least the following attributes:
        ``index_path``, ``path``, ``data``, ``parameter``, ``topo_path``,
        ``zoom_range``, ``caxis``, ``color_values``, ``zbmax``,
        ``minimum_depth``, ``merge``, ``quiet``.

    Raises
    ------
    ValueError
        If ``index_path`` is not set, or if ``topo_path`` is missing when
        required by the selected parameter.
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
    elif (
        twm.parameter == "flood_probability_map"
        or twm.parameter == "flood probability map"
    ):
        if twm.topo_path is None:
            raise ValueError(
                "topo_path must be provided for flood probability map tiles"
            )
        option = "flood_probability_map"
    elif (
        twm.parameter == "topography"
        or twm.parameter == "topo"
        or twm.parameter == "elevation"
    ):
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
            print(f"Processing zoom level {izoom}")

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
                png_file = os.path.join(png_zoom_path_i, f"{j}.png")

                ind = png2int(index_file, -1)

                if option == "flood_probability_map":
                    # valg is actually CDF interpolator to obtain
                    # probability of water level
                    pass

                elif option == "water_level":
                    bathy_file = os.path.join(
                        twm.topo_path, str(izoom), ifolder, f"{j}.png"
                    )
                    if not os.path.exists(bathy_file):
                        continue
                    zb = png2elevation(bathy_file)
                    # Create water level map
                    valt = valg[ind]
                    valt[zb > twm.zbmax] = np.nan
                    valt[ind < 0] = np.nan

                elif option == "floodmap":
                    bathy_file = os.path.join(
                        twm.topo_path, str(izoom), ifolder, f"{j}.png"
                    )
                    if not os.path.exists(bathy_file):
                        continue
                    zb = png2elevation(bathy_file)
                    valt = valg[ind]
                    valt = valt - zb
                    valt[valt < twm.minimum_depth] = np.nan
                    valt[zb < twm.zbmax] = np.nan

                elif option == "topography":
                    bathy_file = os.path.join(
                        twm.topo_path, str(izoom), ifolder, f"{j}.png"
                    )
                    if not os.path.exists(bathy_file):
                        continue
                    zb = png2elevation(bathy_file)
                    valt = zb

                else:  # must be "direct"
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
