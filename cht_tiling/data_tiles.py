import math
import os

import numpy as np

from cht_tiling.utils import makedir, list_folders, list_files, png2int, elevation2png


def make_data_tiles(twm):
    # index path MUST be provided
    if twm.index_path is None:
        raise ValueError("index_path must be provided for data tiles")

    valg = twm.data

    valg = valg.transpose().flatten()

    if not twm.caxis:
        twm.caxis = []
        twm.caxis.append(np.nanmin(valg))
        twm.caxis.append(np.nanmax(valg))

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

                valt = valg[ind]
                valt[ind < 0] = np.nan


                if not path_okay:
                    if not os.path.exists(png_zoom_path_i):
                        makedir(png_zoom_path_i)
                        path_okay = True

                # if os.path.exists(png_file):
                #     # This tile already exists
                #     if twm.merge:
                #         im0 = Image.open(png_file)
                #         rgb = np.array(im)
                #         rgb0 = np.array(im0)
                #         isum = np.sum(rgb, axis=2)
                #         rgb[isum == 0, :] = rgb0[isum == 0, :]
                #         im = Image.fromarray(rgb)

                elevation2png(
                    valt,
                    png_file,
                    encoder=twm.encoder,
                    encoder_vmin=twm.encoder_vmin,
                    encoder_vmax=twm.encoder_vmax,
                )

 