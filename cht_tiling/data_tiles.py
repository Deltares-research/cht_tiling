"""Generate data tiles by mapping grid values through pre-computed index tiles."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from cht_tiling.utils import elevation2png, list_files, list_folders, makedir, png2int

logger = logging.getLogger(__name__)


def make_data_tiles(twm: Any) -> None:
    """Create data-value PNG tiles for each zoom level using index tiles.

    For every existing index tile, the corresponding grid cell values are looked
    up and encoded into a PNG tile using the configured encoder.

    Parameters
    ----------
    twm : Any
        TiledWebMap instance with attributes ``index_path``, ``data``, ``path``,
        ``zoom_range``, ``caxis``, ``quiet``, ``encoder``, ``encoder_vmin``,
        and ``encoder_vmax``.

    Raises
    ------
    ValueError
        If ``twm.index_path`` is None.
    """
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
            logger.info(f"Processing zoom level {izoom}")

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

                valt = valg[ind]
                valt[ind < 0] = np.nan

                if not path_okay:
                    if not os.path.exists(png_zoom_path_i):
                        makedir(png_zoom_path_i)
                        path_okay = True

                elevation2png(
                    valt,
                    png_file,
                    encoder=twm.encoder,
                    encoder_vmin=twm.encoder_vmin,
                    encoder_vmax=twm.encoder_vmax,
                )
