"""Tiled web map creation and management for elevation, flood, and RGBA tile datasets.

Provides the TiledWebMap class for generating, reading, uploading, and managing
slippy-map tile pyramids with support for various encoders (terrarium, mapbox)
and tile types (elevation data, RGBA imagery, index tiles).
"""

import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import boto3
import numpy as np
import toml
import xarray as xr
from botocore import UNSIGNED
from botocore.client import Config

from cht_tiling.data_tiles import make_data_tiles
from cht_tiling.index_tiles import make_index_tiles
from cht_tiling.rgba_tiles import make_rgba_tiles
from cht_tiling.topobathy_tiles import (
    make_topobathy_tiles_lower_levels,
    make_topobathy_tiles_top_level,
)
from cht_tiling.utils import (
    get_zoom_level_for_resolution,
    list_files,
    list_folders,
    num2xy,
    png2elevation,
    xy2num,
)
from cht_tiling.webviewer import write_html
import logging

logger = logging.getLogger(__name__)

class ZoomLevel:
    """Container for tile availability data at a single zoom level.

    Attributes
    ----------
    ntiles : int
        Number of tiles along one axis at this zoom level.
    ij_available : np.ndarray | None
        Flattened array of available tile indices (i * ntiles + j).
    """

    def __init__(self) -> None:
        self.ntiles: int = 0
        self.ij_available: np.ndarray | None = None


class TiledWebMap:
    """Manages a slippy-map tile pyramid for elevation, RGBA, or index data.

    Supports generating tile pyramids from raster data, reading existing
    tile sets, downloading tiles from S3, and uploading to S3. Tile types
    include terrarium/mapbox-encoded elevation, RGBA imagery, and integer
    index tiles.

    Parameters
    ----------
    path : str | Path
        Root directory for the tile pyramid.
    data : object | None
        Input data for tile generation. Type depends on tile type.
    type : str
        Tile type: ``"rgba"`` or ``"data"``.
    parameter : str
        Data parameter: ``"elevation"``, ``"index"``, ``"indices"``, or other.
    encoder : str
        Elevation encoder: ``"terrarium"`` or ``"mapbox"``.
    encoder_vmin : float | None
        Minimum value for custom encoder range.
    encoder_vmax : float | None
        Maximum value for custom encoder range.
    name : str
        Short name of the dataset.
    long_name : str
        Human-readable dataset description.
    url : str | None
        Base URL for the tile service.
    npix : int
        Pixels per tile edge (default 256).
    max_zoom : int
        Maximum zoom level.
    s3_client : object | None
        Pre-configured boto3 S3 client.
    s3_bucket : str | None
        S3 bucket name for downloads/uploads.
    s3_key : str | None
        S3 key prefix for tile storage.
    s3_region : str | None
        AWS region for S3 access.
    source : str
        Data source identifier.
    vertical_reference_level : str
        Vertical datum (e.g. ``"MSL"``).
    vertical_units : str
        Vertical units (e.g. ``"m"``).
    difference_with_msl : float
        Offset from MSL in vertical units.
    index_path : str | None
        Path to pre-existing index tiles.
    topo_path : str | None
        Path to pre-existing topobathy tiles.
    zoom_range : list[int] | None
        Two-element list ``[min_zoom, max_zoom]``.
    color_values : list[dict] | None
        Discrete color definitions for RGBA tiles.
    caxis : list[float] | None
        Color axis range ``[vmin, vmax]``.
    zbmax : float
        Maximum bed level for flood masking.
    minimum_depth : float
        Minimum water depth threshold.
    data_catalog : object | None
        HydroMT data catalog instance.
    lon_range : list[float] | None
        Longitude range ``[lon_min, lon_max]``.
    lat_range : list[float] | None
        Latitude range ``[lat_min, lat_max]``.
    z_range : list[float]
        Valid elevation range ``[z_min, z_max]``.
    dx_max_zoom : float | None
        Resolution in metres for the highest zoom level.
    write_metadata : bool
        Whether to write a metadata TOML file.
    write_availability : bool
        Whether to write an availability NetCDF file.
    make_lower_levels : bool
        Whether to generate lower zoom levels by downsampling.
    make_highest_level : bool
        Whether to generate the highest zoom level tiles.
    skip_existing : bool
        Whether to skip tiles that already exist on disk.
    make_webviewer : bool
        Whether to generate an HTML viewer page.
    merge : bool
        Whether to merge new tiles with existing ones.
    parallel : bool
        Whether to use parallel processing.
    interpolation_method : str
        Interpolation method for resampling (e.g. ``"linear"``).
    quiet : bool
        Whether to suppress progress output.
    """

    def __init__(
        self,
        path: str | Path,
        data: object = None,
        type: str = "rgba",
        parameter: str = "other",
        encoder: str = "terrarium",
        encoder_vmin: float | None = None,
        encoder_vmax: float | None = None,
        name: str = "unknown",
        long_name: str = "Unknown dataset",
        url: str | None = None,
        npix: int = 256,
        max_zoom: int = 0,
        s3_client: object = None,
        s3_bucket: str | None = None,
        s3_key: str | None = None,
        s3_region: str | None = None,
        source: str = "unknown",
        vertical_reference_level: str = "MSL",
        vertical_units: str = "m",
        difference_with_msl: float = 0.0,
        index_path: str | None = None,
        topo_path: str | None = None,
        zoom_range: list[int] | None = None,
        color_values: list[dict] | None = None,
        caxis: list[float] | None = None,
        zbmax: float = 0.0,
        minimum_depth: float = 0.05,
        data_catalog: object = None,
        lon_range: list[float] | None = None,
        lat_range: list[float] | None = None,
        z_range: list[float] = [-999999.0, 999999.0],
        dx_max_zoom: float | None = None,
        write_metadata: bool = False,
        write_availability: bool = True,
        make_lower_levels: bool = True,
        make_highest_level: bool = True,
        skip_existing: bool = False,
        make_webviewer: bool = True,
        merge: bool = True,
        parallel: bool = True,
        interpolation_method: str = "linear",
        quiet: bool = False,
    ) -> None:
        # Set all keyword/value pairs as attributes
        for key, value in locals().items():
            if key in ("self", "path"):
                continue
            setattr(self, key, value)

        self.path = str(path)

        self.read_metadata()
        # Check if available_tiles.nc exists. If not, just read the folders to get the zoom range.
        nc_file = os.path.join(self.path, "available_tiles.nc")
        self.availability_loaded = False
        if os.path.exists(nc_file):
            self.availability_exists = True
        else:
            self.availability_exists = False
        if (
            self.s3_bucket is not None
            and self.s3_key is not None
            and self.s3_region is not None
        ):
            self.download = True
        else:
            self.download = False

    def get_data(
        self,
        xl: list[float],
        yl: list[float],
        max_pixel_size: float,
        crs: object = None,
        waitbox: object = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve elevation data for a bounding box at a given resolution.

        Reads tiles from disk, downloading missing ones from S3 if configured.
        Coordinates are in EPSG:3857.

        Parameters
        ----------
        xl : list[float]
            X extent ``[x_min, x_max]`` in EPSG:3857.
        yl : list[float]
            Y extent ``[y_min, y_max]`` in EPSG:3857.
        max_pixel_size : float
            Maximum pixel size in metres, used to select the zoom level.
        crs : object, optional
            Coordinate reference system (currently unused).
        waitbox : object, optional
            Callable that returns a progress dialog with a ``.close()`` method.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(x, y, z)`` arrays where *x* and *y* are 1-D coordinate arrays
            in EPSG:3857 and *z* is the 2-D elevation grid.
        """
        # Check if availability matrix exists but has not been loaded
        if self.availability_exists and not self.availability_loaded:
            self.read_availability()
            self.availability_loaded = True

        # Determine zoom level
        izoom = get_zoom_level_for_resolution(max_pixel_size)
        izoom = min(izoom, self.max_zoom)

        # Determine the indices of required tiles
        ix0, iy0 = xy2num(xl[0], yl[1], izoom)
        ix1, iy1 = xy2num(xl[1], yl[0], izoom)

        # Make sure indices are within bounds
        ix0 = max(0, ix0)
        iy0 = max(0, iy0)
        iy1 = min(2**izoom - 1, iy1)

        # Create empty array
        nx = (ix1 - ix0 + 1) * 256
        ny = (iy1 - iy0 + 1) * 256
        z = np.empty((ny, nx))
        z[:] = np.nan

        # First try to download missing tiles (it's faster if we can do this in parallel)
        download_file_list = []
        download_key_list = []

        if self.download:
            for i in range(ix0, ix1 + 1):
                itile = np.mod(i, 2**izoom)  # wrap around
                ifolder = str(itile)
                for j in range(iy0, iy1 + 1):
                    png_file = os.path.join(self.path, str(izoom), ifolder, f"{j}.png")
                    if not os.path.exists(png_file):
                        # File does not yet exist
                        if self.availability_exists:
                            # Check availability of the tile in matrix.
                            if not self.check_availability(i, j, izoom):
                                # Tile is also not available for download
                                continue
                        # Add to download_list
                        download_file_list.append(png_file)
                        download_key_list.append(
                            f"{self.s3_key}/{izoom}/{ifolder}/{j}.png"
                        )
                        # Make sure the folder exists
                        if not Path(png_file).parent.exists():
                            Path(png_file).parent.mkdir(parents=True, exist_ok=True)

            # Now download the missing tiles
            if len(download_file_list) > 0:
                if waitbox is not None:
                    wb = waitbox("Downloading topography tiles ...")
                # make boto s
                if self.s3_client is None:
                    self.s3_client = boto3.client(
                        "s3", config=Config(signature_version=UNSIGNED)
                    )
                # Download missing tiles
                with ThreadPool() as pool:
                    pool.starmap(
                        self.download_tile_parallel,
                        [
                            (self.s3_bucket, key, file)
                            for key, file in zip(download_key_list, download_file_list)
                        ],
                    )
                if waitbox is not None:
                    wb.close()

        # Loop over required tiles
        for i in range(ix0, ix1 + 1):
            itile = np.mod(i, 2**izoom)  # wrap around
            ifolder = str(itile)
            for j in range(iy0, iy1 + 1):
                png_file = os.path.join(self.path, str(izoom), ifolder, f"{j}.png")

                if not os.path.exists(png_file):
                    continue

                # Read the png file
                valt = png2elevation(
                    png_file,
                    encoder=self.encoder,
                    encoder_vmin=self.encoder_vmin,
                    encoder_vmax=self.encoder_vmax,
                )

                # Fill array
                ii0 = (i - ix0) * 256
                ii1 = ii0 + 256
                jj0 = (j - iy0) * 256
                jj1 = jj0 + 256
                z[jj0:jj1, ii0:ii1] = valt

        # Compute x and y coordinates
        x0, y0 = num2xy(ix0, iy1 + 1, izoom)  # lower left
        x1, y1 = num2xy(ix1 + 1, iy0, izoom)  # upper right
        # Data is stored in centres of pixels so we need to shift the coordinates
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        x = np.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, nx)
        y = np.linspace(y0 + 0.5 * dy, y1 - 0.5 * dy, ny)
        z = np.flipud(z)

        return x, y, z

    def make(self) -> None:
        """Generate the tile pyramid based on the configured type and parameter."""
        if self.zoom_range is None:
            self.set_zoom_range()

        if self.type == "data":
            if self.parameter == "elevation":
                # Elevation tiles (default terrarium encoding)
                # self.data is a list of dicts (for bathymetry database)
                if self.make_highest_level:
                    # Now loop through datasets in data_list (if zoom range is none and there is an index_path, it is set here)
                    for idata, data_dict in enumerate(self.data):
                        logger.info(
                            f"Processing {data_dict['name']} ... ({idata + 1} of {len(self.data)})"
                        )
                        make_topobathy_tiles_top_level(
                            self,
                            data_dict,
                        )
                if self.make_lower_levels:
                    make_topobathy_tiles_lower_levels(self)
                # For anything but global datasets (that have every tile), make an availability file
                if self.write_availability:
                    self.write_availability_file()
                if self.write_metadata:
                    self.write_metadata_file()

            elif self.parameter == "index" or self.parameter == "indices":
                # Index tiles (int32 encoding)
                # self.data is xugrid
                make_index_tiles(self, topo_path=self.topo_path)

            else:
                # Other data tiles (default float32 encoding)
                # self.data is 1D numpy array, index_path must be provided
                make_data_tiles(self)

        else:
            # RGB tiles (self.parameter can be flood map, topography, or any other string)
            # self.data is 1D numpy array, index_path must be provided
            make_rgba_tiles(self)

        if self.make_webviewer:
            write_html(
                os.path.join(self.path, "index.html"),
                max_native_zoom=self.zoom_range[1],
            )

    def read_metadata(self) -> None:
        """Read metadata from the TOML file in the tile directory, if present."""
        tml_file = os.path.join(self.path, "metadata.tml")
        if os.path.exists(tml_file):
            tml = toml.load(tml_file)
            for key in tml:
                setattr(self, key, tml[key])

    def read_availability(self) -> None:
        """Load the tile availability matrix from ``available_tiles.nc``."""
        nc_file = os.path.join(self.path, "available_tiles.nc")

        with xr.open_dataset(nc_file) as ds:
            self.zoom_levels = []
            # Loop through zoom levels
            for izoom in range(self.max_zoom + 1):
                n = 2**izoom
                iname = f"i_available_{izoom}"
                jname = f"j_available_{izoom}"
                iav = ds[iname].to_numpy()[:]
                jav = ds[jname].to_numpy()[:]
                zoom_level = ZoomLevel()
                zoom_level.ntiles = n
                zoom_level.ij_available = iav * n + jav
                self.zoom_levels.append(zoom_level)

    def set_zoom_range(self) -> None:
        """Determine the zoom range from index tiles, dx, or a default."""
        if self.index_path is None:
            # No index path either
            if self.dx_max_zoom is None:
                # And no dx_max_zoom!
                # Need to determine dx_max_zoom from all the datasets
                # Loop through datasets in datalist to determine dxmin in metres
                self.dx_max_zoom = 10.0
            # Find appropriate zoom level
            zoom_max = get_zoom_level_for_resolution(self.dx_max_zoom)
        else:
            # Index path is provided, so we can determine the max zoom from that
            zoom_levels = list_folders(
                os.path.join(self.index_path, "*"), basename=True
            )
            if zoom_levels:
                # zoom_levels is a list of strings, convert to int and sort
                zoom_levels = [int(z) for z in zoom_levels]
                zoom_levels.sort()
                zoom_max = zoom_levels[-1]
        self.zoom_range = [0, zoom_max]

    def check_availability(self, i: int, j: int, izoom: int) -> bool:
        """Check whether a tile is available at the given zoom level.

        Parameters
        ----------
        i : int
            Tile column index.
        j : int
            Tile row index.
        izoom : int
            Zoom level.

        Returns
        -------
        bool
            True if the tile is available.
        """
        zoom_level = self.zoom_levels[izoom]
        ij = i * zoom_level.ntiles + j
        # Use numpy array for fast search
        available = np.isin(ij, zoom_level.ij_available)
        return available

    def download_tile(self, i: int, j: int, izoom: int) -> bool:
        """Download a single tile from S3.

        Parameters
        ----------
        i : int
            Tile column index.
        j : int
            Tile row index.
        izoom : int
            Zoom level.

        Returns
        -------
        bool
            True if the download succeeded.
        """
        key = f"{self.s3_key}/{izoom}/{i}/{j}.png"
        filename = os.path.join(self.path, str(izoom), str(i), f"{j}.png")
        try:
            self.s3_client.download_file(
                Bucket=self.bucket,
                Key=key,
                Filename=filename,
            )
            logger.info(f"Downloaded {key}")
            okay = True
        except Exception:
            logger.error(f"Failed to download {key}")
            okay = False
        return okay

    def download_tile_parallel(self, bucket: str, key: str, file: str) -> bool:
        """Download a single tile from S3 (for use with ThreadPool).

        Parameters
        ----------
        bucket : str
            S3 bucket name.
        key : str
            S3 object key.
        file : str
            Local file path to save the tile.

        Returns
        -------
        bool
            True if the download succeeded.
        """
        try:
            # Make sure the folder exists
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

            self.s3_client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=file,
            )
            logger.info(f"Downloaded {key}")
            okay = True

        except Exception as e:
            logger.exception(e)
            logger.error(f"Failed to download {key}")
            okay = False

        return okay

    def upload(
        self,
        bucket_name: str,
        s3_folder: str,
        access_key: str,
        secret_key: str,
        region: str,
        parallel: bool = True,
        quiet: bool = True,
    ) -> None:
        """Upload the tile pyramid to an S3 bucket.

        Parameters
        ----------
        bucket_name : str
            Target S3 bucket name.
        s3_folder : str
            S3 folder prefix.
        access_key : str
            AWS access key.
        secret_key : str
            AWS secret key.
        region : str
            AWS region.
        parallel : bool
            Whether to upload files in parallel.
        quiet : bool
            Whether to suppress progress output.
        """
        from cht_utils.remote.s3 import S3Session

        try:
            s3 = S3Session(access_key, secret_key, region)
            s3.upload_folder(
                bucket_name,
                self.path,
                f"{s3_folder}/{self.name}",
                parallel=parallel,
                quiet=quiet,
            )
        except Exception:
            logger.error("An error occurred while uploading !")

    def write_availability_file(self) -> None:
        """Write tile availability to ``available_tiles.nc``."""
        ds = xr.Dataset()
        zoom_level_paths = list_folders(os.path.join(self.path, "*"), basename=True)
        zoom_levels = [int(z) for z in zoom_level_paths]
        zoom_levels.sort()
        for izoom in zoom_levels:
            n = 0
            iav = []
            jav = []
            i_paths = list_folders(
                os.path.join(self.path, str(izoom), "*"), basename=True
            )
            for ipath in i_paths:
                i = int(ipath)
                j_paths = list_files(os.path.join(self.path, str(izoom), ipath, "*"))
                for jpath in j_paths:
                    pngfile = os.path.basename(jpath)
                    j = int(pngfile.split(".")[0])
                    iav.append(i)
                    jav.append(j)
                    n += 1
            iav = np.array(iav)
            jav = np.array(jav)
            dimname = f"n_{izoom}"
            iname = f"i_available_{izoom}"
            jname = f"j_available_{izoom}"
            ds[iname] = xr.DataArray(iav, dims=dimname)
            ds[jname] = xr.DataArray(jav, dims=dimname)

        nc_file = os.path.join(self.path, "available_tiles.nc")
        ds.to_netcdf(nc_file)

        ds.close()

    def write_metadata_file(self) -> None:
        """Write dataset metadata to ``metadata.tml``."""
        metadata = {}

        metadata["longname"] = self.long_name
        metadata["format"] = "tiled_web_map"
        metadata["encoder"] = self.encoder
        if self.encoder_vmin is not None:
            metadata["encoder_vmin"] = self.encoder_vmin
        if self.encoder_vmax is not None:
            metadata["encoder_vmax"] = self.encoder_vmax
        metadata["url"] = self.url
        if self.s3_bucket is not None:
            metadata["s3_bucket"] = self.s3_bucket
        if self.s3_key is not None:
            metadata["s3_key"] = self.s3_key
        if self.s3_region is not None:
            metadata["s3_region"] = self.s3_region
        metadata["max_zoom"] = self.max_zoom
        metadata["interpolation_method"] = "linear"
        metadata["source"] = self.source
        metadata["coord_ref_sys_name"] = "WGS 84 / Pseudo-Mercator"
        metadata["coord_ref_sys_kind"] = "projected"
        metadata["vertical_reference_level"] = self.vertical_reference_level
        metadata["vertical_units"] = self.vertical_units
        metadata["difference_with_msl"] = self.difference_with_msl
        metadata["available_tiles"] = True

        metadata["description"] = {}
        metadata["description"]["title"] = "LONG_NAME"
        metadata["description"]["institution"] = "INST_NAME"
        metadata["description"]["history"] = "created by : AUTHOR"
        metadata["description"]["references"] = "No reference material available"
        metadata["description"]["comment"] = "none"
        metadata["description"]["email"] = "Your email here"
        metadata["description"]["version"] = "1.0"
        metadata["description"]["terms_for_use"] = "Use as you like"
        metadata["description"]["disclaimer"] = (
            "These data are made available in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."
        )

        toml_file = os.path.join(self.path, "metadata.tml")
        with open(toml_file, "w") as f:
            toml.dump(metadata, f)
