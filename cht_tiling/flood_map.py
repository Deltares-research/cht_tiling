import os
from pathlib import Path
from typing import Union

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image
from pyproj import Transformer
from rasterio.warp import Resampling

import cht_tiling.fileops as fo
from cht_tiling.utils import deg2num, num2deg, png2elevation, png2int


class FloodMap:
    def __init__(
        self,
        topobathy_file: Union[str, Path] = None,
        index_file: Union[str, Path] = None,
        zbmin: float = 0.0,
        zbmax: float = 99999.9,
        hmin: float = 0.1,
        max_pixel_size: float = 0.0,
        data_array_name: str = "water_depth",
        cmap: str = None,
        cmin: float = None,
        cmax: float = None,
        color_values: list = None,
    ):
        """
        Initialize the FloodMap class with optional parameters.

        Parameters:
        ----------
        topobathy_file : Union[str, Path], optional
        Topobathy data file (COG).
        index_file : Union[str, Path], optional
        Indices data file (COG).
        zbmin : float, optional
        Minimum allowable topobathy value.
        zbmax : float, optional
        Maximum allowable topobathy value.
        hmin : float, optional
        Minimum allowable water depth.
        max_pixel_size : float, optional
        Maximum pixel size for the appropriate overview level.
        data_array_name : str, optional
        Name of the data array in the output dataset.
        cmap : str, optional
        Colormap to use.
        cmin : float, optional
        Minimum value for colormap normalization.
        cmax : float, optional
        Maximum value for colormap normalization.
        """
        # self.topobathy_file = topobathy_file
        # self.index_file = index_file
        # self.zb = rasterio.open(self.topobathy_file)
        # self.indices = rasterio.open(self.index_file)
        self.topobathy_file = None
        self.index_file = None
        self.zb = None
        self.indices = None
        self.zbmin = 0.0
        self.zbmax = 99999.9
        self.hmin = 0.1
        self.max_pixel_size = 0.0
        self.data_array_name = "water_depth"
        self.color_values = "default"
        self.cmap = "jet"
        self.cmin = 0.0
        self.cmax = 1.0
        self.discrete_colors = False
        self.ds = xr.Dataset()

        self.legend = {}
        self.legend["title"] = "Flood Depth (m)"
        self.legend["contour"] = []
        # Set default
        self.legend["contour"].append(
            {"color": "#FF0000", "lower_value": 2.0, "text": "2.0+ m"}
        )
        self.legend["contour"].append(
            {
                "color": "#FFA500",
                "lower_value": 1.0,
                "upper_value": 2.0,
                "text": "1.0–2.0 m",
            }
        )
        self.legend["contour"].append(
            {
                "color": "#FFFF00",
                "lower_value": 0.3,
                "upper_value": 1.0,
                "text": "0.3–1.0 m",
            }
        )
        self.legend["contour"].append(
            {
                "color": "#00FF00",
                "lower_value": 0.1,
                "upper_value": 0.3,
                "text": "0.1–0.3 m",
            }
        )

    def set_topobathy_file(self, topobathy_file: Union[str, Path]) -> None:
        """
        Set the topobathy file.

        Parameters:
        ----------
        topobathy_file : Union[str, Path]
            Topobathy data file (COG).
        """
        self.topobathy_file = topobathy_file
        self.zb = rasterio.open(self.topobathy_file)

    def set_index_file(self, index_file: Union[str, Path]) -> None:
        """
        Set the index file.

        Parameters:
        ----------
        index_file : Union[str, Path]
            Indices data file (COG).
        """
        self.index_file = index_file
        self.indices = rasterio.open(self.index_file)

    def close(self) -> None:
        """
        Close the topobathy and index files.
        """
        if self.zb is not None:
            self.zb.close()
        if self.indices is not None:
            self.indices.close()
        self.ds.close()

    def read(self, tiffile) -> None:
        """
        Read geotiff file with flood depth data.
        """
        self.ds = xr.Dataset()
        self.ds["water_depth"] = rioxarray.open_rasterio(tiffile, masked=True).squeeze()

    def set_water_level(self, zs: Union[float, np.ndarray]) -> None:
        """
        Set the water level data.

        Parameters:
        ----------
        zs : np.ndarray
            A 1D numpy array containing water level data for each index.
        """
        self.zs = zs

    def make(
        self,
        max_pixel_size: float = 0.0,
        bbox=None,
    ) -> xr.Dataset:
        """
        Generate a flood map geotiff (COG) or netCDF file from water level data, topobathy data, and indices.

        Parameters:
        ----------
        output_file : str
            Path to the output file. The file extension determines the format (e.g., ".tif" for GeoTIFF, ".nc" for netCDF).
        zbmin : float, optional
            Minimum allowable topobathy value. Values below this will be masked. Default is 0.0.
        zbmax : float, optional
            Maximum allowable topobathy value. Values above this will be masked. Default is 99999.9.
        hmin : float, optional
            Minimum allowable water depth. Values below this will be masked. Default is 0.1.
        max_pixel_size : float, optional
            Maximum pixel size for the appropriate overview level. If 0.0, no overviews are used. Default is 0.0.
        data_array_name : str, optional
            Name of the data array in the output dataset. Default is "water_depth".

        Returns:
        -------
        xr.Dataset
            An xarray Dataset containing the computed flood map.

        Notes:
        -----
        - The function reads and processes topobathy and indices data, applies masks based on the provided thresholds,
        and computes water depth.
        - The output file can be saved as a GeoTIFF or netCDF file depending on the file extension.
        """

        # First get the overview level (assuming zb is a string or path)
        overview_level = 0

        if max_pixel_size > 0.0:
            overview_level = get_appropriate_overview_level(self.zb, max_pixel_size)

        # Read the data at the specified overview level
        if overview_level == 0:
            zb = rioxarray.open_rasterio(self.zb)
        else:
            zb = rioxarray.open_rasterio(self.zb, overview_level=overview_level)
        # Remove band dimension if it is 1, and squeeze the array to 2D
        if "band" in zb.dims and zb.sizes["band"] == 1:
            zb = zb.squeeze(dim="band", drop=True)
        if bbox is not None:
            zb = zb.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])

        # Read the data at the specified overview level
        if overview_level == 0:
            indices = rioxarray.open_rasterio(self.indices)
        else:
            # Read the data at the specified overview level
            indices = rioxarray.open_rasterio(
                self.indices, overview_level=overview_level
            )
        # Remove band dimension if it is 1, and squeeze the array to 2D
        if "band" in indices.dims and indices.sizes["band"] == 1:
            indices = indices.squeeze(dim="band", drop=True)
        if bbox is not None:
            indices = indices.rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3]
            )

        # Get the no_data value from the indices array
        nan_val_indices = indices.attrs["_FillValue"]
        # Set the no_data mask
        no_data_mask = indices == nan_val_indices
        # Turn indices into numpy array and set no_data values to 0
        indices = np.squeeze(indices.to_numpy()[:])
        indices[np.where(indices == nan_val_indices)] = 0

        # Compute water depth
        if isinstance(self.zs, float):
            # If zs is a float, use constant water level
            h = np.full(zb.shape, self.zs) - zb.to_numpy()[:]
        else:
            h = self.zs[indices] - zb.to_numpy()[:]
        # Set water depth to NaN where indices are no data
        h[no_data_mask] = np.nan
        # Set water depth to NaN where it is less than hmin
        h[h < self.hmin] = np.nan
        # Set water depth to NaN where zb is less than zbmin
        h[zb.to_numpy()[:] < self.zbmin] = np.nan
        # Set water depth to NaN where zb is greater than zbmax
        h[zb.to_numpy()[:] > self.zbmax] = np.nan

        # Turn h into a DataArray with the same dimensions as zb
        self.ds = xr.Dataset()
        self.ds[self.data_array_name] = xr.DataArray(
            h, dims=["y", "x"], coords={"y": zb.y, "x": zb.x}
        )
        # Use same spatial_ref as zb
        self.ds[self.data_array_name] = self.ds[self.data_array_name].rio.write_crs(
            zb.rio.crs, inplace=True
        )

    def write(
        self,
        output_file: Union[str, Path] = "",
    ) -> None:
        """
        Write the flood map to a file.

        Parameters:
        ----------
        output_file : Union[str, Path]
            Path to the output file. The file extension determines the format (e.g., ".tif" for GeoTIFF, ".nc" for netCDF).

        Returns:
        -------
        None

        Notes:
        -----
        - If the output file is a NetCDF file (".nc"), the dataset is written directly without applying a colormap.
        - If the output file is a GeoTIFF (".tif") and a colormap is provided, the data is normalized and the colormap is applied.
        - If no colormap is provided for a GeoTIFF, the raw data is written as a binary raster.
        """
        if output_file.endswith(".nc"):
            # Write to netcdf
            self.ds.to_netcdf(output_file)

        elif output_file.endswith(".tif"):
            # Write to geotiff
            if self.cmap is not None:
                # Get RBG data array
                rgb_da = get_rgb_data_array(
                    self.ds[self.data_array_name],
                    color_values=self.color_values,
                    cmap=self.cmap,
                    cmin=self.cmin,
                    cmax=self.cmax,
                )

                # Write to file
                rgb_da.rio.to_raster(
                    output_file,
                    driver="COG",
                    compress="deflate",  # or "lzw"
                    blocksize=512,  # optional tuning
                    overview_resampling="nearest",  # controls how overviews are built
                )

            else:
                # Just write binary data
                self.ds[self.data_array_name].rio.to_raster(
                    output_file,
                    driver="COG",
                    compress="deflate",  # or "lzw"
                    blocksize=512,  # optional tuning
                    overview_resampling="nearest",  # controls how overviews are built
                )

    def map_overlay(self, file_name, xlim=None, ylim=None, width=800):
        """
        Create a map overlay of the flood map using the specified colormap and save it to a png file. The CRS is 3857.
        """

        if self.ds is None:
            return

        try:
            # Get the bounds of the data
            lon_min = xlim[0]
            lat_min = ylim[0]
            lon_max = xlim[1]
            lat_max = ylim[1]

            # Get the bounds of the data in EPSG:3857
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x_min, y_min = transformer.transform(lon_min, lat_min)
            x_max, y_max = transformer.transform(lon_max, lat_max)

            # Get required pixel size
            dxy = (x_max - x_min) / width

            # Get bbox in local crs from xlim and ylim
            bbox = reproject_bbox(
                lon_min,
                lat_min,
                lon_max,
                lat_max,
                crs_src="EPSG:4326",
                crs_dst=self.zb.crs,
                buffer=0.05,
            )

            # Compute water depth and store in self.ds
            self.make(max_pixel_size=dxy, bbox=bbox)

            rgb_da = get_rgb_data_array(
                self.ds[self.data_array_name],
                cmap=self.cmap,
                cmin=self.cmin,
                cmax=self.cmax,
                discrete_colors=self.discrete_colors,
                color_values=self.color_values,
            )

            # Now reproject to EPSG:3857 and create a png file
            rgb_3857 = rgb_da.rio.reproject(
                "EPSG:3857", resampling=Resampling.bilinear, nodata=0
            )  # can also use nearest

            # Apply padding
            rgb_3857 = rgb_3857.rio.pad_box(
                minx=x_min, miny=y_min, maxx=x_max, maxy=y_max, constant_values=0
            )

            # Final clip to exact bbox
            rgb_crop = rgb_3857.rio.clip_box(
                minx=x_min, miny=y_min, maxx=x_max, maxy=y_max
            )

            # Convert to numpy array and transpose to (y, x, band)
            rgba = rgb_crop.transpose("y", "x", "band").to_numpy().astype("uint8")

            plt.imsave(file_name, rgba)

            # Need to set the legend dict (used in Guitares map layer)

            if self.discrete_colors:
                self.legend = {}
                self.legend["title"] = "Flood Depth (m)"
                self.legend["contour"] = []

                if isinstance(self.color_values, str):
                    # Use default
                    color_values = []
                    color_values.append(
                        {"color": "lightgreen", "lower_value": 0.1, "upper_value": 0.3}
                    )
                    color_values.append(
                        {"color": "yellow", "lower_value": 0.3, "upper_value": 1.0}
                    )
                    color_values.append(
                        {"color": "#FFA500", "lower_value": 1.0, "upper_value": 2.0}
                    )
                    color_values.append({"color": "red", "lower_value": 2.0})
                else:
                    color_values = self.color_values    

                for cv in color_values:
                    legend_item = {}
                    legend_item["color"] = cv["color"]
                    if "upper_value" in cv and "lower_value" in cv:
                        legend_item["lower_value"] = cv["lower_value"]
                        legend_item["upper_value"] = cv["upper_value"]
                        legend_item["text"] = f"{cv['lower_value']}–{cv['upper_value']} m"
                    elif "upper_value" in cv:
                        legend_item["upper_value"] = cv["upper_value"]
                        legend_item["text"] = f"{cv['lower_value']}- m"
                    else:
                        legend_item["lower_value"] = cv["lower_value"]
                        legend_item["text"] = f"{cv['lower_value']}+ m"
                    self.legend["contour"].append(legend_item)

            else:
                self.legend = {}
                self.legend["title"] = "Flood Depth (m)"
                self.legend["cmin"] = self.cmin
                self.legend["cmax"] = self.cmax
                self.legend["cmap"] = self.cmap

            return True

        except Exception as e:
            return False

    def plot(
        self,
        pngfile,
        zoom=None,
        title="Flood Depth (m)",
        color_values=None,
        cmap="Blues",
        vmin=0.0,
        vmax=5.0,
        lon_lim=None,
        lat_lim=None,
        width=10.0,
        background="EsriWorldImagery",
    ) -> None:
        """
        Plot the flood map using matplotlib and save it to a PNG file.

        Parameters:
        ----------
        pngfile : str
            Path to the output PNG file.
        zoom : int, optional
            Zoom level for the map. Default is None.
        title : str, optional
            Title of the plot. Default is "Flood Depth (m)".
        color_values : list, optional
            List of dictionaries containing color values and ranges for discrete colors. Default is None.
        cmap : str, optional
            Colormap to use for continuous colors. Default is "Blues".
        vmin : float, optional
            Minimum value for color mapping. Default is 0.0.
        vmax : float, optional
            Maximum value for color mapping. Default is 5.0.
        lon_lim : list, optional
            Longitude limits for the plot. Default is None.
        lat_lim : list, optional
            Latitude limits for the plot. Default is None.
        width : float, optional
            Width of the plot in inches. Default is 10.0.
        """

        if lon_lim is None or lat_lim is None:
            lon_min = self.ds.x.min().to_numpy()
            lat_min = self.ds.y.min().to_numpy()
            lon_max = self.ds.x.max().to_numpy()
            lat_max = self.ds.y.max().to_numpy()
            # Use CRS of the data
            crs = self.ds[self.data_array_name].rio.crs
            transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
            x_min, y_min = transformer.transform(lon_min, lat_min)
            x_max, y_max = transformer.transform(lon_max, lat_max)
        else:
            # Reproject bbox to EPSG:3857
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x_min, y_min = transformer.transform(lon_lim[0], lat_lim[0])
            x_max, y_max = transformer.transform(lon_lim[1], lat_lim[1])

        # Reproject to Web Mercator (EPSG:3857)
        da_3857 = self.ds[self.data_array_name].rio.reproject("EPSG:3857")

        if color_values is None:
            discrete_colors = False
        else:
            discrete_colors = True
            if isinstance(color_values, str):
                # Use default
                color_values = []
                color_values.append(
                    {"color": "lightgreen", "lower_value": 0.1, "upper_value": 0.3}
                )
                color_values.append(
                    {"color": "yellow", "lower_value": 0.3, "upper_value": 1.0}
                )
                color_values.append(
                    {"color": "#FFA500", "lower_value": 1.0, "upper_value": 2.0}
                )
                color_values.append({"color": "red", "lower_value": 2.0})

        # Set up the figure
        aspect_ratio = (y_max - y_min) / (x_max - x_min)
        fig, ax = plt.subplots(figsize=(width, aspect_ratio * width))

        if discrete_colors:
            masked = da_3857.where(da_3857 >= color_values[0]["lower_value"])

            classified = xr.full_like(masked, np.nan)
            colors = []
            labels = []
            for icolor, color_value in enumerate(color_values):
                if "upper_value" in color_value:
                    lv = color_value["lower_value"]
                    uv = color_value["upper_value"]
                    classified = classified.where(
                        ~((masked > lv) & (masked <= uv)), icolor + 1
                    )
                    labels.append(f"{lv}–{uv} m")
                else:
                    lv = color_value["lower_value"]
                    classified = classified.where(~(masked > lv), icolor + 1)
                    labels.append(f">{lv} m")
                colors.append(color_value["color"])

            # Discrete colormap
            cmap = ListedColormap(colors)
            # bounds is list from 1 to len(colors) + 1
            # e.g. [1, 2, 3, 4, 5] for 4 colors
            bounds = list(range(1, len(colors) + 2))

            norm = BoundaryNorm(bounds, cmap.N)

            # Plot using xarray's built-in plotting
            classified.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

            # Custom legend
            legend_elements = []
            for i, color_value in enumerate(color_values):
                legend_elements.append(
                    Patch(facecolor=color_value["color"], label=labels[i])
                )
            plt.legend(handles=legend_elements, title="Flood Depth", loc="lower right")

        else:
            # Plot the water depth
            da_3857.plot(
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=True,
                cbar_kwargs={"label": "Flood Depth (m)"},
                alpha=0.75,  # semi-transparent
            )

        # Add OpenStreetMap basemap
        if background.lower() == "osm":
            if zoom is None:
                ctx.add_basemap(
                    ax, crs=da_3857.rio.crs, source=ctx.providers.OpenStreetMap.Mapnik
                )
            else:
                ctx.add_basemap(
                    ax,
                    crs=da_3857.rio.crs,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom=zoom,
                )
        else:
            if zoom is None:
                ctx.add_basemap(
                    ax, crs=da_3857.rio.crs, source=ctx.providers.Esri.WorldImagery
                )
            else:
                ctx.add_basemap(
                    ax,
                    crs=da_3857.rio.crs,
                    source=ctx.providers.Esri.WorldImagery,
                    zoom=zoom,
                )

        # # Add scalebar (in meters)
        # scalebar = ScaleBar(1, units="m", location="lower left")  # scale=1 since coords are in meters
        # ax.add_artist(scalebar)

        # Zoom to bbox
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Clean layout
        ax.set_axis_off()
        plt.title(title)

        # Save to PNG
        plt.tight_layout()
        plt.savefig(pngfile, dpi=300, bbox_inches="tight", pad_inches=0.1)


def get_appropriate_overview_level(
    src: rasterio.io.DatasetReader, max_pixel_size: float
) -> int:
    """
    Given a rasterio dataset `src` and a desired `max_pixel_size`,
    determine the appropriate overview level (zoom level) that fits
    the maximum resolution allowed by `max_pixel_size`.

    Parameters:
    src (rasterio.io.DatasetReader): The rasterio dataset reader object.
    max_pixel_size (float): The maximum pixel size for the resolution.

    Returns:
    int: The appropriate overview level.
    """
    # Get the original resolution (pixel size) in terms of x and y
    original_resolution = src.res  # Tuple of (x_resolution, y_resolution)
    if src.crs.is_geographic:
        original_resolution = (
            original_resolution[0] * 111000,
            original_resolution[1] * 111000,
        )  # Convert to meters
    # Get the overviews for the dataset
    overview_levels = src.overviews(
        1
    )  # Overview levels for the first band (if multi-band, you can adjust this)

    # If there are no overviews, return 0 (native resolution)
    if not overview_levels:
        return 0

    # Calculate the resolution for each overview by multiplying the original resolution by the overview factor
    resolutions = [
        (original_resolution[0] * factor, original_resolution[1] * factor)
        for factor in overview_levels
    ]

    # Find the highest overview level that is smaller than or equal to the max_pixel_size
    selected_overview = 0
    for i, (x_res, y_res) in enumerate(resolutions):
        if x_res <= max_pixel_size and y_res <= max_pixel_size:
            selected_overview = i
        else:
            break

    return selected_overview


def get_rgb_data_array(
    da: xr.DataArray, cmap: str, cmin: float = None, cmax: float = None, color_values=None, discrete_colors=False
) -> xr.DataArray:
    """
    Convert a DataArray to RGB using either a colormap or discrete color values.

    Parameters:
    ----------
    da : xr.DataArray
        The input DataArray to be converted.
    cmap : str, optional
        The colormap to use (e.g., 'viridis'). Used when color_values is None.
    cmin : float, optional
        Minimum value for normalization. If None, the minimum value of the data is used.
    cmax : float, optional
        Maximum value for normalization. If None, the maximum value of the data is used.
    color_values : list, optional
        List of dictionaries containing discrete color definitions with keys:
        'lower_value' (optional), 'upper_value' (optional), and 'rgb' (list of 3 values 0-255).
        At least one of 'lower_value' or 'upper_value' must be specified.
        If provided, this takes precedence over cmap.

    Returns:
    -------
    xr.DataArray
        The RGB DataArray.
    """
    if color_values is not None:
        # Get the shape and flatten the data
        ny, nx = da.shape
        zz = da.to_numpy()
        # Use discrete color values
        rgba = np.zeros((ny, nx, 4), "uint8")

    if discrete_colors:

        if isinstance(color_values, str):
            # Use default
            color_values = []
            color_values.append(
                {"color": "lightgreen", "lower_value": 0.1, "upper_value": 0.3}
            )
            color_values.append(
                {"color": "yellow", "lower_value": 0.3, "upper_value": 1.0}
            )
            color_values.append(
                {"color": "#FFA500", "lower_value": 1.0, "upper_value": 2.0}
            )
            color_values.append({"color": "red", "lower_value": 2.0})

        # Initialize an RGBA array with zeros
        rgba = np.zeros((da.shape[0], da.shape[1], 4), dtype=np.float32)
        # Loop through color classes
        for icolor, color_value in enumerate(color_values):
            # Get rgb float values for this color (between 0.0 and 1.0)
            color_rgba = cm.colors.to_rgba(color_value["color"])
            if "lower_value" in color_values and "upper_value" in color_value:
                rgba[(da >= color_value["lower_value"]) & (da < color_value["upper_value"]), :] = color_rgba
            elif "lower_value" in color_value:
                rgba[(da >= color_value["lower_value"]), :] = color_rgba
            elif "upper_value" in color_value:
                rgba[(da < color_value["upper_value"]), :] = color_rgba
            pass

            # Mask out NaN values in zz
            valid = np.logical_and(inr, ~np.isnan(zz))
            rgba[valid, 0] = color_value["rgb"][0]
            rgba[valid, 1] = color_value["rgb"][1]
            rgba[valid, 2] = color_value["rgb"][2]
            rgba[valid, 3] = 255

    else:
        # Use continuous colormap (existing functionality)
        if cmap is None:
            raise ValueError("Either color_values or cmap must be provided")

        # Normalize the data
        if cmin is None:
            cmin = da.min()
        if cmax is None:
            cmax = da.max()

        # Ensure cmin and cmax are not equal to avoid division by zero
        if cmin == cmax:
            cmin = cmax - 1.0
            cmax = cmax + 1.0
        # Ensure cmin and cmax are not equal to avoid division by zero
        if cmin == cmax:
            cmin = cmax - 1.0
            cmax = cmax + 1.0

        # Normalize to [0, 1]
        normed = (da - cmin) / (cmax - cmin)
        # Normalize to [0, 1]
        normed = (da - cmin) / (cmax - cmin)

        # Get colormap
        cmap_obj = plt.get_cmap(cmap)

        # Apply colormap (returns RGBA)
        rgba = cmap_obj(normed)

        # Convert to 8-bit RGB
        rgba = (rgba[:, :, :] * 255).astype("uint8")

    # Convert to DataArray with 'band' dimension
    rgb_da = xr.DataArray(
        np.moveaxis(rgba, -1, 0),  # shape: (4, height, width)
        dims=("band", "y", "x"),
        coords={"band": [0, 1, 2, 3], "y": da.y, "x": da.x},
        attrs=da.attrs,
    )

    rgb_da.rio.write_crs(da.rio.crs, inplace=True)

    return rgb_da


def reproject_bbox(xmin, ymin, xmax, ymax, crs_src, crs_dst, buffer=0.0):
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    # Buffer the bounding box
    dx = (xmax - xmin) * buffer
    dy = (ymax - ymin) * buffer
    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy

    # Transform all four corners
    x0, y0 = transformer.transform(xmin, ymin)
    x1, y1 = transformer.transform(xmax, ymin)
    x2, y2 = transformer.transform(xmax, ymax)
    x3, y3 = transformer.transform(xmin, ymax)

    # New bounding box
    xs = [x0, x1, x2, x3]
    ys = [y0, y1, y2, y3]

    return min(xs), min(ys), max(xs), max(ys)


# The following methods are used to make flood maps from tiles


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
