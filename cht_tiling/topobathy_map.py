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
from cht_tiling.flood_map import get_appropriate_overview_level, get_rgb_data_array, reproject_bbox


class TopoBathyMap:
    def __init__(
        self,
        topobathy_file: Union[str, Path] = None,
        zbmin: float = -999999.9,
        zbmax: float = 99999.9,
        max_pixel_size: float = 0.0,
        data_array_name: str = "elevation",
        cmap: str = None,
        cmin: float = None,
        cmax: float = None,
        color_values: list = None,
        scale_factor: float = 1.0,
    ):
        """
        Initialize the TopoBathy class with optional parameters.

        Parameters:
        ----------
        topobathy_file : Union[str, Path], optional
            Topobathy data file (COG).
        zbmin : float, optional
            Minimum allowable topobathy value.
        zbmax : float, optional
            Maximum allowable topobathy value.
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
        color_values : list, optional
            List of dictionaries containing color values and ranges for discrete colors.
        """
        self.topobathy_file = topobathy_file
        self.zb = rasterio.open(topobathy_file) if topobathy_file else None
        self.zbmin = zbmin
        self.zbmax = zbmax
        self.max_pixel_size = max_pixel_size
        self.data_array_name = data_array_name
        self.cmap = cmap
        self.cmin = cmin * scale_factor
        self.cmax = cmax * scale_factor
        self.color_values = color_values
        self.scale_factor = scale_factor
        self.ds = xr.Dataset()

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

    def close(self) -> None:
        """
        Close the topobathy file.
        """
        if self.zb is not None:
            self.zb.close()
        self.ds.close()

    def read(self, tiffile) -> None:
        """
        Read geotiff file with elevation data.
        """
        self.ds = xr.Dataset()
        self.ds[self.data_array_name] = rioxarray.open_rasterio(tiffile, masked=True).squeeze()

    def make(
        self,
        max_pixel_size: float = 0.0,
        bbox=None,
    ) -> xr.Dataset:
        """
        Generate a topobathy dataset from elevation data.

        Parameters:
        ----------
        max_pixel_size : float, optional
            Maximum pixel size for the appropriate overview level. If 0.0, no overviews are used. Default is 0.0.
        bbox : tuple, optional
            Bounding box (minx, miny, maxx, maxy) to clip the data.

        Returns:
        -------
        xr.Dataset
            An xarray Dataset containing the elevation data.
        """

        # First get the overview level
        overview_level = 0

        if max_pixel_size > 0.0:
            overview_level = get_appropriate_overview_level(self.zb, max_pixel_size)

        # Read the data at the specified overview level
        if overview_level == 0:
            zb = rioxarray.open_rasterio(self.zb)
        else:
            zb = rioxarray.open_rasterio(self.zb, overview_level=overview_level)
        
        zb = zb * self.scale_factor  # Apply scale factor
        
        # Remove band dimension if it is 1, and squeeze the array to 2D
        if "band" in zb.dims and zb.sizes["band"] == 1:
            zb = zb.squeeze(dim="band", drop=True)
        
        if bbox is not None:
            zb = zb.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])

        # Apply elevation masks
        elevation = zb.to_numpy()[:]
        # Set elevation to NaN where it is less than zbmin
        elevation[elevation < self.zbmin] = np.nan
        # Set elevation to NaN where it is greater than zbmax
        elevation[elevation > self.zbmax] = np.nan

        # Turn elevation into a DataArray with the same dimensions as zb
        self.ds = xr.Dataset()
        self.ds[self.data_array_name] = xr.DataArray(
            elevation, dims=["y", "x"], coords={"y": zb.y, "x": zb.x}
        )
        # Use same spatial_ref as zb
        self.ds[self.data_array_name] = self.ds[self.data_array_name].rio.write_crs(
            zb.rio.crs, inplace=True
        )

        return self.ds

    def write(
        self,
        output_file: Union[str, Path] = "",
    ) -> None:
        """
        Write the topobathy data to a file.

        Parameters:
        ----------
        output_file : Union[str, Path]
            Path to the output file. The file extension determines the format (e.g., ".tif" for GeoTIFF, ".nc" for netCDF).

        Returns:
        -------
        None
        """
        if output_file.endswith(".nc"):
            # Write to netcdf
            self.ds.to_netcdf(output_file)

        elif output_file.endswith(".tif"):
            # Write to geotiff
            if self.cmap is not None:
                # Get RGB data array
                rgb_da = get_rgb_data_array(
                    self.ds[self.data_array_name],
                    cmap=self.cmap,
                    cmin=self.cmin,
                    cmax=self.cmax,
                    color_values=self.color_values,
                )

                # Write to file
                rgb_da.rio.to_raster(
                    output_file,
                    driver="COG",
                    compress="deflate",
                    blocksize=512,
                    overview_resampling="nearest",
                )

            else:
                # Just write binary data
                self.ds[self.data_array_name].rio.to_raster(
                    output_file,
                    driver="COG",
                    compress="deflate",
                    blocksize=512,
                    overview_resampling="nearest",
                )

    def map_overlay(self, file_name, xlim=None, ylim=None, width=800):
        """
        Create a map overlay of the topobathy data using the specified colormap and save it to a png file. The CRS is 3857.
        """

        if self.ds is None:
            return False

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

            self.make(max_pixel_size=dxy, bbox=bbox)

            rgb_da = get_rgb_data_array(
                self.ds[self.data_array_name],
                cmap=self.cmap,
                cmin=self.cmin,
                cmax=self.cmax,
                color_values=self.color_values,
            )

            # Now reproject to EPSG:3857 and create a png file
            rgb_3857 = rgb_da.rio.reproject(
                "EPSG:3857", resampling=Resampling.bilinear, nodata=0
            )

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

            return True

        except Exception as e:
            print(f"Error in map_overlay: {e}")
            return False

    def plot(
        self,
        pngfile,
        zoom=None,
        title="Elevation (m)",
        color_values=None,
        cmap="terrain",
        vmin=None,
        vmax=None,
        lon_lim=None,
        lat_lim=None,
        width=10.0,
        background="EsriWorldImagery",
    ) -> None:
        """
        Plot the topobathy data using matplotlib and save it to a PNG file.

        Parameters:
        ----------
        pngfile : str
            Path to the output PNG file.
        zoom : int, optional
            Zoom level for the map. Default is None.
        title : str, optional
            Title of the plot. Default is "Elevation (m)".
        color_values : list, optional
            List of dictionaries containing color values and ranges for discrete colors. Default is None.
        cmap : str, optional
            Colormap to use for continuous colors. Default is "terrain".
        vmin : float, optional
            Minimum value for color mapping. Default is None (auto-detect).
        vmax : float, optional
            Maximum value for color mapping. Default is None (auto-detect).
        lon_lim : list, optional
            Longitude limits for the plot. Default is None.
        lat_lim : list, optional
            Latitude limits for the plot. Default is None.
        width : float, optional
            Width of the plot in inches. Default is 10.0.
        background : str, optional
            Background map provider. Default is "EsriWorldImagery".
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

        # Auto-detect vmin and vmax if not provided
        if vmin is None:
            vmin = float(da_3857.min())
        if vmax is None:
            vmax = float(da_3857.max())

        if color_values is None:
            discrete_colors = False
        else:
            discrete_colors = True
            if isinstance(color_values, str):
                # Use default elevation color scheme
                color_values = []
                color_values.append(
                    {"color": "blue", "lower_value": -100, "upper_value": 0}
                )
                color_values.append(
                    {"color": "lightblue", "lower_value": 0, "upper_value": 10}
                )
                color_values.append(
                    {"color": "green", "lower_value": 10, "upper_value": 50}
                )
                color_values.append(
                    {"color": "brown", "lower_value": 50, "upper_value": 100}
                )
                color_values.append({"color": "white", "lower_value": 100})

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
            cmap_obj = ListedColormap(colors)
            bounds = list(range(1, len(colors) + 2))
            norm = BoundaryNorm(bounds, cmap_obj.N)

            # Plot using xarray's built-in plotting
            classified.plot(ax=ax, cmap=cmap_obj, norm=norm, add_colorbar=False)

            # Custom legend
            legend_elements = []
            for i, color_value in enumerate(color_values):
                legend_elements.append(
                    Patch(facecolor=color_value["color"], label=labels[i])
                )
            plt.legend(handles=legend_elements, title="Elevation", loc="lower right")

        else:
            # Plot the elevation
            da_3857.plot(
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=True,
                cbar_kwargs={"label": "Elevation (m)"},
                alpha=0.75,
            )

        # Add background map
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

        # Zoom to bbox
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Clean layout
        ax.set_axis_off()
        plt.title(title)

        # Save to PNG
        plt.tight_layout()
        plt.savefig(pngfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()