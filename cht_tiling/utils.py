# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:51:04 2021

@author: ormondt
"""

import glob
import math
import os

import numpy as np

# from matplotlib import cm
# from matplotlib.colors import LightSource
from PIL import Image

# from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator


def get_zoom_level_for_resolution(dx):
    # Get required zoom level
    # Make numpy array with pixel size in meters for all zoom levels
    dxy = 156543.03 / 2 ** np.arange(24)
    # Find zoom level that provides sufficient pixels
    izoom = np.where(dxy < dx)[0]
    if len(izoom) == 0:
        izoom = 23
    else:
        izoom = int(izoom[0])
    return izoom


def get_zoom_level(npixels, lat_range, max_zoom):
    # Get required zoom level
    # lat = np.pi * (lat_range[0] + lat_range[1]) / 360
    # dx = (lon_range[1] - lon_range[0]) * 111111 * np.cos(lat) / npixels[0]
    dxr = (lat_range[1] - lat_range[0]) * 111111 / npixels
    # dxr = min(dx, dy)
    # Make numpy array with pixel size in meters for all zoom levels
    dxy = 156543.03 / 2 ** np.arange(max_zoom + 1)
    # Find zoom level that provides sufficient pixels
    izoom = np.where(dxy < dxr)[0]
    if len(izoom) == 0:
        izoom = max_zoom
    else:
        izoom = izoom[0]
    return izoom


def webmercator_to_lat_lon(easting, northing):
    lon = (easting / 20037508.34) * 180
    lat = (180 / math.pi) * (
        2 * math.atan(math.exp(northing / 20037508.34 * math.pi)) - (math.pi / 2)
    )
    return lat, lon


def lat_lon_to_webmercator(lat, lon):
    # Convert latitude and longitude to Web Mercator coordinates
    x = lon * 20037508.34 / 180
    y = (math.log(math.tan((90 + lat) * math.pi / 360)) / math.pi) * 20037508.34
    return x, y


def lat_lon_to_tile_indices(lat, lon, zoom):
    tile_x = int((lon + 180) / 360 * (2**zoom))
    tile_y = int(
        (
            1
            - (
                math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
                / math.pi
            )
        )
        / 2
        * (2**zoom)
    )
    return tile_x, tile_y


def xy2num(easting, northing, zoom):
    lat, lon = webmercator_to_lat_lon(easting, northing)
    ix, it = lat_lon_to_tile_indices(lat, lon, zoom)
    return ix, it


def deg2num(lat_deg, lon_deg, zoom):
    """Returns column and row index of slippy tile"""
    lat_rad = math.radians(lat_deg)
    n = 2**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """Returns upper left latitude and longitude of slippy tile"""
    # Return upper left corner of tile
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def num2xy(xtile, ytile, zoom):
    """Returns upper left x and y of slippy tile"""
    # Return upper left corner of tile
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    x, y = lat_lon_to_webmercator(lat_deg, lon_deg)
    return x, y


### Old index to degrees functions
def num2deg_ll(xtile, ytile, zoom):
    # Return lower left corner of tile (only used in old format)
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(-lat_rad)
    return (lat_deg, lon_deg)


def num2deg_ur(xtile, ytile, zoom):
    # Return upper_right corner of tile (only used in old format)
    n = 2**zoom
    lon_deg = (xtile + 1) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n)))
    lat_deg = math.degrees(-lat_rad)
    return (lat_deg, lon_deg)


# def num2xy_ll(xtile, ytile, zoom):
#     lat, lon = num2deg_ll(xtile, ytile, zoom)
#     x, y     = lat_lon_to_webmercator(lat, lon)
#     return x, y

# def num2xy_ur(xtile, ytile, zoom):
#     lat, lon = num2deg_ur(xtile, ytile, zoom)
#     x, y     = lat_lon_to_webmercator(lat, lon)
#     return x, y


def get_lower_left_corner(tile_x, tile_y, zoom):
    # Total size of the Web Mercator projection
    total_size = 20037508.34 * 2

    # Calculate tile size
    tile_size = total_size / (2**zoom)

    # Calculate lower-left corner coordinates
    ll_x = tile_x * tile_size - 20037508.34
    ll_y = 20037508.34 - (tile_y + 1) * tile_size

    return ll_x, ll_y


# def rgba2int(rgba):
#     """Convert rgba tuple to int"""
#     r, g, b, a = rgba
#     return (r * 256**3) + (g * 256**2) + (b * 256) + a


### Conversion between elevation and png and vice versa
# Note: we only use the RGB channels for this (and not the alpha channel)


def elevation2png(
    val,
    png_file,
    encoder="terrarium",
    encoder_vmin=0.0,
    encoder_vmax=1.0,
    compress_level=6,
):
    """Convert 256*256 Numpy array to png using terrarium interpretation"""
    if encoder == "terrarium":
        rgb = np.zeros((256, 256, 3), "uint8")
        val += 32768.0
        rgb[:, :, 0] = np.floor(val / 256).astype(int)
        rgb[:, :, 1] = np.floor(val % 256)
        rgb[:, :, 2] = np.floor((val - np.floor(val)) * 256).astype(int)
    elif encoder == "terrarium16":
        rgb = np.zeros((256, 256, 3), "uint8")
        val += 32768.0
        rgb[:, :, 0] = np.floor(val / 256).astype(int)
        rgb[:, :, 1] = np.floor(val % 256).astype(int)
    elif encoder == "uint8":
        # Unsigned integer8

        # Check if any values in val are equal to or larger than 255
        if np.any(val >= 255):
            # Throw an error
            raise ValueError(
                "Some values in are equal to or larger than 255. This is not allowed for encoder 'uint8'."
            )

        rgb = np.zeros((256, 256, 3), "uint8") + 255
        r = val + 0
        r[np.where(val < 0)] = 255
        rgb[:, :, 0] = r

    elif encoder == "uint16":
        # Unsigned integer16

        # Check if any values in val are equal to or larger than 65535
        if np.any(val >= 65535):
            # Throw an error
            raise ValueError(
                "Some values are equal to or larger than 65535. This is not allowed for encoder 'uint16'."
            )

        rgb = np.zeros((256, 256, 3), "uint8") + 255
        r = (val // 256) % 256
        g = val % 256
        r[np.where(val < 0)] = 255
        g[np.where(val < 0)] = 255
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
    elif encoder == "uint24":
        # Unsigned integer24

        # Check if any values in val are equal to or larger than 16777215
        if np.any(val >= 16777215):
            # Throw an error
            raise ValueError(
                "Some values are equal to or larger than 16777215. This is not allowed for encoder 'uint24'."
            )

        rgb = np.zeros((256, 256, 3), "uint8") + 255
        r = (val // 256**2) % 256
        g = (val // 256) % 256
        b = val % 256
        r[np.where(val < 0)] = 255
        g[np.where(val < 0)] = 255
        b[np.where(val < 0)] = 255
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
    elif encoder == "uint32":
        # Unsigned integer32

        # Check if any values in val are equal to or larger than 4294967295
        if np.any(val >= 4294967295):
            # Throw an error
            raise ValueError(
                "Some values are equal to or larger than 4294967295. This is not allowed for encoder 'uint32'."
            )

        rgb = np.zeros((256, 256, 4), "uint8") + 255
        r = (val // 256**3) % 256
        g = (val // 256**2) % 256
        b = (val // 256) % 256
        a = val % 256
        r[np.where(val < 0)] = 255
        g[np.where(val < 0)] = 255
        b[np.where(val < 0)] = 255
        a[np.where(val < 0)] = 255
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        rgb[:, :, 3] = a
    elif encoder == "float8":
        # Stretch to values between 1 and 256^1 - 1. NaN values are set to [0,0,0].
        val = np.maximum(val, encoder_vmin)
        val = np.minimum(val, encoder_vmax)
        val = val - encoder_vmin
        i = np.floor(val * 254 / (encoder_vmax - encoder_vmin)).astype(int) + 1
        i[np.isnan(val)] = 0
        rgb = np.zeros((256, 256, 3), "uint8")
        rgb[:, :, 0] = i
    elif encoder == "float16":
        # Stretch to values between 1 and 256^2 - 1. NaN values are set to [0,0,0].
        val = np.maximum(val, encoder_vmin)
        val = np.minimum(val, encoder_vmax)
        val = val - encoder_vmin
        i = np.floor(val * 65534 / (encoder_vmax - encoder_vmin)).astype(int) + 1
        i[np.isnan(val)] = 0
        rgb = np.zeros((256, 256, 3), "uint8")
        rgb[:, :, 0] = (i // 256) % 256
        rgb[:, :, 1] = i % 256
    elif encoder == "float24":
        # Stretch to values between 1 and 256^3 - 1. NaN values are set to [0,0,0].
        val = np.maximum(val, encoder_vmin)
        val = np.minimum(val, encoder_vmax)
        val = val - encoder_vmin
        i = np.floor(val * 16777214 / (encoder_vmax - encoder_vmin)).astype(int) + 1
        i[np.isnan(val)] = 0
        rgb = np.zeros((256, 256, 3), "uint8")
        rgb[:, :, 0] = (i // 256**2) % 256
        rgb[:, :, 1] = (i // 256) % 256
        rgb[:, :, 2] = i % 256
    elif encoder == "float32":
        # Stretch to values between 1 and 256^4 - 1. NaN values are set to [0,0,0].
        val = np.maximum(val, encoder_vmin)
        val = np.minimum(val, encoder_vmax)
        val = val - encoder_vmin
        i = np.floor(val * 4294967294 / (encoder_vmax - encoder_vmin)).astype(int) + 1
        i[np.isnan(val)] = 0
        rgb = np.zeros((256, 256, 4), "uint8")
        rgb[:, :, 0] = (i // 256**3) % 256
        rgb[:, :, 1] = (i // 256**2) % 256
        rgb[:, :, 2] = (i // 256) % 256
        rgb[:, :, 3] = i % 256

    # Create PIL Image from RGB values and save as PNG
    img = Image.fromarray(rgb)
    img.save(png_file, compress_level=compress_level)


def png2elevation(png_file, encoder="terrarium", encoder_vmin=0.0, encoder_vmax=1.0):
    """Convert png to elevation array based on terrarium interpretation"""
    img = Image.open(png_file)
    # Convert RGB values to elevation values
    if encoder == "terrarium":
        rgb = np.array(img.convert("RGB")).astype(float)
        elevation = (rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256) - 32768.0
        # where val is less than -32767, set to NaN
        elevation[elevation < -32767.0] = np.nan
    elif encoder == "terrarium16":
        rgb = np.array(img.convert("RGB")).astype(float)
        elevation = (rgb[:, :, 0] * 256 + rgb[:, :, 1]) - 32768.0
        # where val is less than -32767, set to NaN
        elevation[elevation < -32767.0] = np.nan
    elif encoder == "uint8":
        rgb = np.array(img.convert("RGB")).astype(int)
        elevation = rgb[:, :, 0]
        elevation[np.where(elevation == 255)] = -1
    elif encoder == "uint16":
        rgb = np.array(img.convert("RGB")).astype(int)
        elevation = rgb[:, :, 0] * 256 + rgb[:, :, 1]
        elevation[elevation == 65535] = -1
    elif encoder == "uint24":
        rgb = np.array(img.convert("RGB")).astype(int)
        elevation = rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]
        elevation[elevation == 16777215] = -1
    elif encoder == "uint32":
        rgb = np.array(img.convert("RGBA")).astype(int)
        elevation = (
            rgb[:, :, 0] * 16777216
            + rgb[:, :, 1] * 65536
            + rgb[:, :, 2] * 256
            + rgb[:, :, 3]
        )
        elevation[elevation == 4294967295] = -1
    elif encoder == "float8":
        rgb = np.array(img.convert("RGB")).astype(float)
        i = rgb[:, :, 0]
        elevation = encoder_vmin + (encoder_vmax - encoder_vmin) * i / 254
        elevation[i == 0] = np.nan
    elif encoder == "float16":
        rgb = np.array(img.convert("RGB")).astype(float)
        i = rgb[:, :, 0] * 256 + rgb[:, :, 1]
        elevation = encoder_vmin + (encoder_vmax - encoder_vmin) * i / 65534
        elevation[i == 0] = np.nan
    elif encoder == "float24":
        rgb = np.array(img.convert("RGB")).astype(float)
        i = rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]
        elevation = encoder_vmin + (encoder_vmax - encoder_vmin) * i / 16777214
        elevation[i == 0] = np.nan
    elif encoder == "float32":
        rgb = np.array(img.convert("RGBA")).astype(float)
        i = (
            rgb[:, :, 0] * 16777216
            + rgb[:, :, 1] * 65536
            + rgb[:, :, 2] * 256
            + rgb[:, :, 3]
        )
        elevation = encoder_vmin + (encoder_vmax - encoder_vmin) * i / 4294967294
        elevation[i == 0] = np.nan
    return elevation


# def elevation2rgb(val):
#     """Convert elevation to rgb tuple"""
#     val += 32768
#     r = np.floor(val / 256)
#     g = np.floor(val % 256)
#     b = np.floor((val - np.floor(val)) * 256)
#     return (r, g, b)

# def rgb2elevation(r, g, b):
#     """Convert rgb tuple to elevation"""
#     val = (r * 256 + g + b / 256) - 32768
#     return val

# def rgb2elevation(rgb):
#     """Convert rgb tuple to elevation"""
#     val = (rgb[:,:,0] * 256 + rgb[:,:,1] + rgb[:,:,2] / 256) - 32768.0
#     # where val is less than -32767, set to NaN
#     val[val<-32767.0] = np.nan
#     return val


### Conversion between int and png and vice versa


def png2int(png_file, idummy):
    """Convert png to int array"""
    # Open the PNG image
    image = Image.open(png_file)
    rgba = np.array(image.convert("RGBA"))
    ind = (
        (rgba[:, :, 0] * 256**3)
        + (rgba[:, :, 1] * 256**2)
        + (rgba[:, :, 2] * 256)
        + rgba[:, :, 3]
    )
    ind[np.where(ind == 4294967295)] = idummy
    return ind


def int2png(val, png_file):
    """Convert int array to png"""
    # Convert index integers to RGBA values
    rgba = np.zeros((256, 256, 4), "uint8") + 255
    r = (val // 256**3) % 256
    g = (val // 256**2) % 256
    b = (val // 256) % 256
    a = val % 256
    r[np.where(val < 0)] = 255
    g[np.where(val < 0)] = 255
    b[np.where(val < 0)] = 255
    a[np.where(val < 0)] = 255
    rgba[:, :, 0] = r
    rgba[:, :, 1] = g
    rgba[:, :, 2] = b
    rgba[:, :, 3] = a
    # Create PIL Image from RGB values and save as PNG
    img = Image.fromarray(rgba)
    img.save(png_file)


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_files(src):
    file_list = []
    full_list = glob.glob(src)
    for item in full_list:
        if os.path.isfile(item):
            file_list.append(item)
    return file_list


def list_folders(src, basename=False):
    folder_list = []
    full_list = glob.glob(src)
    for item in full_list:
        if os.path.isdir(item):
            # folder_list.append(item)
            if basename:
                folder_list.append(os.path.basename(item))
            else:
                folder_list.append(item)

    return folder_list


def interp2(x0, y0, z0, x1, y1):
    f = RegularGridInterpolator((y0, x0), z0, bounds_error=False, fill_value=np.nan)
    # reshape x1 and y1
    sz = x1.shape
    x1 = x1.reshape(sz[0] * sz[1])
    y1 = y1.reshape(sz[0] * sz[1])
    # interpolate
    z1 = f((y1, x1)).reshape(sz)
    return z1


def binary_search(val_array, vals):
    indx = np.searchsorted(val_array, vals)  # ind is size of vals
    not_ok = np.where(indx == len(val_array))[
        0
    ]  # size of vals, points that are out of bounds
    indx[np.where(indx == len(val_array))[0]] = (
        0  # Set to zero to avoid out of bounds error
    )
    is_ok = np.where(val_array[indx] == vals)[0]  # size of vals
    indices = np.zeros(len(vals), dtype=int) - 1
    indices[is_ok] = indx[is_ok]
    indices[not_ok] = -1
    return indices
