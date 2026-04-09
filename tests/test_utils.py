"""Tests for cht_tiling.utils."""

import numpy as np
import pytest

from cht_tiling.utils import (
    deg2num,
    elevation2png,
    get_zoom_level_for_resolution,
    makedir,
    num2deg,
    num2xy,
    png2elevation,
    xy2num,
)


class TestDeg2Num:
    def test_origin_zoom0(self):
        x, y = deg2num(0.0, 0.0, 0)
        assert x == 0
        assert y == 0

    def test_northwest_zoom1(self):
        x, y = deg2num(45.0, -90.0, 1)
        assert x == 0
        assert y == 0

    def test_southeast_zoom1(self):
        x, y = deg2num(-45.0, 90.0, 1)
        assert x == 1
        assert y == 1


class TestNum2Deg:
    def test_origin_zoom0(self):
        lat, lon = num2deg(0, 0, 0)
        assert lon == pytest.approx(-180.0)
        assert lat == pytest.approx(85.05, abs=0.1)

    def test_roundtrip(self):
        lat_in, lon_in = 52.0, 4.0
        zoom = 10
        x, y = deg2num(lat_in, lon_in, zoom)
        lat_out, lon_out = num2deg(x, y, zoom)
        # Should be close (within one tile)
        assert abs(lat_in - lat_out) < 1.0
        assert abs(lon_in - lon_out) < 1.0


class TestXy2NumNum2Xy:
    def test_roundtrip(self):
        zoom = 5
        x_in, y_in = 500000.0, 6000000.0
        tx, ty = xy2num(x_in, y_in, zoom)
        x_out, y_out = num2xy(tx, ty, zoom)
        tile_size = 20037508.34 * 2 / (2**zoom)
        assert abs(x_in - x_out) < tile_size
        assert abs(y_in - y_out) < tile_size

    def test_num2xy_zoom0(self):
        x, y = num2xy(0, 0, 0)
        assert x == pytest.approx(-20037508.34, rel=1e-4)
        assert y == pytest.approx(20037508.34, rel=1e-4)


class TestGetZoomLevel:
    def test_coarse_resolution(self):
        assert get_zoom_level_for_resolution(100000) <= 2

    def test_fine_resolution(self):
        assert get_zoom_level_for_resolution(10) >= 13

    def test_very_fine(self):
        assert get_zoom_level_for_resolution(0.001) == 23


class TestElevationPng:
    def test_terrarium_roundtrip(self, tmp_path):
        z = np.full((256, 256), 100.5)
        z[0, 0] = 0.0
        z[0, 1] = -50.0
        z[1, 0] = 8848.0
        z_orig = z.copy()  # elevation2png modifies input in-place
        png_file = str(tmp_path / "test.png")
        elevation2png(z, png_file)
        result = png2elevation(png_file, encoder="terrarium")
        np.testing.assert_allclose(result, z_orig, atol=1.0)

    def test_terrarium_nodata(self, tmp_path):
        z = np.full((256, 256), 0.0)
        z[0, 0] = -32768.0
        z[255, 255] = -32768.0
        png_file = str(tmp_path / "nodata.png")
        elevation2png(z, png_file)
        result = png2elevation(png_file, encoder="terrarium")
        assert np.isnan(result[0, 0])
        assert np.isnan(result[255, 255])
        assert not np.isnan(result[0, 1])


class TestMakedir:
    def test_creates_nested(self, tmp_path):
        d = str(tmp_path / "a" / "b" / "c")
        makedir(d)
        assert (tmp_path / "a" / "b" / "c").exists()

    def test_existing_is_ok(self, tmp_path):
        makedir(str(tmp_path))
