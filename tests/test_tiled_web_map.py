"""Tests for TiledWebMap class."""

import os

import numpy as np

from cht_tiling import TiledWebMap
from cht_tiling.utils import elevation2png


def _make_index_tiles(root, zoom, tile_indices, npix=256):
    """Create dummy binary index tiles for testing."""
    for tx, ty in tile_indices:
        tile_dir = os.path.join(root, str(zoom), str(tx))
        os.makedirs(tile_dir, exist_ok=True)
        # Create a simple index array (all valid = 0..npix*npix-1)
        ind = np.arange(npix * npix, dtype=np.int32).reshape(npix, npix)
        with open(os.path.join(tile_dir, f"{ty}.dat"), "wb") as f:
            f.write(ind.tobytes())


def _make_elevation_tiles(root, zoom, tile_indices, elevation=100.0, npix=256):
    """Create terrarium-encoded elevation PNG tiles for testing."""
    for tx, ty in tile_indices:
        tile_dir = os.path.join(root, str(zoom), str(tx))
        os.makedirs(tile_dir, exist_ok=True)
        z = np.full((npix, npix), elevation)
        elevation2png(z, os.path.join(tile_dir, f"{ty}.png"))


class TestTiledWebMapInit:
    def test_basic_init(self, tmp_path):
        twm = TiledWebMap(str(tmp_path))
        assert twm.path == str(tmp_path)
        assert twm.encoder == "terrarium"
        assert twm.npix == 256

    def test_custom_params(self, tmp_path):
        twm = TiledWebMap(
            str(tmp_path),
            name="test_ds",
            encoder="terrarium16",
            max_zoom=5,
        )
        assert twm.name == "test_ds"
        assert twm.encoder == "terrarium16"
        assert twm.max_zoom == 5

    def test_s3_download_flag(self, tmp_path):
        twm = TiledWebMap(
            str(tmp_path),
            s3_bucket="bucket",
            s3_key="key",
            s3_region="us-east-1",
        )
        assert twm.download is True

    def test_no_s3_download_flag(self, tmp_path):
        twm = TiledWebMap(str(tmp_path))
        assert twm.download is False


class TestTiledWebMapGetData:
    def test_reads_elevation_tiles(self, tmp_path):
        """Create tiles and read them back via get_data."""
        zoom = 2
        tiles = [(1, 1), (1, 2), (2, 1), (2, 2)]
        _make_elevation_tiles(str(tmp_path), zoom, tiles, elevation=42.0)

        twm = TiledWebMap(str(tmp_path), max_zoom=zoom)

        # Get the bbox of the tiles we created
        from cht_tiling.utils import num2xy

        x0, y0 = num2xy(1, 3, zoom)  # lower-left
        x1, y1 = num2xy(3, 1, zoom)  # upper-right

        x, y, z = twm.get_data(
            [x0 + 1, x1 - 1],
            [y0 + 1, y1 - 1],
            max_pixel_size=200000,
        )

        assert len(x) > 0
        assert len(y) > 0
        assert z.shape[0] == len(y)
        assert z.shape[1] == len(x)
        # Most values should be ~42
        valid = z[~np.isnan(z)]
        if len(valid) > 0:
            np.testing.assert_allclose(np.median(valid), 42.0, atol=1.0)

    def test_empty_bbox_returns_nans(self, tmp_path):
        """No tiles in bbox should return all NaN."""
        twm = TiledWebMap(str(tmp_path), max_zoom=2)
        x, y, z = twm.get_data([100, 200], [100, 200], max_pixel_size=200000)
        assert np.isnan(z).all()


class TestTiledWebMapMake:
    def test_make_dispatches_by_type(self, tmp_path):
        """Verify that make() doesn't crash with valid zoom_range and no data."""
        twm = TiledWebMap(
            str(tmp_path),
            type="data",
            parameter="elevation",
            data=[],
            zoom_range=[0, 0],
            make_highest_level=False,
            make_lower_levels=False,
            write_availability=False,
            write_metadata=False,
            make_webviewer=False,
        )
        # Should complete without error (no data to process)
        twm.make()


class TestTiledWebMapMetadata:
    def test_write_and_read_metadata(self, tmp_path):
        twm = TiledWebMap(
            str(tmp_path),
            long_name="Test Dataset",
            source="unit test",
            max_zoom=5,
            encoder="terrarium",
        )
        twm.write_metadata_file()

        # Verify file was created
        assert os.path.exists(os.path.join(str(tmp_path), "metadata.tml"))

        # Create a new TiledWebMap that reads the metadata
        twm2 = TiledWebMap(str(tmp_path))
        # Metadata stores "longname" not "long_name"
        assert twm2.longname == "Test Dataset"
        assert twm2.max_zoom == 5
        assert twm2.source == "unit test"


class TestTiledWebMapWebviewer:
    def test_webviewer_html(self, tmp_path):
        """Check that index.html is created."""
        from cht_tiling.webviewer import write_html

        html_path = str(tmp_path / "index.html")
        write_html(html_path, max_native_zoom=5)
        assert os.path.exists(html_path)
        content = open(html_path).read()
        assert "leaflet" in content.lower() or "L.tileLayer" in content
