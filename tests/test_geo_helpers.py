"""
Unit tests for geospatial helper utilities.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile

from src.utils.geo_helpers import compute_pixel_area_m2, compute_pixel_area_ha


def create_memory_raster(crs_epsg: int, transform, width: int = 10, height: int = 10):
    """Creates an in-memory raster for testing."""
    memfile = MemoryFile()
    data = np.ones((1, height, width), dtype=np.uint8)
    dataset = memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=f"EPSG:{crs_epsg}",
        transform=transform,
    )
    dataset.write(data)
    return memfile, dataset


def test_compute_pixel_area_geographic():
    """Test area computation on a geographic raster (EPSG:4326)."""
    # 0.00008983 degrees is approx 10m at Punjab latitudes
    transform = from_origin(73.0, 31.5, 0.0000898315, 0.0000898315)
    memfile, dataset = create_memory_raster(crs_epsg=4326, transform=transform)

    area_m2 = compute_pixel_area_m2(dataset)
    area_ha = compute_pixel_area_ha(dataset)

    # 10m x 10m Sentinel-2 pixel in Punjab (UTM 43N) should be ~87.8 m² (0.00878 ha)
    assert 80.0 < area_m2 < 100.0
    assert 0.008 < area_ha < 0.010

    dataset.close()
    memfile.close()


def test_compute_pixel_area_projected():
    """Test area computation on a projected raster (EPSG:32643)."""
    # 10m x 10m pixels directly in UTM meters
    transform = from_origin(300000, 3500000, 10.0, 10.0)
    memfile, dataset = create_memory_raster(crs_epsg=32643, transform=transform)

    area_m2 = compute_pixel_area_m2(dataset)
    area_ha = compute_pixel_area_ha(dataset)

    assert abs(area_m2 - 100.0) < 1e-4
    assert abs(area_ha - 0.01) < 1e-6

    dataset.close()
    memfile.close()
