"""
Unit tests for geospatial helper utilities.
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from shapely.geometry import box

from src.aggregation.aggregate_and_validate import validate_against_housing_societies
from src.utils.geo_helpers import compute_pixel_area_m2, compute_pixel_area_ha, to_metric_crs


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


def test_to_metric_crs_reprojects_geometries():
    """Test that area calculations can use metric units after reprojection."""
    gdf = gpd.GeoDataFrame(geometry=[box(73.0, 31.5, 73.001, 31.501)], crs="EPSG:4326")

    projected = to_metric_crs(gdf)

    assert projected.crs.to_epsg() == 32643
    assert projected.geometry.area.iloc[0] > 0


def test_validation_overlap_is_percentage_without_geographic_area_warning(tmp_path):
    """Test overlap validation uses projected geometry areas and returns percent."""
    conversion = gpd.GeoDataFrame(
        geometry=[box(73.0, 31.5, 73.002, 31.502)], crs="EPSG:4326"
    )
    societies_path = tmp_path / "societies.geojson"
    gpd.GeoDataFrame(
        geometry=[box(73.0, 31.5, 73.001, 31.502)], crs="EPSG:4326"
    ).to_file(societies_path, driver="GeoJSON")

    overlap_pct = validate_against_housing_societies(conversion, societies_path)

    assert 49.0 < overlap_pct < 51.0
