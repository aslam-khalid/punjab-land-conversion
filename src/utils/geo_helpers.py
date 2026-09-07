"""
Geospatial helper functions for area calculations and CRS transformations.
"""

import rasterio
from rasterio.warp import calculate_default_transform


def compute_pixel_area_m2(
    src: rasterio.DatasetReader, target_utm_crs: str = "EPSG:32643"
) -> float:
    """
    Returns per-pixel area in square meters.

    If the source raster is in geographic coordinates (e.g. EPSG:4326),
    it reprojects the bounding box and resolution to a metric UTM projection
    (default: UTM Zone 43N, EPSG:32643, appropriate for Punjab/Faisalabad).
    """
    if src.crs.is_geographic:
        transform, width, height = calculate_default_transform(
            src.crs, target_utm_crs, src.width, src.height, *src.bounds
        )
        pixel_area_m2 = abs(transform.a * transform.e)
    else:
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    return float(pixel_area_m2)


def compute_pixel_area_ha(
    src: rasterio.DatasetReader, target_utm_crs: str = "EPSG:32643"
) -> float:
    """Returns per-pixel area in hectares (1 ha = 10,000 m²)."""
    return compute_pixel_area_m2(src, target_utm_crs=target_utm_crs) / 10_000.0
