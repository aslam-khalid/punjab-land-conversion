"""
Week 2 — Aggregation and validation.

Rolls up pixel-level conversion rasters into district-level statistics,
and cross-checks detected conversion zones against known housing-society
boundaries to sanity-check detection accuracy.
"""

from pathlib import Path
import argparse
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from src.config import HOUSING_SOCIETY_BOUNDARIES_PATH, PROCESSED_DIR
from src.utils.geo_helpers import compute_pixel_area_ha, to_metric_crs


def rasterize_conversion_to_polygons(conversion_tif: Path) -> gpd.GeoDataFrame:
    """Turn a binary conversion raster into vector polygons for overlay/validation."""
    with rasterio.open(conversion_tif) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs

    geoms = [
        {"geometry": shape(geom), "value": val}
        for geom, val in shapes(band, mask=band == 1, transform=transform)
    ]
    return gpd.GeoDataFrame(geoms, crs=crs)


def validate_against_housing_societies(
    conversion_gdf: gpd.GeoDataFrame, societies_path: Path = HOUSING_SOCIETY_BOUNDARIES_PATH
) -> float:
    """Return the fraction of detected conversion area that overlaps known societies."""
    if not societies_path.exists():
        print(f"Warning: {societies_path} not found — skipping validation overlay.")
        return float("nan")

    societies = gpd.read_file(societies_path).to_crs(conversion_gdf.crs)
    overlap = gpd.overlay(conversion_gdf, societies, how="intersection")

    conversion_gdf_utm = to_metric_crs(conversion_gdf)
    overlap_utm = to_metric_crs(overlap)
    detected_area = conversion_gdf_utm.geometry.area.sum()
    overlap_area = overlap_utm.geometry.area.sum()
    return float(overlap_area / detected_area * 100) if detected_area > 0 else float("nan")


def aggregate_by_district(
    district: str, conversion_tif: Path, out_path: Path = None
) -> dict:
    gdf = rasterize_conversion_to_polygons(conversion_tif)
    validation_pct = validate_against_housing_societies(gdf)

    with rasterio.open(conversion_tif) as src:
        band = src.read(1)
        converted_pixels = int((band == 1).sum())
        pixel_area_ha = compute_pixel_area_ha(src)
        converted_area_ha = float(converted_pixels * pixel_area_ha)

    result = {
        "district": district,
        "converted_pixels": converted_pixels,
        "converted_area_ha": converted_area_ha,
        "validation_overlap_pct": validation_pct,
    }

    out_path = out_path or (PROCESSED_DIR / f"{district.lower()}_conversion_summary.geojson")
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Wrote {out_path}")
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate conversion results by district.")
    parser.add_argument("district", type=str)
    parser.add_argument("conversion_tif", type=Path)
    args = parser.parse_args()
    aggregate_by_district(args.district, args.conversion_tif)
