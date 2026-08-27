"""
Week 1 — Data acquisition.

Pulls cloud-masked Sentinel-2 dry-season composites for each Phase-1 district
and each target year, and exports them to Google Drive / GCS as GeoTIFFs.

Usage:
    python -m src.acquisition.fetch_sentinel2 --district Lahore --year 2025
    python -m src.acquisition.fetch_sentinel2 --all
"""

import argparse
import ee

from src.config import (
    PHASE_1_DISTRICTS,
    YEARS,
    GEE_S2_COLLECTION,
    S2_CLOUD_PROB_COLLECTION,
    CLOUD_FILTER_PCT,
    CLOUD_PROB_THRESHOLD,
    COMPOSITE_MONTHS,
    GEE_ADMIN_BOUNDARIES,
    GEE_ADMIN_COUNTRY_FIELD,
    GEE_ADMIN_COUNTRY_VALUE,
    GEE_ADMIN_DISTRICT_FIELD,
    S2_BANDS,
)
from src.utils.gee_auth import init_gee


def get_district_geometry(district: str) -> ee.Geometry:
    """Look up a Punjab district boundary from the GAUL admin-2 dataset."""
    admin = ee.FeatureCollection(GEE_ADMIN_BOUNDARIES)
    feature = admin.filter(
        ee.Filter.And(
            ee.Filter.eq(GEE_ADMIN_COUNTRY_FIELD, GEE_ADMIN_COUNTRY_VALUE),
            ee.Filter.eq(GEE_ADMIN_DISTRICT_FIELD, district),
        )
    ).first()
    return feature.geometry()


def mask_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds using the S2 cloud probability collection."""
    cloud_prob = ee.Image(image.get("cloud_mask")).select("probability")
    is_clear = cloud_prob.lt(CLOUD_PROB_THRESHOLD)
    return image.updateMask(is_clear).copyProperties(image, ["system:time_start"])


def build_composite(district: str, year: int) -> ee.Image:
    """Build a cloud-masked, dry-season median composite for one district/year."""
    geom = get_district_geometry(district)

    # Dry season spans the year boundary (Nov-Feb), so pull Nov(year-1)-Feb(year)
    start = ee.Date.fromYMD(year - 1, COMPOSITE_MONTHS[0], 1)
    end = ee.Date.fromYMD(year, COMPOSITE_MONTHS[-1], 28)

    s2 = (
        ee.ImageCollection(GEE_S2_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER_PCT))
    )
    cloud_prob = (
        ee.ImageCollection(S2_CLOUD_PROB_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
    )

    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=s2,
        secondary=cloud_prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    masked = ee.ImageCollection(joined).map(mask_clouds)
    bands = list(S2_BANDS.values())
    composite = masked.select(bands).median().clip(geom)
    return composite


def export_composite(district: str, year: int) -> None:
    """Kick off a GEE export task to Google Drive."""
    composite = build_composite(district, year)
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=f"S2_{district}_{year}",
        folder="AUC_Punjab",
        fileNamePrefix=f"s2_{district.lower()}_{year}",
        scale=10,
        region=get_district_geometry(district),
        maxPixels=1e10,
    )
    task.start()
    print(f"Started export task: S2_{district}_{year} (task id: {task.id})")


def main():
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 composites for AUC pipeline.")
    parser.add_argument("--district", choices=PHASE_1_DISTRICTS, help="Single district to fetch.")
    parser.add_argument("--year", type=int, choices=YEARS, help="Single year to fetch.")
    parser.add_argument("--all", action="store_true", help="Fetch all districts x all years.")
    args = parser.parse_args()

    init_gee()

    if args.all:
        for district in PHASE_1_DISTRICTS:
            for year in YEARS:
                export_composite(district, year)
    elif args.district and args.year:
        export_composite(args.district, args.year)
    else:
        parser.error("Specify --district and --year, or use --all.")


if __name__ == "__main__":
    main()
