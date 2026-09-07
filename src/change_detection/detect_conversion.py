"""
Week 2 — Change detection.

Compares two classified land-cover rasters (same district, different years)
and flags pixels that transitioned from agricultural -> built_up.
"""

from pathlib import Path
import argparse
import numpy as np
import rasterio

from src.config import LAND_COVER_CLASSES
from src.utils.geo_helpers import compute_pixel_area_ha

AG_CLASS = [k for k, v in LAND_COVER_CLASSES.items() if v == "agricultural"][0]
BUILTUP_CLASS = [k for k, v in LAND_COVER_CLASSES.items() if v == "built_up"][0]


def detect_conversion(before_path: Path, after_path: Path, out_path: Path) -> dict:
    with rasterio.open(before_path) as src_before:
        before = src_before.read(1)
        profile = src_before.profile.copy()
        pixel_area_ha = compute_pixel_area_ha(src_before)
    with rasterio.open(after_path) as src_after:
        after = src_after.read(1)

    if before.shape != after.shape:
        raise ValueError(
            f"Raster shape mismatch: {before_path.name} {before.shape} vs "
            f"{after_path.name} {after.shape}. Reproject/align rasters first."
        )

    converted = (before == AG_CLASS) & (after == BUILTUP_CLASS)

    profile.update(dtype="uint8", count=1, nodata=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(converted.astype("uint8")[None, ...])

    stats = {
        "converted_pixels": int(converted.sum()),
        "converted_area_ha": float(converted.sum() * pixel_area_ha),
        "total_pixels": int(before.size),
    }
    print(f"Wrote {out_path}")
    print(stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ag-to-urban conversion between two years.")
    parser.add_argument("before", type=Path, help="Classified raster for earlier year.")
    parser.add_argument("after", type=Path, help="Classified raster for later year.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    detect_conversion(args.before, args.after, args.out)
