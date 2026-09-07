"""
Week 2 — Full change detection pipeline runner.

Expects the Sentinel-2 GeoTIFF composites to be downloaded from Google Drive
and placed in data/raw/ with the naming convention:
    data/raw/s2_faisalabad_2019.tif
    data/raw/s2_faisalabad_2025.tif

Steps:
    1. Classify both composites with the v4 XGBoost model
    2. Run change detection (agricultural -> built_up)
    3. Vectorise the change pixels into GeoJSON polygons
    4. Print district-level conversion area summary

Usage:
    python -m src.pipeline.run_change_detection --district Faisalabad
"""

import argparse
from pathlib import Path

from src.config import PROCESSED_DIR, MODELS_DIR, RAW_DIR


def main():
    parser = argparse.ArgumentParser(description="Run the full AUC change detection pipeline.")
    parser.add_argument(
        "--district",
        default="Faisalabad",
        help="District name (must match filename convention, e.g. Faisalabad).",
    )
    parser.add_argument(
        "--before-year", type=int, default=2019,
        help="Earlier year for change detection.",
    )
    parser.add_argument(
        "--after-year", type=int, default=2025,
        help="Later year for change detection.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "land_cover_xgboost_v4.joblib",
        help="Path to trained model .joblib file.",
    )
    args = parser.parse_args()

    district_lc = args.district.lower()

    before_tif = RAW_DIR / f"s2_{district_lc}_{args.before_year}.tif"
    after_tif = RAW_DIR / f"s2_{district_lc}_{args.after_year}.tif"

    classified_before = PROCESSED_DIR / f"classified_{district_lc}_{args.before_year}.tif"
    classified_after = PROCESSED_DIR / f"classified_{district_lc}_{args.after_year}.tif"
    conversion_tif = PROCESSED_DIR / f"conversion_{district_lc}_{args.before_year}_{args.after_year}.tif"

    # --- Validate inputs ---
    missing = [p for p in [before_tif, after_tif] if not p.exists()]
    if missing:
        print("\n❌ Missing input GeoTIFFs. Download from Google Drive first:")
        for p in missing:
            print(f"   {p}")
        print(
            "\nGEE export task IDs (check Google Drive / GEE Tasks tab):\n"
            "  Faisalabad 2019: FGOD4N2W6WPURCLXSQRP6XNI\n"
            "  Faisalabad 2025: I74KZYBNCUUL2AZA6BHUOUIV\n"
            "\nDownload the .tif files from your Google Drive 'AUC_Punjab' folder\n"
            "and place them in data/raw/ with the names above."
        )
        return

    from src.classification.classify_raster import classify_raster
    from src.change_detection.detect_conversion import detect_conversion
    from src.aggregation.aggregate_and_validate import aggregate_by_district

    # Step 1: Classify before
    print(f"\n{'='*60}")
    print(f"Step 1/3: Classifying {args.before_year} composite...")
    print(f"{'='*60}")
    classify_raster(before_tif, args.model, classified_before)

    # Step 2: Classify after
    print(f"\n{'='*60}")
    print(f"Step 2/3: Classifying {args.after_year} composite...")
    print(f"{'='*60}")
    classify_raster(after_tif, args.model, classified_after)

    # Step 3: Change detection
    print(f"\n{'='*60}")
    print(f"Step 3/3: Running change detection ({args.before_year} → {args.after_year})...")
    print(f"{'='*60}")
    stats = detect_conversion(classified_before, classified_after, conversion_tif)

    # Aggregation + vectorisation
    agg = aggregate_by_district(args.district, conversion_tif)

    # Final summary
    print(f"\n{'='*60}")
    print("✅ Change Detection Complete")
    print(f"{'='*60}")
    print(f"  District         : {args.district}")
    print(f"  Period           : {args.before_year} → {args.after_year}")
    print(f"  Converted pixels : {stats['converted_pixels']:,}")
    print(f"  Converted area   : {stats['converted_area_ha']:.1f} ha")
    print(f"  Output raster    : {conversion_tif}")
    print(f"  Output GeoJSON   : {PROCESSED_DIR / f'{district_lc}_conversion_summary.geojson'}")


if __name__ == "__main__":
    main()
