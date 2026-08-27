"""
Week 1 — Training data generation.

Generates stratified training pixel samples from GEE by overlaying a Sentinel-2
composite with ESA WorldCover labels, and saves them to a CSV file.
"""

import argparse
from pathlib import Path
import pandas as pd
import ee

from src.config import (
    PHASE_1_DISTRICTS,
    S2_BANDS,
    FEATURE_COLUMNS,
    LAND_COVER_CLASSES,
)
from src.utils.gee_auth import init_gee
from src.acquisition.fetch_sentinel2 import build_composite, get_district_geometry


def add_indices(image: ee.Image) -> ee.Image:
    """Compute and add NDVI, NDBI, and NDWI bands to a Sentinel-2 image."""
    # S2_BANDS mapping:
    # blue: B2, green: B3, red: B4, nir: B8, swir1: B11, swir2: B12
    
    # NDVI = (NIR - Red) / (NIR + Red) -> (B8 - B4) / (B8 + B4)
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    
    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) -> (B11 - B8) / (B11 + B8)
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    
    # NDWI = (Green - NIR) / (Green + NIR) -> (B3 - B8) / (B3 + B8)
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    
    return image.addBands([ndvi, ndbi, ndwi])


def generate_samples(
    district: str,
    year: int,
    num_points_per_class: int,
) -> pd.DataFrame:
    """Retrieve stratified pixel samples from GEE with S2 features and ESA labels."""
    geom = get_district_geometry(district)
    
    print(f"Building Sentinel-2 composite for {district} ({year})...")
    composite = build_composite(district, year)
    composite_with_indices = add_indices(composite)
    
    print("Loading and remapping ESA WorldCover (2020) labels...")
    # Load ESA WorldCover 2020
    esa = ee.Image("ESA/WorldCover/v100/2020").clip(geom)
    
    # Map ESA classes to target classes:
    # 40 (Cropland) -> 0 (agricultural)
    # 50 (Built-up) -> 1 (built_up)
    # All others (Trees, Shrubland, Grassland, Barren, Water, Wetland) -> 2 (other)
    esa_src = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    our_dst = [ 2,  2,  2,  0,  1,  2,  2,  2,  2,  2,   2]
    remapped_label = esa.remap(esa_src, our_dst, 2).rename("label")
    
    # Combine bands
    combined = composite_with_indices.addBands(remapped_label)
    
    # Select bands we need to sample
    bands_to_sample = list(S2_BANDS.values()) + ["NDVI", "NDBI", "NDWI", "label"]
    combined_selected = combined.select(bands_to_sample)
    
    print(f"Sampling {num_points_per_class} points per class...")
    samples = combined_selected.stratifiedSample(
        numPoints=num_points_per_class,
        classBand="label",
        region=geom,
        scale=10,
        classValues=[0, 1, 2],
        classPoints=[num_points_per_class] * 3,
        geometries=False,
    )
    
    print("Downloading samples from Earth Engine (this may take a moment)...")
    samples_info = samples.getInfo()
    features = samples_info.get("features", [])
    
    data = []
    for feat in features:
        props = feat.get("properties", {})
        data.append(props)
        
    df = pd.DataFrame(data)
    
    if df.empty:
        raise RuntimeError("No samples retrieved from Earth Engine.")
        
    # Standardize columns and drop NaNs
    all_cols = FEATURE_COLUMNS + ["label"]
    missing_cols = set(all_cols) - set(df.columns)
    if missing_cols:
        print(f"Warning: Missing columns from sample retrieval: {missing_cols}")
        for col in missing_cols:
            df[col] = None
            
    df = df.dropna(subset=all_cols)
    df["label"] = df["label"].astype(int)
    
    print(f"Retrieved {len(df)} valid training samples.")
    print("Class distribution:")
    for key, name in LAND_COVER_CLASSES.items():
        count = len(df[df["label"] == key])
        print(f"  - {name} ({key}): {count}")
        
    return df[all_cols]


def main():
    parser = argparse.ArgumentParser(
        description="Generate training samples for AUC land-cover classifier."
    )
    parser.add_argument(
        "--district",
        choices=PHASE_1_DISTRICTS,
        default="Faisalabad",
        help="District to sample from.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Year of the Sentinel-2 composite. (Recommended: 2020 to align with ESA WorldCover 2020)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1500,
        help="Number of points to sample per class.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/training_samples.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    init_gee()
    
    df = generate_samples(args.district, args.year, args.num_points)
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved training dataset to {args.out}")


if __name__ == "__main__":
    main()
