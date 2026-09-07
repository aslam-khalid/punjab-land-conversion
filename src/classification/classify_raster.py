"""
Week 2 — Raster classification.

Applies a trained per-pixel land-cover classifier to a Sentinel-2 GeoTIFF
composite and writes a single-band classified output raster.

The input raster must have the six Sentinel-2 bands followed by the two GEE
GLCM texture bands, in the order produced by fetch_sentinel2.py.

Usage:
    python -m src.classification.classify_raster \\
        data/raw/s2_faisalabad_2019.tif \\
        --model models/land_cover_xgboost_v4.joblib \\
        --out data/processed/classified_faisalabad_2019.tif
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import rasterio

from src.config import FEATURE_COLUMNS, S2_BANDS, MODELS_DIR

# Band order expected in the input composite GeoTIFF. Texture bands are
# exported by GEE so inference uses the same computation as training.
RASTER_BAND_ORDER = list(S2_BANDS.values()) + [
    "TEXTURE_CONTRAST",
    "TEXTURE_HOMOGENEITY",
]

# Indices of specific bands within RASTER_BAND_ORDER (0-indexed)
_B3_IDX = RASTER_BAND_ORDER.index("B3")
_B4_IDX = RASTER_BAND_ORDER.index("B4")
_B8_IDX = RASTER_BAND_ORDER.index("B8")
_B11_IDX = RASTER_BAND_ORDER.index("B11")


def compute_indices(bands: np.ndarray) -> tuple:
    """Compute NDVI, NDBI, NDWI from band arrays (H x W each)."""
    eps = 1e-6  # avoid division by zero
    nir = bands[_B8_IDX].astype(float)
    red = bands[_B4_IDX].astype(float)
    green = bands[_B3_IDX].astype(float)
    swir1 = bands[_B11_IDX].astype(float)

    ndvi = (nir - red) / (nir + red + eps)
    ndbi = (swir1 - nir) / (swir1 + nir + eps)
    ndwi = (green - nir) / (green + nir + eps)
    return ndvi, ndbi, ndwi


def classify_raster(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    block_size: int = 512,
) -> None:
    """Apply model to a multi-band Sentinel-2 composite and write classified raster."""
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        n_bands = src.count
        height, width = src.height, src.width
        print(f"Input raster: {width}x{height} px, {n_bands} bands")

        if n_bands < len(RASTER_BAND_ORDER):
            raise ValueError(
                f"Expected {len(RASTER_BAND_ORDER)} bands in input raster "
                "(six Sentinel-2 bands plus two GEE GLCM texture bands), "
                f"got {n_bands}."
            )

        # Read all bands at once (H x W x n_bands)
        print("Reading raster bands...")
        bands = src.read()  # (n_bands, H, W)

    # --- Compute spectral indices ---
    print("Computing spectral indices (NDVI, NDBI, NDWI)...")
    ndvi, ndbi, ndwi = compute_indices(bands)

    # --- Assemble feature matrix ---
    # Order must match FEATURE_COLUMNS exactly
    feature_map = {
        "NDVI": ndvi,
        "NDBI": ndbi,
        "NDWI": ndwi,
        "TEXTURE_CONTRAST": bands[6].astype(float),
        "TEXTURE_HOMOGENEITY": bands[7].astype(float),
        "B2": bands[0].astype(float),
        "B3": bands[1].astype(float),
        "B4": bands[2].astype(float),
        "B8": bands[3].astype(float),
        "B11": bands[4].astype(float),
        "B12": bands[5].astype(float),
    }

    X = np.stack([feature_map[col] for col in FEATURE_COLUMNS], axis=-1)  # (H, W, n_features)
    flat_X = X.reshape(-1, len(FEATURE_COLUMNS))

    # Handle nodata pixels (NaN/inf)
    valid_mask = np.isfinite(flat_X).all(axis=1)
    predictions = np.zeros(flat_X.shape[0], dtype=np.uint8)
    if valid_mask.any():
        print(f"Classifying {valid_mask.sum():,} valid pixels...")
        predictions[valid_mask] = model.predict(flat_X[valid_mask]).astype(np.uint8)

    classified = predictions.reshape(height, width)

    # --- Write output ---
    out_meta = meta.copy()
    out_meta.update(dtype="uint8", count=1, nodata=255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(classified[None, ...])
    print(f"Saved classified raster to {output_path}")

    # Print class distribution
    unique, counts = np.unique(classified[classified != 255], return_counts=True)
    from src.config import LAND_COVER_CLASSES
    print("\nClassified pixel distribution:")
    for cls, cnt in zip(unique, counts):
        label = LAND_COVER_CLASSES.get(int(cls), "unknown")
        pct = 100.0 * cnt / classified.size
        print(f"  {label:15s} ({cls}): {cnt:>10,} px  ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply trained land-cover model to a Sentinel-2 GeoTIFF."
    )
    parser.add_argument("input", type=Path, help="Input Sentinel-2 GeoTIFF composite.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "land_cover_xgboost_v4.joblib",
        help="Path to trained model .joblib file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output classified raster path (.tif).",
    )
    args = parser.parse_args()
    classify_raster(args.input, args.model, args.out)
