"""
Week 1 — Feature extraction.

Computes NDVI, NDBI, and NDWI from downloaded Sentinel-2 GeoTIFFs
(band order must match src.config.S2_BANDS).
"""

from pathlib import Path
import numpy as np
import rasterio

from src.config import S2_BANDS


def _read_band(src: rasterio.DatasetReader, band_name: str) -> np.ndarray:
    band_index = list(S2_BANDS.values()).index(S2_BANDS[band_name]) + 1
    return src.read(band_index).astype("float32")


def compute_indices(tif_path: Path) -> dict[str, np.ndarray]:
    """Return NDVI, NDBI, NDWI arrays for a given composite GeoTIFF."""
    with rasterio.open(tif_path) as src:
        nir = _read_band(src, "nir")
        red = _read_band(src, "red")
        green = _read_band(src, "green")
        swir1 = _read_band(src, "swir1")

    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndbi = (swir1 - nir) / (swir1 + nir + eps)
    ndwi = (green - nir) / (green + nir + eps)

    return {"NDVI": ndvi, "NDBI": ndbi, "NDWI": ndwi}


def save_indices(tif_path: Path, out_dir: Path) -> None:
    """Compute indices and write each as its own single-band GeoTIFF."""
    indices = compute_indices(tif_path)
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tif_path.stem
    for name, array in indices.items():
        out_path = out_dir / f"{stem}_{name}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(array[None, ...])
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute spectral indices for a composite.")
    parser.add_argument("tif_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()
    save_indices(args.tif_path, args.out_dir)
