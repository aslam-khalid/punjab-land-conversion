# Punjab Land Conversion Detection

Automated detection and mapping of agricultural-to-urban land conversion in Punjab, Pakistan, using multi-temporal Sentinel-2 satellite imagery and machine learning. Built as part of a PITB AIRI Team internship project.

## Problem

Punjab has seen rapid, largely unmonitored conversion of agricultural land into private housing societies over the past decade. There is currently no automated, province-wide method to measure how much farmland has been lost to real estate development, or to identify which areas are converting fastest. This project builds a pipeline to detect and quantify that conversion directly from satellite imagery.

## How it works

1. **Data acquisition** — Sentinel-2 imagery (10m resolution) pulled via Google Earth Engine for two time points (e.g. 2019, 2025)
2. **Feature extraction** — spectral indices (NDVI, NDBI, NDWI) plus GLCM texture bands (contrast, homogeneity) computed server-side in GEE
3. **Classification** — an XGBoost pixel classifier labels each pixel as agricultural, built-up, water, or barren
4. **Change detection** — pixel-level comparison between the two years flags agricultural → built-up transitions
5. **Noise filtering** — a minimum mapping unit (MMU) filter removes small isolated pixel noise, followed by a compactness filter that removes elongated linear artifacts (roads, canals, field boundaries) that survive size filtering
6. **Validation** — detected conversion zones are checked against known housing society locations to sanity-check results
7. **Aggregation** — results are vectorized and summarized as total converted hectares, with per-patch detail available for manual inspection

## Pipeline
fetch_sentinel2.py → pulls & exports Sentinel-2 composites from GEE
generate_training_data.py → samples training pixels using ESA WorldCover labels
train_classifier.py → trains the XGBoost land-cover classifier
classify_raster.py → applies the trained model to a full composite
detect_conversion.py → compares two classified rasters, filters noise (MMU + compactness)
aggregate_and_validate.py → vectorizes results, validates against known housing societies
run_change_detection.py → orchestrates the full pipeline end-to-end

Validation utilities:
plot_conversion_vs_societies.py → visual overlay of detected conversion vs. reference societies
inspect_largest_patches.py → prints largest detected patches with coordinates for manual spot-checking

## Setup

bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Copy `.env.example` to `.env` and fill in:

GEE_SERVICE_ACCOUNT=<your-service-account-email>
GEE_KEY_FILE=<path-to-service-account-key.json>


## Usage

bash
# 1. Export satellite composites for a district and year
python -m src.acquisition.fetch_sentinel2 --district Faisalabad --year 2019
python -m src.acquisition.fetch_sentinel2 --district Faisalabad --year 2025

# 2. Download exported tiles from Google Drive into data/raw/, then merge
python merge_tiles.py

# 3. Run the full change detection pipeline
python -m src.pipeline.run_change_detection --district Faisalabad

# 4. Inspect and validate results
python -m src.validation.plot_conversion_vs_societies
python -m src.validation.inspect_largest_patches


## Current results (Faisalabad, 2019–2025)

| Metric | Value |
|---|---|
| Total converted area (post-filtering) | ~412 ha |
| Retained patches (after MMU + compactness filtering) | in progress |
| Validation overlap (6 reference societies) | 0.35% (pre-shape-filter) |

## Known limitations

- **Classifier accuracy**: overall pixel classification accuracy is ~73%. Built-up vs. barren remains the hardest boundary (barren precision ~0.57) since dry soil and concrete can be spectrally similar.
- **Validation reference set is small**: only 6 well-established housing societies are used as ground truth. Most are older developments whose conversion likely predates the 2019–2025 study window, which limits direct overlap validation. Results should be cross-checked manually (see `inspect_largest_patches.py`) rather than trusted on overlap percentage alone.
- **Shape-based false positives**: raw per-pixel change detection is prone to flagging roads, canals, and field boundaries as "conversion." A compactness filter is applied to reduce this, but manual spot-checking of top-detected patches is still recommended before reporting final figures.
- **Single time-period comparison**: results reflect only two snapshots (2019, 2025) and do not capture the pattern or timing of conversion within that window.
- **District scope**: currently implemented and validated for Faisalabad only; not yet run province-wide.

## Tech stack

Google Earth Engine, Python (rasterio, geopandas, scikit-image, scipy), XGBoost, matplotlib

## Author

Muhammad Aslam Khalid — AIRI Team Intern, Punjab Information Technology Board (PITB)
