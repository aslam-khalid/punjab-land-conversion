"""
Central configuration for the AUC (Agricultural-to-Urban Conversion) pipeline.
Edit this file rather than hardcoding paths/params in individual scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Study area — Phase 1 districts (Scope, Section 4 of proposal)
# ---------------------------------------------------------------------------
PHASE_1_DISTRICTS = [
    "Lahore",
    "Sheikhupura",
    "Kasur",
    "Faisalabad",
    "Multan",
    "Rawalpindi",
]

# GAUL / FAO admin boundary dataset used to clip districts in GEE
GEE_ADMIN_BOUNDARIES = "FAO/GAUL/2015/level2"
GEE_ADMIN_COUNTRY_FIELD = "ADM0_NAME"
GEE_ADMIN_COUNTRY_VALUE = "Pakistan"
GEE_ADMIN_DISTRICT_FIELD = "ADM2_NAME"

# ---------------------------------------------------------------------------
# Time points (Methodology, Section 5)
# ---------------------------------------------------------------------------
YEARS = [2016, 2019, 2022, 2025]

# Sentinel-2 surface reflectance collection
GEE_S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
CLOUD_FILTER_PCT = 20          # max scene cloud % to include
CLOUD_PROB_THRESHOLD = 40      # per-pixel cloud probability mask threshold
COMPOSITE_MONTHS = (11, 12, 1, 2)  # Nov-Feb: dry season, minimal cloud cover, post-harvest contrast

# Sentinel-2 band references (10m/20m bands used)
S2_BANDS = {
    "blue": "B2",
    "green": "B3",
    "red": "B4",
    "nir": "B8",
    "swir1": "B11",
    "swir2": "B12",
}

# ---------------------------------------------------------------------------
# Spectral indices thresholds (used for weak labeling / sanity checks only —
# the actual classifier learns its own decision boundary)
# ---------------------------------------------------------------------------
NDVI_VEGETATION_THRESHOLD = 0.3
NDBI_BUILTUP_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
LAND_COVER_CLASSES = {
    0: "agricultural",
    1: "built_up",
    2: "water",
    3: "barren",
}
CLASSIFIER_TYPE = "xgboost"  # "xgboost" | "random_forest" | "unet" (phase 2)
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.2

FEATURE_COLUMNS = [
    "NDVI",
    "NDBI",
    "NDWI",
    "TEXTURE_CONTRAST",
    "TEXTURE_HOMOGENEITY",
    "B2", "B3", "B4", "B8", "B11", "B12",
]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
HOUSING_SOCIETY_BOUNDARIES_PATH = RAW_DIR / "housing_societies.geojson"

# ---------------------------------------------------------------------------
# Dashboard / API
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
CONVERSION_RESULTS_PATH = PROCESSED_DIR / "district_conversion_timeseries.geojson"
