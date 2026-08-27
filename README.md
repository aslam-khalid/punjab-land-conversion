# Automated Detection & Mapping of Agricultural-to-Urban Land Conversion — Punjab

AIRI Team / PITB internship project. Detects and quantifies farmland-to-urban
conversion across Punjab districts using multi-temporal Sentinel-2 imagery.

## Project structure

```
auc-punjab/
├── data/
│   ├── raw/               # downloaded Sentinel-2 composites, housing society boundaries
│   ├── interim/           # spectral indices, intermediate rasters
│   └── processed/         # classified rasters, district-level conversion results
├── src/
│   ├── config.py          # districts, years, GEE settings, all shared params
│   ├── acquisition/       # Week 1 — Sentinel-2 retrieval via Google Earth Engine
│   ├── features/          # Week 1 — NDVI / NDBI / NDWI computation
│   ├── classification/    # Week 1 — XGBoost/Random Forest land-cover classifier
│   ├── change_detection/  # Week 2 — pixel-level ag→urban transition detection
│   ├── aggregation/       # Week 2 — district rollup + housing-society validation
│   └── utils/             # GEE auth, shared helpers
├── dashboard/
│   ├── backend/           # Week 3 — FastAPI serving conversion GeoJSON
│   └── frontend/          # Week 3 — Streamlit map + time slider
├── models/                 # trained classifier artifacts (.joblib)
├── notebooks/               # exploration / QA notebooks
├── reports/                 # district-level report outputs
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# One-time Earth Engine auth (opens a browser)
earthengine authenticate
```

For server/dashboard use, copy `.env.example` to `.env` and fill in a GEE
service account instead of interactive auth.

## Pipeline (matches the 3-week plan in the proposal)

**Week 1 — Data pipeline & model development**
```bash
python -m src.acquisition.fetch_sentinel2 --district Lahore --year 2025
python -m src.features.spectral_indices data/raw/s2_lahore_2025.tif
python -m src.classification.train_classifier data/processed/training_samples.csv
```

**Week 2 — Change detection & validation**
```bash
python -m src.change_detection.detect_conversion \
    data/processed/lahore_2019_classified.tif \
    data/processed/lahore_2025_classified.tif \
    --out data/processed/lahore_2019_2025_conversion.tif

python -m src.aggregation.aggregate_and_validate Lahore \
    data/processed/lahore_2019_2025_conversion.tif
```

**Week 3 — Dashboard**
```bash
uvicorn dashboard.backend.main:app --reload --port 8000
# in a second terminal:
streamlit run dashboard/frontend/app.py
```

## Study area (Phase 1)

Lahore, Sheikhupura, Kasur, Faisalabad, Multan, Rawalpindi — see `src/config.py`
to add districts for the province-wide phase 2 rollout.

## Notes

- Sentinel-2 composites use a Nov–Feb dry-season window per year to minimize
  cloud cover and maximize contrast between bare/harvested farmland and
  built-up surfaces.
- `data/raw/housing_societies.geojson` is expected for validation (Section 5,
  "Aggregation and validation" in the proposal) — source this from PITB/LDA
  records or digitize known DHA/Bahria Town boundaries.
- Classifier defaults to XGBoost per the proposal; switch `CLASSIFIER_TYPE`
  in `src/config.py` to `"random_forest"` if needed. U-Net segmentation is
  scoped as a phase-two extension.
