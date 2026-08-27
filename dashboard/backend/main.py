"""
Week 3 — Dashboard backend.

Serves district-level land conversion time series as GeoJSON for the
map-based frontend (time slider + district choropleth).

Run:
    uvicorn dashboard.backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import geopandas as gpd

from src.config import CONVERSION_RESULTS_PATH, PHASE_1_DISTRICTS, YEARS

app = FastAPI(title="Punjab Ag-to-Urban Conversion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before production deployment
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "districts": PHASE_1_DISTRICTS, "years": YEARS}


@app.get("/conversion")
def get_conversion(district: str | None = None, year: int | None = None):
    """Return district conversion polygons, optionally filtered."""
    if not CONVERSION_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results yet at {CONVERSION_RESULTS_PATH}. Run the pipeline first.",
        )

    gdf = gpd.read_file(CONVERSION_RESULTS_PATH)

    if district:
        gdf = gdf[gdf["district"].str.lower() == district.lower()]
    if year:
        gdf = gdf[gdf["year"] == year]

    return gdf.__geo_interface__


@app.get("/districts")
def list_districts():
    return {"districts": PHASE_1_DISTRICTS}


@app.get("/years")
def list_years():
    return {"years": YEARS}
