"""
Week 3 — Dashboard frontend (Streamlit).

Quick interactive map + time slider consuming the FastAPI backend.
Run:
    streamlit run dashboard/frontend/app.py

(Swap for a Next.js/Leaflet app later if a richer UI is needed — the
FastAPI backend is UI-agnostic either way.)
"""

import requests
import streamlit as st
import pydeck as pdk

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Punjab Ag-to-Urban Conversion", layout="wide")
st.title("Agricultural-to-Urban Land Conversion — Punjab")

try:
    districts = requests.get(f"{API_BASE}/districts", timeout=5).json()["districts"]
    years = requests.get(f"{API_BASE}/years", timeout=5).json()["years"]
except requests.exceptions.ConnectionError:
    st.error("Backend not reachable. Start it with: `uvicorn dashboard.backend.main:app --reload`")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    selected_district = st.selectbox("District", ["All"] + districts)
with col2:
    selected_year = st.select_slider("Year", options=years, value=years[-1])

params = {"year": selected_year}
if selected_district != "All":
    params["district"] = selected_district

resp = requests.get(f"{API_BASE}/conversion", params=params, timeout=10)

if resp.status_code == 404:
    st.warning("No results yet — run the pipeline (acquisition -> classification -> "
               "change detection -> aggregation) to populate data/processed/.")
else:
    geojson = resp.json()
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=31.5, longitude=73.5, zoom=6),
            layers=[
                pdk.Layer(
                    "GeoJsonLayer",
                    geojson,
                    filled=True,
                    get_fill_color=[220, 80, 40, 140],
                    pickable=True,
                )
            ],
        )
    )
    st.caption(f"Showing conversion zones — {selected_district}, {selected_year}")
