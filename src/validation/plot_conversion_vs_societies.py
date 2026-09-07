"""Plot detected conversion zones against known housing societies."""

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.config import HOUSING_SOCIETY_BOUNDARIES_PATH, PROCESSED_DIR


DEFAULT_CONVERSION_PATH = PROCESSED_DIR / "faisalabad_conversion_summary.geojson"
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "validation_check.png"
METRIC_CRS = "EPSG:32643"


def plot_validation_check(
    conversion_path: Path = DEFAULT_CONVERSION_PATH,
    societies_path: Path = HOUSING_SOCIETY_BOUNDARIES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Save a metric-CRS overlay and print both layers' bounding boxes."""
    conversion = gpd.read_file(conversion_path)
    societies = gpd.read_file(societies_path)

    conversion_utm = conversion.to_crs(METRIC_CRS)
    societies_utm = societies.to_crs(METRIC_CRS)

    fig, ax = plt.subplots(figsize=(12, 12))
    conversion_utm.plot(
        ax=ax,
        color="red",
        alpha=0.5,
        label="Detected conversion",
        rasterized=True,
    )
    societies_utm.plot(
        ax=ax,
        color="royalblue",
        edgecolor="black",
        alpha=0.6,
        label="Known housing societies",
    )

    for _, row in societies_utm.iterrows():
        name = row.get("name", "Unnamed society")
        centroid = row.geometry.centroid
        ax.annotate(
            name,
            xy=(centroid.x, centroid.y),
            fontsize=9,
            ha="center",
            color="darkblue",
        )

    try:
        import contextily as ctx
    except ImportError:
        ctx = None

    if ctx is not None:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=METRIC_CRS)
    else:
        ax.set_facecolor("#eef2f3")
        print("Note: contextily is not installed; plotted without an OSM basemap.")

    ax.set_title("Faisalabad: Detected Conversion (2019->2025) vs Known Housing Societies")
    ax.legend(
        handles=[
            Patch(facecolor="red", alpha=0.5, label="Detected conversion"),
            Patch(
                facecolor="royalblue",
                edgecolor="black",
                alpha=0.6,
                label="Known housing societies",
            ),
        ]
    )
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved validation plot to {output_path}")
    print("\nConversion zones bounding box (UTM):", conversion_utm.total_bounds)
    print("Housing societies bounding box (UTM):", societies_utm.total_bounds)


def main() -> None:
    plot_validation_check()


if __name__ == "__main__":
    main()
