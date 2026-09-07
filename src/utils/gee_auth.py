"""
Google Earth Engine authentication helper.

First-time setup (run once, interactively, outside of any script):
    earthengine authenticate

Or, for a service account (recommended for the dashboard/server context):
    1. Create a service account in Google Cloud Console with Earth Engine access.
    2. Download its JSON key.
    3. Set GEE_SERVICE_ACCOUNT and GEE_KEY_FILE below (or via environment
       variables of the same name), or place the key at
       data/raw/gee_service_account.json and set GEE_SERVICE_ACCOUNT only.
"""

import os
import ee


def init_gee() -> None:
    """Initialize the Earth Engine API, preferring a service account if configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    service_account = os.environ.get("GEE_SERVICE_ACCOUNT")
    key_file = os.environ.get("GEE_KEY_FILE")
    project = os.environ.get("GEE_PROJECT")

    try:
        if service_account and key_file:
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials, project=project)
        else:
            # Falls back to locally cached user credentials from
            # `earthengine authenticate`
            ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Earth Engine initialization failed. Run `earthengine authenticate` "
            "once locally, or set GEE_SERVICE_ACCOUNT / GEE_KEY_FILE env vars "
            f"for service-account auth. Original error: {exc}"
        ) from exc


if __name__ == "__main__":
    init_gee()
    print("Earth Engine initialized successfully.")
