import os
import time
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime


load_dotenv()

API_KEY = os.getenv("EIA_API_KEY")
ROOT_URL = "https://api.eia.gov/v2/electricity"

BASE_PARAMS = {
    "api_key": API_KEY
}

DATA_PARAMS = {
    "api_key": API_KEY,
    "frequency": "monthly",
    "data[]": ["price", "sales", "revenue", "customers"]
}

# Folder setup
RAW_PATH = "data/raw"
LOG_PATH = "logs/ingestion.log"

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ---------------- UTILITIES ----------------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def get_json(url, params=None):
    """Safely fetch JSON from API."""
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        log(f"❌ Error fetching {url}: {e}")
        return None


def save_json(data, path):
    """Save any dictionary as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_dataframe(df, route_id):
    """Save data as CSV and Parquet."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    folder = os.path.join(RAW_PATH, route_id)
    os.makedirs(folder, exist_ok=True)

    csv_path = os.path.join(folder, f"{route_id}_{ts}.csv")
    parquet_path = os.path.join(folder, f"{route_id}_{ts}.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    log(f"📦 Saved {len(df)} rows to {csv_path}")


# ---------------- FETCH ROUTES ----------------
def get_routes():
    """Fetch all child routes under /electricity"""
    log("🔍 Fetching electricity routes...")
    data = get_json(f"{ROOT_URL}", BASE_PARAMS)
    if not data:
        return []

    routes = data["response"]["routes"]
    route_ids = [r["id"] for r in routes]
    log(f"✅ Found {len(route_ids)} routes: {route_ids}")
    return route_ids


# ---------------- FETCH METADATA ----------------
def get_metadata(route_id):
    """Fetch metadata for a given route (facets, frequencies, etc.)"""
    url = f"{ROOT_URL}/{route_id}"
    meta = get_json(url, BASE_PARAMS)
    if meta:
        folder = os.path.join(RAW_PATH, route_id)
        os.makedirs(folder, exist_ok=True)
        meta_path = os.path.join(folder, f"{route_id}_metadata.json")
        save_json(meta, meta_path)
        log(f"📘 Saved metadata for {route_id}")
    return meta


# ---------------- FETCH DATA ----------------
def get_data(route_id):
    """Fetch paginated data from a given route's /data endpoint"""
    url = f"{ROOT_URL}/{route_id}/data"
    offset = 0
    length = 5000
    all_rows = []

    while True:
        params = DATA_PARAMS.copy()
        params.update({"offset": offset, "length": length})
        res = get_json(url, params)
        if not res or "response" not in res:
            break

        rows = res["response"].get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        log(f"📄 {route_id}: fetched {len(rows)} rows (offset={offset})")

        offset += length
        time.sleep(1)

        if len(rows) < length:
            break

    if all_rows:
        df = pd.DataFrame(all_rows)
        save_dataframe(df, route_id)
    else:
        log(f"⚠️ No data found for {route_id}")


# ---------------- MAIN ----------------
def run_ingestion():
    log("🚀 Starting multi-route ingestion pipeline")
    routes = get_routes()

    for route_id in routes:
        log(f"🔸 Processing route: {route_id}")
        get_metadata(route_id)

        # Some routes may not have 'data', so handle gracefully
        try:
            get_data(route_id)
        except Exception as e:
            log(f"⚠️ Skipped data fetch for {route_id}: {e}")

    log("✅ All routes processed successfully!")


if __name__ == "__main__":
    run_ingestion()
