import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from pyproj import Transformer

# === Paths ===
SUBS_PATH = "SwinburneData/Subcatchments/subcatchments.gpkg"
RASTER_CATALOG = "SwinburneData/FloodMaps/raster_catalog.csv"
OUTPUT_FILE = "SwinburneData/FloodAI/flood_metrics_by_subcatchment.csv"

# === Load subcatchments ===
print("📥 Reading subcatchments...")
subs = gpd.read_file(SUBS_PATH)
subs["geometry"] = subs.geometry.centroid
subs = subs.to_crs(epsg=4326)
subs["lon"] = subs.geometry.x
subs["lat"] = subs.geometry.y
print(f"✅ Loaded {len(subs)} subcatchment centroids")

# === Load raster catalog ===
catalog = pd.read_csv(RASTER_CATALOG)
catalog = catalog[catalog["crs"].notnull()]

# === Prepare output dataframe ===
results = subs[["subcatchment_id", "lat", "lon"]].copy()

# === Sampling function ===
def sample_raster_value(raster_path, lat, lon, raster_crs):
    """Sample raster value at given lat/lon (auto reproject)"""
    try:
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        with rasterio.open(raster_path) as src:
            # handle multi-band rasters
            val = list(src.sample([(x, y)]))[0]
            if len(val) > 1:
                val = val[0]
            return float(val)
    except Exception as e:
        print(f"⚠️ Sampling failed for {raster_path}: {e}")
        return None

# === Iterate through rasters ===
for _, row in catalog.iterrows():
    raster_path = row["full_path"]
    metric = row["metric"].replace(".grd", "")
    period = row["return_period"]
    raster_crs = row["crs"]

    col_name = f"{period}_{metric}"
    print(f"🌍 Sampling: {col_name}")

    values = []
    for _, s in subs.iterrows():
        v = sample_raster_value(raster_path, s["lat"], s["lon"], raster_crs)
        values.append(v)

    results[col_name] = values

# === Save merged output ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Flood metrics saved → {OUTPUT_FILE}")
print(f"Columns: {len(results.columns)} (includes {len(catalog)} raster metrics)")
