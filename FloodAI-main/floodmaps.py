import os
import rasterio
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
HOUSE_CSV = "houses/houses_in_flood_region.csv"
FLOODMAP_ROOT = "SwinburneData/FloodMaps/FrankstonSouth"
OUTPUT_CSV = "houses/house_flood_exposure.csv"

# Flood scenarios
RETURN_PERIODS = ["001y", "002y", "005y", "010y", "020y", "050y", "100y"]

# Raster identifiers by category
RASTER_KEYWORDS = {
    "dmax": ["dmax", "d(maxmax)", "_d("],
    "hmax": ["hmax", "h(maxmax)", "_h("],
    "Vmax": ["Vmax", "2dm"],  # handle 100y special case
    "Z0max": ["Z0max", "Z(maxmax)", "_Z("],
}

# ---------------------------------------------------------------------
# HELPER: Find the matching raster file by keyword
# ---------------------------------------------------------------------
def find_raster(base_path, keywords):
    for root, _, files in os.walk(base_path):
        for f in files:
            name = f.lower()
            if name.endswith(".grd"):
                for kw in keywords:
                    if kw.lower() in name:
                        return os.path.join(root, f)
    return None

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
houses = pd.read_csv(HOUSE_CSV)
print(f"🏠 Loaded {len(houses)} houses")

for period in RETURN_PERIODS:
    print(f"\n>>> Processing {period} ...")
    period_path = os.path.join(FLOODMAP_ROOT, period)

    if not os.path.exists(period_path):
        print(f"⚠️ Folder missing: {period_path}")
        continue

    for key, patterns in RASTER_KEYWORDS.items():
        raster_file = find_raster(period_path, patterns)
        if raster_file is None:
            print(f"⚠️ No raster for {key} in {period}")
            houses[f"{key}_{period}"] = None
            continue

        print(f" Extracting {key} from {raster_file} ...")

        with rasterio.open(raster_file) as src:
            coords = [(x, y) for x, y in zip(houses["lon"], houses["lat"])]
            values = []
            for coord in coords:
                try:
                    for val in src.sample([coord]):
                        v = float(val[0])
                        if v > 250 or v < -999:
                            v = None  # filter no-data and invalid values
                        values.append(v)
                except Exception:
                    values.append(None)
            houses[f"{key}_{period}"] = values

# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------
houses.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved exposure table → {OUTPUT_CSV}")
