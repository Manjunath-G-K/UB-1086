import os
import rasterio
import pandas as pd

BASE_DIR = "SwinburneData/FloodMaps"
OUTPUT_CSV = os.path.join(BASE_DIR, "raster_catalog.csv")

records = []

print(f"🔍 Scanning flood maps in: {BASE_DIR}\n")
import re


def detect_metric(fname):
    """Detect flood metric type from filename (case-insensitive)."""
    fname = fname.lower()

    # direct matches
    for m in ["dmax", "hmax", "vmax", "z0max"]:
        if m in fname:
            return m

    # variants with (maxmax)
    if re.search(r"d\(maxmax\)", fname):
        return "dmaxmax"
    if re.search(r"h\(maxmax\)", fname):
        return "hmaxmax"

    return "Unknown"



for root, dirs, files in os.walk(BASE_DIR):
    for fname in files:
        if fname.lower().endswith(".tif"):
            fpath = os.path.join(root, fname)
            try:
                with rasterio.open(fpath) as src:
                    crs = src.crs.to_string() if src.crs else "Unknown"
                    bounds = src.bounds
                    width, height = src.width, src.height

                name = os.path.basename(fname)
                tokens = name.split("_")
                return_period = next((t for t in tokens if "y" in t and t[:-1].isdigit()), "Unknown")
                metric = detect_metric(fname)


                records.append({
                    "file_name": fname,
                    "full_path": fpath,
                    "return_period": return_period,
                    "metric": metric,
                    "crs": crs,
                    "width": width,
                    "height": height,
                    "xmin": bounds.left,
                    "ymin": bounds.bottom,
                    "xmax": bounds.right,
                    "ymax": bounds.top
                })
                print(f"✅ {fname} ({metric}, {return_period})")

            except Exception as e:
                print(f"⚠️ Could not read {fname}: {e}")

if records:
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved catalog with {len(df)} entries → {OUTPUT_CSV}")
else:
    print("\n⚠️ No raster files found!")
