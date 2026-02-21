# build_house_exposure_from_gpkg.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point

# ----------------------------
# CONFIG
# ----------------------------
PROPERTY_GPKG = "SwinburneData/Property/Properties.gpkg"   # your property polygons
FLOOD_ROOT    = "SwinburneData/FloodMaps/FrankstonSouth"   # root containing 001y,002y,...,100y
OUT_CSV       = "houses/house_flood_exposure_from_gpkg.csv"

# Which suburb to keep for MVP (change/remove filter if you want all)
KEEP_SUBURB = "FRANKSTON SOUTH"   # case-insensitive

# Scenarios → subfolders
SCEN_DIR = {
    "001y": "001y/Mapping",
    "002y": "002y/Mapping",
    "005y": "005y/Mapping",
    "010y": "010y/Mapping",
    "020y": "020y/Mapping",
    "050y": "050y/Mapping",
    "100y": "100y/Final",
}

# For each metric, try these file name patterns (we’ll try .tab first, then .grd)
# {scen} will be 001y/002y/.../100y (without leading FS_001_)
PATTERNS = {
    "dmax": [
        "FS_001_{scen}_010m_dmax",                 # regular sets
        "FS_{scen}_010m_102_d(maxmax)_g002",       # 100y alt
        "FS_{scen}_020m_301_d(maxmax)_g002",       # 100y alt
    ],
    "hmax": [
        "FS_001_{scen}_010m_hmax",
        "FS_{scen}_010m_102_h(maxmax)_g002",
        "FS_{scen}_020m_301_h(maxmax)_g002",
    ],
    "Vmax": [
        "FS_001_{scen}_010m_Vmax",
        "FS_{scen}_010m_102_V(maxmax)_g002",
        "FS_{scen}_020m_301_V(maxmax)_g002",
        "FS_{scen}_010m_102.2dm_g002",            # sometimes velocity is packaged as 2dm
        "FS_{scen}_020m_301.2dm_g002",
    ],
    "Z0max": [
        "FS_001_{scen}_010m_Z0max",
        "FS_{scen}_010m_102_Z(maxmax)_g002",
        "FS_{scen}_020m_301_Z(maxmax)_g002",
    ],
}

# ----------------------------
# Helpers
# ----------------------------
def _find_raster(folder, stem):
    grd = os.path.join(folder, f"{stem}.grd")
    tab = os.path.join(folder, f"{stem}.tab")
    if os.path.exists(grd): return grd
    if os.path.exists(tab): return tab
    return None


def _open_best_band(path):
    """
    Open raster and pick a sensible data band:
      - prefer float dtype bands (f32/f64) with nodata (often real data)
      - else use band 1, but treat 255 and -1e37 as nodata later
    Returns (dataset, band_index), where band_index is 1-based.
    """
    ds = rasterio.open(path)
    # If single band, use it
    if ds.count == 1:
        return ds, 1

    # Try to find a float band
    for b in range(1, ds.count + 1):
        if np.issubdtype(ds.dtypes[b - 1], np.floating):
            return ds, b

    # Fall back to first band
    return ds, 1

def _sample_points(ds, band_idx, xy_list):
    """Sample a list of (x,y) in ds CRS, return np.array values with nodata handled."""
    vals = []
    nodata = ds.nodatavals[band_idx - 1]
    for x, y in xy_list:
        # quick bounds check
        if not (ds.bounds.left <= x <= ds.bounds.right and ds.bounds.bottom <= y <= ds.bounds.top):
            vals.append(np.nan)
            continue
        try:
            v = list(ds.sample([(x, y)]))[0][band_idx - 1]
        except Exception:
            v = np.nan
        # normalize nodata-ish values
        if v is None:
            vals.append(np.nan)
        else:
            if nodata is not None and np.isclose(v, nodata):
                vals.append(np.nan)
            elif v == 255 or v == -1e37:  # common placeholders
                vals.append(np.nan)
            else:
                vals.append(float(v))
    return np.array(vals, dtype=float)

def _extract_metric_for_scenario(gdf_wgs84_points, scen_key, metric_key):
    """
    For a scenario (e.g., 001y) and metric (dmax/hmax/Vmax/Z0max),
    try each naming pattern until we find a raster.
    Sample values at property centroids.
    """
    folder = os.path.join(FLOOD_ROOT, SCEN_DIR[scen_key])
    patterns = PATTERNS[metric_key]
    raster_path = None
    for p in patterns:
        stem = p.format(scen=scen_key)
        candidate = _find_raster(folder, stem)
        if candidate:
            raster_path = candidate
            break

    col_name = f"{metric_key}_{scen_key}"
    if not raster_path:
        print(f" ⚠️  No raster for {metric_key} in {scen_key}")
        return pd.Series(np.nan, index=gdf_wgs84_points.index, name=col_name)

    # Open and pick best band
    ds, band_idx = _open_best_band(raster_path)
    print(f"Extracting {metric_key} for {scen_key} from {raster_path} (band {band_idx}) ...")

    # Reproject points to raster CRS and sample
    pts = gdf_wgs84_points.to_crs(ds.crs)
    xy = [(p.x, p.y) for p in pts.geometry]
    vals = _sample_points(ds, band_idx, xy)
    ds.close()
    return pd.Series(vals, index=gdf_wgs84_points.index, name=col_name)

# ----------------------------
# Pipeline
# ----------------------------
def main():
    # Load property polygons
    print("🏠 Loading property polygons from GPKG ...")
    props = gpd.read_file(PROPERTY_GPKG)
    props = props.to_crs(epsg=4326)

    

    # Optional suburb filter for MVP speed/clarity
    if "Suburb" in props.columns and KEEP_SUBURB:
        keep = props["Suburb"].astype(str).str.upper() == KEEP_SUBURB.upper()
        props = props.loc[keep].copy()
        if props.empty:
            print(f"⚠️ No properties found for suburb '{KEEP_SUBURB}'.")
            return

    # Build clean address + centroid
    def s(x): return "" if pd.isna(x) else str(x).strip()
    if not {"House","Street","Suburb","Postcode"}.issubset(props.columns):
        print("⚠️ Properties.gpkg missing one of [House, Street, Suburb, Postcode].")
    props["Full_Address"] = (
        props.get("House", "").astype(str).fillna("").str.strip() + " " +
        props.get("Street","").astype(str).fillna("").str.strip() + ", " +
        props.get("Suburb","").astype(str).fillna("").str.upper() + " VIC " +
        props.get("Postcode","").astype(str).fillna("").str.strip()
    ).str.replace(r"\s+,", ",", regex=True).str.replace(r",\s+VIC\s+$", "", regex=True)

    pts = props.copy()
    pts["geometry"] = props.geometry.centroid
    pts["lat"] = pts.geometry.y
    pts["lon"] = pts.geometry.x
    pts["house_id"] = np.arange(1, len(pts) + 1)

    # Compute metrics per scenario
    out = pts[["house_id","Full_Address","lat","lon","Suburb","Postcode"]].copy()
    for scen in SCEN_DIR.keys():
        for metric in ["dmax","hmax","Vmax","Z0max"]:
            series = _extract_metric_for_scenario(pts, scen, metric)
            out[series.name] = series.values

    # Optional: drop houses with all-NaN across scenarios (outside rasters)
    metric_cols = [c for c in out.columns if any(k in c for k in ["dmax_","hmax_","Vmax_","Z0max_"])]
    keep_mask = ~out[metric_cols].isna().all(axis=1)
    kept = out.loc[keep_mask].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    kept.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved exposure table → {OUT_CSV}")
    print(f"Total houses processed: {len(kept)} (from {len(out)})")

if __name__ == "__main__":
    main()
