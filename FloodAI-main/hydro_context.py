import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

def get_subcatchment(point: Point, subcatchments_gdf: gpd.GeoDataFrame):
    """
    Identify which subcatchment a given point falls within.
    """
    hits = subcatchments_gdf[subcatchments_gdf.intersects(point)]
    return hits.iloc[0]['subcatchment_id'] if not hits.empty else None


def get_ifd_curve(station_table: pd.DataFrame, station_id=None, duration_min=None):
    """
    Builds an IFD curve dictionary (ARI → rainfall mm) for a given duration.
    Works with IFD tables where columns are percentages (AEPs).
    """
    # Normalize headers
    station_table.columns = station_table.columns.str.strip().str.lower()

    # Identify duration column
    duration_col = 'duration in min' if 'duration in min' in station_table.columns else None
    if duration_col is None:
        raise KeyError("❌ Expected 'Duration in min' column missing.")

    # Filter for selected duration
    rows = station_table[station_table[duration_col] == duration_min]
    if rows.empty:
        raise ValueError(f"No rows found for duration {duration_min} min")

    row = rows.iloc[0]

    # Extract percentage-based columns (AEPs)
    percent_cols = [c for c in row.index if '%' in c]

    # Convert to dictionary: ARI → rainfall value
    ifd_curve = {}
    for col in percent_cols:
        try:
            aep = float(col.replace('%', '')) / 100.0
            ari = round(1.0 / aep, 2)
            ifd_curve[ari] = float(row[col])
        except ValueError:
            continue

    if not ifd_curve:
        raise ValueError("No valid rainfall values found in percentage columns.")

    return ifd_curve


def rain_to_ari(rain_mm, duration_min, ifd_curve, uplift=0.0):
    """
    Convert observed rainfall (mm) to its corresponding ARI and AEP using interpolation.
    """
    target = rain_mm * (1.0 + uplift / 100.0)
    aris = sorted(ifd_curve.keys())
    values = [ifd_curve[a] for a in aris]

    lo_idx = max([i for i, v in enumerate(values) if v <= target], default=0)
    hi_idx = min([i for i, v in enumerate(values) if v >= target], default=len(values) - 1)

    lo_ari, hi_ari = aris[lo_idx], aris[hi_idx]
    lo_val, hi_val = values[lo_idx], values[hi_idx]

    if lo_ari == hi_ari or hi_val == lo_val:
        aep = 1.0 / lo_ari
        return float(lo_ari), aep

    # Linear interpolation
    ratio = (target - lo_val) / (hi_val - lo_val)
    interp_aep = (1.0 / lo_ari) + ((1.0 / hi_ari) - (1.0 / lo_ari)) * ratio
    return 1.0 / interp_aep, interp_aep
