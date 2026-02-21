import streamlit as st
from shapely.geometry import Point
import datetime as dt
from geolocate import load_properties, resolve_address
from hydro_context import get_subcatchment, get_ifd_curve, rain_to_ari
from rasterio_sampler import sample_raster, interp_log_ari
import pandas as pd
import numpy as np
import re
import geopandas as gpd

st.set_page_config(page_title="FloodAI Alpha", layout="wide")
st.title("🌧️ FloodAI — Alpha Prototype")


# Load core layers
props_gdf = load_properties()
subcatchments_gdf = gpd.read_file("SwinburneData/Subcatchments/Subcatchments.gpkg")
station_ifd_table = pd.read_csv("SwinburneData/IFD/ifd_table.csv")
raster_catalog = pd.read_csv("SwinburneData/FloodMaps/raster_catalog.csv")


# User input
st.subheader("💬 Describe rainfall event")
text = st.text_area("Event description", "60 mm rainfall in 45 minutes at Frankston South")
rain_mm = float(re.search(r"(\d+(?:\.\d+)?)\s*mm", text).group(1)) if re.search(r"(\d+(?:\.\d+)?)\s*mm", text) else 50
minutes = float(re.search(r"(\d+(?:\.\d+)?)\s*(minute|min|mins)", text, re.I).group(1)) if re.search(r"(\d+(?:\.\d+)?)\s*(minute|min|mins)", text, re.I) else 60


addr = re.search(r"in ([A-Za-z ]+)", text)
addr_text = addr.group(1) if addr else "Frankston South"
prop = resolve_address(addr_text, props_gdf)


if not prop:
    st.error("Could not resolve address.")
    st.stop()


lat, lon = prop['lat'], prop['lon']
point = Point(lon, lat)


sub_id = get_subcatchment(point, subcatchments_gdf)
station_id = 1001 # placeholder: assign per subcatchment
ifd_curve = get_ifd_curve(station_ifd_table, station_id, int(minutes))
ari, aep = rain_to_ari(rain_mm, minutes, ifd_curve)


# --- Select RPs for sampling ---
rps = sorted([int(str(r).replace('y','')) for r in raster_catalog['return_period'].unique()])
lo = max([r for r in rps if r <= ari], default=rps[0])
hi = min([r for r in rps if r >= ari], default=rps[-1])

metrics = {}

# --- Determine which column stores raster paths ---
path_col = None
for c in raster_catalog.columns:
    if c.strip().lower() in ['path', 'file', 'filepath', 'filename', 'full_path']:
        path_col = c
        break
if path_col is None:
    raise KeyError("❌ Could not find a valid raster path column (expected 'path', 'file', or 'filepath')")

# --- Normalize for consistent matching ---
raster_catalog['return_period'] = raster_catalog['return_period'].astype(str).str.lower().str.strip()
raster_catalog['metric'] = raster_catalog['metric'].astype(str).str.lower().str.strip()

# --- Define helper for robust matching ---
def match_rp(val, target):
    """Handles variations like '001y', '01Y', '1y', or '1'."""
    return val.replace('y', '').lstrip('0') == str(target).lstrip('0')

# --- Loop through metrics and sample rasters ---
for metric in ['dmax', 'hmax', 'vmax']:
    row_lo = raster_catalog[raster_catalog.apply(lambda r: match_rp(r['return_period'], lo) and r['metric'] == metric, axis=1)]
    row_hi = raster_catalog[raster_catalog.apply(lambda r: match_rp(r['return_period'], hi) and r['metric'] == metric, axis=1)]

    if row_lo.empty or row_hi.empty:
        st.error(f"⚠️ No matching raster found for metric '{metric}' and return period {lo}y/{hi}y.")
        st.stop()

    path_lo = row_lo.iloc[0][path_col]
    path_hi = row_hi.iloc[0][path_col]

    v_lo, _ = sample_raster(lon, lat, path_lo)
    v_hi, _ = sample_raster(lon, lat, path_hi)
    st.write(f"Sampling at lon={lon}, lat={lat}, CRS={src_crs}")



    metrics[metric] = interp_log_ari(lo, hi, v_lo, v_hi, ari)




st.markdown(f"**Resolved property:** {prop['Full_Address']}")
st.markdown(f"**Subcatchment:** {sub_id} | **Event ARI:** {ari:.1f} years | **AEP:** {aep*100:.2f}%")


results = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value (m or m/s)'])
st.dataframe(results)


risk = "High" if metrics['dmax']>0.6 else "Moderate" if metrics['dmax']>0.3 else "Low"
st.warning(f"Risk level: {risk}")


st.caption(f"Timestamp: {dt.datetime.now().isoformat(timespec='seconds')}")