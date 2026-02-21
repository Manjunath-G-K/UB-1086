# -------------------------------------------------------------
# FloodAI Interactive MVP v3
# - Property lookup directly from .gpkg (no CSV)
# - Dynamic rasters 001y..100y with non-linear (log/log) interpolation
# - Zone visualization + nearby properties
# - Tone-aware summary + VIC safety tips
# - Modern pydeck map
# -------------------------------------------------------------
# Usage:
#   streamlit run floodai_interactive_v3.py
# -------------------------------------------------------------

import re
import os
import json
import math
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio

import streamlit as st
import pydeck as pdk

# -------------------------
# CONFIG (update paths if needed)
# -------------------------
PROPERTY_GPKG = "SwinburneData/Property/Properties.gpkg"
FLOODMAP_ROOT = "SwinburneData/FloodMaps/FrankstonSouth"

# Melbourne/Frankston projected CRS (meters) -> GDA2020 / MGA Zone 55 (EPSG:7855)
PROJ_CRS = "EPSG:7855"   # good for buffering/measuring
GEO_CRS  = "EPSG:4326"

RETURN_PERIODS = [1, 2, 5, 10, 20, 50, 100]  # years
# Colors for depth classes (m)
DEPTH_CLASSES = [
    (0.00, 0.05, [200, 200, 200]),  # Dry / ~NoData
    (0.05, 0.15, [180, 255, 180]),  # very shallow
    (0.15, 0.30, [255, 255, 150]),  # shallow
    (0.30, 0.60, [255, 210, 130]),  # moderate
    (0.60, 1.20, [255, 160, 120]),  # deep
    (1.20, 99.0, [255, 90, 90])     # very deep
]

# Raster file patterns per scenario (001y..050y same naming; 100y has "Final")
SCENARIO_SPECS = {
    "001y": {
        "base": "001y/Mapping",
        "files": {
            "dmax": "FS_001_001y_010m_dmax.grd",
            "hmax": "FS_001_001y_010m_hmax.grd",
            "Vmax": "FS_001_001y_010m_Vmax.grd",
            "Z0max": "FS_001_001y_010m_Z0max.grd",
        },
    },
    "002y": {
        "base": "002y/Mapping",
        "files": {
            "dmax": "FS_001_002y_010m_dmax.grd",
            "hmax": "FS_001_002y_010m_hmax.grd",
            "Vmax": "FS_001_002y_010m_Vmax.grd",
            "Z0max": "FS_001_002y_010m_Z0max.grd",
        },
    },
    "005y": {
        "base": "005y/Mapping",
        "files": {
            "dmax": "FS_001_005y_010m_dmax.grd",
            "hmax": "FS_001_005y_010m_hmax.grd",
            "Vmax": "FS_001_005y_010m_Vmax.grd",
            "Z0max": "FS_001_005y_010m_Z0max.grd",
        },
    },
    "010y": {
        "base": "010y/Mapping",
        "files": {
            "dmax": "FS_001_010y_010m_dmax.grd",
            "hmax": "FS_001_010y_010m_hmax.grd",
            "Vmax": "FS_001_010y_010m_Vmax.grd",
            "Z0max": "FS_001_010y_010m_Z0max.grd",
        },
    },
    "020y": {
        "base": "020y/Mapping",
        "files": {
            "dmax": "FS_001_020y_010m_dmax.grd",
            "hmax": "FS_001_020y_010m_hmax.grd",
            "Vmax": "FS_001_020y_010m_Vmax.grd",
            "Z0max": "FS_001_020y_010m_Z0max.grd",
        },
    },
    "050y": {
        "base": "050y/Mapping",
        "files": {
            "dmax": "FS_001_050y_010m_dmax.grd",
            "hmax": "FS_001_050y_010m_hmax.grd",
            "Vmax": "FS_001_050y_010m_Vmax.grd",
            "Z0max": "FS_001_050y_010m_Z0max.grd",
        },
    },
    "100y": {
        "base": "100y/Final",
        "files": {
            "dmax": "FS_100y_010m_102_d(maxmax)_g002.grd",
            "hmax": "FS_100y_010m_102_h(maxmax)_g002.grd",
            # No Vmax/Z0max provided for 100y in your dataset; we won’t error if missing.
            "Vmax": None,
            "Z0max": None,
        },
    },
}


# -------------------------
# Utilities
# -------------------------
def now_iso() -> str:
    return dt.datetime.now().isoformat()


@st.cache_data(show_spinner=False)
def load_properties() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(PROPERTY_GPKG)
    # Ensure geometry column present
    if gdf.geometry is None:
        raise ValueError("No geometry column found in the GPKG file.")
    
    # Only set CRS if missing
    if gdf.crs is None:
        gdf = gdf.set_crs(GEO_CRS)
    # Normalize to geographic lat/lon for display
    elif gdf.crs.to_string() != GEO_CRS:
        gdf = gdf.to_crs(GEO_CRS)

    # Keep minimal useful columns
    keep = [
        "Prefix","Unit","Level","House","Street","Suburb","Postcode",
        "PropName","PropNum","PropertyType","PR_PFI","geometry"
    ]
    gdf = gdf[[c for c in keep if c in gdf.columns] + ["geometry"]]

    # Clean up strings
    for col in ["House","Street","Suburb","Postcode"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str).str.strip()
    return gdf



def address_string(row) -> str:
    parts = []
    if pd.notna(row.get("House")) and str(row["House"]).lower() != "nan":
        parts.append(str(row["House"]))
    if pd.notna(row.get("Street")) and str(row["Street"]).lower() != "nan":
        parts.append(str(row["Street"]))
    if pd.notna(row.get("Suburb")) and str(row["Suburb"]).lower() != "nan":
        parts.append(str(row["Suburb"]).upper())
    if pd.notna(row.get("Postcode")) and str(row["Postcode"]).lower() != "nan":
        parts.append(str(row["Postcode"]))
    if not parts:
        return "Unknown"
    # Format e.g. "10 McMahons Road, FRANKSTON VIC 3199"
    if len(parts) >= 3:
        # insert "VIC" before postcode if we have a suburb and postcode
        if parts[-1].isdigit():
            parts.insert(-1, "VIC")
    return ", ".join(parts)


def parse_text_event(text: str) -> Dict:
    """Extract rain (mm), duration (hr), and any coarse location phrases."""
    rain = None
    dur_hr = None
    addr_hint = None

    # rain mm
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mm|millimetre|millimeter|millimetres|millimeters)", text, re.I)
    if m:
        rain = float(m.group(1))

    # duration (minutes/hours)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hr|hours|hrs|minute|min|minutes|mins)", text, re.I)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("hour") or unit.startswith("hr"):
            dur_hr = val
        else:
            dur_hr = val / 60.0

    # coords?
    mc = re.search(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", text)
    latlon = None
    if mc:
        latlon = (float(mc.group(1)), float(mc.group(2)))

    # simple suburb keywords (Frankston, Seaford, Langwarrin, Carrum Downs)
    for suburb in ["Frankston South","Frankston North","Frankston","Seaford","Langwarrin","Carrum Downs","Skye","Sandhurst","Langwarrin South"]:
        if re.search(rf"\b{re.escape(suburb)}\b", text, re.I):
            addr_hint = suburb.upper()

    return {
        "rain_mm": rain,
        "duration_hr": dur_hr,
        "latlon": latlon,
        "addr_hint": addr_hint,
        "timestamp": now_iso(),
        "severity_flag": (rain or 0) >= 60.0
    }


def quality_gate(parsed: Dict) -> Tuple[bool, List[str]]:
    """Validate & tell user what’s missing. We treat rain & location as must-haves."""
    msgs = []
    if parsed.get("rain_mm") is None:
        msgs.append("• I couldn’t find the rainfall amount (mm).")
    if parsed.get("duration_hr") is None:
        msgs.append("• I couldn’t find how long it rained (minutes or hours).")
    if parsed.get("latlon") is None and parsed.get("addr_hint") is None:
        msgs.append("• I couldn’t find a location (address, suburb, or GPS).")
    return (len(msgs) == 0, msgs)


def infer_return_period(rain_mm: float, duration_hr: float) -> int:
    """
    Very coarse rule-of-thumb mapping (for MVP demo only).
    Uses intensity (mm/hr) to choose 1,2,5,10,20,50,100.
    """
    if duration_hr is None or duration_hr <= 0:
        duration_hr = 1.0
    I = rain_mm / duration_hr

    # Crude breaks based on Melbourne-ish short-duration intensities
    if I < 10:   return 1
    if I < 20:   return 2
    if I < 35:   return 5
    if I < 50:   return 10
    if I < 70:   return 20
    if I < 100:  return 50
    return 100


def find_best_property(gdf_props: gpd.GeoDataFrame,
                       text_hint: Optional[str],
                       latlon: Optional[Tuple[float, float]]) -> Optional[pd.Series]:
    """
    Robust version for EPSG:7855 property polygons.
    Supports lookup by lat/lon or text suburb/street.
    """

    # --- Ensure geometry and CRS ---
    if not isinstance(gdf_props, gpd.GeoDataFrame):
        raise TypeError("Expected GeoDataFrame for properties input.")
    if "geometry" not in gdf_props.columns:
        raise ValueError("No geometry column found in the property GeoDataFrame.")
    if gdf_props.crs is None:
        gdf_props = gdf_props.set_crs("EPSG:7855")
    elif gdf_props.crs.to_string() != "EPSG:7855":
        gdf_props = gdf_props.to_crs("EPSG:7855")

    # --- Coordinate-based lookup ---
    if latlon:
        lat, lon = latlon
        try:
            # create target point and project to EPSG:7855
            target = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:7855").iloc[0]
            # compute centroids of property polygons
            gdf_props["centroid"] = gdf_props.geometry.centroid
            # find nearest centroid
            dists = gdf_props["centroid"].distance(target)
            idx = int(dists.idxmin())
            row = gdf_props.loc[idx].copy()
            # project centroid back to lat/lon for output
            cen = gpd.GeoSeries([row["centroid"]], crs="EPSG:7855").to_crs("EPSG:4326").iloc[0]
            row["lat"] = cen.y
            row["lon"] = cen.x
            row["Full_Address"] = address_string(row)
            return row
        except Exception as e:
            st.error(f"Coordinate lookup failed: {e}")
            return None

    # --- Text hint lookup (suburb or street) ---
    if text_hint:
        hint = text_hint.strip().upper()
        # Suburb match
        if "Suburb" in gdf_props.columns:
            submatch = gdf_props[gdf_props["Suburb"].str.upper() == hint]
            if len(submatch):
                sub_proj = submatch.copy()
                sub_proj["centroid"] = sub_proj.geometry.centroid
                centroid = sub_proj.unary_union.centroid
                sub_proj["dist"] = sub_proj["centroid"].distance(centroid)
                row = submatch.loc[sub_proj["dist"].idxmin()].copy()
                cen = gpd.GeoSeries([row.geometry.centroid], crs="EPSG:7855").to_crs("EPSG:4326").iloc[0]
                row["lat"] = cen.y
                row["lon"] = cen.x
                row["Full_Address"] = address_string(row)
                return row

        # Street match fallback
        if "Street" in gdf_props.columns:
            streetmatch = gdf_props[gdf_props["Street"].str.upper().str.contains(hint, na=False)]
            if len(streetmatch):
                row = streetmatch.iloc[0].copy()
                cen = gpd.GeoSeries([row.geometry.centroid], crs="EPSG:7855").to_crs("EPSG:4326").iloc[0]
                row["lat"] = cen.y
                row["lon"] = cen.x
                row["Full_Address"] = address_string(row)
                return row

    return None




def open_raster_safe(path: str):
    """Open raster if exists; returns (dataset, chosen_band) or (None, None)."""
    if not path:
        return None, None
    path = os.path.join(FLOODMAP_ROOT, path).replace("\\","/")
    if not os.path.exists(path):
        return None, None
    try:
        ds = rasterio.open(path)
        # Your .grd often stores data in band 4, others 1..3 are 255; auto-pick non-constant band.
        chosen = 1
        if ds.count >= 4:
            chosen = 4
        return ds, chosen
    except Exception:
        return None, None


def sample_raster(ds, band_idx: int, lon: float, lat: float) -> Optional[float]:
    """Sample single pixel; treat 255 and -1e37 as NoData."""
    if ds is None:
        return None
    # project to ds CRS
    pt = gpd.GeoSeries([Point(lon, lat)], crs=GEO_CRS).to_crs(ds.crs).iloc[0]
    x, y = pt.x, pt.y
    # quick bounds check
    b = ds.bounds
    if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
        return None
    try:
        val = list(ds.sample([(x, y)]))[0][band_idx - 1]
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            val = float(val[0])
        if val in (255, -1e37, -1.0e37):
            return None
        # Some rasters are stored as cm; your grids appear to be meters,
        # but if you see very large values, apply a sanity clamp here.
        return float(val)
    except Exception:
        return None


def collect_metrics_for_point(lon: float, lat: float) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Returns dictionary:
       { "001y": {"dmax":..., "hmax":..., "Vmax":..., "Z0max":...}, ... }
    Missing rasters or out-of-bounds -> None
    """
    out = {}
    for scen, spec in SCENARIO_SPECS.items():
        base = spec["base"]
        files = spec["files"]
        out[scen] = {}
        for metric, fname in files.items():
            if not fname:
                out[scen][metric] = None
                continue
            ds, band = open_raster_safe(f"{base}/{fname}")
            if ds is None:
                out[scen][metric] = None
                continue
            out[scen][metric] = sample_raster(ds, band, lon, lat)
            ds.close()
    return out


def logspace_interpolate(x_known: List[float], y_known: List[float], x_target: float) -> Optional[float]:
    """Interpolate in log-log space (power-law). Requires positive x,y."""
    xk = np.array([x for x, y in zip(x_known, y_known) if (x is not None and y is not None and x>0 and (y or 0)>0)], dtype=float)
    yk = np.array([y for x, y in zip(x_known, y_known) if (x is not None and y is not None and x>0 and (y or 0)>0)], dtype=float)
    if len(xk) < 2:
        return None
    lx = np.log(xk)
    ly = np.log(yk)
    xt = np.log(np.array([x_target], dtype=float))
    yt = np.interp(xt, lx, ly, left=ly[0], right=ly[-1])
    return float(np.exp(yt)[0])


def fill_missing_with_interpolation(metrics_by_scen: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    """For each metric, fill missing scenarios using power-law interpolation across return periods."""
    scen_to_T = {"001y":1, "002y":2, "005y":5, "010y":10, "020y":20, "050y":50, "100y":100}
    metrics = ["dmax","hmax","Vmax","Z0max"]
    for m in metrics:
        xs, ys = [], []
        for s in SCENARIO_SPECS.keys():
            xs.append(scen_to_T[s])
            ys.append(metrics_by_scen[s].get(m))
        # Fill each missing
        for idx, s in enumerate(SCENARIO_SPECS.keys()):
            if ys[idx] is None:
                y_est = logspace_interpolate(xs, ys, xs[idx])
                metrics_by_scen[s][m] = y_est
    return metrics_by_scen


def classify_depth(val: Optional[float]) -> Tuple[str, List[int]]:
    if val is None:
        return "No Data", [200, 200, 200]
    for lo, hi, col in DEPTH_CLASSES:
        if lo <= val < hi:
            if hi <= 0.05:
                return "Dry", col
            if hi <= 0.15:
                return "Very shallow", col
            if hi <= 0.30:
                return "Shallow", col
            if hi <= 0.60:
                return "Moderate", col
            if hi <= 1.20:
                return "Deep", col
            return "Very deep", col
    return "Very deep", [255, 90, 90]


def zone_buffers(lon: float, lat: float) -> Dict[str, Polygon]:
    pt = gpd.GeoSeries([Point(lon, lat)], crs=GEO_CRS).to_crs(PROJ_CRS).iloc[0]
    return {
        "150m": gpd.GeoSeries([pt.buffer(150)], crs=PROJ_CRS).to_crs(GEO_CRS).iloc[0],
        "300m": gpd.GeoSeries([pt.buffer(300)], crs=PROJ_CRS).to_crs(GEO_CRS).iloc[0],
        "600m": gpd.GeoSeries([pt.buffer(600)], crs=PROJ_CRS).to_crs(GEO_CRS).iloc[0],
    }


def pydeck_map(center: Tuple[float,float],
               points_df: pd.DataFrame,
               zones: Dict[str, Polygon]) -> pdk.Deck:
    """
    points_df columns: lat, lon, dmax_Chosen, depth_class_color [r,g,b], Full_Address
    zones: dict with GeoJSON-like polygons
    """
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=points_df,
        get_position='[lon, lat]',
        get_fill_color='depth_class_color',
        get_radius=6,
        pickable=True,
        radius_min_pixels=3,
        radius_max_pixels=30,
        stroked=True,
        get_line_color=[40, 40, 40],
    )

    zone_features = []
    for name, poly in zones.items():
        zone_features.append({
            "type": "Feature",
            "properties": {"name": name},
            "geometry": json.loads(gpd.GeoSeries([poly], crs=GEO_CRS).to_json())["features"][0]["geometry"]
        })

    zones_layer = pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": zone_features},
        stroked=True,
        filled=False,
        get_line_color=[120, 120, 120, 180],
        get_line_width=2,
        pickable=False
    )

    tooltip = {
        "html": "<b>{Full_Address}</b><br/>Depth (chosen): {dmax:.2f} m",
        "style": {"backgroundColor": "white", "color": "black"}
    }

    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=14, pitch=0)
    deck = pdk.Deck(layers=[zones_layer, scatter], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9", tooltip=tooltip)
    return deck


def empathetic_message(depth: Optional[float], severity_flag: bool, suburb: Optional[str]) -> str:
    tone = "moderate"
    if severity_flag or (depth or 0) >= 0.6:
        tone = "high"
    if (depth or 0) < 0.15:
        tone = "low"

    suburb_txt = suburb or "your area"

    if tone == "high":
        return (
            f"Thanks for reporting this. Conditions around {suburb_txt} could be **hazardous**. "
            f"If water is rising or moving, stay away from drains and creeks, keep children and pets close, "
            f"and don’t drive through floodwater. If you feel unsafe, move to higher ground and call **000**."
        )
    if tone == "moderate":
        return (
            f"Appreciate the update. There may be **localized flooding** in {suburb_txt}. "
            "Avoid walking or riding through water, watch for road closures, and check on neighbours who may need help."
        )
    return (
        f"Thanks — current signal for {suburb_txt} looks **low**. "
        "Please keep an eye on conditions and let us know if anything changes."
    )


def vic_safety_guidance() -> List[str]:
    return [
        "For flood help call **SES Victoria 132 500** (life-threatening emergencies: **000**).",
        "Use the **VicEmergency** app/website for warnings and relief centre updates.",
        "Never drive, ride, or walk through floodwater — it can be deeper and faster than it looks.",
        "Move vehicles and valuables to higher ground; switch off power at the mains if safe.",
        "Keep an emergency kit (medications, torch, radio, charger, water, copies of documents).",
        "Check **VicTraffic** for road closures; follow local council (City of Frankston) updates.",
        "If trapped by rising water, move to the highest available place and call **000**."
    ]


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="FloodAI MVP v3", layout="wide")
st.title("🌧️ FloodAI — Interactive MVP (v3)")
st.caption("Properties from GPKG • Dynamic flood rasters (001y–100y) • Power-law interpolation • Risk zones • VIC safety")

# 1) Input panel
st.subheader("🗣️ Describe the rainfall or flood event")
default_text = "Severe rainfall of 80 mm near -38.1243, 145.1245 lasting 1 hour in Seaford"
text = st.text_input("Type in natural language (address, suburb or GPS OK):", value=default_text)

with st.expander("Assist me (what should I write?)"):
    st.write("- Include a rainfall amount and duration, e.g. **60 mm in 45 minutes**.")
    st.write("- Include a **suburb** (Frankston, Seaford, Langwarrin, …) or **lat,lon** (-38.1, 145.1).")
    st.write("- You can also click a location on the map (after parsing).")

parsed = parse_text_event(text)
ok, missing_msgs = quality_gate(parsed)

colA, colB = st.columns([1, 1])

with colA:
    st.write("**Parsed (draft):**")
    st.json(parsed, expanded=False)

with colB:
    if not ok:
        st.warning("I need a bit more info:")
        for m in missing_msgs:
            st.write(m)
        st.info("Tip: you can also provide coordinates like: `-38.1643, 145.1245` or a suburb name.")
    else:
        st.success("Looks good — I have rain, duration and a location hint.")

st.divider()

# 2) Property lookup
st.subheader("🏠 Property match")
props_gdf = load_properties()

# Try best match from parsed text
prop_row = find_best_property(props_gdf, parsed.get("addr_hint"), parsed.get("latlon"))

# Optional: map click to refine
st.caption("Optional: click map to refine the location (then press 'Use clicked location').")

# Initial map center
center_latlon = (-38.146, 145.133)  # Frankston CBD-ish
if prop_row is not None:
    center_latlon = (prop_row.get("lat", prop_row.geometry.y), prop_row.get("lon", prop_row.geometry.x))

clicked = st.map(pd.DataFrame({"lat":[center_latlon[0]],"lon":[center_latlon[1]]}), zoom=13)

# Streamlit's simple map doesn't give click events directly; keep the found property for now.
if prop_row is None and parsed.get("latlon") is None and parsed.get("addr_hint") is None:
    st.stop()

# Confirm property
if prop_row is None:
    st.error("I couldn't confidently match a property. Please add a suburb or GPS coordinates.")
    st.stop()

st.success(f"Matched: **{address_string(prop_row)}**")

lat = float(prop_row.get("lat", prop_row.geometry.y))
lon = float(prop_row.get("lon", prop_row.geometry.x))
suburb = prop_row.get("Suburb", None)

# 3) Choose scenario
auto_T = infer_return_period(parsed.get("rain_mm") or 0, parsed.get("duration_hr") or 1.0)
scenario_choice = st.selectbox(
    "Scenario (return period)",
    ["Auto (from rainfall)"] + [f"{T}y" for T in RETURN_PERIODS],
    index=0
)
if scenario_choice.startswith("Auto"):
    chosen_T = auto_T
else:
    chosen_T = int(scenario_choice.replace("y",""))

# 4) Sample rasters at the property location
with st.spinner("Sampling rasters…"):
    metrics_all = collect_metrics_for_point(lon, lat)
    metrics_all = fill_missing_with_interpolation(metrics_all)

# Pull chosen scenario depth
scen_key = f"{chosen_T:03d}y"
d_chosen = metrics_all.get(scen_key, {}).get("dmax", None)
h_chosen = metrics_all.get(scen_key, {}).get("hmax", None)
depth_label, depth_color = classify_depth(d_chosen)

# 5) Build zones + nearby properties
zones = zone_buffers(lon, lat)
props_near = props_gdf.copy()
props_near["Full_Address"] = props_near.apply(address_string, axis=1)
props_near["lat"] = props_near.geometry.y
props_near["lon"] = props_near.geometry.x

# Keep within 600m
poly_600 = zones["600m"]
props_near = props_near[props_near.within(poly_600)].copy()

# Sample dmax for chosen scenario for each nearby point (quick)
vals = []
for _, r in props_near.iterrows():
    vv = metrics_all[scen_key]["dmax"] if (abs(r["lat"]-lat)<1e-9 and abs(r["lon"]-lon)<1e-9) else None
    # sample individually only when needed (to keep demo snappy)
    if vv is None:
        # micro-sample this single point (cheap)
        metrics_this = collect_metrics_for_point(r["lon"], r["lat"])
        vv = fill_missing_with_interpolation(metrics_this)[scen_key]["dmax"]
    vals.append(vv)

props_near["dmax"] = vals
props_near["depth_class"] = [classify_depth(v)[0] for v in vals]
props_near["depth_class_color"] = [classify_depth(v)[1] for v in vals]

# For the focal property row (ensure visible)
me_df = pd.DataFrame([{
    "lat": lat, "lon": lon,
    "Full_Address": address_string(prop_row),
    "dmax": d_chosen,
    "depth_class_color": [0, 0, 0]  # highlight with black dot overlay
}])

map_df = pd.concat([
    props_near[["lat","lon","Full_Address","dmax","depth_class_color"]],
    me_df
], ignore_index=True)

deck = pydeck_map((lat, lon), map_df, zones)
st.pydeck_chart(deck, use_container_width=True)

# 6) Summary + empathy
st.subheader("📄 Flood Event Summary")
col1, col2 = st.columns([1.1, 1])
with col1:
    st.markdown(f"- **Rainfall:** {parsed.get('rain_mm') or '—'} mm")
    st.markdown(f"- **Duration:** {parsed.get('duration_hr') or '—'} hr")
    st.markdown(f"- **Scenario:** {chosen_T}y")
    st.markdown(f"- **Flood Depth (dmax):** {('%.2f' % d_chosen) if d_chosen is not None else 'No Data'} m  ({depth_label})")
    st.markdown(f"- **Location:** {address_string(prop_row)}")
    st.markdown(f"- **Suburb:** {suburb or '—'}")
    st.markdown(f"- **Timestamp:** {parsed.get('timestamp')}")

with col2:
    st.info(empathetic_message(d_chosen, parsed.get("severity_flag", False), suburb))

with st.expander("Victoria safety pointers (save/share)"):
    for tip in vic_safety_guidance():
        st.write(f"- {tip}")

# 7) Full scenario table (after interpolation)
st.subheader("📊 Depth/Speed by Scenario (at this property)")
tbl = []
for scen, vals in metrics_all.items():
    row = {"Scenario": scen}
    row.update(vals)
    tbl.append(row)
st.dataframe(pd.DataFrame(tbl).sort_values("Scenario").reset_index(drop=True), use_container_width=True)

st.caption("Note: Missing rasters are filled using **power-law (log/log) interpolation** across return periods, not linear.")

