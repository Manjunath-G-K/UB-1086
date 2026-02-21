# floodai_app_v6.py
import os, re, datetime, math
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from rapidfuzz import process, fuzz

st.set_page_config(page_title="FloodAI — MVP v5", layout="wide")

# -----------------------------
# PATHS
# -----------------------------
PROPERTY_GPKG        = "SwinburneData/Property/Properties.gpkg"
FLOODMAP_ROOT        = "SwinburneData/FloodMaps/FrankstonSouth"
SUBCATCHMENT_GPKG    = "SwinburneData/Subcatchments/Subcatchments.gpkg"
IFD_CSV              = "SwinburneData/IFD/ifd_table.csv"
RASTER_CATALOG_CSV   = "SwinburneData/FloodMaps/raster_catalog.csv"

# -----------------------------
# DATA LOAD
# -----------------------------
subcatchments_gdf = gpd.read_file(SUBCATCHMENT_GPKG).to_crs(epsg=4326)

# IFD table
ifd_table = pd.read_csv(IFD_CSV)
# normalise IFD col names
ifd_table.columns = (
    ifd_table.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)

# Raster catalog
raster_catalog_raw = pd.read_csv(RASTER_CATALOG_CSV)

# -----------------------------
# NORMALISE RASTER CATALOG
# -----------------------------
def normalise_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_period"] = df["return_period"].astype(str).str.lower().str.strip()
    df["metric"]        = df["metric"].astype(str).str.lower().str.strip()
    df["full_path"]     = df["full_path"].astype(str).str.replace("\\", "/", regex=False)

    # unify metric naming
    metric_map = {
        "dmaxmax": "dmax",
        "hmaxmax": "hmax",
        "z0":      "z0max",
        "z0_max":  "z0max",
    }
    df["metric"] = df["metric"].replace(metric_map, regex=False)

    # check file presence
    df["exists"] = df["full_path"].apply(lambda p: os.path.exists(p))
    missing = df.loc[~df["exists"], "full_path"].tolist()
    if len(missing):
        st.warning(f"⚠️ Missing raster files (showing up to 5): {missing[:5]}")

    return df

raster_catalog = normalise_catalog(raster_catalog_raw)

YEARS_ORDER = sorted(
    raster_catalog["return_period"].unique(),
    key=lambda x: int(x.replace("y",""))
)
YEAR_VALS   = np.array([int(x.replace("y","")) for x in YEARS_ORDER], dtype=float)

# -----------------------------
# STREAMLIT SETUP
# -----------------------------

st.title("🌧️ FloodAI — MVP v5 (Alpha Merge: GPKG + Rasters + Fuzzy Address + IFD)")

# -----------------------------
# PROPERTY LOADER
# -----------------------------
@st.cache_data(show_spinner=False)
def load_properties() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(PROPERTY_GPKG)

    # Expect EPSG:7855 for local metric work
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=7855)

    # Add Full_Address column for display / fuzzy matching
    def _safe(s): return s.fillna("").astype(str).str.strip()
    gdf["Full_Address"] = (
        _safe(gdf.get("House",   pd.Series([""]*len(gdf)))) + " " +
        _safe(gdf.get("Street",  pd.Series([""]*len(gdf)))) + ", " +
        _safe(gdf.get("Suburb",  pd.Series([""]*len(gdf)))) + " VIC " +
        _safe(gdf.get("Postcode",pd.Series([""]*len(gdf))))
    ).str.replace(r"\s+,", ",", regex=True)\
     .str.replace(r",\s+VIC\s+$"," VIC", regex=True)

    return gdf

props_gdf = load_properties()
st.caption(f"📍 Properties loaded: {len(props_gdf):,} | CRS: {props_gdf.crs}")

# cached WGS84 copy if needed later
@st.cache_data(show_spinner=False)
def props_wgs84(_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return _gdf.to_crs(epsg=4326)
props_gdf_wgs = props_wgs84(props_gdf)

# -----------------------------
# TEXT PARSERS (rain, duration, coords)
# -----------------------------
def parse_rain(text:str):
    m = re.search(r'(\d+(?:\.\d+)?)\s*(mm|millimet(re|er|res|ers)?)', text, re.I)
    return float(m.group(1)) if m else None

def parse_duration(text:str):
    # hours
    m = re.search(r'(\d+(?:\.\d+)?)\s*(hour|hr|h|hours)', text, re.I)
    if m:
        return float(m.group(1))
    # minutes
    m = re.search(r'(\d+(?:\.\d+)?)\s*(minute|min|mins|m)', text, re.I)
    if m:
        return float(m.group(1))/60.0
    return None

def parse_coords(text:str):
    # look for "num num"
    m = re.search(r'(-?\d{1,3}\.\d+)[,\s]+(-?\d{1,3}\.\d+)', text)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    # heuristic: Melbourne-ish lat ~ -38, lon ~ 145
    if abs(a) < 90 and abs(b) > 90:
        return (a, b)  # (lat, lon)
    if abs(b) < 90 and abs(a) > 90:
        return (b, a)
    return (a, b)

def parse_address_hint(text:str):
    # "in Frankston South", "at 24 Dell Road"
    m = re.search(r'(?:near|in|at|around)\s+([A-Za-z0-9 ,\-\/]+)', text, re.I)
    return m.group(1).strip() if m else None

def clarification_needed(rain, dur, coords, addr_hint):
    need = []
    if rain is None: need.append("rainfall (mm)")
    if dur is None:  need.append("duration (hours)")
    if coords is None and not addr_hint: need.append("location (address or coordinates)")
    return need

# -----------------------------
# ADDRESS NORMALISATION / FUZZY MATCH HELPERS
# -----------------------------
STREET_EQUIV = {
    "ST": "STREET", "ST.": "STREET",
    "RD": "ROAD",   "RD.": "ROAD",
    "CT": "COURT",  "CT.": "COURT",
    "AVE": "AVENUE","AV": "AVENUE",
    "PL": "PLACE",
    "CR": "CRESCENT", "CRES": "CRESCENT"
}

def normalize_street(s: str) -> str:
    """Expand common street abbreviations and uppercase."""
    s = s.upper().strip()
    for abbr, full in STREET_EQUIV.items():
        s = re.sub(rf"\b{abbr}\b", full, s)
    return s

def parse_address_parts(addr: str):
    """
    Take a freeform address-ish string, pull out:
    - house/unit number
    - street name
    - suburb guess
    """
    addr = addr.strip().upper()
    parts = re.split(r"[ ,]+", addr)

    house_no = None
    street   = None
    suburb   = None

    # find first number-looking token → house/unit
    for p in parts:
        if re.fullmatch(r"\d+[A-Z]?", p):
            house_no = p
            break

    # guess suburb as last token or last two tokens
    if len(parts) >= 2:
        suburb = " ".join(parts[-2:]) if len(parts[-1]) < 4 else parts[-1]

    # what remains is street-ish
    street_parts = [p for p in parts if p not in (house_no, suburb)]
    if street_parts:
        street = " ".join(street_parts)

    return house_no, street, suburb

# -----------------------------
# PROPERTY LOCATOR (FUZZY ADDRESS OR COORDS)
# -----------------------------
def best_property(addr_hint: str | None, latlon: tuple[float, float] | None):
    gdf = props_gdf

    # --- 1. coordinate-based path
    if latlon:
        lat, lon = latlon
        pt = gpd.GeoSeries(
            [Point(lon, lat)],
            crs="EPSG:4326"
        ).to_crs(gdf.crs).iloc[0]

        gdf["centroid_tmp"] = gdf.geometry.centroid
        gdf["dist"] = gdf["centroid_tmp"].distance(pt)

        row = gdf.loc[gdf["dist"].idxmin()]
        cen_wgs = gpd.GeoSeries(
            [row["centroid_tmp"]], crs=gdf.crs
        ).to_crs(4326).iloc[0]

        st.success("📍 Location resolved using coordinates")
        return {
            "match_type": "coords",
            "Full_Address": row.get("Full_Address"),
            "Suburb": row.get("Suburb"),
            "Postcode": row.get("Postcode"),
            "lat": cen_wgs.y,
            "lon": cen_wgs.x,
            "prop_idx": int(row.name),
        }

    # --- 2. fuzzy address path
    if addr_hint:
        addr_hint_norm = normalize_street(addr_hint)

        house_no, street, suburb = parse_address_parts(addr_hint_norm)
        street = normalize_street(street or "")
        suburb = suburb.upper() if suburb else ""

        candidates = gdf.copy()
        candidates["StreetNorm"] = candidates["Street"].astype(str).apply(normalize_street)

        # fuzzy suburb filter first
        if suburb:
            sub_match = process.extractOne(
                suburb,
                candidates["Suburb"].astype(str).unique(),
                scorer=fuzz.partial_ratio
            )
            if sub_match and sub_match[1] > 60:
                candidates = candidates[candidates["Suburb"].str.upper() == sub_match[0].upper()]

        # fuzzy street match
        if street:
            street_matches = process.extract(
                street,
                candidates["StreetNorm"].unique(),
                scorer=fuzz.partial_ratio,
                limit=10
            )
            if street_matches:
                best_streets = [s for s, score, _ in street_matches if score > 60]
                candidates = candidates[candidates["StreetNorm"].isin(best_streets)]

        # optional house number filter
        if house_no:
            candidates = candidates[
                candidates["House"].astype(str).str.contains(house_no, na=False)
            ]

        if candidates.empty:
            return None

        # rank by similarity of full address to input
        candidates["score"] = candidates["Full_Address"].apply(
            lambda a: fuzz.token_set_ratio(a.upper(), addr_hint_norm.upper()) / 100.0
        )
        best_row = candidates.sort_values("score", ascending=False).iloc[0]

        if best_row["score"] < 0.45:
            return None

        cen_wgs = gpd.GeoSeries(
            [best_row.geometry.centroid], crs=gdf.crs
        ).to_crs(4326).iloc[0]

        st.success(f"✅ Matched: {best_row['Full_Address']} (Similarity {best_row['score']*100:.1f}%)")

        return {
            "match_type": "address (fuzzy component match)",
            "Full_Address": best_row.get("Full_Address"),
            "Suburb": best_row.get("Suburb"),
            "Postcode": best_row.get("Postcode"),
            "lat": cen_wgs.y,
            "lon": cen_wgs.x,
            "prop_idx": int(best_row.name),
        }

    return None

# -----------------------------
# FLOOD RASTER SAMPLING
# -----------------------------
def sample_raster_value(raster_path, lon, lat, prefer_band=1, max_depth_m=2.0):
    """
    Sample flood raster at lat/lon.
    Scales uint8 rasters (0-255) to a depth in metres (0 - max_depth_m).
    Returns 0.0 if outside raster or nodata.
    """
    try:
        with rasterio.open(raster_path) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])

            inside = (
                src.bounds.left   <= xs[0] <= src.bounds.right and
                src.bounds.bottom <= ys[0] <= src.bounds.top
            )
            if not inside:
                return 0.0

            val = list(src.sample([(xs[0], ys[0])], indexes=prefer_band))[0][0]

            if val is None or np.isnan(val):
                return 0.0

            # uint8 → scale
            if src.dtypes[0] == "uint8":
                return round((val / 255.0) * max_depth_m, 3)

            # float-ish → assume already metres
            return round(float(val), 3)

    except Exception as e:
        print(f"⚠️ Raster sampling error for {raster_path}: {e}")
        return 0.0

def collect_metrics_for_point(lat, lon):
    """
    Build dict:
    {
      "001y": { "dmax": val, "hmax": val, "vmax": val, "z0max": val },
      ...
    }
    using first matching raster per metric.
    """
    metrics = {
        y: {"dmax": np.nan, "hmax": np.nan, "vmax": np.nan, "z0max": np.nan}
        for y in YEARS_ORDER
    }

    for y in YEARS_ORDER:
        subset = raster_catalog[raster_catalog["return_period"] == y]
        for metric in ["dmax", "hmax", "vmax", "z0max"]:
            row = subset[subset["metric"].str.startswith(metric)]
            if not row.empty:
                path = row.iloc[0]["full_path"]
                metrics[y][metric] = sample_raster_value(path, lon, lat)

    return metrics

# -----------------------------
# RETURN PERIOD INTERPOLATION
# -----------------------------
def interp_log_year(metric_by_year: dict[str, float]) -> dict[str, float]:
    xs, ys = [], []
    for ytag, val in metric_by_year.items():
        yr = int(ytag.replace("y",""))
        if not np.isnan(val):
            xs.append(yr)
            ys.append(val)
    if len(xs) < 2:
        return metric_by_year
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    logx = np.log(xs)
    for ytag in metric_by_year.keys():
        yr = int(ytag.replace("y",""))
        if np.isnan(metric_by_year[ytag]):
            metric_by_year[ytag] = float(np.interp(np.log(yr), logx, ys))
    return metric_by_year

# -----------------------------
# RISK + SAFETY
# -----------------------------
def summarize_risk(dmax_m: float|None, vmax_ms: float|None) -> str:
    if dmax_m is None or np.isnan(dmax_m):
        return "Unknown risk — no flood depth at this location."
    if dmax_m < 0.05:
        return "Very low risk — nuisance ponding only."
    if dmax_m < 0.3:
        return "Low risk — shallow overland flow possible."
    if dmax_m < 0.6:
        return "Moderate risk — curb-depth flow; avoid driving."
    if dmax_m < 1.0:
        return "High risk — building ingress possible; avoid floodwater."
    return "Extreme risk — life-threatening conditions possible."

def safety_tips_vic() -> list[str]:
    return [
        "Never enter floodwater — it can be fast-moving or contaminated.",
        "Do not drive through floodwater. As little as 15 cm can float a small car.",
        "Stay informed via VicEmergency app/website and ABC local radio.",
        "For SES flood/storm assistance call 132 500. Call 000 in life-threatening emergencies.",
        "Move vehicles and valuables to higher ground if safe to do so.",
        "Turn off electricity/gas at the mains if flooding is imminent and you can do it safely.",
        "Prepare sandbags for low doorways/vents.",
        "Avoid walking near drains/culverts.",
    ]

# -----------------------------
# IFD → RETURN PERIOD
# -----------------------------
def estimate_return_period_from_ifd(rain_mm: float, duration_h: float, lat: float, lon: float, ifd_df: pd.DataFrame) -> float:
    """
    Map the user's rainfall event to an approximate return period.
    """
    try:
        intensity = rain_mm / duration_h               # mm per hour
        duration_min = duration_h * 60                 # to minutes

        # nearest "station"
        idx = ((ifd_df['lat'] - lat)**2 + (ifd_df['lon'] - lon)**2).idxmin()
        station_id = ifd_df.loc[idx, 'station_id']
        site_ifd = ifd_df.loc[ifd_df['station_id'] == station_id].copy()

        site_ifd['duration_in_min'] = site_ifd['duration_in_min'].astype(float)
        # pick closest duration row
        row = site_ifd.iloc[(site_ifd['duration_in_min'] - duration_min).abs().argsort()[:1]]

        aep_cols = ['63.20%', '50%', '20%', '10%', '5%', '2%', '1%']
        return_periods = [1, 2, 5, 10, 20, 50, 100]
        aep_values = row[aep_cols].values.flatten().astype(float)

        # interpolate intensity → RP
        rp = np.interp(intensity, aep_values, return_periods)
        return float(round(rp, 1))
    except Exception as e:
        print(f"⚠️ IFD estimation error: {e}")
        return np.nan

# -----------------------------
# UI — INPUT & CLARIFICATION
# -----------------------------
with st.container():
    st.subheader("💬 Describe the rainfall/flood observation")

    default_txt = "Severe rainfall of 60 mm in Frankston South for 45 minutes"
    user_txt = st.text_area("Free text", height=90, value=default_txt)

    rain = parse_rain(user_txt)
    dur  = parse_duration(user_txt)
    coords = parse_coords(user_txt)
    addr_hint = parse_address_hint(user_txt)

    missing = clarification_needed(rain, dur, coords, addr_hint)

    if missing:
        st.warning("I need a bit more info to proceed:")
        col1,col2,col3 = st.columns(3)

        with col1:
            rain = st.number_input("Rainfall (mm)", value=rain if rain else 50.0, min_value=0.0, step=1.0)

        with col2:
            dur = st.number_input("Duration (hours)", value=dur if dur else 1.0, min_value=0.0, step=0.25)

        with col3:
            mode = st.radio("Location input", ["Address","Coordinates"], index=0 if not coords else 1, horizontal=True)

        if mode == "Coordinates":
            lat = st.number_input("Latitude (e.g., -38.15)",  value=coords[0] if coords else -38.1539, step=0.0001, format="%.6f")
            lon = st.number_input("Longitude (e.g., 145.10)", value=coords[1] if coords else 145.1038, step=0.0001, format="%.6f")
            coords = (lat, lon)
            addr_hint = None

        else:
            st.markdown("**Enter address details**")
            col_a, col_b, col_c = st.columns([1, 2, 2])

            with col_a:
                house_input = st.text_input("🏠 Unit / House number", value="", placeholder="e.g. 24")

            with col_b:
                street_input = st.text_input("🛣️ Street name", value="", placeholder="e.g. Dell Road")

            with col_c:
                suburb_input = st.text_input("🏙️ Suburb", value="", placeholder="e.g. Frankston")

            addr_hint = " ".join(
                [x.strip() for x in [house_input, street_input, suburb_input] if x]
            ).strip()

            coords = None

    st.divider()

# -----------------------------
# PROPERTY RESOLUTION (GPKG + subcatchment)
# -----------------------------
prop = best_property(addr_hint, coords)

if not prop:
    st.error("❌ I couldn't resolve a property from that input. Please provide a clearer address or coordinates.")
    st.stop()

# subcatchment lookup
prop_point = Point(prop["lon"], prop["lat"])
sub_match = subcatchments_gdf[subcatchments_gdf.contains(prop_point)]
if not sub_match.empty:
    sub_name = sub_match.iloc[0].get("Subcatchment", "Unknown")
    st.caption(f"📍 Subcatchment: {sub_name}")
else:
    st.caption("📍 Outside known subcatchments.")

st.success(f"📍 Matched property: **{prop['Full_Address']}**")
st.caption(f"Lat/Lon: {prop['lat']:.6f}, {prop['lon']:.6f} | Suburb: {prop['Suburb']} {prop['Postcode']} | via {prop['match_type']}")

# -----------------------------
# FLOOD METRICS FOR THIS PROPERTY
# -----------------------------
st.subheader("📈 Flood metrics by return period (at property location)")
raw = collect_metrics_for_point(prop["lat"], prop["lon"])

# Interpolate (log in return period space)
interp = {}
for mkey in ["dmax","hmax","vmax","z0max"]:
    series = {y: raw[y][mkey] for y in YEARS_ORDER}
    interp[mkey] = interp_log_year(series)

df = pd.DataFrame({
    "ReturnPeriod": YEARS_ORDER,
    "Depth_dmax_m":       [interp["dmax"][y] for y in YEARS_ORDER],
    "WaterLevel_hmax_m":  [interp["hmax"][y] for y in YEARS_ORDER],
    "Velocity_vmax_ms":   [interp["vmax"][y] for y in YEARS_ORDER],
    "HazardIndex_z0":     [interp["z0max"][y] for y in YEARS_ORDER],
})

st.dataframe(df, use_container_width=True)

# -----------------------------
# CHOOSE SCENARIO FROM RAINFALL (IFD-BASED)
# -----------------------------
if rain is not None and dur is not None:
    scen_rp = estimate_return_period_from_ifd(rain, dur, prop["lat"], prop["lon"], ifd_table)

    if scen_rp is None or np.isnan(scen_rp):
        scen = "010y"
        st.caption("🌀 Unable to estimate return period — defaulting to 10-year event (010y).")
    else:
        # round RP to nearest 5 years and turn into label like "010y"
        scen_rounded = int(round(scen_rp / 5) * 5)
        scen = f"{scen_rounded:03d}y"
        st.caption(f"🌀 Estimated event intensity corresponds to ~{scen_rp:.1f}-year storm ({scen})")
else:
    scen = "010y"
    st.caption("🌀 Missing rainfall or duration — defaulting to 10-year event (010y).")

d_here = interp["dmax"].get(scen, np.nan)
v_here = interp["vmax"].get(scen, np.nan)

st.info(
    f"Scenario selected from rainfall: **{scen}** → "
    f"Depth ≈ **{d_here if not np.isnan(d_here) else 'N/A'} m**, "
    f"Velocity ≈ **{v_here if not np.isnan(v_here) else 'N/A'} m/s**"
)
st.warning(summarize_risk(d_here, v_here))

# -----------------------------
# MAP + ZONES
# -----------------------------
st.subheader("🗺️ Map & Safety Zones")

m = folium.Map(location=[prop["lat"], prop["lon"]], zoom_start=15)

popup_html = (
    f"{prop['Full_Address']}<br>"
    f"<b>{scen}</b>: depth={d_here if not np.isnan(d_here) else 'N/A'} m"
)

folium.Marker(
    [prop["lat"], prop["lon"]],
    popup=popup_html,
    icon=folium.Icon(color="red")
).add_to(m)

# Approximate hazard rings 100m / 250m / 500m
for radius, col in [(100, "#FF0000"), (250, "#FFA500"), (500, "#3388ff")]:
    folium.Circle(
        location=[prop["lat"], prop["lon"]],
        radius=radius,
        color=col,
        fill=True,
        fill_opacity=0.08,
        weight=1,
        tooltip=f"Zone {radius} m"
    ).add_to(m)

st_folium(m, height=520, use_container_width=True)

# -----------------------------
# ADVISORY / SAFETY
# -----------------------------
st.subheader("🧠 Advisory")

sev = (
    "severe"   if ((rain and rain>=60) or (d_here and not np.isnan(d_here) and d_here>=0.6))
    else "moderate" if (rain and rain>=20)
    else "low"
)

st.write(
    "**Overall sentiment:** "
    + (
        "🚨 High Concern" if sev=="severe"
        else "⚠️ Elevated Caution" if sev=="moderate"
        else "✅ Low Concern"
    )
)

with st.expander("Safety guidance (Victoria / SES-aligned)"):
    for tip in safety_tips_vic():
        st.markdown(f"- {tip}")

# -----------------------------
# SUMMARY + EXPORT
# -----------------------------
st.divider()
st.markdown("### 📋 Summary")
st.markdown(f"""
- **Rainfall reported:** {rain if rain is not None else 'Not provided'} mm  
- **Duration:** {dur if dur is not None else 'Not provided'} h  
- **Matched property:** {prop['Full_Address']}  
- **Chosen scenario:** {scen}  
- **Estimated depth:** {d_here if d_here is not None and not np.isnan(d_here) else 'No data'} m  
- **Estimated velocity:** {v_here if v_here is not None and not np.isnan(v_here) else 'No data'} m/s  
- **Risk level:** {('Extreme/High' if sev=='severe' else 'Moderate' if sev=='moderate' else 'Low')}  
- **Timestamp:** {datetime.datetime.now().isoformat(timespec='seconds')}
""")

st.download_button(
    "⬇️ Download Flood Report (CSV)",
    df.to_csv(index=False).encode('utf-8'),
    "flood_metrics.csv",
    "text/csv"
)
