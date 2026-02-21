import spacy
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

# -------------------------------------------------------------
# Load NLP model and property dataset
# -------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")
PROPERTY_GPKG = "SwinburneData/Property/Properties.gpkg"

# Load and prepare property polygons
props = gpd.read_file(PROPERTY_GPKG)
props = props.to_crs(epsg=4326)
KNOWN_SUBURBS = set(props["Suburb"].dropna().str.upper().unique())

# -------------------------------------------------------------
# Extraction functions
# -------------------------------------------------------------
def extract_rainfall(text):
    m = re.search(r'(\d+(?:\.\d+)?)\s*(mm|millimeter|millimetre|millimetres)', text, re.I)
    return float(m.group(1)) if m else None

def extract_duration(text):
    if m := re.search(r'(\d+(?:\.\d+)?)\s*(h|hr|hour|hours)', text, re.I):
        return float(m.group(1))
    elif m := re.search(r'(\d+(?:\.\d+)?)\s*(m|min|mins|minute|minutes)', text, re.I):
        return float(m.group(1)) / 60.0
    return None

def extract_coords(text):
    m = re.search(r'(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)', text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return lat, lon
    return None

def extract_location(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text.strip()

    for suburb in KNOWN_SUBURBS:
        if suburb in text.upper():
            return suburb.title()
    return None

def find_suburb_from_coords(lat, lon):
    """Spatial lookup: find property polygon containing or nearest to the given coordinates."""
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")

    # First try 'within' match
    join = gpd.sjoin(pt, props[["Suburb", "geometry"]], how="left", predicate="within")

    if len(join) and pd.notna(join.iloc[0]["Suburb"]):
        return join.iloc[0]["Suburb"]

    # If not within, try nearest (for coordinates just outside suburb polygon)
    join_near = gpd.sjoin_nearest(pt, props[["Suburb", "geometry"]], how="left", distance_col="dist_m")
    if len(join_near) and pd.notna(join_near.iloc[0]["Suburb"]):
        return join_near.iloc[0]["Suburb"]

    return None

def extract_severity(text):
    return any(k in text.lower() for k in ["heavy", "intense", "severe", "extreme", "massive"])

# -------------------------------------------------------------
# Main parser
# -------------------------------------------------------------
def parse_event(text):
    rain = extract_rainfall(text)
    dur = extract_duration(text)
    coords = extract_coords(text)
    severity = extract_severity(text)

    loc_text = extract_location(text)
    loc_from_coords = None
    if coords:
        lat, lon = coords
        loc_from_coords = find_suburb_from_coords(lat, lon)

    final_loc = loc_from_coords or loc_text

    return {
        "rain_mm": rain,
        "duration_hr": dur,
        "address_text": final_loc,
        "timestamp": datetime.now().isoformat(),
        "severity_flag": severity
    }

# -------------------------------------------------------------
# Example tests
# -------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        "Heavy rainfall of 60 mm near -38.1643, 145.1245 lasting 45 minutes",
        "Severe rainfall of 80 millimetres in Frankston for 1 hour",
        "Light rain of 15mm in Seaford",
        "Moderate rainfall at coordinates -38.140, 145.136 lasting 2 hours"
    ]

    for txt in examples:
        parsed = parse_event(txt)
        print(f"\n💧 Input: {txt}")
        print("🧩 Parsed event:", parsed)
