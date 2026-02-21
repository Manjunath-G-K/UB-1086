import spacy
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")
PROPERTY_GPKG = "SwinburneData/Property/Properties.gpkg"

# Load property polygons
print("🔍 Loading property data ...")
props = gpd.read_file(PROPERTY_GPKG)
props = props.to_crs(epsg=4326)

# Prepare a helper column for full addresses
props["Full_Address"] = (
    props["House"].fillna("").astype(str).str.strip() + " " +
    props["Street"].fillna("").astype(str).str.strip() + ", " +
    props["Suburb"].fillna("").astype(str).str.strip() + " VIC " +
    props["Postcode"].fillna("").astype(str)
).str.strip()

# -------------------------------------------------------------
# ENTITY EXTRACTION HELPERS
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

# In extract_coords():
def extract_coords(text):
    m = re.search(r'(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)', text)
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    # Heuristic: latitude ~ -38 in Melbourne, longitude ~ 145
    if abs(x) > abs(y):  # reversed
        return y, x
    return x, y


def extract_location(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            return ent.text.strip()
    # Fallback: match street-like patterns
    m = re.search(r'\b(\d+\s+[A-Za-z\s]+(?:Street|St|Road|Rd|Avenue|Ave|Court|Ct|Drive|Dr|Way|Parade))[, ]*(\w+)?', text, re.I)
    if m:
        return m.group(0)
    return None


def extract_severity(text):
    severe_terms = ["heavy", "intense", "severe", "extreme", "massive", "torrential"]
    return any(term in text.lower() for term in severe_terms)

# -------------------------------------------------------------
# PROPERTY LOOKUP FUNCTIONS
# -------------------------------------------------------------
def find_property_from_coords(lat, lon):
    """Return the nearest property info for given coordinates."""
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")

    # Check if inside any parcel
    join = gpd.sjoin(pt, props, how="left", predicate="within")
    if len(join) and pd.notna(join.iloc[0]["Full_Address"]):
        row = join.iloc[0]
        return {
            "matched": True,
            "Full_Address": row["Full_Address"],
            "Suburb": row["Suburb"],
            "Postcode": row["Postcode"],
            "lat": lat,
            "lon": lon
        }

    # Otherwise, find the nearest one
    join_near = gpd.sjoin_nearest(pt, props, how="left", distance_col="dist_m")
    if len(join_near):
        row = join_near.iloc[0]
        return {
            "matched": False,
            "Full_Address": row["Full_Address"],
            "Suburb": row["Suburb"],
            "Postcode": row["Postcode"],
            "lat": lat,
            "lon": lon
        }

    return {"matched": False, "Full_Address": None, "Suburb": None, "Postcode": None, "lat": lat, "lon": lon}

def find_property_from_suburb(suburb_name):
    """Return a random representative property from a suburb."""
    sub = props[props["Suburb"].str.upper() == suburb_name.upper()]
    if len(sub):
        row = sub.sample(1).iloc[0]
        return {
            "matched": True,
            "Full_Address": row["Full_Address"],
            "Suburb": row["Suburb"],
            "Postcode": row["Postcode"],
            "lat": row.geometry.centroid.y,
            "lon": row.geometry.centroid.x
        }
    return None

# -------------------------------------------------------------
# MAIN PARSER
# -------------------------------------------------------------
def parse_event(text):
    rain = extract_rainfall(text)
    dur = extract_duration(text)
    coords = extract_coords(text)
    severity = extract_severity(text)
    loc_text = extract_location(text)

    prop_data = None
    if coords:
        lat, lon = coords
        prop_data = find_property_from_coords(lat, lon)
    elif loc_text:
        prop_data = find_property_from_suburb(loc_text)

    event = {
        "rain_mm": rain,
        "duration_hr": dur,
        "timestamp": datetime.now().isoformat(),
        "severity_flag": severity
    }

    if prop_data:
        event.update(prop_data)
    else:
        event.update({
            "Full_Address": None,
            "Suburb": None,
            "Postcode": None,
            "lat": None,
            "lon": None
        })

    return event

# -------------------------------------------------------------
# DEMO RUN
# -------------------------------------------------------------
if __name__ == "__main__":
    test_inputs = [
        "Heavy rainfall of 60 mm near -38.1643, 145.1245 lasting 45 minutes",
        "Severe rainfall of 80 millimetres in Frankston South for 1 hour",
        "Light rain of 25mm near 145.132, -38.144 lasting 30 minutes",
        "Extreme rainfall reported around 20 Smith Street, Frankston",
        "Moderate rain of 12mm in Seaford for 15 minutes"
    ]

    for txt in test_inputs:
        result = parse_event(txt)
        print("\n💧 Input:", txt)
        print("🧭 Output:", result)
