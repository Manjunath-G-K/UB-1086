# Handles property and address resolution.
#
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import re


def load_properties(path="SwinburneData/Property/Properties.gpkg"):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=7855)
    gdf["Full_Address"] = (
    gdf["House"].fillna('').astype(str) + ' ' + gdf["Street"].fillna('').astype(str) + ', ' +
    gdf["Suburb"].fillna('').astype(str) + ' VIC ' + gdf["Postcode"].fillna('').astype(str)        ).str.replace(r"\s+,", ",", regex=True)
    return gdf


def resolve_address(address_text:str, props_gdf:gpd.GeoDataFrame):
    tokens = re.split(r"[ ,]+", address_text.upper().strip())
    sub = next((t for t in tokens if t.isalpha()), None)
    subset = props_gdf[props_gdf['Suburb'].astype(str).str.upper().str.contains(sub, na=False)] if sub else props_gdf
    matches = subset[subset['Street'].astype(str).str.upper().str.contains(tokens[0], na=False)] if len(tokens)>0 else subset
    if matches.empty:
        return None
    row = matches.iloc[0]
    centroid = row.geometry.centroid
    return {
    'Full_Address': row['Full_Address'],
    'lat': centroid.y,
    'lon': centroid.x,
    'prop_idx': int(row.name)
    }