import geopandas as gpd
import rasterio
from shapely.geometry import box
import pandas as pd
from shapely.errors import ShapelyDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# ------------------------------------------------------------------
# CONFIG PATHS
# ------------------------------------------------------------------
PROPERTY_PATH = "SwinburneData/Property/Properties.gpkg"
FLOOD_RASTER_PATH = "SwinburneData/FloodMaps/FrankstonSouth/001y/Mapping/FS_001_001y_010m_dmax.grd"
OUTPUT_CSV = "houses/houses_in_flood_region.csv"

# ------------------------------------------------------------------
# 1. Load raster and get its bounding box
# ------------------------------------------------------------------
with rasterio.open(FLOOD_RASTER_PATH) as src:
    bounds = src.bounds
    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    raster_crs = src.crs
    print(f"🗺️ Flood map bounding box (m): {bounds}")
    print(f"Raster CRS: {raster_crs}")

# ------------------------------------------------------------------
# 2. Load property polygons
# ------------------------------------------------------------------
props = gpd.read_file(PROPERTY_PATH)
print(f"🏠 Loaded {len(props)} property polygons.")

# Ensure CRS consistency
if props.crs != raster_crs:
    print(f"🔄 Reprojecting properties to match raster CRS...")
    props = props.to_crs(raster_crs)

# ------------------------------------------------------------------
# 3. Compute centroids & spatial join
# ------------------------------------------------------------------
props["centroid"] = props.geometry.centroid
centroids = gpd.GeoDataFrame(props, geometry="centroid", crs=raster_crs)

flood_area = gpd.GeoDataFrame(geometry=[bbox], crs=raster_crs)

# FIX 1: use “intersects” instead of “within” — flood bounds may slightly clip edges
houses_in_flood = gpd.sjoin(centroids, flood_area, how="inner", predicate="intersects")

print(f"✅ Found {len(houses_in_flood)} property centroids inside/intersecting flood raster extent.")

# ------------------------------------------------------------------
# 4. Convert to lat/lon for readability
# ------------------------------------------------------------------
houses_in_flood = houses_in_flood.to_crs(epsg=4326)
houses_in_flood["lat"] = houses_in_flood.geometry.y
houses_in_flood["lon"] = houses_in_flood.geometry.x

# ------------------------------------------------------------------
# 5. Clean address information
# ------------------------------------------------------------------
def clean_str(val):
    return str(val).strip() if pd.notna(val) else ""

houses_in_flood["Full_Address"] = (
    houses_in_flood["House"].apply(clean_str) + " " +
    houses_in_flood["Street"].apply(clean_str) + ", " +
    houses_in_flood["Suburb"].apply(clean_str) + " VIC " +
    houses_in_flood["Postcode"].apply(clean_str)
).str.replace(" ,", ",").str.replace("  ", " ").str.strip()

houses_in_flood["Suburb_Assigned"] = houses_in_flood["Suburb"].apply(clean_str)
houses_in_flood["Suburb_OSM"] = houses_in_flood["Suburb_Assigned"]
houses_in_flood["Full_OSM_Address"] = houses_in_flood["Full_Address"]
houses_in_flood["Match"] = True

# ------------------------------------------------------------------
# 6. Filter only Frankston South region + cleanup
# ------------------------------------------------------------------


# Remove incomplete addresses and duplicates
houses_in_flood = houses_in_flood[houses_in_flood["Full_Address"].str.len() > 10]
houses_in_flood = houses_in_flood.drop_duplicates(subset=["lat", "lon"])

# If too many, pick nearest 100 to flood center
if len(houses_in_flood) > 100:
    flood_center_lat, flood_center_lon = -38.150, 145.135
    houses_in_flood["dist"] = ((houses_in_flood["lat"] - flood_center_lat)**2 + 
                               (houses_in_flood["lon"] - flood_center_lon)**2)**0.5
    houses_in_flood = houses_in_flood.sort_values("dist").head(100)

# ------------------------------------------------------------------
# 7. Final formatting and export
# ------------------------------------------------------------------
houses_in_flood = houses_in_flood.reset_index(drop=True)
houses_in_flood["house_id"] = houses_in_flood.index + 1

output_df = houses_in_flood[
    ["house_id", "House", "Full_Address", "lat", "lon", 
     "Suburb_Assigned", "Suburb_OSM", "Full_OSM_Address", "Match"]
].rename(columns={"House": "address"})

output_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Exported {len(output_df)} houses → {OUTPUT_CSV}")
