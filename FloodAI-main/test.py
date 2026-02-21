import geopandas as gpd

catch = gpd.read_file("SwinburneData/RORB_GIS_ESRI/FM2023_24_Kananook_RORB_catchment.shp")
node = gpd.read_file("SwinburneData/RORB_GIS_ESRI/FM2023-24_Kananook_RORB_node.shp")
# node1 = gpd.read_file("SwinburneData/RORB_GIS_ESRI/FM2023-24_Kananook_RORB_node.prj")
# node2 = gpd.read_file("SwinburneData/RORB_GIS_ESRI/FM2023-24_Kananook_RORB_node.qmd")
node3= gpd.read_file("SwinburneData/RORB_GIS_ESRI/FM2023-24_Kananook_RORB_node.dbf")
node4= gpd.read_file("SwinburneData/RORB_GIS_ESRI/FM2023-24_Kananook_RORB_node.shx")


print("\n--- Catchment Columns ---")
print(catch.columns.tolist())
# print(node1.columns.tolist())
# print(node2.columns.tolist())
print(node3.columns.tolist())
print(node4.columns.tolist())

print("\n--- Node Columns ---")
print(node.columns.tolist())

print("\nSample catchment record:")
print(catch.head(3))

print("\nSample node record:")
print(node.head(3))
