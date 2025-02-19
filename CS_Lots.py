#!/usr/bin/env python
# coding: utf-8

# # Public EV Charger Placement Optimization Project

# ## EVCS-OPTIM
# Electric Vehicle Charging Station - Optimization Placement Tool via Intersection Modeling

# In[1]:


import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import matplotlib.pyplot as plt


import folium
from folium.plugins import HeatMap
import branca

import osmnx as ox
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, mapping, MultiPolygon


# **Brainstorm**
# 
# Top Charging Locations by Type:
# - Shopping Centers
# - Other commercial destinations
# - Offices (Semi-Private)
# - Parks
# - Libraries
# - Schools
# - Other public spaces
# - Hospitals/Clinics
# - Public Transit Stations
# - Rest/Refuel Stations
# - High Density Residential/Mixed
# - Community Centers / Churches
# - Stadiums

# **Consumer Costs**
# 

# ## Data Gathering

# ### Charger Data

# Let's start our data gathering process by collecting information on public EV chargers in California through AFDC database

# In[2]:


# Charger Data Query
url = "https://developer.nrel.gov/api/alt-fuel-stations/v1.json"
params = {
    'format': 'json',  # Output response format
    'api_key': 'Tbqhfv28h6gIIxbaMTByUNg4ByP2vf1E1A3XYgGa',  # Your developer API key
    'status': 'all',  # Return stations that match the given status
    # 'access': 'public',  # Return stations with the given access type
    'fuel_type': 'ELEC', # Return stations that supply any of the given fuel types
    'state': 'CA',
    'country': 'US',
    
}

response = requests.get(url, params=params)
if response.status_code == 200:

    data = response.json()


# ### SDG&E Territory Zip Codes

# We will need to use SDG&E territory zip codes to filter much of our data by the relevant geographical area

# In[3]:


sdge_zips_data = pd.read_excel("data/SDGE Service Territory Zip Code List Q2-2021.xlsx")
sdge_zips = list(sdge_zips_data['ZIP_CODE'])


# ### Zoning Data

# Zoning regulations and land-use can be a useful determinant of where EV chargers might be located

# In[4]:


zoning_data = gpd.read_file("shapefiles/zoning_datasd.shp")

zoning_data = zoning_data.to_crs(epsg=4326)  # Convert to WGS84 (EPSG:4326)

# Convert zone codes to types
zoning_categories = {
    'Commercial': ['CC', 'CN', 'CV', 'CP', 'CR', 'CCPD'],
    'Office': ['CO'],
    'Residential High': ['RH', 'RM-3', 'RM-4'],
    'Residential Medium': ['RM-2', 'RM-1'],
    'Residential Low': ['RS', 'RL'],
    'Residential Mixed': ['RMX'],
    'Industrial': ['IP', 'IL', 'IH', 'IS', 'IBT'],
    'Mixed Use': ['MU', 'EMX'],
    'Agricultural': ['AG', 'AR'],
    'Open Space': ['OS'],
    'Planned': ['BLPD', 'MBPD', 'GQPD', 'MPD', 'CUPD', 'LJPD', 'LJSPD'],
    'Transit': ['OTOP', 'OTRM', 'OTCC'],
    'Other': ['UNZONED'],
}

def map_zoning_category(zone_code):
    if isinstance(zone_code, str):  # Check if zone_code is a string
        for category, prefixes in zoning_categories.items():
            if any(zone_code.startswith(prefix) for prefix in prefixes):
                return category
    return 'Other'  # Return 'Other' for NaN or non-string values

zoning_data.fillna({'zone_name':'Unknown'}, inplace=True)
zoning_data['zone_type'] = zoning_data['zone_name'].apply(map_zoning_category)


# ### DMV Registration Data

# Gather information on the local registration information of vehicles according to the DMV

# In[5]:


dmv_data = pd.read_csv("vehicle-fuel-type-count-by-zip-code-20231.csv")


# ### Installation Costs

# - 2015 EVCS/EVSE Installation Costs: https://afdc.energy.gov/files/u/publication/evse_cost_report_2015.pdf
#   - Average public AC level 2 EVCS installation cost in San Diego is ~4000 (33% higher than average)
#     
#   - Average public DCFC EVCS installation cost ~24000 (33% figure would entail ~32000 cost in San Diego)
# 
# 

# ## Pre-Processing

# ### Zip Codes

# In[6]:


pop_gdf = gpd.read_file("population.geojson", driver="GeoJSON")
income_gdf = gpd.read_file("income.geojson", driver="GeoJSON")
# pop_gdf.head()


# In[7]:


pop_gdf


# ### Parking Lots

# In[8]:


zip_shape = gpd.read_file("shapefiles/sdge_zcta.shp")
# zip_shape


# In[9]:


import osmnx as ox
import geopandas as gpd
import folium

# Define the area of interest
place_name = "San Diego, California, USA"

# Query OSM for parking lots
parking_lots = ox.geometries_from_place(place_name, tags={"amenity": "parking"})

# Filter for public parking lots
public_parking = parking_lots[
    (parking_lots.get("parking") == "public") |  # Explicitly public parking
    (parking_lots.get("access").isin([None, "yes", "permissive"]))  # Exclude private/restricted lots
]

# Convert to a GeoDataFrame
public_parking_gdf = gpd.GeoDataFrame(public_parking, geometry="geometry", crs="EPSG:4326")
parking_cols = ['osmid','geometry','amenity','access','fee','parking','capacity']
public_parking_gdf = public_parking_gdf.reset_index()[parking_cols]

# Create a folium map centered on San Diego
m = folium.Map(location=[32.7157, -117.1611], zoom_start=12)

# Add public parking lots to the map
for _, row in public_parking_gdf.iterrows():
    if row.geometry.geom_type == "Point":
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup="Public Parking Lot"
        ).add_to(m)
    elif row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
        folium.GeoJson(
            row.geometry,
            style_function=lambda x: {"color": "blue", "fillColor": "blue", "weight": 1, "fillOpacity": 0.4},
            tooltip="Non-Private Parking Lot"
        ).add_to(m)
print(public_parking_gdf.shape)
print(public_parking_gdf.columns)
m


# In[10]:


public_parking_gdf.head(5)


# In[11]:


# intersections_gdf = intersections_gdf.set_crs("EPSG:4326", allow_override=True)
zoning_data = zoning_data.set_crs("EPSG:4326", allow_override=True)

# Perform the spatial join to map intersections to zoning data
lots = gpd.sjoin(public_parking_gdf, zoning_data, how="left", predicate="within")

# Drop irrelevant columns
lots = lots.drop(columns=['index_right','imp_date','ordnum','objectid'])

# Remove intersections with null zoning
lots.loc[pd.isna(lots['zone_type']), 'zone_type'] = "Multiple"

print(lots.shape)
lots.head()


# In[12]:


lots.value_counts('zone_type')


# In[13]:


import folium
import branca.colormap as cm

# Create a folium map centered on San Diego
m2 = folium.Map(location=[32.7157, -117.1611], zoom_start=12)

# Define color mapping for zoning types
color_mapping = {
    'Commercial': 'lightblue',
    'Office': 'lightgreen',
    'Residential High': 'lightcoral',
    'Residential Medium': 'khaki',
    'Residential Low': 'lightyellow',
    'Residential Mixed': 'lightpink',
    'Industrial': 'lightgray',
    'Mixed Use': 'lightseagreen',
    'Agricultural': 'lightgoldenrodyellow',
    'Open Space': 'lightblue',
    'Planned': 'lightcyan',
    'Transit': 'lavender',
    'Other': 'gray',
}

# Function to style zoning polygons
def zoning_style(feature):
    zone_type = feature['properties']['zone_type']
    color = color_mapping.get(zone_type, 'gray')  # Default to gray if not found
    return {
        'fillColor': color, 
        'color': 'black', 
        'weight': 1,
        'fillOpacity': 0.3,
    }
    
# Add zoning overlay to the map
folium.GeoJson(
    zoning_data,  # GeoJSON file with zoning polygons
    style_function=zoning_style
).add_to(m2)

# Add a legend for zoning categories
opacity = 0.8
legend_html = f"""
<div style="position: fixed; 
             top: 10px; left: 10px; 
             width: 200px; height: auto; 
             background-color: rgba(255, 255, 255, {opacity}); 
             border:2px solid grey; 
             z-index: 9999; 
             font-size:14px;">
    <b>Zoning Categories</b><br>
    <div style="line-height: 1.5;">
"""
for category, color in color_mapping.items():
    legend_html += f'<div><i style="background:{color}; width: 20px; height: 20px; display: inline-block;"></i> {category}</div>'
legend_html += "</div></div>"
m2.get_root().html.add_child(folium.Element(legend_html))

# Add parking lots with zoning type tooltips
for _, row in lots.iterrows():
    zone_type = row['zone_type'] if 'zone_type' in row else "Unknown"
    
    if row.geometry.geom_type == "Point":
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.7,
            popup=f"Zone Type: {zone_type}",
            tooltip=f"Zone: {zone_type}"
        ).add_to(m2)
    
    elif row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
        folium.GeoJson(
            row.geometry,
            style_function=lambda x: {"color": "green", "fillColor": "green", "weight": 1, "fillOpacity": 0.4},
            tooltip=f"Zone: {zone_type}"
        ).add_to(m2)


# Save map as an HTML file (optional)
m2.save("parking_lots_with_zoning.html")

# Display the map
print("Map of Public Parking Lots Overlaid by City Zoning")
m2


# ### AFDC Charging Stations

# In[14]:


# CS - Charging Stations
cs = pd.DataFrame(data['fuel_stations'])
cs = cs[cs['zip'] != 'CA']
cs['zip'] = cs['zip'].astype(int)
cs = cs[cs['zip'].isin(sdge_zips)]
cs = cs[cs['access_code']=="public"]
cs = cs[[
    'id', 'access_code','facility_type','latitude','longitude','zip','ev_connector_types',
    'ev_dc_fast_num','ev_level1_evse_num','ev_level2_evse_num','ev_network'
]]
cs = cs.reset_index(drop=True)

print(cs.shape)
# cs.columns
cs.head()


# In[15]:


get_ipython().run_cell_magic('time', '', "# Create a geometry column for EV chargers\ngeometry = [Point(xy) for xy in zip(cs['longitude'], cs['latitude'])]\ncs_gdf = gpd.GeoDataFrame(cs, geometry=geometry)\ncs_gdf.set_crs(epsg=4326, inplace=True)\n\n# Create a one-hot encoding of the 'ev_connector_types' column\n# Step 1: Explode the 'ev_connector_types' column to create individual rows\ncs_gdf_exploded = cs_gdf.explode('ev_connector_types', ignore_index=True)\n\n# Step 2: One-hot encode the exploded 'ev_connector_types' column\none_hot_encoded = pd.get_dummies(cs_gdf_exploded['ev_connector_types'], prefix='connector')\nconnector_cols = list(one_hot_encoded.columns)\n\n# Step 3: Combine the one-hot encoded DataFrame with the exploded DataFrame\ncs_gdf_exploded = pd.concat([cs_gdf_exploded, one_hot_encoded], axis=1)\ncs_gdf_exploded = cs_gdf_exploded.groupby('id')[connector_cols].sum().reset_index()\n\n# Step 4: Aggregate back to the original DataFrame, summing up the one-hot columns\ncs_gdf = cs_gdf.merge(cs_gdf_exploded, how='left', on='id')\ncs_gdf = cs_gdf.drop(columns=['ev_connector_types'])\n\n# Optional: Filter chargers by a specific zip code (e.g., 92122)\n# zip_list = [92122, 92093, 92037, 92117, 92161, 92092, 92121, 92145]\n# cs_gdf = cs_gdf[cs_gdf['zip'].isin(zip_list)]\n\n# Check the resulting DataFrame\nprint(cs_gdf.shape)\ncs_gdf.head()\n")


# In[16]:


# %%time
# # Map Chargers to Parking Lots

# def nearest_lot(charger_row, lots_gdf):
#     charger_coords = (charger_row['latitude'], charger_row['longitude'])
#     distances = lots_gdf.geometry.apply(lambda x: geodesic(charger_coords, (x.centroid.y, x.centroid.x)).meters if isinstance(x, (Point, Polygon, MultiPolygon)) else float('inf'))
#     nearest_index = distances.idxmin()
#     return lots_gdf.loc[nearest_index]

# def chargers_within_radius(lot, chargers_df, radius=400/2):  # radius in meters (1 mile = 1609.34 meters)
#     lot_coords = (lot.geometry.centroid.y, lot.geometry.centroid.x)
#     chargers_coords = list(zip(chargers_df['latitude'], chargers_df['longitude']))
#     distances = [geodesic(lot_coords, charger).meters for charger in chargers_coords]
#     return sum(d <= radius for d in distances)

# def distance_to_nearest_charger(lot, chargers_df):
#     lot_coords = (lot.geometry.centroid.y, lot.geometry.centroid.x)
#     chargers_coords = list(zip(chargers_df['latitude'], chargers_df['longitude']))
#     distances = [geodesic(lot_coords, charger).meters for charger in chargers_coords]
#     return min(distances)

# # Apply function to assign nearest lot to each charger
# chargers_with_lots = cs_gdf.apply(nearest_lot, lots_gdf=lots, axis=1)

# cs_by_lots = cs_gdf.merge(chargers_with_lots[['osmid']], how='left', left_index=True, right_index=True)

# type_cols = ['ev_dc_fast_num','ev_level1_evse_num','ev_level2_evse_num']

# cs_by_lots = cs_by_lots.groupby('osmid')[type_cols+connector_cols].sum().reset_index()
# lots_cs = lots.merge(cs_by_lots, how='left', on='osmid')
# lots_cs[type_cols] = lots_cs[type_cols].fillna(0)
# lots_cs[connector_cols] = lots_cs[connector_cols].fillna(0)
# lots_cs['cs_total'] = lots_cs[type_cols].sum(axis=1)
# lots_cs[connector_cols] = lots_cs[connector_cols] > 0

# # Apply function to calculate the number of chargers within a radius for each intersection
# lots_cs['chargers_in_radius'] = lots_cs.apply(chargers_within_radius, chargers_df=cs_gdf, axis=1)

# # Apply function to calculate the nearest charger distance for each intersection
# lots_cs['distance_to_nearest_charger'] = lots_cs.apply(distance_to_nearest_charger, chargers_df=cs_gdf, axis=1)

# lots_cs.to_file('lots_cs.geojson', driver='GeoJSON')

# print(cs_lots.shape)
# lots_cs.head()


# In[17]:


lots_cs = gpd.read_file("lots_cs.geojson")
# lots_cs = lots_cs.iloc[:,1:]
lots_cs.head(2)


# In[18]:


# road_traffic = gpd.read_file("traffic_data.geojson")
# road_traffic.head(2)


# In[19]:


rl = gpd.read_file("data/roads_datasd.geojson")
print(rl.shape)
rl.head(2)


# In[21]:


tc = pd.read_csv("data/traffic_counts_datasd.csv")
# print(tc.shape)
tc['date_count'] = pd.to_datetime(tc['date_count'])
tc_sorted = tc.sort_values(by=['street_name', 'date_count'], ascending=[True, False])
tc_recent = tc_sorted.drop_duplicates(subset='street_name')
print(tc_recent.shape)
tc_recent.head(2)


# In[40]:


from shapely.geometry import MultiLineString
rl = rl.to_crs(epsg=4326)

# Merge based on street names and ensure the result is a GeoDataFrame
tc_rel = tc_recent[['id','street_name','total_count']]
rl_rel = rl[['rd20full','geometry']]
merged = tc_rel.merge(rl_rel, left_on='street_name', right_on='rd20full', how='inner')

# Convert the merged DataFrame to a GeoDataFrame
road_traffic = gpd.GeoDataFrame(merged, geometry='geometry')

# Group by 'id', aggregate geometries into MultiLineString, and calculate the average 'total_count'
road_traffic_grouped = road_traffic.groupby('street_name').agg({
    'geometry': lambda x: MultiLineString(x.tolist()),
    'total_count': 'mean'
}).reset_index()

# Convert the grouped DataFrame to a GeoDataFrame and set the CRS
road_traffic_grouped = gpd.GeoDataFrame(road_traffic_grouped, geometry='geometry')
road_traffic_grouped.crs = 'EPSG:4326'

# Print the final road_traffic_grouped GeoDataFrame and count of NaN in 'total_count'
print(road_traffic_grouped['total_count'].isna().sum())
road_traffic_grouped.head(2)


# In[41]:


get_ipython().run_cell_magic('time', '', '# Assuming you have your dataframes ready: road_traffic and cs_lots\n# Ensure they both have a consistent coordinate reference system (CRS)\nroad_traffic_grouped = road_traffic_grouped.to_crs(\'EPSG:4326\')\nlots_cs = lots_cs.to_crs(\'EPSG:4326\')\n\ncs_lots_traffic = lots_cs.copy()\n\n# Re-project to a projected CRS for accurate buffer operation (e.g., EPSG:3857)\nbuffered_cs_lots = lots_cs.to_crs(\'EPSG:3857\')\nroad_traffic_projected = road_traffic_grouped.to_crs(\'EPSG:3857\')\n\n# Buffer the parking lots to consider "adjacent" roads within a given distance (e.g., 5 meters)\nbuffered_cs_lots[\'geometry\'] = buffered_cs_lots[\'geometry\'].buffer(50)\n\n# Calculate the total adjacent traffic volume for each parking lot\ndef calculate_total_adjacent_traffic(parking_lot, road_traffic_grouped):\n    total_traffic = 0\n    for road in road_traffic_grouped.itertuples():\n        if parking_lot.intersects(road.geometry):\n            total_traffic += road.total_count\n    return total_traffic\n\n# Add a new column to cs_lots for total adjacent traffic\ncs_lots_traffic[\'traffic\'] = buffered_cs_lots.apply(\n    lambda row: calculate_total_adjacent_traffic(row[\'geometry\'], road_traffic_projected), axis=1\n).astype(int)  # Calculate total adjacent traffic using roads.\n\n# Re-project back to the original CRS\ncs_lots_traffic = cs_lots_traffic.to_crs(\'EPSG:4326\')\n\n# Print the results\ncs_lots_traffic\n')


# In[42]:


cs_lots_traffic['traffic'].mean()


# In[46]:


# import folium
# import geopandas as gpd
# import branca.colormap as cm

# # Define a colormap for the traffic values
# traffic_colormap = cm.linear.YlOrRd_09.scale(cs_lots_traffic['traffic'].min(), cs_lots_traffic['traffic'].max())

# # Ensure CRS is WGS84 (EPSG:4326) for compatibility with Folium
# cs_lots_traffic = cs_lots_traffic.to_crs(epsg=4326)
# road_traffic = road_traffic.to_crs(epsg=4326)

# # Initialize the folium map centered on San Diego
# m = folium.Map(location=[32.7157, -117.1611], zoom_start=11, tiles="cartodbpositron")

# # Add the color scale legend to the map
# traffic_colormap.caption = "Traffic Volume"
# traffic_colormap.add_to(m)

# # Add road traffic data to the map
# for _, row in road_traffic.iterrows():
#     if row['geometry'].geom_type == 'LineString':
#         folium.PolyLine(
#             locations=[(point[1], point[0]) for point in row['geometry'].coords],
#             color='blue',
#             weight=2,
#             tooltip=folium.Tooltip(f'Traffic: {row["total_count"]}')
#         ).add_to(m)

# # Add parking lots to the map
# for _, row in cs_lots_traffic.iterrows():
#     color = traffic_colormap(row['traffic'])
    
#     tooltip_text = (
#         f"Parking Lot OSMID: {row.osmid}<br>"
#         f"Traffic: {row.traffic}<br>"
#     )

#     if isinstance(row.geometry, Point):
#         # Add a small circle for point geometries
#         folium.Circle(
#             location=(row.geometry.y, row.geometry.x),
#             radius=20,  # Small circle radius
#             color=color,
#             fill=True,
#             fill_opacity=0.6,
#             tooltip=folium.Tooltip(tooltip_text),
#         ).add_to(m)
#     else:
#         # Add the parking lot geometry to the map
#         folium.GeoJson(
#             row.geometry,
#             style_function=lambda feature, color=color: {
#                 'fillColor': color,
#                 'color': color,
#                 'weight': 1,
#                 'fillOpacity': 0.6,
#                 'opacity': 0.6
#             },
#             tooltip=folium.Tooltip(tooltip_text),
#         ).add_to(m)

# # Display the map
# m


# In[18]:


# # Manually weighted demand-score
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# cs_lots_scored = cs_lots.copy()

# features = ['distance_to_nearest_charger', 'chargers_in_radius', 'cs_total']
# scaler = StandardScaler()
# standardized_features = scaler.fit_transform(cs_lots[features])

# cs_lots_scored[['distance_score', 'radius_score', 'cs_total_score']] = standardized_features

# cs_lots_scored['demand_score'] = (
#     0.6 * cs_lots_scored['distance_score'] +
#     -0.2 * cs_lots_scored['radius_score'] +
#     -0.1 * cs_lots_scored['cs_total_score']
# )

# min_max_scaler = MinMaxScaler()
# cs_lots_scored['demand_score'] = min_max_scaler.fit_transform(cs_lots_scored[['raw_demand_score']])

# print(cs_lots_scored.shape)
# cs_lots_scored.head(10)


# In[19]:


# # Principal Component Analysis Scoring
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# cs_lots_scored = cs_lots.copy()

# # Standardize the features
# features = ['distance_to_nearest_charger', 'chargers_in_radius', 'cs_total']
# scaler = StandardScaler()
# standardized_features = scaler.fit_transform(cs_lots[features])

# # Apply PCA
# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(standardized_features)

# cs_lots_scored[['pc1', 'pc2']] = principal_components

# # Calculate demand score based on principal components
# cs_lots_scored['raw_demand_score'] = (
#     0.6 * cs_lots_scored['pc1'] +
#     0.4 * cs_lots_scored['pc2']
# )

# # Normalize the demand score to a scale of 0 to 1
# min_max_scaler = MinMaxScaler()
# cs_lots_scored['demand_score'] = min_max_scaler.fit_transform(cs_lots_scored[['raw_demand_score']])

# print(cs_lots_scored.shape)
# cs_lots_scored.head(2)


# In[49]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

cs_lots_scored = cs_lots_traffic.copy()

# Calculate percentile rank for each feature
features = ['distance_to_nearest_charger', 'chargers_in_radius', 'cs_total', 'traffic']
for feature in features:
    cs_lots_scored[f'percentile_{feature}'] = cs_lots_scored[feature].rank(pct=True)

# Manually assign weights based on expert knowledge
cs_lots_scored['distance_score'] = cs_lots_scored['percentile_distance_to_nearest_charger'] * 0.5
cs_lots_scored['radius_score'] = cs_lots_scored['percentile_chargers_in_radius'] * -0.3
cs_lots_scored['cs_total_score'] = cs_lots_scored['percentile_cs_total'] * -0.2
cs_lots_scored['traffic_score'] = cs_lots_scored['percentile_traffic'] * 0.1

# Calculate the combined demand score including the traffic score
cs_lots_scored['demand_score'] = (
    cs_lots_scored['distance_score'] +
    cs_lots_scored['radius_score'] +
    cs_lots_scored['cs_total_score'] +
    cs_lots_scored['traffic_score']
)

# Normalize the demand score to a scale of 0 to 1
min_max_scaler = MinMaxScaler()
cs_lots_scored['demand_score'] = min_max_scaler.fit_transform(cs_lots_scored[['demand_score']])

print(cs_lots_scored.shape)
cs_lots_scored.head(2)


# In[50]:


get_ipython().run_cell_magic('time', '', '\n# Define a colormap for the demand score\ndemand_score_colormap = cm.linear.YlOrRd_09.scale(cs_lots_scored[\'demand_score\'].min(), cs_lots_scored[\'demand_score\'].max())\n\n# Ensure CRS is WGS84 (EPSG:4326) for compatibility with Folium\ncs_lots_scored = cs_lots_scored.to_crs(epsg=4326)\n\n# Initialize the folium map centered on San Diego\nm4 = folium.Map(location=[32.7157, -117.1611], zoom_start=11, tiles="cartodbpositron")\n\n# Add the color scale legend to the map\ndemand_score_colormap.caption = "Demand Score"\ndemand_score_colormap.add_to(m4)\n\n# Add parking lots to the map\nfor _, row in cs_lots_scored.iterrows():\n    color = demand_score_colormap(row[\'demand_score\'])\n    \n    tooltip_text = (\n        f"Parking Lot OSMID: {row.osmid}<br>"\n        f"Demand Score: {row.demand_score:.2f}<br>"\n        f"Zone Name: {row.zone_name}<br>"\n        f"Distance to Nearest Charger: {row.distance_to_nearest_charger:.2f} meters<br>"\n        f"Chargers in Radius: {row.chargers_in_radius}<br>"\n        f"CS Total: {row.cs_total}<br>"\n        f"Traffic: {row.traffic}<br>"\n        f"Percentile Distance Score: {row.percentile_distance_to_nearest_charger:.2f}<br>"\n        f"Percentile Radius Score: {row.percentile_chargers_in_radius:.2f}<br>"\n        f"Percentile CS Total Score: {row.percentile_cs_total:.2f}<br>"\n        f"Distance Score: {row.distance_score:.2f}<br>"\n        f"Radius Score: {row.radius_score:.2f}<br>"\n        f"CS Total Score: {row.cs_total_score:.2f}"\n    )\n\n    if isinstance(row.geometry, Point):\n        # Add a small circle for point geometries\n        folium.Circle(\n            location=(row.geometry.y, row.geometry.x),\n            radius=20,  # Small circle radius\n            color=color,\n            fill=True,\n            fill_opacity=0.6,\n            tooltip=folium.Tooltip(tooltip_text),\n        ).add_to(m4)\n    else:\n        # Add the parking lot geometry to the map\n        folium.GeoJson(\n            row.geometry,\n            style_function=lambda feature, color=color: {\n                \'fillColor\': color,\n                \'color\': color,\n                \'weight\': 1,\n                \'fillOpacity\': 0.6,\n                \'opacity\': 0.6\n            },\n            tooltip=folium.Tooltip(tooltip_text),\n        ).add_to(m4)\n\n# Display the map\nprint("Map of parking lots with color-coded demand scores")\nm4\n')


# In[87]:


get_ipython().run_cell_magic('time', '', '# Ensure CRS is WGS84 (EPSG:4326) for compatibility with Folium\ncs_lots_scored = cs_lots_scored.to_crs(epsg=4326)\n\n# Initialize the folium map centered on San Diego\nm3 = folium.Map(location=[32.7157, -117.1611], zoom_start=11, tiles="cartodbpositron")\n\n# Function to get coordinates based on geometry type\ndef get_coords(geometry):\n    if isinstance(geometry, (Point, Polygon, MultiPolygon)):\n        centroid = geometry.centroid\n        return centroid.y, centroid.x\n    return np.nan, np.nan\n\n# Add intersection points to the map\nfor _, row in cs_lots_scored.iterrows():\n    lat, lon = get_coords(row.geometry)\n    if np.isnan(lat) or np.isnan(lon):\n        continue  # Skip invalid coordinates\n\n    # Add a circle marker for each intersection\n    folium.CircleMarker(\n        location=(lat, lon),  # Latitude and Longitude\n        radius=3,\n        color="red",\n        fill=True,\n        fill_opacity=0.5,\n        popup=folium.Popup(f"Intersection OSMID: {row.osmid}<br>Zone Name: {row.zone_name}", max_width=250),\n    ).add_to(m3)\n\n    # Add buffer zones as circles\n    folium.Circle(\n        location=(lat, lon),\n        radius=row.distance_to_nearest_charger,  # Distance in meters\n        color="blue",  # Circle line color\n        weight=2,  # Line thickness\n        opacity=0.3,  # Line opacity\n        fill=True,\n        fill_opacity=0.01,  # Adjust the fill opacity separately\n    ).add_to(m3)\n\n# Display the map\nprint("Map of parking lots with a radius to the nearest public charger")\nm3\n')


# In[13]:


# %%time

# type_cols = ['ev_dc_fast_num','ev_level1_evse_num','ev_level2_evse_num']

# cs_by_intersection = cs_gdf.groupby('osmid')[type_cols+connector_cols].sum().reset_index()
# intersection_cs = intersections.merge(cs_by_intersection, how='left', on='osmid')

# intersection_cs[type_cols] = intersection_cs[type_cols].fillna(0)
# intersection_cs[connector_cols] = intersection_cs[connector_cols].fillna(0)
# intersection_cs['cs_total'] = intersection_cs[type_cols].sum(axis=1)

# intersection_cs[connector_cols] = intersection_cs[connector_cols] > 0

# print(intersection_cs.shape)
# intersection_cs.sort_values(by='cs_total', ascending=False).head(20)


# ### DMV Registration Data

# Now filter the charger data to get all public AFDC chargers in the SDG&E territory

# In[14]:


# # DMV
# dmv = dmv_data.copy()
# dmv = dmv[dmv['ZIP Code']!="OOS"]
# dmv['ZIP Code'] = dmv['ZIP Code'].astype(int)
# dmv = dmv[dmv['ZIP Code'].isin(sdge_zips)]
# print(dmv.shape)
# dmv.head()


# ### Feature Engineering
