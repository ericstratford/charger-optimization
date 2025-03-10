import requests
import json
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from scipy.stats import pearsonr, spearmanr
from shapely.geometry import Point
from collections import Counter
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

# process zip code list of sdge service territory
sdge_zip = pd.read_excel("data/sdge_zip.xlsx", sheet_name=0)
unique_zip = sdge_zip['ZIP_CODE'].unique().tolist()
print(unique_zip)
print("There are " + str(len(unique_zip)) + " unique zip codes")

# load full afdc data into a dataframe
with open('data/afdc_full.json', 'r') as f:
    afdc = json.load(f)
fs = afdc.get('fuel_stations', [])
afdc_df = pd.DataFrame(fs)
# start narrow down to electric car charger within SDG&E service territory
sdge = afdc_df[(afdc_df['fuel_type_code'] == 'ELEC')][['id', 'zip', 'latitude', 'longitude']]
sdge['zip'] = pd.to_numeric(sdge['zip'], errors='coerce').astype('Int64')
sdge = sdge[sdge['zip'].isin(unique_zip)]
sdge = sdge.reset_index(drop=True)
print(sdge.head())
print("Number of electric vehicle chargers in SDG&E service territory:", len(sdge))

# overpass API url and query
# A node is a single point in space defined by its latitude and longitude coordinates.
# A way represents linear or area features by connecting nodes together.
# A relation is more complex type that defines relationships between multiple nodes, ways, or other relations.

lat_min = 32.54
lon_min = -118.06
lat_max = 33.73
lon_max = -116.37

url = "https://overpass-api.de/api/interpreter"
query = f"""
[out:json];
(
  node["shop"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["shop"]({lat_min},{lon_min},{lat_max},{lon_max});
  relation["shop"]({lat_min},{lon_min},{lat_max},{lon_max});
  
  node["amenity"="marketplace"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["amenity"="marketplace"]({lat_min},{lon_min},{lat_max},{lon_max});
  relation["amenity"="marketplace"]({lat_min},{lon_min},{lat_max},{lon_max});
  
  node["building"="retail"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["building"="retail"]({lat_min},{lon_min},{lat_max},{lon_max});
  relation["building"="retail"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out center;
"""

# request
response = requests.get(url, params={"data": query})

# check status code, save data to json, create html
if response.status_code == 200:
    data = response.json()
    print(f"{len(data['elements'])} results.")

    # # save to json
    # with open("shop_sd.json", "w") as file:
    #     json.dump(data, file, indent=2)
    #     print("json saved to shop_sd.json")
else:
    # error handling
    print("Error:", response.status_code)
    print(f"Error: {response.status_code}")

# get nodes from dataset
elements = data["elements"]
df_raw = pd.json_normalize(
    elements,
    sep="_",
    record_path=None,
)

# select wanted cols for nodes
df_nodes = df_raw[df_raw['type']=='node'][["id", "type", "lat", "lon", "tags_name", "tags_shop", "tags_addr:postcode"]]
# rename cols
df_nodes.columns = ["id", "geo_type", "latitude", "longitude", "name", "shop_type", "postal_code"]
print(len(df_nodes))

# select wanted cols for ways
df_ways = df_raw[df_raw['type']=='way'][["id", "type", "center_lat", "center_lon", "tags_name", "tags_shop", "tags_addr:postcode", "nodes"]]
# rename cols
df_ways.columns = ["id", "geo_type", "latitude", "longitude", "name", "shop_type", "postal_code", "nodes"]
print(len(df_ways))

# concat nodes and ways dfs
df = pd.concat([df_nodes, df_ways], ignore_index=True)

# Truncate postal codes to 5 digits
df['postal_code'] = df['postal_code'].str.extract(r'(\d{5})')

# filter df so that it only has zip codes in unique_zip
unique_zip = [str(z) for z in unique_zip]
df['postal_code'] = df['postal_code'].astype(str)
df = df[df['postal_code'].isin(unique_zip)]

# Get missing zip codes for rows in df

shapefile = "shapefiles/sdge_zcta.shp"
gdf = gpd.read_file(shapefile)

if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4269")

gdf = gdf.to_crs(epsg=4326)

geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Perform spatial join to find matching ZCTAs
result = gpd.sjoin(geo_df, gdf, how='left', predicate='within')

# Update postal_code column with ZCTA values for missing rows
df['postal_code'] = result['ZCTA5CE20'].combine_first(df['postal_code'])

# # save to csv
# df.to_csv("shop_sd.csv", index=False)
# print("combined df saved to shop_sd.csv")
print(df.head())
print(len(df))

# create Folium map
sd_map = folium.Map(location=[32.7157, -117.1611], zoom_start=12)
# add elements
for _, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    name = row['name']
    shop_type = row['shop_type']
    # add marker
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,  # Size of the dot
        color="blue",  # Outline color
        fill=True,  # Fill the circle
        fill_color="blue",  # Fill color
        fill_opacity=0.7,  # Opacity of the fill
        popup=f"{name} ({shop_type})"  # Popup text
    ).add_to(sd_map)

for _, row in sdge.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color="orange",
        fill=True,
        fill_color="orange",
        fill_opacity=0.7,
        popup=f"Charger ID: {row['id']}; location: {row['latitude'], row['longitude']}"
    ).add_to(sd_map)
# save to html
sd_map.save("data/shop_and_evcharger.html")
print("map saved to shop_and_evcharger.html")

# basic information
print("Basic Info:")
print(df.info())

# check missing values
print("Missing Values Count:")
print(df.isnull().sum())

# plot missing data
plt.figure(figsize=(6,3))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# distribution of some of the shop types
plt.figure(figsize=(6,3))
sns.countplot(data=df, x='shop_type', order=df['shop_type'].value_counts().index[:20])
plt.title('Top 20 Shop Type Distribution')
plt.xticks(rotation=90)
plt.show()

# Distribution of Postal Codes (if available)
plt.figure(figsize=(6,3))
df['postal_code'].value_counts().head(20).plot(kind='bar')
plt.title('Top 20 Postal Codes Distribution')
plt.xlabel('Postal Code')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate distance to nearest charging station
df['nearest_charger_distance'] = df.apply(
    lambda row: min(haversine(row['latitude'], row['longitude'], cs['latitude'], cs['longitude']) for _, cs in sdge.iterrows()),
    axis=1
)

# Calculate density of charging stations within 1 km
df['charger_density'] = df.apply(
    lambda row: sum(haversine(row['latitude'], row['longitude'], cs['latitude'], cs['longitude']) <= 1 for _, cs in sdge.iterrows()),
    axis=1
)

# One-hot encode shop_type
df = pd.get_dummies(df, columns=['shop_type'], prefix='shop')

# Drop non-numeric columns before correlation analysis
df_numeric = df.drop(columns=['id', 'nodes', 'name', 'geo_type'], errors='ignore')

# Correlation analysis
correlation_matrix = df_numeric.corr()
print(correlation_matrix['nearest_charger_distance'])

coords = df[['latitude', 'longitude']]

# Perform K-Means clustering
kmeans = KMeans(n_clusters=250, random_state=42)
df['cluster'] = kmeans.fit_predict(coords)

# Calculate convex hull area for each cluster
def calculate_convex_hull_area(cluster_points):
    if len(cluster_points) >= 3:
        hull = ConvexHull(cluster_points)
        polygon = Polygon(cluster_points[hull.vertices])
        return polygon.area
    else:
        return 0

# Create a dictionary to store cluster info
cluster_info = {}

for cluster in df['cluster'].unique():
    cluster_points = df[df['cluster'] == cluster][['latitude', 'longitude']].values
    area = calculate_convex_hull_area(cluster_points)
    shop_count = len(cluster_points)
    cluster_info[cluster] = {'area': area, 'shop_count': shop_count}

# Convert cluster_info to a DataFrame for easier sorting
cluster_df = pd.DataFrame.from_dict(cluster_info, orient='index')

# Rank clusters: first by shop count (descending), then by area (ascending)
cluster_df['rank'] = cluster_df.apply(lambda x: (-x['shop_count'], x['area']), axis=1)
cluster_df = cluster_df.sort_values(by='rank').reset_index()

# Select the top 100 clusters
top_100_clusters = cluster_df.head(100)['index'].tolist()

# Label shops in the top 100 clusters
df['in_top_100'] = df['cluster'].apply(lambda x: x in top_100_clusters)

# Create a base map
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)

# Add markers to the map
for idx, row in df.iterrows():
    if row['in_top_100']:
        color = 'black'
    else:
        color = 'blue'
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(m)

# Add boundary lines for the top 100 clusters
for cluster in top_100_clusters:
    cluster_points = df[df['cluster'] == cluster][['latitude', 'longitude']].values
    if len(cluster_points) >= 3:
        hull = ConvexHull(cluster_points)
        boundary = cluster_points[hull.vertices]
        folium.PolyLine(
            locations=boundary,
            color='red',
            weight=2,
            opacity=0.7
        ).add_to(m)

# Display the map
m


