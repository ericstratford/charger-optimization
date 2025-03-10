# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# %%
# Reading in the traffic dataset
traffic_df = pd.read_csv('data/traffic_counts_datasd.csv', usecols = ['id', 'street_name', 'limits', 'total_count', 'date_count'])
traffic_df = traffic_df[['street_name', 'total_count']].groupby('street_name').mean()
traffic_df = traffic_df.reset_index()
traffic_df

# %%
# Reading in the roads dataset
roads_df = gpd.read_file('data/roads_datasd.geojson')
roads_df

# %%
roads_df = roads_df[['objectid', 'rd30full', 'speed', 'postdate', 'geometry']]

# %% [markdown]
# # Data Cleaning

# %%
roads_df_sorted = roads_df.sort_values(by=['rd30full', 'postdate'], ascending=[True, False])
roads_df_sorted = roads_df_sorted.drop_duplicates(subset='rd30full', keep='first')

# %%
# Checking which street name abreviations don't match
traf_suff = traffic_df['street_name'].str.split().str[-1].unique()
suff = roads_df_sorted['rd30full'].str.split().str[-1].unique()

for x in traf_suff:
    if x not in suff:
        print(x)

# %%
traffic_df[traffic_df['street_name'].str.split().str[-1] == ('(SB)')]

# %%
roads_df_sorted[roads_df_sorted['rd30full'].str.contains('8')]

# %%
def ending_street_names(name):
    name_spt = name.split()
    if name_spt[-1] == 'AV':
        name_spt[-1] = 'AVE'
    elif name_spt[-1] == 'WY':
        name_spt[-1] = 'WAY'
    elif name_spt[-1] == 'BL':
        name_spt[-1] = 'BLVD'
    elif (name_spt[-1] == 'CIRCLE') or (name_spt[-1] == 'CR'):
        name_spt[:-1] = 'CIR'
    elif name_spt[-1] == 'PY':
        name_spt[:-1] = 'PKY'
    elif name_spt[-1] == 'ML':
        name_spt[-1] = 'MALL'
    elif name_spt[-1] == 'RL':
        return  'EL CAMINO REAL'
    elif (name_spt[-1] == 'TL') or (name_spt[-1] == 'TRAIL'):
        name_spt[-1] = 'TRL'
    elif name_spt[-1] == 'GL':
        name_spt[-1] = 'GLEN'
    elif name_spt[-1] == 'TR':
        name_spt[-1] = 'TER'
    elif name_spt[-1] == 'HY':
        name_spt[-1] = 'HWY'
    return ' '.join(name_spt)

# %%
def num_street_names(name):
    st_name = name
    name_spt = name.split()
    if name_spt[0][:2].isnumeric():
        name_spt[0] = name_spt[0][:2]
    return ' '.join(name_spt)

# %%
# Changing the street names so that they both match
traffic_df['street_name'] = traffic_df['street_name'].apply(ending_street_names)
roads_df_sorted['rd30full'] = roads_df_sorted['rd30full'].apply(num_street_names)

# %%
# Merging the traffic and road datasets
merged_df = roads_df_sorted.merge(traffic_df, left_on = 'rd30full', right_on = 'street_name')
merged_df

# %% [markdown]
# # Plots

# %%
merged_df.plot(column='total_count', cmap = 'Reds', legend=True)


# %%
#San Diego cities
geo_df5 = gpd.read_file('data/zip_codes.geojson')
geo_df5['name'] = geo_df5['community']
geo_df5 = geo_df5[['name', 'geometry']]

sdge_df = geo_df5 
sdge_df['name'] = sdge_df['name'].str.strip()

# %%
#Converting the CSV files into dataframes
col_index = [i for i in range(29)] + [i for i in range(41, 50)] + [70, 71, 72]
df = pd.read_csv('data/ev_data.csv', usecols=[i for i in col_index], dtype={'zip': str, 'ev_network_web': str, 'ev_renewable_source': str, 'ev_other_evse': str, 'ev_workplace_charging': str}) 
# sdge_areas = pd.read_csv('SDGE_service_list.csv', usecols=['ZipCode'])

#Getting the zip codes for all the areas at SDGE serves
sdge_zip_codes = [str(element) for element in sdge_df['name'].unique()]

#Querying the data to only include chargers within the areas that SDGE serves
df = df[df['city'].isin(sdge_zip_codes)]

# %%
#Creating new columns 

#Cleaning the data to get rid of any NaN values in the open_date column
time_df = df.dropna(subset=['open_date'])

#Making the open_date column into a datetime object
time_df['open_date'] = pd.to_datetime(time_df['open_date'])

#Creating a dataframe for the number of EV chargers opened every year for every city
time_graph = time_df
time_graph['open_year'] = df['open_date'].str[:4]
time_graph['open_year'] = time_graph['open_year'].dropna()
time_graph['open_year'] = time_graph['open_year'].astype(int)
time_graph = time_graph.groupby(['open_year', 'city']).count()
time_graph = time_graph.reset_index()
time_graph = time_graph[['open_year', 'city', 'station_name']]
time_graph['station_name'] = time_graph['station_name'].astype(float)

#Array of all of the names of the cities that SDGE serves
city_names = time_graph['city'].unique()
city_names

# %%
#Creating a map of how close EV chargers are to traffic areas

sub_merged_df = merged_df[merged_df['total_count'] >10000]
#Data for every transit stop in San Diego county

ax = sdge_df.plot(figsize=(10, 8))

# Plot the second GeoDataFrame on top
gdf_test = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
ax1 = gdf_test.plot(ax=ax, color = 'yellow', markersize = 5, alpha = .3)
sub_merged_df.plot(ax=ax1, color = 'red', markersize = 20)

# sub_merged_df.plot(ax=ax1, column='total_count', cmap = 'Reds', legend=True, markersize = 10)

plt.title('Traffic Areas (Red) with EV Charger Locations (Yellow)');
plt.axis('off');

# %%
transit = gpd.read_file('data/transit_stops.geojson')

# %%
transit_centers = transit[transit['stop_name'].str.contains('Transit Center')]

# %%
ax = sdge_df.plot(figsize=(10, 8))

# Plot the second GeoDataFrame on top
gdf_test = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
ax1 = gdf_test.plot(ax=ax, color = 'yellow', markersize = 5, alpha = 0.3)
transit_centers.plot(ax=ax1, color = 'red', legend=True, markersize = 1)

plt.title('Transit Centers (Red) with EV Charger Locations (Yellow)');
plt.axis('off');

# %% [markdown]
# # Proportion of EV Chargers

# %%
high_traffic_areas = merged_df[merged_df['total_count'] > 10000]
high_traffic_areas = high_traffic_areas.to_crs(epsg=3395)  # UTM projection (meters)

# Create a 1-mile buffer around the high-traffic areas (1 mile = 1609.34 meters)
high_traffic_buffer = high_traffic_areas.buffer(1609.34)

ev_chargers = gdf_test.set_crs('EPSG:4326', allow_override=True)

ev_chargers = ev_chargers.to_crs(high_traffic_areas.crs)

# Spatial join to find EV chargers within high-traffic area buffers
ev_chargers_within_buffer = ev_chargers[ev_chargers.geometry.within(high_traffic_buffer.unary_union)]

ev_chargers_within_buffer_unique = ev_chargers_within_buffer.drop_duplicates(subset='station_name') 

# Calculate the proportion
total_ev_chargers = len(ev_chargers)
ev_chargers_within = len(ev_chargers_within_buffer_unique)
proportion = ev_chargers_within / total_ev_chargers

# Output the proportion
print(f"Proportion of EV chargers within 1 mile of a high-traffic area: {proportion:.2f}")

# %%
transit_stops_projected = transit.to_crs(epsg=26911)  

# Reproject the EV chargers to the same CRS 
ev_chargers_projected = ev_chargers.to_crs(epsg=26911)

# Create a 1-mile buffer (1 mile = 1609.34 meters)
transit_buffers = transit_stops_projected.copy()
transit_buffers['geometry'] = transit_buffers.geometry.buffer(1609.34)

ev_chargers_within_buffer = gpd.sjoin(ev_chargers_projected, transit_buffers, how='inner', op='within')

ev_chargers_within_buffer_unique = ev_chargers_within_buffer.drop_duplicates(subset='station_name')  # Replace 'EV_charger_id' with your unique identifier

# Calculating proportions
ev_chargers_within = len(ev_chargers_within_buffer_unique)
proportion = ev_chargers_within / total_ev_chargers
print(f"Proportion of EV chargers within 1 mile of transit stops: {proportion:.2f}")

# %%
transit_centers_projected = transit_centers.to_crs(epsg=26911)  # UTM Zone 11N for San Diego (use your relevant CRS)

# Reproject the EV chargers to the same CRS 
ev_chargers_projected = ev_chargers.to_crs(epsg=26911)

# Create a 1-mile buffer (1 mile = 1609.34 meters)
transit_centers_buffers = transit_centers_projected.copy()
transit_centers_buffers['geometry'] = transit_centers_buffers.geometry.buffer(1609.34)

ev_chargers_within_buffer_centers = gpd.sjoin(ev_chargers_projected, transit_centers_buffers, how='inner', op='within')

ev_chargers_within_buffer_centers_unique = ev_chargers_within_buffer_centers.drop_duplicates(subset='station_name')  # Replace 'EV_charger_id' with your unique identifier

# Calculating proportions
ev_chargers_within = len(ev_chargers_within_buffer_centers_unique)
proportion = ev_chargers_within / total_ev_chargers
print(f"Proportion of EV chargers within 1 mile of transit centers: {proportion:.2f}")

# %%
