#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import geopandas as gpd
import pandas as pd
from cenpy import products
import folium
import pandas as pd
from folium.plugins import MarkerCluster
import requests
import json


# %%


sd_pop_df = products.ACS(2019).from_county('San Diego, CA', level='tract', variables='B01003_001E')
# oc_pop_df = products.ACS(2019).from_county('Orange County, CA', level='tract', variables='B01003_001E')
# pop_df = pd.concat([sd_pop_df, oc_pop_df])
sd_pop_df.rename(columns={'B01003_001E': 'Population'}, inplace=True)


sd_income_df = products.ACS(2019).from_county('San Diego, CA', level='tract', variables='B06011_001E')
# oc_income_df = products.ACS(2019).from_county('Orange County, CA', level='tract', variables='B06011_001E')
# income_df = pd.concat([sd_income_df, oc_income_df])
sd_income_df.rename(columns={'B06011_001E': 'Median Income'}, inplace=True)


if 'geometry' not in sd_pop_df.columns:
    raise ValueError("Population dataframe does not have a 'geometry' column. Ensure cenpy output includes geometry.")

if 'geometry' not in sd_income_df.columns:
    raise ValueError("Income dataframe does not have a 'geometry' column. Ensure cenpy output includes geometry.")


pop_gdf = gpd.GeoDataFrame(sd_pop_df, geometry=sd_pop_df.geometry)
pop_gdf = pop_gdf.set_crs(epsg=4326, allow_override=True) 

income_gdf = gpd.GeoDataFrame(sd_income_df, geometry=sd_income_df.geometry)
income_gdf = income_gdf.set_crs(epsg=4326, allow_override=True) 

pop_gdf.to_file("population.geojson", driver="GeoJSON")
income_gdf.to_file("income.geojson", driver="GeoJSON")




# %%


pop_gdf = gpd.read_file('population.geojson')


# %%


pop_gdf


# %%


gpd.read_file('income.geojson')


# %%
pop_df.to_csv('population.csv', index=False)


# %%
income_df.to_csv('income.csv', index=False)

# %%
income_df

# %%

# %%


sd_pop_df = products.ACS(2019).from_county('San Diego, CA', level='tract', variables='B01003_001E')
oc_pop_df = products.ACS(2019).from_county('Orange County, CA', level='tract', variables='B01003_001E')


pop_df = pd.concat([sd_pop_df, oc_pop_df])
pop_df.rename(columns={'B01003_001E': 'Population'}, inplace=True)
pop_df = pop_df.to_crs(epsg=4326)  


sd_income_df = products.ACS(2019).from_county('San Diego, CA', level='tract', variables='B06011_001E')
oc_income_df = products.ACS(2019).from_county('Orange County, CA', level='tract', variables='B06011_001E')


income_df = pd.concat([sd_income_df, oc_income_df])
income_df.rename(columns={'B06011_001E': 'Median Income'}, inplace=True)
income_df = income_df.to_crs(epsg=4326)  

df = pd.read_csv('filtered_alt_fuel_stations.csv')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)


m = folium.Map(location=[32.7157, -117.1611], zoom_start=10)


population_layer = folium.FeatureGroup(name='Population Distribution')
folium.GeoJson(
    pop_df,
    style_function=lambda feature: {
        'fillColor': '#d73027' if feature['properties']['Population'] > 10000 else
                     '#fc8d59' if feature['properties']['Population'] > 7500 else
                     '#fee08b' if feature['properties']['Population'] > 5000 else
                     '#d9ef8b' if feature['properties']['Population'] > 3000 else
                     '#91cf60',
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['Population'],
        aliases=['Population:'],
        localize=True
    )
).add_to(population_layer)


income_layer = folium.FeatureGroup(name='Income Distribution')
folium.GeoJson(
    income_df,
    style_function=lambda feature: {
        'fillColor': '#d3d3d3' if feature['properties']['Median Income'] is None else  
                     '#313695' if feature['properties']['Median Income'] > 100000 else
                     '#4575b4' if feature['properties']['Median Income'] > 75000 else
                     '#74add1' if feature['properties']['Median Income'] > 50000 else
                     '#abd9e9' if feature['properties']['Median Income'] > 30000 else
                     '#e0f3f8',
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['Median Income'],
        aliases=['Median Income ($):'],
        localize=True
    )
).add_to(income_layer)


population_layer.add_to(m)
income_layer.add_to(m)


marker_cluster = MarkerCluster().add_to(m)
for _, row in gdf.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(f"""
            <b>Station Name:</b> {row['station_name']}<br>
            <b>Address:</b> {row['street_address']}, {row['city']}, {row['state']}<br>
            <b>ZIP Code:</b> {row['zip']}<br>
            <b>Network:</b> {row['ev_network']}
        """, max_width=300),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)


folium.LayerControl().add_to(m)


m.save("Population_Income_EV_Map.html")
print("Interactive map saved as 'Population_Income_EV_Map.html'.")
m


# %%





# %%

# %%

# %%

# %%
charger-optimization
