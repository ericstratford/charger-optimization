import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
from folium.plugins import MarkerCluster
import branca.colormap as cm
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import wkt
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import osmnx as ox
import requests


st.set_page_config(layout="wide")
st.title("EV Charger Placement Optimization Tool")


if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'map_data' not in st.session_state:
    st.session_state.map_data = None

@st.cache_data(ttl=3600)
def load_data():

    with st.spinner('Loading base data...'):

        sdge_zips_data = pd.read_excel("data/SDGE Service Territory Zip Code List Q2-2021.xlsx")
        sdge_zips = list(sdge_zips_data['ZIP_CODE'])
        
  
        zoning_data = gpd.read_file("data/zoning_datasd.geojson")
        zoning_data = zoning_data.to_crs(epsg=4326)
        

        place_name = "San Diego, California, USA"
        with st.spinner('Fetching parking data from OpenStreetMap...'):
            parking_lots = ox.geometries_from_place(place_name, tags={"amenity": "parking"})
        
        public_parking = parking_lots[
            (parking_lots.get("parking") == "public") |
            (parking_lots.get("access").isin([None, "yes", "permissive"]))
        ]
        
        public_parking_gdf = gpd.GeoDataFrame(public_parking, geometry="geometry", crs="EPSG:4326")
        parking_cols = ['osmid','geometry','amenity','access','fee','parking','capacity']
        public_parking_gdf = public_parking_gdf.reset_index()[parking_cols]
        
        return sdge_zips, zoning_data, public_parking_gdf

@st.cache_data(ttl=3600)
def load_charging_stations():

    with st.spinner('Loading charging station data...'):
        url = "https://developer.nrel.gov/api/alt-fuel-stations/v1.json"
        params = {
            'format': 'json',
            'api_key': 'Tbqhfv28h6gIIxbaMTByUNg4ByP2vf1E1A3XYgGa',
            'status': 'all',
            'fuel_type': 'ELEC',
            'state': 'CA',
            'country': 'US',
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return None

@st.cache_data(ttl=3600)
def process_charging_stations(data, sdge_zips):

    with st.spinner('Processing charging station data...'):
        cs = pd.DataFrame(data['fuel_stations'])
        cs = cs[cs['zip'] != 'CA']
        cs['zip'] = cs['zip'].astype(int)
        cs = cs[cs['zip'].isin(sdge_zips)]
        cs = cs[cs['access_code'] == "public"]
        cs = cs[[
            'id', 'access_code', 'facility_type', 'latitude', 'longitude', 'zip',
            'ev_connector_types', 'ev_dc_fast_num', 'ev_level1_evse_num',
            'ev_level2_evse_num', 'ev_network'
        ]]
        return cs.reset_index(drop=True)

def calculate_metrics(lot_geometry, chargers_df):

    lot_coords = (lot_geometry.centroid.y, lot_geometry.centroid.x)
    chargers_coords = list(zip(chargers_df['latitude'], chargers_df['longitude']))
    distances = [geodesic(lot_coords, charger).meters for charger in chargers_coords]
    min_distance = min(distances) if distances else float('inf')
    chargers_in_radius = sum(d <= 200 for d in distances)  
    return min_distance, chargers_in_radius

@st.cache_data(ttl=3600)
def prepare_analysis_data(_cs_df, _parking_gdf, _zoning_data):
    with st.spinner('Preparing analysis data...'):
        lots_data = []
        for idx, lot in _parking_gdf.iterrows():
            min_distance, chargers_in_radius = calculate_metrics(lot.geometry, _cs_df)
            lot_data = {
                'osmid': lot.osmid,
                'geometry': lot.geometry,
                'min_distance': min_distance,
                'chargers_in_radius': chargers_in_radius
            }
            lots_data.append(lot_data)
        lots_gdf = gpd.GeoDataFrame(lots_data, crs=_parking_gdf.crs)
        lots_gdf = gpd.sjoin(lots_gdf, _zoning_data, how="left", predicate="within")
        lots_gdf = lots_gdf.drop(columns=['index_right'])
        lots_gdf['distance_percentile'] = lots_gdf['min_distance'].rank(pct=True)
        lots_gdf['chargers_percentile'] = lots_gdf['chargers_in_radius'].rank(pct=True)
        lots_gdf['demand_score'] = (
            lots_gdf['distance_percentile'] * 0.6 +
            (1 - lots_gdf['chargers_percentile']) * 0.4
        )
        return lots_gdf

def create_demand_map(scored_data):
    m = folium.Map(location=[32.7157, -117.1611], zoom_start=11, tiles="cartodbpositron")
    colormap = cm.linear.YlOrRd_09.scale(
        scored_data['demand_score'].min(),
        scored_data['demand_score'].max()
    )
    colormap.caption = "Demand Score"
    colormap.add_to(m)
    for idx, row in scored_data.iterrows():
        color = colormap(row['demand_score'])
        tooltip_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>Parking Lot ID:</b> {row.osmid}<br>
                <b>Demand Score:</b> {row['demand_score']:.2f}<br>
                <b>Distance to Nearest Charger:</b> {row['min_distance']:.0f}m<br>
                <b>Chargers within 200m:</b> {row['chargers_in_radius']}<br>
                <b>Zone:</b> {row.get('zone_name', 'N/A')}
            </div>
        """
        if isinstance(row.geometry, Point):
            folium.Circle(
                location=(row.geometry.y, row.geometry.x),
                radius=20,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(tooltip_html, max_width=300)
            ).add_to(m)
        else:
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style={
                    'fillColor': color,
                    'color': color,
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(tooltip_html, max_width=300)
            ).add_to(m)
    return m

def create_parking_coverage_map(parking_gdf):
    m = folium.Map(location=[32.7157, -117.1611], zoom_start=12)
    for _, row in parking_gdf.iterrows():
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
                row.geometry.__geo_interface__,
                style_function=lambda x: {
                    "color": "blue",
                    "fillColor": "blue",
                    "weight": 1,
                    "fillOpacity": 0.4
                },
                tooltip="Non-Private Parking Lot"
            ).add_to(m)
    return m

def create_zoning_map(parking_gdf, zoning_data):
    m = folium.Map(location=[32.7157, -117.1611], zoom_start=12)
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
    folium.GeoJson(
        zoning_data.__geo_interface__,
        style_function=lambda feature: {
            'fillColor': color_mapping.get(feature['properties'].get('zone_type', 'Other'), 'gray'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.3,
        }
    ).add_to(m)
    legend_html = f"""
    <div style="position: fixed; 
                 top: 10px; left: 10px; 
                 width: 200px; height: auto; 
                 background-color: rgba(255, 255, 255, 0.8); 
                 border:2px solid grey; 
                 z-index: 9999; 
                 font-size:14px;">
        <b>Zoning Categories</b><br>
        <div style="line-height: 1.5;">
    """
    for category, color in color_mapping.items():
        legend_html += f'<div><i style="background:{color}; width: 20px; height: 20px; display: inline-block;"></i> {category}</div>'
    legend_html += "</div></div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    for _, row in parking_gdf.iterrows():
        zone_type = row.get('zone_type', 'Unknown')
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
            ).add_to(m)
        elif row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {
                    "color": "green",
                    "fillColor": "green",
                    "weight": 1,
                    "fillOpacity": 0.4
                },
                tooltip=f"Zone: {zone_type}"
            ).add_to(m)
    return m

def create_radius_map(scored_data):
    m = folium.Map(location=[32.7157, -117.1611], zoom_start=11, tiles="cartodbpositron")
    for idx, row in scored_data.iterrows():
        centroid = row.geometry.centroid
        folium.CircleMarker(
            location=(centroid.y, centroid.x),
            radius=3,
            color="red",
            fill=True,
            fill_opacity=0.7,
            popup=f"ID: {row.osmid}<br>Distance: {row['min_distance']:.0f}m"
        ).add_to(m)
        folium.Circle(
            location=(centroid.y, centroid.x),
            radius=200, 
            color="blue",
            weight=1,
            opacity=0.3,
            fill=True,
            fill_opacity=0.1
        ).add_to(m)
    return m

def create_population_income_map():
    pop_df = pd.read_csv("data/population_data.csv")
    pop_df['geometry'] = pop_df['geometry'].apply(wkt.loads)
    pop_df = gpd.GeoDataFrame(pop_df, geometry='geometry', crs="EPSG:4326")
    
    income_df = pd.read_csv("data/income_data.csv")
    income_df['geometry'] = income_df['geometry'].apply(wkt.loads)
    income_df = gpd.GeoDataFrame(income_df, geometry='geometry', crs="EPSG:4326")
    
    df = pd.read_csv('data/filtered_alt_fuel_stations.csv')
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
            'fillColor': (
                '#d73027' if feature['properties']['Population'] > 10000 else
                '#fc8d59' if feature['properties']['Population'] > 7500 else
                '#fee08b' if feature['properties']['Population'] > 5000 else
                '#d9ef8b' if feature['properties']['Population'] > 3000 else
                '#91cf60'
            ),
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
            'fillColor': (
                '#d3d3d3' if feature['properties']['Median Income'] is None else  
                '#313695' if feature['properties']['Median Income'] > 100000 else
                '#4575b4' if feature['properties']['Median Income'] > 75000 else
                '#74add1' if feature['properties']['Median Income'] > 50000 else
                '#abd9e9' if feature['properties']['Median Income'] > 30000 else
                '#e0f3f8'
            ),
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
    
    ev_layer = folium.FeatureGroup(name="EV Charger Station")
    marker_cluster = MarkerCluster().add_to(ev_layer)
    for idx, row in gdf.iterrows():
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
    ev_layer.add_to(m)
    
    folium.LayerControl(collapsed=False, position='topleft').add_to(m)
    return m

def main():
    try:
        sdge_zips, zoning_data, public_parking_gdf = load_data()
        charging_data = load_charging_stations()
        
        if charging_data is None:
            st.error("Failed to load charging station data")
            return
            
        if st.session_state.processed_data is None:
            cs_df = process_charging_stations(charging_data, sdge_zips)
            st.session_state.processed_data = prepare_analysis_data(
                cs_df, public_parking_gdf, zoning_data
            )
        
        st.sidebar.header("Visualization Controls")
        map_type = st.sidebar.selectbox(
            "Select Map Type",
            [
                "Demand Score Map", 
                "Charger Radius Map", 
                "Parking Coverage Map", 
                "Zoning Overview Map",
                "Population & Income Map"
            ]
        )
        
        st.markdown("""
            <style>
            .stDataFrame {
                width: 100% !important;
                max-width: 100% !important;
            }
            .row-widget.stDataFrame > div {
                width: 100% !important;
                max-width: 100% !important;
            }
            </style>
        """, unsafe_allow_html=True)
        

        col1, col2, col3 = st.columns([5, 0.2, 2])
        with col1:
            st.header("Map Visualization")
            if map_type == "Population & Income Map":
                map_obj = create_population_income_map()
                st.write("This map displays population and median income distribution along with EV charging stations.")
            elif map_type == "Demand Score Map":
                map_obj = create_demand_map(st.session_state.processed_data)
                st.write("This map shows the demand score for potential charger locations.")
            elif map_type == "Charger Radius Map":
                map_obj = create_radius_map(st.session_state.processed_data)
                st.write("This map shows the coverage radius of existing chargers.")
            elif map_type == "Parking Coverage Map":
                map_obj = create_parking_coverage_map(public_parking_gdf)
                st.write("This map shows the distribution of public parking lots in San Diego.")
            elif map_type == "Zoning Overview Map":
                map_obj = create_zoning_map(public_parking_gdf, zoning_data)
                st.write("This map shows the zoning overlay with public parking locations.")
            
            folium_static(map_obj, width=None)
        
        with col2:
            st.empty()
        
        with col3:
            st.header("Analysis")
            st.subheader("Top 10 Priority Locations")
            top_locations = st.session_state.processed_data.nlargest(10, 'demand_score')[
                ['osmid', 'zone_name', 'demand_score', 'min_distance', 'chargers_in_radius']
            ].rename(columns={
                'min_distance': 'Distance (m)',
                'chargers_in_radius': 'Nearby Chargers',
                'demand_score': 'Priority Score'
            })
            st.dataframe(
                top_locations.style.format({
                    'Distance (m)': '{:.0f}',
                    'Priority Score': '{:.2f}'
                }),
                height=300,
                use_container_width=True
            )
            
            st.subheader("Zone Type Distribution")
            if 'zone_type' in st.session_state.processed_data.columns:
                zone_dist = st.session_state.processed_data['zone_type'].value_counts()
                st.bar_chart(zone_dist)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
