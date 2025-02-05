import requests
import json
import folium
import pandas as pd

# overpass API url and query
# A node is a single point in space defined by its latitude and longitude coordinates.
# A way represents linear or area features by connecting nodes together.
# A relation is more complex type that defines relationships between multiple nodes, ways, or other relations.
url = "https://overpass-api.de/api/interpreter"
query = """
[out:json];
(
  node["shop"](32.5557,-117.2549,33.1140,-116.9812);
  way["shop"](32.5557,-117.2549,33.1140,-116.9812);
  relation["shop"](32.5557,-117.2549,33.1140,-116.9812);
  
  node["amenity"="marketplace"](32.5557,-117.2549,33.1140,-116.9812);
  way["amenity"="marketplace"](32.5557,-117.2549,33.1140,-116.9812);
  relation["amenity"="marketplace"](32.5557,-117.2549,33.1140,-116.9812);
  
  node["building"="retail"](32.5557,-117.2549,33.1140,-116.9812);
  way["building"="retail"](32.5557,-117.2549,33.1140,-116.9812);
  relation["building"="retail"](32.5557,-117.2549,33.1140,-116.9812);
);
out center;
"""

# request
response = requests.get(url, params={"data": query})

# check status code, save data to json, create html
if response.status_code == 200:
    data = response.json()
    print(f"{len(data['elements'])} results.")

    # save to json
    with open("shopping_sd.json", "w") as file:
        json.dump(data, file, indent=2)
        print("json saved to shopping_sd.json")
else:
    # error handling
    print("Error:", response.status_code)
    print(f"Error: {response.status_code}")


elements = data["elements"]
df = pd.json_normalize(
    elements,
    sep="_",
    record_path=None,
)
print(df[df["type"]=="way"].head())
# select wanted columns
df = df[["id", "type", "lat", "lon", "tags_name", "tags_shop", "tags_addr:postcode"]]
# rename columns
df.columns = ["id", "geo_type", "latitude", "longitude", "name", "shop_type", "postal_code"]
# save to csv
df.to_csv("shopping_sd.csv", index=False)
print("csv saved to shopping_sd.csv")


# create Folium map
san_diego_map = folium.Map(location=[32.7157, -117.1611], zoom_start=12)
# add elements
for element in data['elements']:
     # coordinates
    if "lat" in element and "lon" in element:  # nodes
        lat = element["lat"]
        lon = element["lon"]
    elif "center" in element:  # ways and relations
        lat = element["center"]["lat"]
        lon = element["center"]["lon"]
    else:
        continue
    # get name / fallback to ID
    name = element.get("tags", {}).get("name", f"ID: {element['id']}")
    shop_type = element.get("tags", {}).get("shop", "Unknown")
    # add marker
    folium.Marker(
        location=[lat, lon],
        popup=f"{name} ({shop_type})",
        icon=folium.Icon(color="blue", icon="shopping-cart", prefix="fa"),
    ).add_to(san_diego_map)
# save to html
san_diego_map.save("shopping_sd.html")
print("map saved to shopping_sd.html")