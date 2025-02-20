# EVCS - OPTIM
Electric Vehicle Charging Stations - Optimizing Placement Tool via Infrastructure Modeling

## Overview
The EVCS-OPTIM project provides an interactive tool designed to recommend optimal locations for new electric vehicle (EV) charging stations. By integrating multiple datasets and employing geospatial, temporal, and optimization analyses, our tool assists urban planners, utility companies, and community advocates in making data-driven decisions for infrastructure deployment. The current model evaluates public parking lots, scoring each based on factors such as proximity to key infrastructures, traffic density, population demographics, and grid constraints before ranking them according to user-defined criteria.


### Reproducing
#### CS_Lots

** Downloads **
To re-run the code provided in CS_Lots, one needs to download the data, shapefiles, and python notebook provided in the main repository.

** APIs **
We use an API to query data on public electric vehicle charging station locations from the Alternative Fuels Data Center via this [API setup](https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/). To query from the Python file, an API key is needed, which can be obtained by signing up through the aforementioned link. Simply replace the "***" in the data query cell with the newly obtained API key in order to properly query the charging station data later used.

** Libraries/Requirements **
- pandas
- numpy
- requests
- geopandas
- matplotlib
- folium
- osmnx
- geopy
- shapely
- scikit-learn
- branca
