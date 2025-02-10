# EVCS - OPTIM
Electric Vehicle Charging Stations - Optimizing Placement Tool via Infrastructure Modeling

## Overview
This project aims to provide an interactive tool which recommends the best locations to construct additional electric vehicle charging stations. We primarily use demand-incentive based data, performing geospatial and temporal analysis in order to inform an optimization tool. For the current iteration of the model, we evaluate public parkings lots and score them based on an optimization model prior to ranking and recommending lots according to user-inputted criteria.

### Reproducing
#### CS_Lots

** Downloads **
To re-run the code provided in CS_Lots, one needs to download the data, shapefiles, and python notebook provided in the main repository. The income.geojson and population.geojson files provided in the income and population branches are also imported in the notebook as pre-ran datafiles.

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
