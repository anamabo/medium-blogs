"""
This script is meant to explore the data used in this project
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import plotly.express as px

housing = fetch_california_housing()

# create a data frame from the input data
# Note: Each value of the target corresponds to the average house value in units of 100,000
features = pd.DataFrame(housing.data, columns=housing.feature_names)
target = pd.Series(housing.target, name=housing.target_names[0])

# plot points on a map to see their geographical dispersion
scatter_map = px.scatter_mapbox(features, lat="Latitude", lon="Longitude", size_max=10,
                                color=target.values,
                                height=850, width=850,
                                )
scatter_map.update_layout(mapbox_style="stamen-terrain",
                          mapbox_zoom=5,
                          mapbox_center_lat=features['Latitude'].mean(),
                          mapbox_center_lon=features['Longitude'].mean())
scatter_map.show()

# Checking the distribution of the target
histogram = px.histogram(target.values, nbins=100,
                         labels={'value': 'house price'})
histogram.show()

# Checking the logarithmic distribution of the target
histogram = px.histogram(np.log1p(target.values), nbins=100,
                         labels={'value': 'house price'})
histogram.show()
