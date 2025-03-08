import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from shapely.geometry import Point

# Load dataset: Assume we have hydrogen demand data with locations
data = pd.read_csv("hydrogen_demand.csv")  # Columns: latitude, longitude, demand

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

# Clustering using K-Means to identify optimal refueling station locations
num_stations = 10  # Adjust as needed
kmeans = KMeans(n_clusters=num_stations, random_state=42)
data["cluster"] = kmeans.fit_predict(data[["latitude", "longitude"]])

# Extract cluster centers
centers = pd.DataFrame(kmeans.cluster_centers_, columns=["latitude", "longitude"])
centers_gdf = gpd.GeoDataFrame(centers, geometry=gpd.points_from_xy(centers.longitude, centers.latitude))

# Plot results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data.longitude, y=data.latitude, hue=data.cluster, palette="tab10", alpha=0.6)
sns.scatterplot(x=centers.longitude, y=centers.latitude, color="red", marker="X", s=200, label="Proposed Stations")
plt.title("Optimal Hydrogen Refueling Station Locations")
plt.legend()
plt.show()

# Save results
data.to_csv("clustered_hydrogen_demand.csv", index=False)
centers.to_csv("proposed_station_locations.csv", index=False)

print("Optimization complete. Proposed refueling station locations saved.")
