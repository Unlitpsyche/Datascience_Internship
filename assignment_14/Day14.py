import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r'D:\studies\Programs\LabAssignment_Day14\LabAssignment_Day14\Dataset_Day14.csv')

print("Question 1:")

# Check missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Scale numeric columns using StandardScaler
scaler = StandardScaler()
cols_to_scale = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Scale and assign back to df columns (no 'method' function in pandas)
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("\nData after scaling (first 5 rows):")
print(df.head())

# Prepare data for clustering
X = df[cols_to_scale].values

print("\nQuestion 2:")

# DBSCAN with default eps=0.8, min_samples=4
dbscan_default = DBSCAN(eps=0.8, min_samples=4)
dbscan_default.fit(X)

# Assign cluster labels
df['Cluster'] = dbscan_default.labels_

# Show species distribution in each cluster
distribution = df.groupby(['Cluster', 'Species']).size().unstack(fill_value=0)
print("Species distribution per cluster:")
print(distribution)

print("\nQuestion 3:")

# Use nearest neighbor distances to find optimal eps
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
sorted_distances = np.sort(distances[:, 3])  # 4th nearest neighbor distances

# Plot elbow graph for eps estimation
plt.figure(figsize=(8,4))
plt.plot(sorted_distances)
plt.title('Nearest Neighbor distances for eps estimation')
plt.xlabel('Points sorted by distance')
plt.ylabel('4th Nearest Neighbor Distance')
plt.grid(True)
plt.show()

print("Sample of sorted 4th nearest neighbor distances (first 10):")
print(sorted_distances[:10])

print("\nQuestion 4:")

# Choose eps from elbow plot, for example 0.5
eps_opt = 0.5
best_score = -1
best_min_samples = None

for ms in range(2, 11):
    dbscan = DBSCAN(eps=eps_opt, min_samples=ms)
    labels = dbscan.fit_predict(X)
    if len(set(labels)) > 1 and len(set(labels)) < len(X):
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_min_samples = ms

print(f"Optimal min_samples using silhouette score with eps={eps_opt}: {best_min_samples} (score: {best_score:.4f})")

print("\nQuestion 5:")

# Find outliers with optimal parameters
dbscan_opt = DBSCAN(eps=eps_opt, min_samples=best_min_samples)
dbscan_opt.fit(X)
df['Cluster_opt'] = dbscan_opt.labels_

outliers = df[df['Cluster_opt'] == -1]
print(f"Outliers detected (total {len(outliers)}):")
print(outliers[['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']])
