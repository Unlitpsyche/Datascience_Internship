import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

# Step 1: Load the dataset
df = pd.read_csv(r"D:\studies\Programs\LabAssignment_Day3\LabAssignment_Day3\Dataset_Day3.csv")

# Step 2: Define Haversine formula to compute distance between two lat-long points
def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r

# Step 3: Add new column "distance"
df['distance'] = df.apply(lambda row: haversine(
    row['pickup_latitude'], row['pickup_longitude'],
    row['dropoff_latitude'], row['dropoff_longitude']
), axis=1)

# Show sample of distance values (Answer to Q1)
print("Q1: Sample Distances Calculated (First 5 rows)")
print(df[['key', 'pickup_latitude', 'pickup_longitude', 
          'dropoff_latitude', 'dropoff_longitude', 'distance']].head())
print("\n")

# Step 4: Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Step 5: Remove outliers from fare_amount, passenger_count, and distance
df_clean = remove_outliers(df, 'fare_amount')
df_clean = remove_outliers(df_clean, 'passenger_count')
df_clean = remove_outliers(df_clean, 'distance')

# Step 6: Find removed keys (Answer to Q2)
removed_keys = list(set(df['key']) - set(df_clean['key']))

print("Q2: Sample Outlier Keys Removed (First 10 shown)")
print(removed_keys[:10])  # Only show first 10 outlier keys
print(f"\nTotal outlier rows removed: {len(removed_keys)}\n")

# Step 7: Show scatterplot (Answer to Q3)
print("Q3: Scatterplot of Distance vs Fare Amount")

plt.figure(figsize=(10, 6))
plt.scatter(df_clean['distance'], df_clean['fare_amount'], alpha=0.3, color='teal')
plt.title('Distance vs Fare Amount')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()

