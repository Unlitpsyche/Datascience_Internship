import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('D:\studies\Programs\LabAssignment_Day2\LabAssignment_Day2\Dataset_Day2.csv')

# 1. Find all unique values of columns "name", "mfr", and "vitamins" and store them in separate numpy arrays, then print them
unique_names = np.array(df['name'].unique())
unique_mfr = np.array(df['mfr'].unique())
unique_vitamins = np.array(df['vitamins'].unique())

print("Unique cereal names:")
print(unique_names)
print("\nUnique manufacturers:")
print(unique_mfr)
print("\nUnique vitamins values:")
print(unique_vitamins)

# 2. Create a new dataframe with all columns where 'sodium' > 100 AND 'protein' < 3
df_HighSodLowProt = df[(df['sodium'] > 100) & (df['protein'] < 3)]
print("\nDataframe with sodium > 100 and protein < 3:")
print(df_HighSodLowProt)

# 3. From the dataframe in 2, print the average 'calories' by 'mfr' and also print the 'mfr' with the highest average 'calories'
avg_calories_by_mfr = df_HighSodLowProt.groupby('mfr')['calories'].mean()
print("\nAverage calories by manufacturer (mfr):")
print(avg_calories_by_mfr)

mfr_highest_avg_calories = avg_calories_by_mfr.idxmax()
highest_avg_calories = avg_calories_by_mfr.max()
print(f"\nManufacturer with highest average calories: {mfr_highest_avg_calories} ({highest_avg_calories:.2f} calories)")
