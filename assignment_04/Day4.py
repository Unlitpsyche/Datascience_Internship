import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'D:\studies\Programs\LabAssignment_Day4\LabAssignment_Day4\Dataset_Day4.csv')

# List of numeric columns (excluding 'Prefix')
numeric_cols = ['Assignment', 'Tutorial', 'Midterm', 'TakeHome', 'Final']

# 1. Show missing values before treatment
print("Missing values before treatment:")
print(df.isnull().sum())

# 2. Impute missing values with mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 3. Show missing values after treatment
print("\nMissing values after treatment:")
print(df.isnull().sum())

# 4. Descriptive statistics
print("\nDescriptive statistics:")
print(df[numeric_cols].describe())

# 5. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# 6. Pairplot
sns.pairplot(df[numeric_cols])
plt.suptitle('Pairplot of Dataset Variables', y=1.02)
plt.show()

# 7. Histograms for each numeric variable
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# 8. Scatterplots: Show relationship of each feature with Final marks
for col in ['Assignment', 'Tutorial', 'Midterm', 'TakeHome']:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df['Final'])
    plt.title(f'{col} vs Final')
    plt.xlabel(col)
    plt.ylabel('Final')
    plt.show()

