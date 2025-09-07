import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set a random state for reproducibility as specified in the assignment
RANDOM_STATE = 50

# Load the dataset
try:
    df = pd.read_csv(r'D:\studies\Programs\LabAssignment_Day10\LabAssignment_Day10\Dataset_Day10.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Dataset_Day10.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# --- Problem 1: Replace all Missing values with relevant figures ---
# As per the assignment, 0 values in specific columns are considered missing.
# Columns to treat 0 as missing: Glucose, BloodPressure, BMI, DiabetesPedigreeFunction
# It's generally better to replace missing numerical data with the median to avoid skewing distributions
# caused by outliers, or the mean if the data is normally distributed and no outliers are present.
# For this dataset, median is a safer choice given potential skewness.

# Columns where 0 should be treated as missing based on Assignment Problems_Day10.docx
columns_with_zero_as_missing = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']

# Replace 0s with NaN for easier handling of missing values
for col in columns_with_zero_as_missing:
    df[col] = df[col].replace(0, np.nan)
    print(f"Replaced 0s with NaN in column: {col}")

# Impute missing values (NaNs) with the median of their respective columns
# 'Pregnancies', 'Age', and 'SkinThickness' (if present and not already handled)
# 'SkinThickness' was causing KeyError, implying it's not in the dataset or handled differently.
# Based on the problem description, only Glucose, BloodPressure, BMI, DiabetesPedigreeFunction
# are explicitly mentioned for 0-replacement and likely imputation.
numerical_cols_to_impute_median = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']

for col in numerical_cols_to_impute_median:
    median_val = df[col].median()
    df.fillna({col: median_val}, inplace=True)
    print(f"Imputed NaN values in column '{col}' with median: {median_val}")

# --- Problem 2: Remove all existing outliers and get the final data for classification ---
# Outlier removal using IQR method as seen in similar codebooks.
# Assuming numerical columns for outlier detection (all except 'Outcome')
Q1 = df.drop(columns=['Outcome']).quantile(0.25)
Q3 = df.drop(columns=['Outcome']).quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create a boolean mask for outliers
# For each row, check if any feature is outside its IQR bounds
outlier_mask = ((df.drop(columns=['Outcome']) < lower_bound) | (df.drop(columns=['Outcome']) > upper_bound)).any(axis=1)

# Remove outliers
df_cleaned = df[~outlier_mask].copy() # Using .copy() to avoid SettingWithCopyWarning
print(f"\nOriginal data shape: {df.shape}")
print(f"Data shape after outlier removal: {df_cleaned.shape}")

# Separate features (X) and target (y)
X = df_cleaned.drop('Outcome', axis=1)
y = df_cleaned['Outcome']

# --- Problem 3: Split the data into 80% training and 20% testing data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- Problem 4: Use a Decision Tree classifier algorithm with target variable as ‘Outcome’ ---
# --- Problem 5: Print the default model performance metrics: Accuracy, Precision, Recall, F1Score ---

# Initialize the Decision Tree Classifier with 'entropy' criterion and specified random_state
# The DT_Codebook mentioned 'entropy' as criterion for the final model.
# For default metrics, we will use a base DecisionTreeClassifier first.
dt_classifier_default = DecisionTreeClassifier(random_state=RANDOM_STATE)

# Fit the model to the training data
dt_classifier_default.fit(X_train, y_train)

# Make predictions on the test data
y_pred_default = dt_classifier_default.predict(X_test)

# Calculate and print performance metrics
print("\n--- Default Decision Tree Model Performance Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_default):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_default):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_default):.4f}")

# --- Problem 6: Plot a Precision & Recall vs max_leaf_nodes & max_depth curve ---
# (Consider a range of numbers for both parameters)
# (both Prec and Rec on the same graph).
# Find the parameter values for which F1-score is the highest. (Use ‘entropy’ as criterion)

# Define ranges for max_leaf_nodes and max_depth
max_leaf_nodes_range = range(2, 50, 2)  # From 2 to 48, step by 2
max_depth_range = range(1, 20)           # From 1 to 19

# Store metrics for plotting
precision_scores_leaf = []
recall_scores_leaf = []
f1_scores_leaf = []

precision_scores_depth = []
recall_scores_depth = []
f1_scores_depth = []

best_f1_score = -1
best_params = {'max_leaf_nodes': None, 'max_depth': None}

print("\n--- Hyperparameter Tuning (max_leaf_nodes) ---")
for nodes in max_leaf_nodes_range:
    dt = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=nodes, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    precision_scores_leaf.append(prec)
    recall_scores_leaf.append(rec)
    f1_scores_leaf.append(f1)

    if f1 > best_f1_score:
        best_f1_score = f1
        best_params['max_leaf_nodes'] = nodes

print("--- Hyperparameter Tuning (max_depth) ---")
for depth in max_depth_range:
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    precision_scores_depth.append(prec)
    recall_scores_depth.append(rec)
    f1_scores_depth.append(f1)

    if f1 > best_f1_score:
        best_f1_score = f1
        best_params['max_depth'] = depth


# Plotting Precision & Recall vs max_leaf_nodes
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(max_leaf_nodes_range, precision_scores_leaf, label='Precision')
plt.plot(max_leaf_nodes_range, recall_scores_leaf, label='Recall')
plt.plot(max_leaf_nodes_range, f1_scores_leaf, label='F1-Score', linestyle='--', color='green')
plt.xlabel('max_leaf_nodes')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score vs. max_leaf_nodes (Criterion: Entropy)')
plt.legend()
plt.grid(True)

# Plotting Precision & Recall vs max_depth
plt.subplot(1, 2, 2)
plt.plot(max_depth_range, precision_scores_depth, label='Precision')
plt.plot(max_depth_range, recall_scores_depth, label='Recall')
plt.plot(max_depth_range, f1_scores_depth, label='F1-Score', linestyle='--', color='green')
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score vs. max_depth (Criterion: Entropy)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nParameter values for which F1-score is the highest (from individual tuning):")
print(f"Optimal parameter found (either max_leaf_nodes or max_depth): {best_params}")
print(f"Highest F1-Score achieved: {best_f1_score:.4f}")


