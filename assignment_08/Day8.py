import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import zscore

# Load the dataset
try:
    df = pd.read_csv(r"D:\studies\Programs\LabAssignment_Day8\LabAssignment_Day8\Dataset_Day8.csv")
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print("Error: 'Dataset_Day8.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- Problem 1: Data Preprocessing ---

print("--- Data Preprocessing ---")

# Replace 0 values in specified columns with NaN
cols_to_impute = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']
for col in cols_to_impute:
    df[col] = df[col].replace(0, np.nan)
print(f"Replaced 0s with NaN in {cols_to_impute}.\n")

# Impute missing values with the median for numerical columns
# For 'Outcome', it's a target variable, no imputation needed for 0s as per problem description.
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ['int64', 'float64']:
            median_val = df[col].median()
            df.fillna({col:median_val}, inplace=True)
            print(f"Imputed missing values in '{col}' with median: {median_val}")
        else:
            # For non-numeric columns (if any, though not expected in this dataset after 0 replacement)
            mode_val = df[col].mode()[0]
            df.fillna({col:median_val}, inplace=True)
            print(f"Imputed missing values in '{col}' with mode: {mode_val}")
print("\nMissing values imputation complete.\n")

# Outlier removal using Z-score (similar to ClassificationP2, assuming a threshold of 3)
# Identify numerical columns for outlier detection (excluding 'Outcome' as it's the target)
# Corrected numerical_cols based on 'Dataset_Day8.csv' content from Assignment Problems_Day8.docx
numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Calculate Z-scores for each numerical column
for col in numerical_cols:
    df[f'{col}_zscore'] = np.abs(zscore(df[col]))

# Filter out rows where any Z-score is greater than 3
outlier_threshold = 3
original_rows = len(df)
df_outlier_free = df[~(df[[f'{col}_zscore' for col in numerical_cols]] > outlier_threshold).any(axis=1)].copy()
removed_rows = original_rows - len(df_outlier_free)
print(f"Removed {removed_rows} outlier rows (approximately {removed_rows/original_rows*100:.2f}% of data) using Z-score > {outlier_threshold} criterion.\n")

# Drop the z-score columns
df_outlier_free = df_outlier_free.drop(columns=[f'{col}_zscore' for col in numerical_cols])

# Display info of the cleaned dataset
print("Info of the final preprocessed DataFrame:")
df_outlier_free.info()
print("\nFirst 5 rows of the preprocessed DataFrame:")
print(df_outlier_free.head())

# --- Data Splitting ---

print("\n--- Data Splitting ---")
X = df_outlier_free.drop('Outcome', axis=1) # Features
y = df_outlier_free['Outcome'] # Target variable

# Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
print(f"Data split into 70% training ({len(X_train)} samples) and 30% testing ({len(X_test)} samples).")
print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Testing set shape: X_test {X_test.shape}, y_test {y_test.shape}\n")

# --- Problem 2: k-Nearest Neighbor Algorithm with fixed distance metric and plotting Precision/Recall vs k ---

print("--- k-NN Algorithm: Default Performance & P-R Curve ---")

# Default k value can be sqrt(len(X_train)) or a common choice like 5
# Here, let's use k=5 as a default starting point as the prompt asks for "default model performance metrics"
default_k = 5
fixed_distance_metric = 'euclidean' # Using Euclidean distance as fixed for plotting

knn_default = KNeighborsClassifier(n_neighbors=default_k, metric=fixed_distance_metric)
knn_default.fit(X_train, y_train)
y_pred_default = knn_default.predict(X_test)

# Print default model performance metrics
accuracy = accuracy_score(y_test, y_pred_default)
precision = precision_score(y_test, y_pred_default, zero_division=0) # zero_division to handle cases with no positive predictions
recall = recall_score(y_test, y_pred_default, zero_division=0)
f1 = f1_score(y_test, y_pred_default, zero_division=0)

print(f"Default k-NN Model Performance (k={default_k}, metric='{fixed_distance_metric}'):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}\n")

print("Insight: The default k-NN model provides a good starting point. The F1-score balances precision and recall, indicating a reasonable overall performance for predicting diabetes.")

# Plot Precision & Recall vs k curve
k_range = range(1, int(np.sqrt(len(X_train))) + 10) # Test k values up to sqrt(N) + 10 for a broader view
precision_scores = []
recall_scores = []
f1_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric=fixed_distance_metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

plt.figure(figsize=(10, 6))
plt.plot(k_range, precision_scores, label='Precision', marker='o', linestyle='-')
plt.plot(k_range, recall_scores, label='Recall', marker='x', linestyle='--')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title(f'Precision & Recall vs. k (Metric: {fixed_distance_metric})')
plt.xticks(k_range[::2]) # Show every second k value for better readability
plt.grid(True)
plt.legend()
plt.show()

# Find the k for which F1-score is the highest
best_k_f1 = k_range[np.argmax(f1_scores)]
max_f1_score = np.max(f1_scores)
print(f"The k value with the highest F1-score ({max_f1_score:.4f}) is: {best_k_f1}\n")
print("Insight: The Precision-Recall curve helps visualize the trade-off between these two metrics as 'k' changes. The optimal 'k' for the highest F1-score indicates the best balance between correctly identifying positive cases and avoiding false positives.")


# --- Problem 3: Find the best distance metric and number of neighbors combination ---

print("--- Finding Best Distance Metric and k Combination ---")

distance_metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev'] # Common distance metrics

best_f1_overall = -1
best_k_overall = -1
best_metric_overall = ''
all_results = []

for metric in distance_metrics:
    print(f"Evaluating with metric: {metric}")
    for k in k_range: # Re-evaluate over the same range of k values
        try:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)

            all_results.append({
                'k': k,
                'metric': metric,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

            if f1 > best_f1_overall:
                best_f1_overall = f1
                best_k_overall = k
                best_metric_overall = metric
        except ValueError as e:
            print(f"Skipping k={k} with metric={metric} due to error: {e}")
            continue

results_df = pd.DataFrame(all_results)
print("\nAll combinations performance (first 5 rows):")
print(results_df.head())

print(f"\nBest k-NN Configuration:")
print(f"Best Distance Metric: {best_metric_overall}")
print(f"Best Number of Neighbors (k): {best_k_overall}")
print(f"Highest F1-Score Achieved: {best_f1_overall:.4f}\n")

# Retrieve and print performance metrics for the best combination
best_config_row = results_df[(results_df['k'] == best_k_overall) & (results_df['metric'] == best_metric_overall)].iloc[0]
print(f"Performance metrics for the best combination (k={best_k_overall}, metric='{best_metric_overall}'):")
print(f"Accuracy: {best_config_row['accuracy']:.4f}")
print(f"Precision: {best_config_row['precision']:.4f}")
print(f"Recall: {best_config_row['recall']:.4f}")
print(f"F1-Score: {best_config_row['f1_score']:.4f}\n")

print("Insight: By iterating through different distance metrics and k values, we identified the combination that yields the highest F1-score. This comprehensive search helps in optimizing the k-NN model's performance by finding the most suitable parameters for the given dataset and problem, offering a more robust classification model.")

