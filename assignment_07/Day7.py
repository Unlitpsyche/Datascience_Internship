import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Set Random State as specified in the assignment
RANDOM_STATE = 203

# Load the dataset
try:
    df = pd.read_csv(r"D:\studies\Programs\LabAssignment_Day7\LabAssignment_Day7\Dataset_Day7.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset_Day7.csv not found. Please ensure the file is in the correct directory.")
    exit()

# --- Problem 1: Replace missing values (0 in specified columns) with relevant figures ---
# Columns to check for missing values (where 0 indicates missing)
missing_value_cols = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']

print("\n--- Handling Missing Values ---")
for col in missing_value_cols:
    # Replace 0s with NaN for easier handling of missing values
    df[col] = df[col].replace(0, np.nan)
    print(f"Missing values (0s replaced with NaN) in '{col}': {df[col].isnull().sum()} samples")

    # Impute missing values with the median (as it's robust to outliers)
    # Reassign the result back to the column to avoid 'inplace' FutureWarnings
    df[col] = df[col].fillna(df[col].median())
    print(f"Missing values in '{col}' after median imputation: {df[col].isnull().sum()} samples")

print("\nDataFrame info after missing value imputation:")
df.info()

# --- Problem 2: Remove all existing outliers ---
# Using Z-score method for outlier detection (values beyond 3 standard deviations)
# This approach is common and robust enough for general use.
print("\n--- Outlier Removal (using Z-score) ---")

# Select numerical columns for outlier detection, excluding 'Outcome'
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('Outcome') # 'Outcome' is the target variable, not a feature for outlier detection

initial_rows = df.shape[0]
outlier_indices = set()

for col in numerical_cols:
    mean = df[col].mean()
    std = df[col].std()
    # Identify outliers using Z-score > 3 or < -3
    col_outliers = df[(np.abs(df[col] - mean) / std > 3)].index
    outlier_indices.update(col_outliers)

# Drop the rows identified as outliers
df_cleaned = df.drop(list(outlier_indices)).reset_index(drop=True)

print(f"Initial number of rows: {initial_rows}")
print(f"Number of outlier rows removed: {len(outlier_indices)}")
print(f"Final number of rows after outlier removal: {df_cleaned.shape[0]}")

# --- Problem 3: Split the data into 70% training and 30% testing data ---
print("\n--- Data Splitting ---")
X = df_cleaned.drop('Outcome', axis=1)
y = df_cleaned['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
# stratify=y ensures that the proportion of 'Outcome' values is the same in both train and test sets.

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# --- Problem 4: Create a logistic regression model with target variable as ‘Outcome’ ---
print("\n--- Logistic Regression Model Training ---")
# Initialize and train the Logistic Regression model
# solver='liblinear' is a good choice for small datasets
# random_state ensures reproducibility of the model
LR_model = LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', max_iter=200)
LR_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.")

# --- Problem 5: Print the default model performance metrics: Accuracy, Precision, Recall, F1-Score & AIC ---
print("\n--- Model Performance Metrics ---")

# Predictions on the test set
y_pred = LR_model.predict(X_test)
y_pred_proba = LR_model.predict_proba(X_test)[:, 1] # Probability of the positive class (Outcome=1)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Calculate AIC (Akaike Information Criterion)
# AIC is typically calculated using `statsmodels` as `sklearn` does not directly provide it.
# We need to add a constant to the independent variables for statsmodels.
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(disp=0) # disp=0 suppresses convergence output
aic = result.aic
print(f"AIC (Akaike Information Criterion): {aic:.4f}")


# --- Problem 6: Plot a F1_score vs threshold curve. Find the threshold for which f1-score is the highest. ---
print("\n--- F1-Score vs. Threshold Curve ---")

# Calculate ROC curve to get various thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate F1-scores for all thresholds
f1_scores = []
for thresh in thresholds:
    # Convert probabilities to binary predictions based on the current threshold
    y_pred_threshold = (y_pred_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_threshold))

# Find the threshold with the highest F1-score
optimal_threshold_index = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_index]
max_f1_score = f1_scores[optimal_threshold_index]

print(f"Optimal Threshold for highest F1-score: {optimal_threshold:.4f}")
print(f"Highest F1-score achieved: {max_f1_score:.4f}")

# Plotting the F1-score vs. Threshold curve
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, marker='o', linestyle='-', markersize=4, label='F1-score')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.4f})')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.title('F1-score vs. Classification Threshold')
plt.grid(True)
plt.legend()
plt.show()

print("\n--- End of Analysis ---")

