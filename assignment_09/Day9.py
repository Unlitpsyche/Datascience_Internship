import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import zscore

# Set a random state for reproducibility as specified in the assignment
RANDOM_STATE = 1234

# 1. Load Data
try:
    df = pd.read_csv(r"D:\studies\Programs\LabAssignment_Day9\LabAssignment_Day9\Dataset_Day9.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset_Day9.csv not found. Please ensure the file is in the correct directory.")
    exit()

print(f"\nInitial DataFrame shape: {df.shape}")
print("\nInitial DataFrame info:")
df.info()
print("\nInitial DataFrame head:")
print(df.head())

# Define columns where 0 should be considered as missing data
missing_value_cols = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# 2. Replace Missing values (0s) with relevant figures (median)
# Using df.replace({col: value}, inplace=True) and df.fillna({col: value}, inplace=True)
print("\nHandling missing values (0s treated as missing) and NA values...")
for col in missing_value_cols:
    # Replace 0 values with the median of the column (excluding 0s for median calculation)
    median_val = df[df[col] != 0][col].median()
    if pd.isna(median_val): # Fallback if all values are 0 or NaN
        median_val = df[col].median() # Use overall median if no non-zero values
    df.replace({col: 0}, value=median_val, inplace=True)
    # Fill any actual NaN values that might exist (though the prompt suggests 0s are the missing markers)
    if df[col].isnull().any():
        df.fillna({col: df[col].median()}, inplace=True)
print("Missing values (0s replaced with median) and NA values filled.")
print("\nDataFrame info after handling 0s as missing:")
df.info()

# 3. Remove existing outliers
# Using Z-score method for outlier detection and removal
print("\nRemoving outliers using Z-score...")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude 'Outcome' from outlier detection as it's the target variable
if 'Outcome' in numeric_cols:
    numeric_cols.remove('Outcome')

# Calculate Z-scores for relevant numeric columns
for col in numeric_cols:
    df[f'zscore_{col}'] = np.abs(zscore(df[col]))

# Define a threshold for outliers (e.g., Z-score > 3)
outlier_threshold = 3
outlier_rows_mask = (df[[f'zscore_{col}' for col in numeric_cols]] > outlier_threshold).any(axis=1)
original_rows = df.shape[0]
df_cleaned = df[~outlier_rows_mask].copy() # Create a copy to avoid SettingWithCopyWarning

rows_removed = original_rows - df_cleaned.shape[0]
print(f"Removed {rows_removed} outlier rows ({(rows_removed/original_rows)*100:.2f}% of original data).")
df_cleaned = df_cleaned.drop(columns=[f'zscore_{col}' for col in numeric_cols]) # Drop zscore columns
print(f"DataFrame shape after outlier removal: {df_cleaned.shape}")
print("\nDataFrame head after outlier removal:")
print(df_cleaned.head())


# 4. Split the data into 75% training and 25% testing data
X = df_cleaned.drop('Outcome', axis=1)
y = df_cleaned['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

print(f"\nTraining data shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Testing data shape (X_test, y_test): {X_test.shape}, {y_test.shape}")

# 5. Print the default model performance metrics: Accuracy, Precision, Recall, F1-Score
print("\n--- Default SVM Classifier Performance ---")
# Using a common default kernel like 'rbf' and C=1, gamma='scale'
default_svm_clf = svm.SVC(random_state=RANDOM_STATE)
default_svm_clf.fit(X_train, y_train)
y_pred_default = default_svm_clf.predict(X_test)

accuracy_default = accuracy_score(y_test, y_pred_default)
# Added zero_division=0 to handle cases where precision is undefined
precision_default = precision_score(y_test, y_pred_default, zero_division=0)
recall_default = recall_score(y_test, y_pred_default)
f1_default = f1_score(y_test, y_pred_default)

print(f"Accuracy: {accuracy_default:.4f}")
print(f"Precision: {precision_default:.4f}")
print(f"Recall: {recall_default:.4f}")
print(f"F1-Score: {f1_default:.4f}")

# 6. Print Precision & Recall & F1-Score vs kernel type curve
# Find the kernel type for which F1-score is the highest. Take (C = 0.001,0.01,0.1,1,10)
print("\n--- Hyperparameter Tuning: Finding Best Kernel ---")
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
C_values_kernel = [0.001, 0.01, 0.1, 1, 10]
gamma_value = 'scale' # Using default gamma for this part or a fixed value like 'scale'

performance_by_kernel = []

for kernel in kernel_types:
    for C in C_values_kernel:
        print(f"Testing Kernel: {kernel}, C: {C}, Gamma: {gamma_value}")
        svm_clf = svm.SVC(kernel=kernel, C=C, gamma=gamma_value, random_state=RANDOM_STATE)
        svm_clf.fit(X_train, y_train)
        y_pred = svm_clf.predict(X_test)

        # Added zero_division=0 to handle cases where precision is undefined
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        performance_by_kernel.append({'Kernel': kernel, 'C': C, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})

performance_df_kernel = pd.DataFrame(performance_by_kernel)
print("\nPerformance by Kernel Type and C values:")
print(performance_df_kernel)

# Plotting Precision, Recall, F1-Score vs Kernel Type
plt.figure(figsize=(12, 6))
sns.lineplot(data=performance_df_kernel, x='Kernel', y='Precision', marker='o', label='Precision')
sns.lineplot(data=performance_df_kernel, x='Kernel', y='Recall', marker='o', label='Recall')
sns.lineplot(data=performance_df_kernel, x='Kernel', y='F1-Score', marker='o', label='F1-Score')
plt.title('Precision, Recall, F1-Score vs Kernel Type (for various C values)')
plt.xlabel('Kernel Type')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Find the kernel type for which F1-score is the highest
best_kernel_row = performance_df_kernel.loc[performance_df_kernel['F1-Score'].idxmax()]
best_kernel = best_kernel_row['Kernel']
best_c_for_kernel_search = best_kernel_row['C'] # Keep track of this C for the next step, as it gave the max F1
print(f"\nKernel type with the highest F1-Score: {best_kernel} (C={best_c_for_kernel_search}, F1-Score={best_kernel_row['F1-Score']:.4f})")


# 7. Plot a curve on Precision & Recall & F1-Score vs appropriate range of C
# using the best kernel type obtained in question(3b)
# Find the C for which F1-score is the highest for the given kernel type.
print(f"\n--- Hyperparameter Tuning: Finding Best C for {best_kernel} kernel ---")
# Take C between (0,10) in small increments of 0.05
C_values_range = np.arange(0.05, 10.05, 0.05)
gamma_value_best_kernel = 'scale' # Stick with 'scale' for gamma or if the best_kernel_row had a specific gamma, use that.

performance_by_C = []

for C in C_values_range:
    # Ensure C is not 0 for SVM
    if C == 0:
        continue
    svm_clf = svm.SVC(kernel=best_kernel, C=C, gamma=gamma_value_best_kernel, random_state=RANDOM_STATE)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    # Added zero_division=0 to handle cases where precision is undefined
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    performance_by_C.append({'C': C, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})

performance_df_C = pd.DataFrame(performance_by_C)
print("\nPerformance by C value for the best kernel:")
print(performance_df_C.head()) # Print head as it can be a large DataFrame

# Plotting Precision, Recall, F1-Score vs C value for the best kernel
plt.figure(figsize=(12, 6))
sns.lineplot(data=performance_df_C, x='C', y='Precision', marker='o', label='Precision')
sns.lineplot(data=performance_df_C, x='C', y='Recall', marker='o', label='Recall')
sns.lineplot(data=performance_df_C, x='C', y='F1-Score', marker='o', label='F1-Score')
plt.title(f'Precision, Recall, F1-Score vs C Value for {best_kernel} Kernel')
plt.xlabel('C Value')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Find the C for which F1-score is the highest for the given kernel type
best_c_row = performance_df_C.loc[performance_df_C['F1-Score'].idxmax()]
best_C_final = best_c_row['C']
print(f"\nBest C value for '{best_kernel}' kernel with the highest F1-Score: {best_C_final:.2f} (F1-Score={best_c_row['F1-Score']:.4f})")
