import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

# Load the dataset
try:
    df = pd.read_csv(r'D:\studies\Programs\LabAssignment_Day12\LabAssignment_Day12\Dataset_Day12.csv')
except FileNotFoundError:
    print("Error: 'Dataset_Day12.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Define columns where 0 should be considered missing
missing_value_cols = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']

# Replace 0 values with NaN for specified columns
for col in missing_value_cols:
    df[col] = df[col].replace(0, np.nan)

# Display missing values after replacement
print("\nMissing values after replacing 0's with NaN:")
print(df.isnull().sum())

# Replace missing values with the median of their respective columns
for col in missing_value_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df.fillna({col: median_val}, inplace=True)

# Display missing values after imputation
print("\nMissing values after imputing with median:")
print(df.isnull().sum())

# Outlier removal using the IQR method
for col in df.columns:
    if df[col].dtype in ['float64', 'int64'] and col != 'Outcome':  # Exclude 'Outcome' as it's a target variable
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove outliers using boolean indexing
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Display shape after outlier removal
print(f"\nShape of data after outlier removal: {df.shape}")

# Split the data into features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naïve Bayes Classifier
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_model.predict(X_test)

# Print default model performance metrics
print("\nNaïve Bayes Classifier - Default Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_nb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_nb):.4f}")

# k-Fold Cross Validation for Naïve Bayes
print("\nNaïve Bayes Classifier - k-Fold Cross Validation Performance:")
kf = KFold(n_splits=10, shuffle=True, random_state=42) # Using 10 splits as a common practice
accuracy_scores_nb = cross_val_score(nb_model, X, y, cv=kf, scoring='accuracy')
precision_scores_nb = cross_val_score(nb_model, X, y, cv=kf, scoring='precision')
recall_scores_nb = cross_val_score(nb_model, X, y, cv=kf, scoring='recall')
f1_scores_nb = cross_val_score(nb_model, X, y, cv=kf, scoring='f1')

print(f"Mean Accuracy (k-Fold): {accuracy_scores_nb.mean():.4f}")
print(f"Mean Precision (k-Fold): {precision_scores_nb.mean():.4f}")
print(f"Mean Recall (k-Fold): {recall_scores_nb.mean():.4f}")
print(f"Mean F1 Score (k-Fold): {f1_scores_nb.mean():.4f}")

# Adaboost with Naïve Bayes as base estimator (Optional)
print("\nAdaboost with Naïve Bayes - Model Performance:")
# Using the Gaussian Naive Bayes classifier as the base estimator
adaboost_model = AdaBoostClassifier(estimator=nb_model, random_state=42)

# Train the Adaboost model
adaboost_model.fit(X_train, y_train)

# Make predictions with Adaboost
y_pred_adaboost = adaboost_model.predict(X_test)

# Print Adaboost model performance metrics
print(f"Accuracy (Adaboost): {accuracy_score(y_test, y_pred_adaboost):.4f}")
print(f"Precision (Adaboost): {precision_score(y_test, y_pred_adaboost):.4f}")
print(f"Recall (Adaboost): {recall_score(y_test, y_pred_adaboost):.4f}")
print(f"F1 Score (Adaboost): {f1_score(y_test, y_pred_adaboost):.4f}")

# k-Fold Cross Validation for Adaboost with Naïve Bayes
print("\nAdaboost with Naïve Bayes - k-Fold Cross Validation Performance:")
accuracy_scores_adaboost = cross_val_score(adaboost_model, X, y, cv=kf, scoring='accuracy')
precision_scores_adaboost = cross_val_score(adaboost_model, X, y, cv=kf, scoring='precision')
recall_scores_adaboost = cross_val_score(adaboost_model, X, y, cv=kf, scoring='recall')
f1_scores_adaboost = cross_val_score(adaboost_model, X, y, cv=kf, scoring='f1')

print(f"Mean Accuracy (Adaboost k-Fold): {accuracy_scores_adaboost.mean():.4f}")
print(f"Mean Precision (Adaboost k-Fold): {precision_scores_adaboost.mean():.4f}")
print(f"Mean Recall (Adaboost k-Fold): {recall_scores_adaboost.mean():.4f}")
print(f"Mean F1 Score (Adaboost k-Fold): {f1_scores_adaboost.mean():.4f}")
