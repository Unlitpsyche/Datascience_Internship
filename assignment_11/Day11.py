import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r'D:\studies\Programs\LabAssignment_Day11\LabAssignment_Day11\Dataset_Day11.csv')

# Step 1: Replace 0s with NaN in specified columns (considered missing)
cols_missing = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']
for col in cols_missing:
    df.replace({col: 0}, inplace=True)

# Step 2: Impute missing values with median of each column
for col in cols_missing:
    median_val = df[col].median()
    df.fillna({col: median_val}, inplace=True)

print("\nData after handling missing values (first 5 rows):")
print(df.head())
print("\nMissing values count per column after imputation:")
print(df.isnull().sum())

# Step 3: Remove outliers using IQR method for the same columns
for col in cols_missing:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\nData after removing outliers (first 5 rows):")
print(df.head())
print("\nData shape after outlier removal:", df.shape)

# Step 4: Split data into train and test sets (80% train, 20% test)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Function to print performance metrics
def print_metrics(y_true, y_pred, model_name):
    print(f"Performance metrics for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print("-" * 40)

# Step 5: Bagging with Decision Trees, vary n_estimators 2 to 25
bagging_f1 = []
bagging_acc = []
n_estimators_range = range(2, 26)
for n in n_estimators_range:
    bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=n, random_state=99)
    bag.fit(X_train, y_train)
    y_pred = bag.predict(X_test)
    bagging_f1.append(f1_score(y_test, y_pred))
    bagging_acc.append(accuracy_score(y_test, y_pred))

# Print metrics for default Bagging (n_estimators=10)
bag_default = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=99)
bag_default.fit(X_train, y_train)
y_pred_default = bag_default.predict(X_test)
print_metrics(y_test, y_pred_default, "Bagging Classifier (n_estimators=10)")

# Plot F1 Score & Accuracy vs n_estimators for Bagging
plt.figure(figsize=(10,5))
plt.plot(n_estimators_range, bagging_f1, label='F1 Score')
plt.plot(n_estimators_range, bagging_acc, label='Accuracy')
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.title('Bagging Classifier Performance')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Random Forest, vary n_estimators 2 to 25, plot F1 Score * Accuracy
rf_f1 = []
rf_acc = []
for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, random_state=99)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_f1.append(f1_score(y_test, y_pred))
    rf_acc.append(accuracy_score(y_test, y_pred))

# Print metrics for default Random Forest (n_estimators=10)
rf_default = RandomForestClassifier(n_estimators=10, random_state=99)
rf_default.fit(X_train, y_train)
y_pred_rf_default = rf_default.predict(X_test)
print_metrics(y_test, y_pred_rf_default, "Random Forest (n_estimators=10)")

# Plot F1 Score * Accuracy vs n_estimators for Random Forest
rf_product = [f*a for f,a in zip(rf_f1, rf_acc)]
plt.figure(figsize=(10,5))
plt.plot(n_estimators_range, rf_product, marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('F1 Score * Accuracy')
plt.title('Random Forest Performance')
plt.grid(True)
plt.show()

# Step 7: AdaBoost with Decision Trees, default parameters
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=99)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print_metrics(y_test, y_pred_ada, "AdaBoost Classifier")
