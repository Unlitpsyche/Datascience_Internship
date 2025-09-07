import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r'D:\studies\Programs\LabAssignment_Day6\LabAssignment_Day6\Dataset_Day6.csv')

# 1. Replace missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)


cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)


# Remove outliers using IQR method except for 'Price'
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in num_cols:
    if col != 'Price':
        df = remove_outliers(df, col)

# For 'Price', remove outliers only if data volume reduces by less than 30%
original_len = len(df)
df_price_filtered = remove_outliers(df, 'Price')
if len(df_price_filtered) >= 0.7 * original_len:
    df = df_price_filtered

# 2. One Hot Encoding for categorical variables
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3. Split data into train and test with random_state=50
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Adjusted R2 calculation
def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Multiple Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
adj_r2_lr = adjusted_r2(r2_lr, X_test.shape[0], X_test.shape[1])
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print("Linear Regression Performance:")
print(f"R2: {r2_lr:.4f}")
print(f"Adjusted R2: {adj_r2_lr:.4f}")
print(f"MAE: {mae_lr:.2f}")

# 4. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

r2_ridge = r2_score(y_test, y_pred_ridge)
adj_r2_ridge = adjusted_r2(r2_ridge, X_test.shape[0], X_test.shape[1])
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

print("\nRidge Regression Performance:")
print(f"R2: {r2_ridge:.4f}")
print(f"Adjusted R2: {adj_r2_ridge:.4f}")
print(f"MAE: {mae_ridge:.2f}")

#5. Lasso Regression
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

r2_lasso = r2_score(y_test, y_pred_lasso)
adj_r2_lasso = adjusted_r2(r2_lasso, X_test.shape[0], X_test.shape[1])
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print("\nLasso Regression Performance:")
print(f"R2: {r2_lasso:.4f}")
print(f"Adjusted R2: {adj_r2_lasso:.4f}")
print(f"MAE: {mae_lasso:.2f}")
