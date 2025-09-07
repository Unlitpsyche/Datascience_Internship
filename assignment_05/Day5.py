import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv(r'D:\studies\Programs\LabAssignment_Day5\LabAssignment_Day5\Dataset_Day5.csv')

# 2. Select predictor and target
X = df[['RM']]
y = df['MEDV']

# 3. Split into train and test sets (80% train, 20% test), with random_state=100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# 4. Create and fit the model on training data
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Print model coefficients
print("Regression Model: MEDV = {:.4f} + {:.4f} * RM".format(model.intercept_, model.coef_[0]))

# 6. Predict on test data
y_pred = model.predict(X_test)

# 7. Print performance metrics on test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
print(f"R-squared (R2) on test set: {r2:.4f}")

# 8. Predict the price for RM = 7 (use DataFrame to avoid warnings)
rm_value = pd.DataFrame({'RM': [7]})
predicted_price = model.predict(rm_value)
print(f"Predicted MEDV for RM = 7: {predicted_price[0]:.4f}")

# 9. (Optional) Plot regression line with test data
plt.figure(figsize=(7,5))
plt.scatter(X_test, y_test, color='blue', label='Actual Test Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line (Test)')
plt.xlabel('RM (Average number of rooms per dwelling)')
plt.ylabel('MEDV (Median value of owner-occupied homes in $1000s)')
plt.title('Linear Regression: MEDV vs RM (Test Set)')
plt.legend()
plt.tight_layout()
plt.show()
