# implementation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- 1. Load the Data ---
try:
    df = pd.read_csv('01-linear-regression/Real estate.csv')
except FileNotFoundError:
    print("Error: 'data/real_estate_data.csv' not found.")
    print("Please make sure you have created the file and are running the script from the correct directory (ML-algorithm).")
    exit()

# --- 2. Data Cleaning (THE FIX IS HERE) ---
# The original column names from the file are messy. Let's rename them.
# We will drop the 'No' column as it is just an index.
df = df.drop('No', axis=1)

# Now, we rename the remaining columns to be clean and easy to use.
df.columns = [
    'transaction_date', 'house_age', 'distance_to_mrt', 
    'convenience_stores', 'latitude', 'longitude', 'price_per_unit_area'
]

print("--- Cleaned Dataset (First 5 Rows) ---")
print(df.head())
print("-" * 40)


# --- 3. Prepare Data for Modeling ---
# Define our features (the X values) and the target (the Y value).
features = ['transaction_date', 'house_age', 'distance_to_mrt', 'convenience_stores', 'latitude', 'longitude']
target = 'price_per_unit_area'

X = df[features]
y = df[target]

# Split the data into a training set and a testing set for evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 4. Simple Linear Regression (One Feature) ---
print("\n--- Model 1: Simple Linear Regression (Distance to MRT vs. Price) ---")
X_train_simple = X_train[['distance_to_mrt']]
X_test_simple = X_test[['distance_to_mrt']]

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)
y_pred_simple = simple_model.predict(X_test_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print(f"Coefficient for 'distance_to_mrt': {simple_model.coef_[0]:.4f}")
print(f"Evaluation (R-squared): {r2_simple:.4f}")


# --- 5. Multiple Linear Regression (All Features) ---
print("\n--- Model 2: Multiple Linear Regression (All Features vs. Price) ---")
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)
y_pred_multiple = multiple_model.predict(X_test)
r2_multiple = r2_score(y_test, y_pred_multiple)

print("Coefficients for each feature:")
for feature, coef in zip(features, multiple_model.coef_):
    print(f"  - {feature}: {coef:.4f}")
print(f"\nEvaluation (R-squared): {r2_multiple:.4f}")


# --- 6. Polynomial Regression (Modeling a Curve) ---
print("\n--- Model 3: Polynomial Regression (Curved Relationship) ---")
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_transformer.fit_transform(X_train_simple)
X_test_poly = poly_transformer.transform(X_test_simple)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial model R-squared: {r2_poly:.4f}")