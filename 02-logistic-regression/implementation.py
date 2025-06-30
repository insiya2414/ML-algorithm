import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# --- 1. Load the Data ---
df = pd.read_csv('02-logistic-regression/Social_Network_Ads.csv')

# --- 2. Data Cleaning & Preprocessing ---
# Drop 'User ID' as it's not a useful feature
df = df.drop('User ID', axis=1)

# Encode 'Gender' as 0/1
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# --- 3. Prepare Data for Modeling ---
features = ['Gender', 'Age', 'EstimatedSalary']
target = 'Purchased'

X = df[features]
y = df[target]

# Feature scaling (important for logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 4. Simple Logistic Regression (One Feature: Age) ---
print("\n--- Model 1: Simple Logistic Regression (Age vs. Purchased) ---")
X_train_simple = X_train[:, [1]].reshape(-1, 1)  # Age column
X_test_simple = X_test[:, [1]].reshape(-1, 1)

simple_model = LogisticRegression()
simple_model.fit(X_train_simple, y_train)
y_pred_simple = simple_model.predict(X_test_simple)
acc_simple = accuracy_score(y_test, y_pred_simple)

print(f"Coefficient for 'Age': {simple_model.coef_[0][0]:.4f}")
print(f"Intercept: {simple_model.intercept_[0]:.4f}")
print(f"Accuracy: {acc_simple:.4f}")
print(confusion_matrix(y_test, y_pred_simple))
print(classification_report(y_test, y_pred_simple))

# --- 5. Multiple Logistic Regression (All Features) ---
print("\n--- Model 2: Multiple Logistic Regression (All Features vs. Purchased) ---")
multiple_model = LogisticRegression()
multiple_model.fit(X_train, y_train)
y_pred_multiple = multiple_model.predict(X_test)
acc_multiple = accuracy_score(y_test, y_pred_multiple)

print("Coefficients for each feature:")
for feature, coef in zip(features, multiple_model.coef_[0]):
    print(f"  - {feature}: {coef:.4f}")
print(f"Intercept: {multiple_model.intercept_[0]:.4f}")
print(f"Accuracy: {acc_multiple:.4f}")
print(confusion_matrix(y_test, y_pred_multiple))
print(classification_report(y_test, y_pred_multiple))

# --- 6. Polynomial Logistic Regression (Modeling Nonlinear Boundaries) ---
print("\n--- Model 3: Polynomial Logistic Regression (Age & Salary, degree=2) ---")
# We'll use Age and EstimatedSalary for polynomial features
X_train_poly_base = X_train[:, [1, 2]]
X_test_poly_base = X_test[:, [1, 2]]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_poly_base)
X_test_poly = poly.transform(X_test_poly_base)

poly_model = LogisticRegression(max_iter=1000)
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
acc_poly = accuracy_score(y_test, y_pred_poly)

print(f"Polynomial model accuracy: {acc_poly:.4f}")
print(confusion_matrix(y_test, y_pred_poly))
print(classification_report(y_test, y_pred_poly))