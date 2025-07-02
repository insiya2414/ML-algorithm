import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Data ---
df = pd.read_csv('04-random-forest/car_prediction_data.csv')

# --- 2. Data Cleaning & Preprocessing ---
# Encode categorical features (except Car_Name, which we drop for simplicity)
df = df.drop('Car_Name', axis=1)

categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

features = [col for col in df.columns if col != 'Selling_Price']
target = 'Selling_Price'

X = df[features]
y = df[target]

# --- 3. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Random Forest Regression ---
print("\n--- Random Forest Regressor (Car Price Prediction) ---")
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

# --- 5. Feature Importance ---
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=[features[i] for i in indices])
plt.title("Feature Importance in Random Forest Regressor")
plt.show()

# --- 6. Visualize a Single Tree (Optional) ---
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], feature_names=features, filled=True, rounded=True, max_depth=3)
plt.title("Example Tree from Random Forest Regressor (truncated to depth 3)")
plt.show()

# --- Notes ---
# - All categorical features are label-encoded for compatibility with RandomForest.
# - The model predicts the car's selling price (regression).
# - Prints MSE, MAE, R^2, and feature importances.
# - Visualizes one tree from the forest for interpretability.