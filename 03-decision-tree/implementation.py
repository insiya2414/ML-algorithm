import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load the Data ---
df = pd.read_csv('03-decision-tree/winequality-red.csv')

# --- 2. Data Cleaning & Preprocessing ---
# No missing values or categorical columns in this dataset, so we can use it as is.

# --- 3. Decision Tree Classification ---
# We'll treat wine quality as a classification problem: good (quality >= 7), bad (quality < 7)
df['quality_label'] = (df['quality'] >= 7).astype(int)  # 1 = good, 0 = bad

features = df.columns[:-2]  # all columns except 'quality' and 'quality_label'
target_class = 'quality_label'

X = df[features]
y_class = df[target_class]

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

print("\n--- Decision Tree Classifier (Good/Bad Wine) ---")
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred_class)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))
print("Classification Report:")
print(classification_report(y_test, y_pred_class))

# Optional: Visualize the tree
plt.figure(figsize=(16, 8))
plot_tree(clf, feature_names=features, class_names=['Bad', 'Good'], filled=True, rounded=True)
plt.title("Decision Tree Classifier for Wine Quality")
plt.show()

# --- 4. Decision Tree Regression ---
# Now treat wine quality as a regression problem (predict the actual score)
target_reg = 'quality'
y_reg = df[target_reg]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

print("\n--- Decision Tree Regressor (Predict Quality Score) ---")
reg = DecisionTreeRegressor(max_depth=4, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_reg = reg.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_reg)
r2 = r2_score(y_test_r, y_pred_reg)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Optional: Visualize the regression tree
plt.figure(figsize=(16, 8))
plot_tree(reg, feature_names=features, filled=True, rounded=True)
plt.title("Decision Tree Regressor for Wine Quality")
plt.show()


# What this does:

# Classification: Predicts if wine is "good" (quality ≥ 7) or "bad" using a decision tree classifier.
# Regression: Predicts the actual wine quality score using a decision tree regressor.
# Metrics: Prints accuracy/confusion/classification report for classification, and MSE/R² for regression.
# Visualization: Plots both trees for interpretation.