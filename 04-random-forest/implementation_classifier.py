import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Data ---
df = pd.read_csv('04-random-forest/car_evaluation.csv')

# --- 2. Data Cleaning & Preprocessing ---
# Encode all categorical features and the target
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features = df.columns[:-1]  # all columns except 'decision'
target = 'decision'

X = df[features]
y = df[target]

# --- 3. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Random Forest Classification ---
print("\n--- Random Forest Classifier (Car Evaluation) ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoders[target].classes_))

# --- 5. Feature Importance ---
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=[features[i] for i in indices])
plt.title("Feature Importance in Random Forest")
plt.show()

# --- 6. Visualize a Single Tree (Optional) ---
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], feature_names=features, class_names=label_encoders[target].classes_, filled=True, rounded=True, max_depth=3)
plt.title("Example Tree from Random Forest (truncated to depth 3)")
plt.show()

# --- Notes ---
# - All categorical features are label-encoded for compatibility with RandomForest.
# - The model predicts the car's acceptability class (unacc, acc, good, vgood).
# - Prints accuracy, confusion matrix, classification report, and feature importances.
# - Visualizes one tree from the forest for interpretability.

