import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Data ---
df = pd.read_csv('05-gradient-boosting/ionosphere.csv')

# --- 2. Data Cleaning & Preprocessing ---
# Encode the target class ('g'/'b') as 1/0
df['class'] = LabelEncoder().fit_transform(df['class'])

features = df.columns[:-1]  # all columns except 'class'
target = 'class'

X = df[features]
y = df[target]

# --- 3. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Gradient Boosting Classification ---
print("\n--- Gradient Boosting Classifier (Ionosphere Dataset) ---")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['bad', 'good']))

# --- 5. Feature Importance ---
importances = gb.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices])
plt.title("Feature Importance in Gradient Boosting")
plt.show()

# --- Notes ---
# - The target is binary: 'g' (good) and 'b' (bad), encoded as 1 and 0.
# - The model predicts if a radar return is 'good' or 'bad'.
# - Prints accuracy, confusion matrix, classification report, and feature importances.