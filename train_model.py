import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset (update path as needed)
df = pd.read_csv("visioncare_extended_data.csv")

# Separate features and target
X = df.drop("Vision_Risk", axis=1)
y = df["Vision_Risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "vision_model.pkl")
print("✅ Model saved as vision_model.pkl")
