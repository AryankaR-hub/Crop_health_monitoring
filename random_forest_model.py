import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------------
# Load AUGMENTED dataset
# -----------------------------
df = pd.read_csv("plant_data_augmented.csv")

X = df[["Temperature_C", "Humidity_%", "Soil_Moisture_%",
        "Soil_pH", "Nutrient_Level", "Light_Intensity_lux"]]

y = df["Health_Status"]

# -----------------------------
# Train-Test Split (200 unseen)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45, stratify=y
)

# -----------------------------
# Model
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight={0: 3, 1: 1},   # focus on unhealthy plants
    random_state=45
)

# -----------------------------
# 5-Fold Cross Validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=45)

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')

print("Random Forest CV Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# -----------------------------
# Train final model
# -----------------------------
rf_model.fit(X_train, y_train)

# -----------------------------
# Prediction with THRESHOLD FIX
# -----------------------------
y_prob = rf_model.predict_proba(X_test)[:, 1]

# 👇 IMPORTANT: lower threshold to detect unhealthy
y_pred = (y_prob > 0.4).astype(int)

# -----------------------------
# Results
# -----------------------------
print("\n--- Final Test Results (Improved Random Forest) ---")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(rf_model, "random_forest_model.pkl")