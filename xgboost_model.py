import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

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
# Handle imbalance (VERY IMPORTANT)
# -----------------------------
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos   # focuses on minority class

# -----------------------------
# Model
# -----------------------------
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=45,
    eval_metric='logloss'   # removes warning
)

# -----------------------------
# 5-Fold Cross Validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=45)

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')

print("XGBoost CV Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# -----------------------------
# Train final model
# -----------------------------
xgb_model.fit(X_train, y_train)

# -----------------------------
# Prediction with THRESHOLD FIX
# -----------------------------
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# 👇 IMPORTANT: lower threshold
y_pred = (y_prob > 0.4).astype(int)

# -----------------------------
# Results
# -----------------------------
print("\n--- Final Test Results (Improved XGBoost) ---")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(xgb_model, "xgboost_model.pkl")