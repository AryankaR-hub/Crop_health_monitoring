import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load and Split
df = pd.read_csv("plant_moniter_health_data.csv")
X = df[["Temperature_C", "Humidity_%", "Soil_Moisture_%", "Soil_pH", "Nutrient_Level", "Light_Intensity_lux"]]
y = df["Health_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=45)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Calculate ratio for scale_pos_weight
ratio = np.bincount(y_train)[1] / np.bincount(y_train)[0]

# Train XGBoost
xgb_model = XGBClassifier(scale_pos_weight=ratio, n_estimators=220, learning_rate=0.05, random_state=45
                          )
xgb_model.fit(X_train_res, y_train_res)

# Results
y_pred = xgb_model.predict(X_test)
print("--- Balanced XGBoost ---")
print(classification_report(y_test, y_pred))
joblib.dump(xgb_model, "xgb_balanced.pkl")