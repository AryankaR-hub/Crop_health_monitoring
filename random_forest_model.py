import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE 

# Load and Split
df = pd.read_csv("plant_moniter_health_data.csv")
X = df[["Temperature_C", "Humidity_%", "Soil_Moisture_%", "Soil_pH", "Nutrient_Level", "Light_Intensity_lux"]]
y = df["Health_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45, stratify=y)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=45)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=220, class_weight='balanced', random_state=45)
rf_model.fit(X_train_res, y_train_res)

# Results
y_pred = rf_model.predict(X_test)
print("--- Balanced Random Forest ---")
print(classification_report(y_test, y_pred))
joblib.dump(rf_model, "rf_balanced.pkl")