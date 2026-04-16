import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("plant_moniter_health_data.csv")

# Separate healthy and unhealthy
healthy = df[df["Health_Status"] == 1]
unhealthy = df[df["Health_Status"] == 0]

# Number of synthetic samples to generate
n_samples = 500

# Generate synthetic UNHEALTHY plants
synthetic_unhealthy = pd.DataFrame({
    "Temperature_C": np.random.uniform(30, 45, n_samples),
    "Humidity_%": np.random.uniform(20, 50, n_samples),
    "Soil_Moisture_%": np.random.uniform(5, 30, n_samples),
    "Soil_pH": np.random.uniform(3, 5.5, n_samples),
    "Nutrient_Level": np.random.uniform(10, 40, n_samples),
    "Light_Intensity_lux": np.random.uniform(5000, 15000, n_samples),
    "Health_Score": np.random.uniform(20, 50, n_samples),
    "Health_Status": 0
})

# Add Plant IDs
synthetic_unhealthy["Plant_ID"] = ["Synthetic_" + str(i) for i in range(len(synthetic_unhealthy))]

# Combine with original dataset
df_final = pd.concat([df, synthetic_unhealthy], ignore_index=True)

# Save file (IMPORTANT LINE)
df_final.to_csv("plant_data_augmented.csv", index=False)

# Confirm
print("✅ Synthetic dataset generated successfully!")
print("Rows in new dataset:", len(df_final))