# train.py — Model training script for EV regenerative braking energy prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# -------------------------------
# Step 1: Load your datasets
# -------------------------------
try:
    df1 = pd.read_csv("ADAS_EV_Dataset.csv")
    df2 = pd.read_csv("EV_Energy_Consumption_Dataset.csv")
    df3 = pd.read_csv("nev_energy_management_dataset.csv")

    # Merge all datasets into one DataFrame
    df = pd.concat([df1, df2, df3], ignore_index=True)

except Exception as e:
    print("❌ Error loading datasets:", e)
    exit()

# -------------------------------
# Step 2: Create synthetic training features
# -------------------------------
# If your dataset doesn't already have km_travelled or brakes columns,
# we'll simulate them for training.

if "km_travelled" not in df.columns:
    df["km_travelled"] = np.random.uniform(1, 500, len(df))  # random km between 1 and 500

if "no_of_brakes" not in df.columns:
    df["no_of_brakes"] = np.random.randint(1, 200, len(df))  # random brakes between 1 and 200

# Create a synthetic target (energy recovered) if not present
if "energy_recovered" not in df.columns:
    # Basic formula: more km and more brakes → more recovered energy
    df["energy_recovered"] = 0.05 * df["km_travelled"] + 0.2 * df["no_of_brakes"] + np.random.normal(0, 2, len(df))

# -------------------------------
# Step 3: Split into features (X) and target (y)
# -------------------------------
X = df[["km_travelled", "no_of_brakes"]]
y = df["energy_recovered"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 4: Train model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluate model
# -------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ Model trained successfully!")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# -------------------------------
# Step 6: Save model
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/regen_model.pkl")
print("✅ Model saved at: models/regen_model.pkl")
