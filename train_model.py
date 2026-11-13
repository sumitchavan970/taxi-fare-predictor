"""
train_model.py
--------------
Trains a Machine Learning model that predicts taxi fare from
trip distance (km) and trip duration (minutes).

Steps:
  1. Load the CSV dataset
  2. Clean the data (missing, duplicate, invalid values)
  3. Basic data analysis + charts (saved in the 'plots' folder)
  4. Train a Linear Regression model
  5. Evaluate with MAE, RMSE and R2
  6. Save the trained model with Joblib (models/taxi_fare_model.pkl)

Run:  python train_model.py
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # draw charts to files (no pop-up window needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
# Paths are built from this file's location, so the script works
# no matter which folder you start it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "taxi_fare.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "taxi_fare_model.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

FEATURES = ["trip_distance_km", "trip_duration_min"]  # model inputs
TARGET = "fare_amount"                                # what we predict

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
print(f"Rows loaded: {len(df)}")

# ---------------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------------
before = len(df)
df = df.dropna(subset=FEATURES + [TARGET])   # remove missing values
print(f"Removed missing rows   : {before - len(df)}")

before = len(df)
df = df.drop_duplicates()                    # remove duplicate rows
print(f"Removed duplicate rows : {before - len(df)}")

before = len(df)
# invalid = distance, duration or fare that is zero or negative
df = df[(df["trip_distance_km"] > 0) &
        (df["trip_duration_min"] > 0) &
        (df[TARGET] > 0)]
print(f"Removed invalid rows   : {before - len(df)}")
print(f"Clean rows remaining   : {len(df)}\n")

# ---------------------------------------------------------------
# 3. Basic data analysis and visualisation
# ---------------------------------------------------------------
print("Summary statistics:")
print(df[FEATURES + [TARGET]].describe().round(2), "\n")

print("Correlation with fare_amount:")
print(df[FEATURES + [TARGET]].corr()[TARGET].round(3), "\n")

# Chart 1: distance vs fare
plt.figure(figsize=(6, 4))
plt.scatter(df["trip_distance_km"], df[TARGET], s=8, alpha=0.5)
plt.xlabel("Trip Distance (km)")
plt.ylabel("Fare Amount")
plt.title("Distance vs Fare")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "distance_vs_fare.png"), dpi=120)
plt.close()

# Chart 2: duration vs fare
plt.figure(figsize=(6, 4))
plt.scatter(df["trip_duration_min"], df[TARGET], s=8, alpha=0.5, color="green")
plt.xlabel("Trip Duration (minutes)")
plt.ylabel("Fare Amount")
plt.title("Duration vs Fare")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "duration_vs_fare.png"), dpi=120)
plt.close()

# Chart 3: fare distribution
plt.figure(figsize=(6, 4))
plt.hist(df[TARGET], bins=30, color="orange", edgecolor="black")
plt.xlabel("Fare Amount")
plt.ylabel("Number of Trips")
plt.title("Fare Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fare_distribution.png"), dpi=120)
plt.close()

# ---------------------------------------------------------------
# 4. Train the model
# ---------------------------------------------------------------
X = df[FEATURES]
y = df[TARGET]

# 80% of data to learn from, 20% kept aside to test the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 5. Evaluate the model on the unseen test data
# ---------------------------------------------------------------
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("Model evaluation (test data):")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R2   : {r2:.4f}")

# Chart 4: actual vs predicted
plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred, s=8, alpha=0.5, color="purple")
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=120)
plt.close()

# ---------------------------------------------------------------
# 6. Save the trained model (Joblib)
# ---------------------------------------------------------------
joblib.dump(model, MODEL_FILE)

# Save the scores and data ranges so the website can display them
with open(METRICS_FILE, "w") as f:
    json.dump({
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 4),
        "rows_used": int(len(df)),
        "distance_range": [float(df["trip_distance_km"].min()), float(df["trip_distance_km"].max())],
        "duration_range": [float(df["trip_duration_min"].min()), float(df["trip_duration_min"].max())],
    }, f, indent=2)

print(f"\nModel saved to {MODEL_FILE}")
print(f"Charts saved in the '{PLOTS_DIR}' folder")
