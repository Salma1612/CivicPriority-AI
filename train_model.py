# train_model.py
# CivicPriority AI - Incident Resolution Time Prediction

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Support CSVs in either the repository root or a data/ folder
def find_csv(filename):
    for folder in (DATA_DIR, BASE_DIR):
        path = folder / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Put it in the repository root "
        "or inside a folder named 'data'."
    )


def main():
    print("Loading CivicPriority AI datasets...")

    incidents = pd.read_csv(find_csv("incidents.csv"))
    locations = pd.read_csv(find_csv("locations.csv"))

    required_incident_cols = {
        "location_id", "resolution_hours", "severity",
        "incident_type", "injuries"
    }
    required_location_cols = {
        "location_id", "zone", "road_type", "road_condition",
        "population_density_per_km2", "near_school",
        "near_hospital", "pedestrian_activity"
    }

    missing_incident = required_incident_cols - set(incidents.columns)
    missing_location = required_location_cols - set(locations.columns)

    if missing_incident:
        raise ValueError(
            f"Missing columns in incidents.csv: {sorted(missing_incident)}"
        )
    if missing_location:
        raise ValueError(
            f"Missing columns in locations.csv: {sorted(missing_location)}"
        )

    # Each incident receives contextual features from its location.
    data = incidents.merge(
        locations,
        on="location_id",
        how="left",
        validate="many_to_one",
        suffixes=("_incident", "_location")
    )

    # The target must be known and non-negative.
    data["resolution_hours"] = pd.to_numeric(
        data["resolution_hours"], errors="coerce"
    )
    data = data[
        data["resolution_hours"].notna()
        & (data["resolution_hours"] >= 0)
    ].copy()

    if len(data) < 20:
        raise ValueError(
            "Not enough incidents with valid resolution_hours "
            "to train and evaluate a meaningful model."
        )

    # Only use features that could plausibly be known when an incident
    # is reported. Do not include resolution_hours, resolved, or
    # other post-resolution outcome fields as input features.
    categorical_features = [
        "severity",
        "incident_type",
        "zone",
        "road_type",
        "road_condition",
    ]

    numeric_features = [
        "injuries",
        "population_density_per_km2",
        "near_school",
        "near_hospital",
        "pedestrian_activity",
    ]

    feature_columns = categorical_features + numeric_features

    # Convert numeric inputs safely.
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    X = data[feature_columns].copy()
    y = data["resolution_hours"].astype(float)

    # Hold out test records so evaluation uses unseen examples.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print(f"Usable records: {len(data)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print("Training Random Forest model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Baseline: always predict the median resolution time from training data.
    baseline_prediction = np.full(
        len(y_test), y_train.median(), dtype=float
    )

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    baseline_mae = mean_absolute_error(y_test, baseline_prediction)
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_prediction)
    )

    metrics = {
        "project": "CivicPriority AI",
        "task": "Incident resolution time prediction",
        "model": "RandomForestRegressor",
        "target": "resolution_hours",
        "target_unit": "hours",
        "evaluation": {
            "method": "Random train/test split",
            "test_size": 0.20,
            "random_state": 42,
            "training_records": int(len(X_train)),
            "testing_records": int(len(X_test)),
            "usable_records": int(len(data)),
        },
        "random_forest": {
            "mae_hours": round(float(model_mae), 4),
            "rmse_hours": round(float(model_rmse), 4),
        },
        "median_baseline": {
            "mae_hours": round(float(baseline_mae), 4),
            "rmse_hours": round(float(baseline_rmse), 4),
        },
        "features": feature_columns,
        "note": (
            "Evaluation uses the available dataset. If the data is synthetic, "
            "these metrics do not establish real-world performance."
        ),
    }

    model_path = MODEL_DIR / "resolution_model.joblib"
    metrics_path = MODEL_DIR / "model_metrics.json"

    # Save the full preprocessing + model pipeline.
    joblib.dump({
        "model": model,
        "feature_columns": feature_columns,
        "target": "resolution_hours",
        "target_unit": "hours",
    }, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n--- Evaluation Results (hours) ---")
    print(f"Random Forest MAE:  {model_mae:.2f}")
    print(f"Random Forest RMSE: {model_rmse:.2f}")
    print(f"Median baseline MAE:  {baseline_mae:.2f}")
    print(f"Median baseline RMSE: {baseline_rmse:.2f}")

    if model_mae < baseline_mae:
        print("The model improves on the median baseline by MAE.")
    else:
        print("The model does not beat the median baseline by MAE.")

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
