# predict.py
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resolution_model.joblib"


def predict_resolution_time(incident):
    """
    incident: dictionary containing the model's input features.
    Returns the predicted resolution time in hours.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python train_model.py"
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    row = pd.DataFrame([incident])

    missing = [col for col in feature_columns if col not in row.columns]
    if missing:
        raise ValueError(f"Missing input features: {missing}")

    row = row[feature_columns]
    prediction = float(model.predict(row)[0])

    return max(0.0, prediction)


if __name__ == "__main__":
    sample_incident = {
        "severity": "High",
        "incident_type": "Road Damage",
        "zone": "Zone A",
        "road_type": "Urban",
        "road_condition": "Poor",
        "injuries": 0,
        "population_density_per_km2": 5000,
        "near_school": 1,
        "near_hospital": 0,
        "pedestrian_activity": 3,
    }

    hours = predict_resolution_time(sample_incident)
    print(f"Estimated resolution time: {hours:.2f} hours")
