"""ml_utils.py - data loading, model loading, validation and prediction helpers (no UI code)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "resolution_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

CATEGORICAL = ["severity", "incident_type", "zone", "road_type", "road_condition", "pedestrian_activity"]
NUMERIC = ["injuries", "population_density_per_km2", "near_school", "near_hospital"]
EXPECTED_FEATURES = CATEGORICAL + NUMERIC

# Natural ordering for ordinal labels; labels not listed here are appended alphabetically.
ORDER = {
    "severity": ["Low", "Medium", "High", "Critical"],
    "road_condition": ["Good", "Fair", "Poor", "Critical"],
    "pedestrian_activity": ["Low", "Medium", "High"],
}
LABELS = {
    "severity": "Severity", "incident_type": "Incident type", "zone": "Zone", "road_type": "Road type",
    "road_condition": "Road condition", "pedestrian_activity": "Pedestrian activity",
    "injuries": "Number of injuries", "population_density_per_km2": "Population density (per km²)",
    "near_school": "Near a school", "near_hospital": "Near a hospital",
}


def find_csv(name: str) -> Path:
    for folder in (BASE_DIR / "data", BASE_DIR):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"{name} was not found in the repository root or a data/ folder.")


@st.cache_data(show_spinner=False)
def load_data():
    """Returns (incidents merged with location attributes, locations, complaints)."""
    incidents = pd.read_csv(find_csv("incidents.csv"))
    locations = pd.read_csv(find_csv("locations.csv"))
    try:
        complaints = pd.read_csv(find_csv("public_complaints.csv"))
    except FileNotFoundError:
        complaints = pd.DataFrame()
    merged = incidents.merge(locations, on="location_id", how="left", validate="many_to_one")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged["month"] = merged["timestamp"].dt.to_period("M").astype(str)
    return merged, locations, complaints


def ordered(column: str, values) -> list:
    vals = [v for v in pd.Series(values).dropna().unique()]
    first = [v for v in ORDER.get(column, []) if v in vals]
    rest = sorted([v for v in vals if v not in first], key=str)
    return first + rest


def category_options(df: pd.DataFrame) -> dict:
    return {c: ordered(c, df[c]) for c in CATEGORICAL if c in df.columns}


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    """Returns (bundle, error_message). Never trains. Verifies the artifact before use."""
    if not MODEL_PATH.exists():
        return None, ("Model file not found at models/resolution_model.joblib. "
                      "Run `python ml_train.py` and commit the generated models/ folder.")
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:  # incompatible sklearn version, corrupt file, ...
        return None, f"The model file could not be loaded ({type(exc).__name__}: {exc}). Re-train with the same scikit-learn version as the app."
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        return None, "The model file has an unexpected structure (needs 'model' and 'feature_columns')."
    if set(bundle["feature_columns"]) != set(EXPECTED_FEATURES):
        return None, ("The model was trained on different features than this app supplies: "
                      f"{sorted(set(bundle['feature_columns']) ^ set(EXPECTED_FEATURES))}")
    return bundle, None


@st.cache_data(show_spinner=False)
def load_metrics():
    """Returns metrics dict or None."""
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def load_optional_csv(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def validate_input(values: dict, options: dict) -> list:
    """Returns a list of human-readable problems (empty list = valid)."""
    problems = []
    for col in CATEGORICAL:
        if values.get(col) not in options.get(col, []):
            problems.append(f"{LABELS[col]}: choose one of the listed values.")
    inj = values.get("injuries")
    if inj is None or inj < 0 or int(inj) != inj:
        problems.append("Number of injuries must be a whole number of 0 or more.")
    dens = values.get("population_density_per_km2")
    if dens is None or dens < 0:
        problems.append("Population density must be 0 or more.")
    for col in ("near_school", "near_hospital"):
        if values.get(col) not in (0, 1):
            problems.append(f"{LABELS[col]}: choose Yes or No.")
    return problems


def predict_hours(bundle: dict, values: dict) -> float:
    """Builds a one-row frame with the exact training columns and returns predicted hours."""
    cols = list(bundle["feature_columns"])
    row = pd.DataFrame([{c: values[c] for c in cols}])[cols]
    for c in NUMERIC:
        if c in row:
            row[c] = pd.to_numeric(row[c])
    return float(max(0.0, bundle["model"].predict(row)[0]))


def hours_to_text(h: float) -> str:
    d, rem = divmod(h, 24)
    return f"{int(d)} d {rem:.1f} h" if d >= 1 else f"{h:.1f} h"


def verdict(metrics: dict):
    """Honest comparison of the selected model with baselines, derived only from the metrics file.
    Returns (kind, headline, detail) where kind is 'good' | 'bad' | 'mixed' | 'unknown'."""
    if not metrics:
        return "unknown", "No evaluation metrics available", "Run ml_train.py to generate models/model_metrics.json."
    m = metrics.get("selected_model") or metrics.get("random_forest")
    med, mean = metrics.get("median_baseline"), metrics.get("mean_baseline")
    if not (m and med):
        return "unknown", "Metrics incomplete", "The metrics file does not include a baseline comparison."
    d_mae = med["mae_hours"] - m["mae_hours"]   # positive = model better
    best_rmse = min(med["rmse_hours"], mean["rmse_hours"]) if mean else med["rmse_hours"]
    d_rmse = best_rmse - m["rmse_hours"]
    detail = (f"MAE {m['mae_hours']:.2f} h vs median baseline {med['mae_hours']:.2f} h "
              f"({d_mae:+.2f} h). RMSE {m['rmse_hours']:.2f} h vs best baseline {best_rmse:.2f} h ({d_rmse:+.2f} h).")
    if d_mae > 0 and d_rmse > 0:
        return "good", "Model beats the baselines on held-out data", detail
    if d_mae <= 0 and d_rmse <= 0:
        return "bad", "Model does not beat the baseline", detail
    return "mixed", "Mixed result: model does not clearly beat the baseline", detail
