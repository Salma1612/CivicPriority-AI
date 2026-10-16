"""
ml_train.py  -  CivicPriority AI: resolution-time regression (training + evaluation).

Run from the repository root:   python ml_train.py
Optional:                        python ml_train.py --resolved-only

Outputs (read by ml_app.py):
    models/resolution_model.joblib   {"model", "feature_columns", "target", "target_unit", ...}
    models/model_metrics.json        evaluation metrics (all numbers come from this run)
    models/test_predictions.csv      held-out actual vs predicted values
    models/feature_importance.csv    permutation importance on the held-out set

Evaluation protocol
    1. 80/20 random split (random_state=42). The test set is touched exactly once, at the end.
    2. Candidate models and hyperparameters are compared ONLY with 5-fold CV on the training split.
    3. The best *learned* candidate (lowest CV MAE) is evaluated once on the test set and
       compared with median and mean baselines. If it does not beat the baseline, the
       metrics file says so.

Fixes relative to the original train_model.py:
    * 'pedestrain_activity' typo -> 'pedestrian_activity' (the old script raised KeyError).
    * near_school / near_hospital are 0/1 flags and stay numeric; pedestrian_activity
      (Low/Medium/High) is categorical, matching the CSV.
"""
import argparse, json, warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

CATEGORICAL = ["severity", "incident_type", "zone", "road_type", "road_condition", "pedestrian_activity"]
NUMERIC = ["injuries", "population_density_per_km2", "near_school", "near_hospital"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "resolution_hours"
SEED = 42


def find_csv(name):
    for folder in (BASE_DIR / "data", BASE_DIR):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"{name} not found in repository root or data/.")


def preprocessing():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def make_pipeline(regressor):
    return Pipeline([("preprocessing", preprocessing()), ("regressor", regressor)])


def metrics_of(y_true, y_pred):
    return {"mae_hours": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse_hours": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)}


def main(resolved_only=False):
    MODEL_DIR.mkdir(exist_ok=True)
    incidents = pd.read_csv(find_csv("incidents.csv"))
    locations = pd.read_csv(find_csv("locations.csv"))
    data = incidents.merge(locations, on="location_id", how="left", validate="many_to_one")
    if resolved_only:
        data = data[data["resolved"] == 1]
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data[data[TARGET].notna() & (data[TARGET] >= 0)].copy()
    for c in NUMERIC:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    if len(data) < 100:
        raise ValueError("Not enough usable records to train and evaluate.")

    X, y = data[FEATURES], data[TARGET].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
    print(f"Usable: {len(data)} | train: {len(X_tr)} | test: {len(X_te)}")

    # ---- Model selection: training-only cross-validation -------------------------------
    candidates = {
        "Median baseline (DummyRegressor)": make_pipeline(DummyRegressor(strategy="median")),
        "Ridge": make_pipeline(Ridge(alpha=10.0)),
        "RandomForestRegressor (default, as in train_model.py)": make_pipeline(
            RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
        "HistGradientBoostingRegressor": make_pipeline(
            HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, random_state=SEED)),
    }
    cv_rows = []
    for name, pipe in candidates.items():
        s = -cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        cv_rows.append({"model": name, "cv_mae_hours": round(float(s.mean()), 4),
                        "cv_mae_std": round(float(s.std()), 4), "tuned": False})
        print(f"CV MAE {name}: {s.mean():.3f} +/- {s.std():.3f}")

    grid = GridSearchCV(
        make_pipeline(RandomForestRegressor(n_estimators=250, random_state=SEED, n_jobs=-1)),
        {"regressor__min_samples_leaf": [5, 20, 50], "regressor__max_depth": [4, 8, None]},
        cv=5, scoring="neg_mean_absolute_error", n_jobs=1)
    grid.fit(X_tr, y_tr)
    tuned_name = "RandomForestRegressor (tuned, CV grid)"
    cv_rows.append({"model": tuned_name, "cv_mae_hours": round(float(-grid.best_score_), 4),
                    "cv_mae_std": None, "tuned": True,
                    "params": {k.replace("regressor__", ""): v for k, v in grid.best_params_.items()}})
    print(f"CV MAE {tuned_name}: {-grid.best_score_:.3f}  {grid.best_params_}")

    learned = [r for r in cv_rows if not r["model"].startswith("Median baseline")]
    best = min(learned, key=lambda r: r["cv_mae_hours"])
    best_name = best["model"]
    final = grid.best_estimator_ if best_name == tuned_name else candidates[best_name]
    final.fit(X_tr, y_tr)

    # ---- Single final evaluation on the untouched test set ----------------------------
    pred = np.clip(final.predict(X_te), 0, None)
    model_m = metrics_of(y_te, pred)
    med_m = metrics_of(y_te, np.full(len(y_te), y_tr.median()))
    mean_m = metrics_of(y_te, np.full(len(y_te), y_tr.mean()))
    beats_mae = model_m["mae_hours"] < med_m["mae_hours"]
    beats_rmse = model_m["rmse_hours"] < min(med_m["rmse_hours"], mean_m["rmse_hours"])

    # Original RF configuration, reported for continuity with earlier results.
    orig = candidates["RandomForestRegressor (default, as in train_model.py)"].fit(X_tr, y_tr)
    orig_m = metrics_of(y_te, np.clip(orig.predict(X_te), 0, None))

    imp = permutation_importance(final, X_te, y_te, n_repeats=5, random_state=SEED,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
    pd.DataFrame({"feature": FEATURES, "importance_mae_increase_hours": imp.importances_mean,
                  "std": imp.importances_std}).sort_values(
        "importance_mae_increase_hours", ascending=False).to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    pd.DataFrame({"actual_hours": y_te.values, "predicted_hours": pred}).to_csv(
        MODEL_DIR / "test_predictions.csv", index=False)

    metrics = {
        "project": "CivicPriority AI", "task": "Incident resolution time prediction",
        "model": best_name, "regressor_class": type(final.named_steps["regressor"]).__name__,
        "target": TARGET, "target_unit": "hours",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "training_subset": "resolved incidents only" if resolved_only else "all incidents with a valid resolution_hours",
        "evaluation": {"method": "Random 80/20 split; model selection by 5-fold CV on training split only",
                       "test_size": 0.20, "random_state": SEED,
                       "training_records": int(len(X_tr)), "testing_records": int(len(X_te)),
                       "usable_records": int(len(data))},
        "selected_model": {**model_m, "cv_mae_hours": best["cv_mae_hours"], "params": best.get("params")},
        "random_forest": orig_m,   # original RF configuration (kept for compatibility)
        "median_baseline": med_m, "mean_baseline": mean_m,
        "beats_median_baseline_mae": bool(beats_mae), "beats_baselines_rmse": bool(beats_rmse),
        "cv_results": cv_rows, "features": FEATURES,
        "note": "Dataset is synthetic; these metrics do not establish real-world performance.",
    }
    (MODEL_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"model": final, "feature_columns": FEATURES, "target": TARGET, "target_unit": "hours",
                 "model_name": best_name, "sklearn_version": sklearn.__version__},
                MODEL_DIR / "resolution_model.joblib", compress=3)

    print("\n--- Held-out test results (hours) ---")
    print(f"Selected ({best_name}): MAE {model_m['mae_hours']:.2f}  RMSE {model_m['rmse_hours']:.2f}")
    print(f"Median baseline       : MAE {med_m['mae_hours']:.2f}  RMSE {med_m['rmse_hours']:.2f}")
    print(f"Mean baseline         : MAE {mean_m['mae_hours']:.2f}  RMSE {mean_m['rmse_hours']:.2f}")
    print("Beats median baseline on MAE." if beats_mae else "Does NOT beat the median baseline on MAE.")
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved-only", action="store_true", help="train only on incidents with resolved == 1")
    main(ap.parse_args().resolved_only)
