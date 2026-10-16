# CivicPriority AI - ML Resolution-Time App

A separate Streamlit application that predicts incident **resolution time (hours)** with a scikit-learn regression model.
It does not modify the baseline `app.py` (weighted priority heuristic). The weighted score is **not** an ML prediction and is not used here.

## Files (all in the repository root, next to `app.py`)

```
app.py                      <- baseline, UNCHANGED
ml_app.py                   <- NEW: ML app entry point
ml_components.py            <- NEW: styling, KPI cards, chart defaults
ml_utils.py                 <- NEW: data/model loading, validation, prediction
ml_train.py                 <- NEW: training + evaluation (writes models/)
requirements.txt            <- REPLACE with the version in this package (superset of the old one)
README_ML.md                <- NEW
incidents.csv, locations.csv, public_complaints.csv   <- existing, required
models/                     <- created by `python ml_train.py`; commit it
    resolution_model.joblib
    model_metrics.json
    test_predictions.csv
    feature_importance.csv
```
`train_model.py` and `predict.py` are left as they are. See "Known issues" below before using them.

## Architecture

`ml_train.py` (offline) -> `models/*` artifacts -> `ml_app.py` (read-only).
The app never trains. It loads the saved pipeline (preprocessing + estimator in one object), validates that its feature list matches
what the form supplies, and shows every metric from `model_metrics.json`. To change the model, re-run `ml_train.py` and replace the
files in `models/`; no UI changes are needed.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python ml_train.py                 # about a minute; creates models/
streamlit run ml_app.py
```

Optional: `python ml_train.py --resolved-only` trains only on incidents with `resolved == 1` (see Known issues).

### Verify

1. **Overview**: KPIs show 10,000 incidents; the model-status panel names the model and shows the baseline comparison.
2. **Predict**: press *Predict Resolution Time* with defaults; an hours value appears with the "not a guarantee" notice. Press *Reset form*; the result clears.
3. **Missing-model behaviour**: temporarily rename `models/`; the Predict page must say prediction is unavailable and show no number. Rename it back.
4. **Explorer**: filter by severity, check the match count changes, download the CSV.
5. **Analytics / Performance / Location / Info**: each page loads without errors; Performance shows MAE, RMSE, CV MAE, set sizes and the baseline comparison.

## Deploy as a separate Streamlit Community Cloud app

1. Add the new files to the **same GitHub repository** (upload via the GitHub web UI), including the `models/` folder generated above, and replace `requirements.txt`. Do not edit `app.py`.
2. In Streamlit Community Cloud choose **Create app** -> *Deploy a public app from GitHub*.
3. Repository: `Salma1612/CivicPriority-AI`; Branch: `main`; **Main file path: `ml_app.py`**; choose a different subdomain from the baseline.
4. Deploy. The existing app at `civicpriority-ai.streamlit.app` keeps running `app.py`.
5. Both apps read the same root `requirements.txt`; the new file only adds packages, so the baseline is unaffected.

**Version match:** the model file is a pickle. Train with the same scikit-learn major/minor version Streamlit Cloud installs
(the metrics file records `sklearn_version`, and the Model Performance page shows it). If the app says the model "could not be loaded",
pin `scikit-learn==<version used for training>` in `requirements.txt` or re-train in an environment with the deployed version.

**Large model file:** GitHub rejects files over 100 MB. The default model is well under 1 MB. If you later train a larger one:
raise `min_samples_leaf`, lower `n_estimators` in `ml_train.py`, keep `compress=3`, or use Git LFS.

## Evaluation protocol

- 80/20 random split, `random_state=42`.
- Candidate models and hyperparameters are compared **only** with 5-fold CV on the training split.
- The test set is evaluated once for the selected model, a median baseline and a mean baseline.
- The app states plainly when the model does not beat the baseline.

## Known issues in the existing files

- `train_model.py` raises `KeyError` because `categorical_features` contains `"pedestrain_activity"` (typo). `ml_train.py` uses `pedestrian_activity`.
- `predict.py`'s sample passes `"pedestrian_activity": 3`, but the data uses `Low` / `Medium` / `High`.
- `incidents.csv` has 2,874 incidents with `resolved = 0` that still have a `resolution_hours` value. Confirm what that value means; use `--resolved-only` if it is not a real resolution time.
- Coordinates in `locations.csv` are simulated; the map page says so.
