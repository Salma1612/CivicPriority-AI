"""
ml_app.py - CivicPriority AI: ML Resolution-Time Platform (separate from the baseline app.py).
Run:  streamlit run ml_app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ml_utils as U
from ml_components import (BLUE, NAVY, SEVERITY_COLORS, SLATE, TEAL, inject_css, kpi_row, note,
                           page_header, plot)

st.set_page_config(page_title="CivicPriority AI | ML Resolution Time", page_icon=None, layout="wide")
inject_css()

# ------------------------------------------------------------------ data & model (cached)
try:
    DATA, LOCATIONS, COMPLAINTS = U.load_data()
except FileNotFoundError as exc:
    st.error(f"Data files missing: {exc} Place incidents.csv and locations.csv in the repository root or a data/ folder.")
    st.stop()
OPTIONS = U.category_options(DATA)
BUNDLE, MODEL_ERROR = U.load_model_bundle()
METRICS = U.load_metrics()
SEV_ORDER = U.ORDER["severity"]

# ------------------------------------------------------------------ sidebar
PAGES = ["ML Overview", "Predict Resolution Time", "Incident Explorer", "Resolution Analytics",
         "Model Performance", "Location Insights", "Data & Model Information"]
with st.sidebar:
    st.markdown("### CivicPriority AI")
    st.caption("Resolution-time prediction (ML)")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    if BUNDLE is not None:
        st.caption(f"Model loaded: {BUNDLE.get('model_name', type(BUNDLE['model'].named_steps['regressor']).__name__)}")
    else:
        st.caption("Model: not available")
    st.caption("Synthetic prototype data. Not for operational decisions.")


def model_label():
    if METRICS and METRICS.get("model"):
        return METRICS["model"]
    if BUNDLE is not None:
        return BUNDLE.get("model_name", "Unknown")
    return "Not trained"


# =================================================================== 1. OVERVIEW
def page_overview():
    page_header("ML Overview", "Historical incident data and the status of the resolution-time regression model.")
    h = DATA["resolution_hours"]
    kpi_row([
        ("Incidents", f"{len(DATA):,}", f"{DATA['timestamp'].min():%b %Y} to {DATA['timestamp'].max():%b %Y}"),
        ("Incident types", str(DATA["incident_type"].nunique()), f"{DATA['zone'].nunique()} zones"),
        ("Mean resolution", f"{h.mean():.1f} h", "historical"),
        ("Median resolution", f"{h.median():.1f} h", "historical"),
        ("Resolved flag", f"{(DATA['resolved'] == 1).mean():.0%}", "incidents with resolved = 1"),
    ])
    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.histogram(DATA, x="resolution_hours", nbins=50, title="Resolution time distribution (historical)",
                           labels={"resolution_hours": "Resolution time (hours)"})
        fig.add_vline(x=h.median(), line_dash="dash", line_color=TEAL, annotation_text=f"median {h.median():.1f} h")
        fig.update_layout(yaxis_title="Incidents")
        plot(fig, 340)
    with c2:
        sev = DATA["severity"].value_counts().reindex(SEV_ORDER).dropna().reset_index()
        sev.columns = ["severity", "count"]
        fig = px.bar(sev, x="severity", y="count", color="severity", color_discrete_map=SEVERITY_COLORS,
                     title="Severity distribution (historical)")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Incidents")
        plot(fig, 340)

    st.markdown("#### Model status")
    kind, headline, detail = U.verdict(METRICS)
    if BUNDLE is None:
        note(f"<b>Prediction unavailable.</b> {MODEL_ERROR}", "warn")
    cA, cB = st.columns([2, 3])
    with cA:
        st.markdown(f'<div class="panel"><b>Model:</b> {model_label()}<br><b>Target:</b> resolution_hours (hours)<br>'
                    f'<b>Status:</b> {"Trained and loaded" if BUNDLE is not None else "Not available"}<br>'
                    f'<b>Evaluated:</b> {METRICS["trained_at_utc"][:10] if METRICS else "no metrics file"}</div>',
                    unsafe_allow_html=True)
        st.write("")
        note(f"<b>{headline}.</b><br>{detail}", {"good": "good", "bad": "bad", "mixed": "mixed"}.get(kind, "warn"))
    with cB:
        comparison_chart()


def comparison_chart(height=300):
    if not METRICS:
        st.info("No evaluation metrics found. Run `python ml_train.py` to create models/model_metrics.json.")
        return
    m = METRICS.get("selected_model") or METRICS.get("random_forest")
    rows = [("Trained model", m)] + [(n, METRICS[k]) for n, k in
                                     (("Median baseline", "median_baseline"), ("Mean baseline", "mean_baseline")) if k in METRICS]
    df = pd.DataFrame([{"Model": n, "MAE": v["mae_hours"], "RMSE": v["rmse_hours"]} for n, v in rows])
    long = df.melt("Model", var_name="Metric", value_name="Hours")
    fig = px.bar(long, x="Metric", y="Hours", color="Model", barmode="group", text_auto=".2f",
                 title="Held-out test error (lower is better)",
                 color_discrete_sequence=[BLUE, SLATE, "#9FB3C8"])
    fig.update_layout(yaxis_title="Hours", xaxis_title="", legend_title_text="", legend_y=-0.12)
    plot(fig, height)


# =================================================================== 2. PREDICT
DEFAULTS = {}


def _defaults():
    d = {"severity": "Medium", "incident_type": OPTIONS["incident_type"][0], "zone": OPTIONS["zone"][0],
         "road_type": OPTIONS["road_type"][0], "road_condition": "Fair", "pedestrian_activity": "Medium"}
    for k, v in list(d.items()):
        if v not in OPTIONS[k]:
            d[k] = OPTIONS[k][0]
    d.update({"injuries": 0, "population_density_per_km2": int(DATA["population_density_per_km2"].median()),
              "near_school": "No", "near_hospital": "No"})
    return d


def reset_form():
    for k, v in _defaults().items():
        st.session_state[f"f_{k}"] = v
    st.session_state.pop("last_result", None)


def page_predict():
    page_header("Predict Resolution Time", "Enter incident details to get a model estimate of resolution time in hours.")
    if BUNDLE is None:
        note(f"<b>Prediction is unavailable.</b> {MODEL_ERROR}<br>No substitute value is shown.", "bad")
        return
    d = _defaults()
    for k, v in d.items():
        st.session_state.setdefault(f"f_{k}", v)

    with st.form("predict_form"):
        st.markdown("##### Incident")
        a, b, c = st.columns(3)
        a.selectbox(U.LABELS["severity"], OPTIONS["severity"], key="f_severity")
        b.selectbox(U.LABELS["incident_type"], OPTIONS["incident_type"], key="f_incident_type")
        c.number_input(U.LABELS["injuries"], min_value=0, max_value=50, step=1, key="f_injuries")
        st.markdown("##### Location context")
        a, b, c = st.columns(3)
        a.selectbox(U.LABELS["zone"], OPTIONS["zone"], key="f_zone")
        b.selectbox(U.LABELS["road_type"], OPTIONS["road_type"], key="f_road_type")
        c.selectbox(U.LABELS["road_condition"], OPTIONS["road_condition"], key="f_road_condition")
        a, b, c = st.columns(3)
        a.selectbox(U.LABELS["pedestrian_activity"], OPTIONS["pedestrian_activity"], key="f_pedestrian_activity")
        b.number_input(U.LABELS["population_density_per_km2"], min_value=0, step=100, key="f_population_density_per_km2")
        with c:
            s1, s2 = st.columns(2)
            s1.radio(U.LABELS["near_school"], ["No", "Yes"], horizontal=True, key="f_near_school")
            s2.radio(U.LABELS["near_hospital"], ["No", "Yes"], horizontal=True, key="f_near_hospital")
        b1, b2, _ = st.columns([1.3, 1, 3])
        go_pred = b1.form_submit_button("Predict Resolution Time", type="primary")
        b2.form_submit_button("Reset form", on_click=reset_form)

    if go_pred:
        values = {k: st.session_state[f"f_{k}"] for k in d}
        values["near_school"] = 1 if values["near_school"] == "Yes" else 0
        values["near_hospital"] = 1 if values["near_hospital"] == "Yes" else 0
        problems = U.validate_input(values, OPTIONS)
        if problems:
            st.error("Please fix the following:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            try:
                st.session_state["last_result"] = (values, U.predict_hours(BUNDLE, values))
            except Exception as exc:
                st.session_state.pop("last_result", None)
                st.error(f"The model could not produce a prediction ({type(exc).__name__}). Check that the model file matches this app's scikit-learn version.")
                st.caption(str(exc))

    if "last_result" in st.session_state:
        values, hours = st.session_state["last_result"]
        left, right = st.columns([2, 3])
        with left:
            st.markdown(f'<div class="result"><div class="l" style="color:{SLATE};font-size:.75rem;text-transform:uppercase;'
                        f'letter-spacing:.06em;font-weight:600">Estimated resolution time</div>'
                        f'<div class="big">{hours:.1f} hours</div>'
                        f'<div style="color:{SLATE}">{U.hours_to_text(hours)}</div></div>', unsafe_allow_html=True)
        with right:
            peers = DATA[(DATA["incident_type"] == values["incident_type"]) & (DATA["severity"] == values["severity"])]["resolution_hours"]
            st.markdown("**Historical context (not a prediction)**")
            if len(peers) >= 5:
                st.write(f"{len(peers):,} past incidents of type **{values['incident_type']}** with severity **{values['severity']}**: "
                         f"median {peers.median():.1f} h, middle 50% between {peers.quantile(.25):.1f} h and {peers.quantile(.75):.1f} h.")
            st.write(f"All incidents: median {DATA['resolution_hours'].median():.1f} h, "
                     f"middle 50% between {DATA['resolution_hours'].quantile(.25):.1f} h and {DATA['resolution_hours'].quantile(.75):.1f} h.")
        note("<b>This is a model estimate, not a guaranteed resolution time.</b> The model was trained on synthetic data.", "warn")
        kind, headline, detail = U.verdict(METRICS)
        if kind in ("bad", "mixed"):
            note(f"<b>Low confidence.</b> {headline}. {detail} Treat this estimate as no better than the historical median shown above.", "bad")
        elif kind == "unknown":
            note("No evaluation metrics were found, so the quality of this estimate is unknown.", "warn")
        with st.expander("Inputs sent to the model"):
            st.dataframe(pd.DataFrame([values]), hide_index=True, width="stretch")


# =================================================================== 3. EXPLORER
def page_explorer():
    page_header("Incident Explorer", "Search, filter and export historical incident records.")
    c1, c2, c3 = st.columns(3)
    sev = c1.multiselect("Severity", SEV_ORDER, default=[])
    typ = c2.multiselect("Incident type", OPTIONS["incident_type"], default=[])
    zon = c3.multiselect("Zone", OPTIONS["zone"], default=[])
    c4, c5, c6, c7 = st.columns(4)
    rt = c4.multiselect("Road type", OPTIONS["road_type"], default=[])
    rc = c5.multiselect("Road condition", OPTIONS["road_condition"], default=[])
    status = c6.selectbox("Resolved flag", ["All", "Resolved (1)", "Not resolved (0)"])
    q = c7.text_input("Search incident or location ID", placeholder="e.g. I000123 or L0116")
    dmin, dmax = DATA["timestamp"].min().date(), DATA["timestamp"].max().date()
    dr = st.date_input("Reported between", (dmin, dmax), min_value=dmin, max_value=dmax)
    hmax = float(DATA["resolution_hours"].max())
    hr = st.slider("Resolution time (hours)", 0.0, hmax, (0.0, hmax))

    f = DATA
    for col, sel in (("severity", sev), ("incident_type", typ), ("zone", zon), ("road_type", rt), ("road_condition", rc)):
        if sel:
            f = f[f[col].isin(sel)]
    if status != "All":
        f = f[f["resolved"] == (1 if status.startswith("Resolved") else 0)]
    if q.strip():
        f = f[f["incident_id"].str.contains(q.strip(), case=False, na=False) | f["location_id"].str.contains(q.strip(), case=False, na=False)]
    if isinstance(dr, tuple) and len(dr) == 2:
        f = f[(f["timestamp"].dt.date >= dr[0]) & (f["timestamp"].dt.date <= dr[1])]
    f = f[f["resolution_hours"].between(*hr)]

    if f.empty:
        st.info("No incidents match these filters. Clear a filter or widen the date and hour ranges.")
        return
    kpi_row([("Matching incidents", f"{len(f):,}", f"{len(f) / len(DATA):.0%} of all"),
             ("Mean resolution", f"{f['resolution_hours'].mean():.1f} h", ""),
             ("Median resolution", f"{f['resolution_hours'].median():.1f} h", ""),
             ("90th percentile", f"{f['resolution_hours'].quantile(.9):.1f} h", ""),
             ("Total injuries", f"{int(f['injuries'].sum()):,}", "")])
    st.write("")
    cols = ["incident_id", "timestamp", "incident_type", "severity", "injuries", "zone", "road_type", "road_condition",
            "pedestrian_activity", "resolved", "resolution_hours"]
    st.dataframe(f[cols].sort_values("timestamp", ascending=False), hide_index=True, width="stretch", height=420)
    st.download_button("Download filtered records (CSV)", f[cols].to_csv(index=False).encode("utf-8"),
                       "filtered_incidents.csv", "text/csv")


# =================================================================== 4. ANALYTICS
def page_analytics():
    page_header("Resolution Analytics", "Historical observations only. Charts on this page do not use model predictions.")
    note("Patterns shown are associations in the dataset and do not establish cause and effect.")
    t1, t2, t3, t4 = st.tabs(["Distributions", "By category", "Numeric relationships", "Long-resolution incidents"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            plot(px.histogram(DATA, x="resolution_hours", nbins=60, title="Resolution time (hours)",
                              labels={"resolution_hours": "Hours"}), 340)
        with c2:
            plot(px.box(DATA, x="resolved", y="resolution_hours", title="Resolution time by resolved flag",
                        labels={"resolved": "resolved (0/1)", "resolution_hours": "Hours"}), 340)
        st.caption("Incidents flagged as not resolved (resolved = 0) still carry a resolution_hours value in this dataset; "
                   "how that value was generated should be confirmed before relying on it.")
    with t2:
        dim = st.selectbox("Group by", ["severity", "incident_type", "zone", "road_type", "road_condition", "pedestrian_activity"])
        order = U.ordered(dim, DATA[dim]) if dim != "zone" else sorted(DATA[dim].unique())
        plot(px.box(DATA, x=dim, y="resolution_hours", category_orders={dim: order},
                    title=f"Resolution time by {U.LABELS[dim].lower()}", labels={"resolution_hours": "Hours"}), 400)
        g = DATA.groupby(dim)["resolution_hours"].agg(["count", "mean", "median"]).reindex(order).round(1)
        g.columns = ["Incidents", "Mean (h)", "Median (h)"]
        st.dataframe(g, width="stretch")
    with t3:
        num = st.selectbox("Numeric feature", ["population_density_per_km2", "injuries"])
        if num == "injuries":
            g = DATA.groupby("injuries")["resolution_hours"].agg(["count", "mean", "median"]).reset_index()
            fig = px.bar(g, x="injuries", y="mean", text=g["count"].map(lambda n: f"n={n}"),
                         title="Mean resolution time by injuries", labels={"mean": "Mean hours"})
            plot(fig, 360)
        else:
            s = DATA.sample(min(3000, len(DATA)), random_state=1)
            fig = px.scatter(s, x=num, y="resolution_hours", opacity=.35, trendline="ols" if _has_statsmodels() else None,
                             title="Population density vs resolution time (sample of up to 3,000 incidents)",
                             labels={num: U.LABELS[num], "resolution_hours": "Hours"})
            plot(fig, 400)
        r = DATA[[num, "resolution_hours"]].corr().iloc[0, 1]
        st.write(f"Pearson correlation with resolution time: **{r:+.3f}** (a value near 0 means little linear relationship).")
    with t4:
        q1, q3 = DATA["resolution_hours"].quantile([.25, .75])
        cut = q3 + 1.5 * (q3 - q1)
        out = DATA[DATA["resolution_hours"] > cut]
        st.write(f"Outlier rule: above Q3 + 1.5 x IQR = **{cut:.1f} h**. {len(out):,} incidents ({len(out) / len(DATA):.1%}) exceed it.")
        n = st.slider("Show longest N incidents", 10, 200, 25, 5)
        cols = ["incident_id", "timestamp", "incident_type", "severity", "zone", "road_condition", "injuries", "resolution_hours"]
        st.dataframe(DATA.nlargest(n, "resolution_hours")[cols], hide_index=True, width="stretch")


def _has_statsmodels():
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


# =================================================================== 5. MODEL PERFORMANCE
def page_performance():
    page_header("Model Performance & Evaluation", "All figures are read from models/model_metrics.json and the saved test predictions.")
    if not METRICS:
        note("<b>No evaluation metrics found.</b> Run <code>python ml_train.py</code> and commit models/model_metrics.json.", "warn")
        return
    kind, headline, detail = U.verdict(METRICS)
    note(f"<b>{headline}.</b><br>{detail}", {"good": "good", "bad": "bad", "mixed": "mixed"}.get(kind, "warn"))
    m = METRICS.get("selected_model") or METRICS["random_forest"]
    ev = METRICS.get("evaluation", {})
    kpi_row([("MAE", f"{m['mae_hours']:.2f} h", "mean absolute error"), ("RMSE", f"{m['rmse_hours']:.2f} h", "root mean squared error"),
             ("CV MAE", f"{m['cv_mae_hours']:.2f} h" if m.get("cv_mae_hours") else "n/a", "training-only 5-fold"),
             ("Training set", f"{ev.get('training_records', 0):,}", "incidents"), ("Test set", f"{ev.get('testing_records', 0):,}", "incidents")])
    st.write("")
    st.caption(f"Model: {METRICS.get('model')} | Trained: {METRICS.get('trained_at_utc', 'n/a')} | "
               f"scikit-learn {METRICS.get('sklearn_version', 'n/a')} | Data: {METRICS.get('training_subset', 'n/a')} | "
               f"Protocol: {ev.get('method', 'n/a')}")
    c1, c2 = st.columns(2)
    with c1:
        comparison_chart(340)
    with c2:
        cv = pd.DataFrame(METRICS.get("cv_results", []))
        if not cv.empty:
            fig = px.bar(cv.sort_values("cv_mae_hours"), x="cv_mae_hours", y="model", orientation="h", text_auto=".2f",
                         title="Cross-validated MAE on training data (model selection)", labels={"cv_mae_hours": "Hours", "model": ""})
            fig.update_yaxes(autorange="reversed", automargin=True)
            fig.update_layout(margin=dict(l=10, r=20, t=48, b=44))
            plot(fig, 340)
    preds = U.load_optional_csv(str(U.PREDICTIONS_PATH))
    if preds is not None and {"actual_hours", "predicted_hours"} <= set(preds.columns):
        preds = preds.assign(error=preds["predicted_hours"] - preds["actual_hours"])
        c1, c2 = st.columns(2)
        with c1:
            s = preds.sample(min(1500, len(preds)), random_state=1)
            fig = px.scatter(s, x="actual_hours", y="predicted_hours", opacity=.4, title="Predicted vs actual (test set)",
                             labels={"actual_hours": "Actual (h)", "predicted_hours": "Predicted (h)"})
            mx = float(max(preds["actual_hours"].max(), preds["predicted_hours"].max()))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect prediction", line=dict(dash="dash", color=TEAL)))
            plot(fig, 360)
        with c2:
            plot(px.histogram(preds, x="error", nbins=50, title="Residuals: predicted minus actual (test set)",
                              labels={"error": "Error (h)"}), 360)
        st.caption("Predictions that cluster in a narrow band regardless of the actual value indicate the model is mostly predicting a typical value.")
    imp = U.load_optional_csv(str(U.IMPORTANCE_PATH))
    if imp is not None:
        fig = px.bar(imp.sort_values("importance_mae_increase_hours"), x="importance_mae_increase_hours", y="feature", orientation="h",
                     title="Permutation importance on test set (increase in MAE when the feature is shuffled)",
                     labels={"importance_mae_increase_hours": "MAE increase (h)", "feature": ""})
        plot(fig, 360)
        st.caption("Values at or below zero mean shuffling the feature did not hurt the model, so it contributed no measurable signal.")
    st.markdown("#### What the metrics mean")
    st.markdown("- **MAE** is the average size of the miss in hours; it treats all misses equally.\n"
                "- **RMSE** squares misses first, so a few very large misses raise it more than many small ones.\n"
                "- **Median / mean baselines** always predict one constant from the training data. A useful model must beat them on unseen data.\n"
                "- **Cross-validation** on the training split was used for model selection; the test set was used once for the final numbers.")


# =================================================================== 6. LOCATION
def page_location():
    page_header("Location & Civic Insights", "Incident activity aggregated by location.")
    note("The latitude and longitude values in locations.csv are simulated for this prototype and do not correspond to real places.", "warn")
    c1, c2, c3 = st.columns(3)
    zon = c1.multiselect("Zone", OPTIONS["zone"], default=[])
    typ = c2.multiselect("Incident type", OPTIONS["incident_type"], default=[])
    sev = c3.multiselect("Severity", SEV_ORDER, default=[])
    f = DATA
    if zon: f = f[f["zone"].isin(zon)]
    if typ: f = f[f["incident_type"].isin(typ)]
    if sev: f = f[f["severity"].isin(sev)]
    if f.empty:
        st.info("No incidents match these filters.")
        return
    rank = {s: i for i, s in enumerate(SEV_ORDER)}
    g = (f.assign(sev_rank=f["severity"].map(rank))
          .groupby(["location_id", "zone", "latitude", "longitude", "road_type", "road_condition"])
          .agg(incidents=("incident_id", "count"), mean_hours=("resolution_hours", "mean"), top=("sev_rank", "max"))
          .reset_index())
    g["highest_severity"] = g["top"].map(lambda r: SEV_ORDER[int(r)])
    g["mean_hours"] = g["mean_hours"].round(1)
    kpi_row([("Locations shown", f"{len(g):,}", ""), ("Incidents", f"{len(f):,}", ""),
             ("Busiest location", f"{int(g['incidents'].max())}", g.loc[g['incidents'].idxmax(), 'location_id']),
             ("Longest mean resolution", f"{g['mean_hours'].max():.1f} h", g.loc[g['mean_hours'].idxmax(), 'location_id'])])
    st.write("")
    mapper = getattr(px, "scatter_map", None) or px.scatter_mapbox
    kwargs = dict(lat="latitude", lon="longitude", color="highest_severity", size="incidents", size_max=16,
                  category_orders={"highest_severity": SEV_ORDER}, color_discrete_map=SEVERITY_COLORS,
                  hover_name="location_id", hover_data={"zone": True, "road_type": True, "road_condition": True, "incidents": True,
                                                        "mean_hours": True, "latitude": False, "longitude": False},
                  zoom=9, height=520)
    fig = mapper(g, **kwargs)
    fig.update_layout(map_style="open-street-map") if hasattr(px, "scatter_map") else fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, title=""))
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("Marker color is the highest severity recorded at the location; marker size is incident count.")
    z = f.groupby("zone").agg(incidents=("incident_id", "count"), mean_hours=("resolution_hours", "mean"),
                              median_hours=("resolution_hours", "median")).round(1).reset_index()
    if not COMPLAINTS.empty and "location_id" in COMPLAINTS:
        cz = COMPLAINTS.merge(LOCATIONS[["location_id", "zone"]], on="location_id").groupby("zone").size().rename("complaints")
        z = z.merge(cz, on="zone", how="left")
    st.markdown("##### Zone summary")
    st.dataframe(z.sort_values("incidents", ascending=False), hide_index=True, width="stretch")


# =================================================================== 7. INFO
def page_info():
    page_header("Data & Model Information", "What data is used, what the model predicts and what it cannot tell you.")
    st.markdown(f"""
**Data.** {len(DATA):,} incidents joined to {len(LOCATIONS):,} locations (`location_id`), plus {len(COMPLAINTS):,} public complaints used only for location summaries.
The dataset is **synthetic**; coordinates are simulated.

**What the model predicts.** The regression model estimates `resolution_hours`, the time to resolve an incident, in hours.
It does **not** predict incident priority. The weighted priority score in the baseline app is a rule-based heuristic and is not part of this model.

**Features supplied to the model.** {", ".join(f"`{c}`" for c in U.EXPECTED_FEATURES)}.

**Preprocessing (saved inside the model pipeline).** Numeric features: missing values filled with the training median.
Categorical features: missing values filled with the most frequent label, then one-hot encoded; labels not seen in training are ignored rather than causing an error.

**Evaluation.** Random 80/20 split. Candidate models and settings were compared with 5-fold cross-validation on the training split only; the test split was used once for the final MAE and RMSE.
""")
    note("<b>Limitations.</b> Training data is synthetic, so metrics do not show real-world performance. "
         "In this dataset the input features show little relationship with resolution time (see Resolution Analytics and the model's permutation importance). "
         "The `resolved` flag and `resolution_hours` are not consistent for every incident. "
         "A prediction is a statistical estimate, never a guarantee, and should not drive operational decisions.", "warn")
    st.markdown("**Artifacts.** " + ("Model and metrics were found." if BUNDLE is not None and METRICS else
                                      "One or more artifacts are missing; run `python ml_train.py`."))


{"ML Overview": page_overview, "Predict Resolution Time": page_predict, "Incident Explorer": page_explorer,
 "Resolution Analytics": page_analytics, "Model Performance": page_performance, "Location Insights": page_location,
 "Data & Model Information": page_info}[page]()
