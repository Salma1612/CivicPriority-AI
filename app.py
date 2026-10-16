import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="CivicPriority AI", page_icon="🏙️", layout="wide")
st.title("🏙️ CivicPriority AI")
st.caption("Explainable Smart City intervention-priority dashboard | Hackathon prototype")

DATA_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    locations = pd.read_csv(DATA_DIR / "locations.csv")
    incidents = pd.read_csv(DATA_DIR / "incidents.csv", parse_dates=["timestamp"])
    complaints = pd.read_csv(DATA_DIR / "public_complaints.csv", parse_dates=["created_at"])
    return locations, incidents, complaints
try:
    locations, incidents, complaints = load_data()
except Exception as e:
    st.error(f"Could not load the three CSV files. Keep them in the same folder as app.py. Details: {e}")
    st.stop()

# Aggregate events at location level.
sev_weight = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
incidents = incidents.copy()
incidents["severity_points"] = incidents["severity"].map(sev_weight).fillna(0)
incidents["is_unresolved"] = incidents["resolved"].eq(0).astype(int)
incident_agg = incidents.groupby("location_id").agg(
    incident_count=("incident_id", "count"),
    severity_points=("severity_points", "sum"),
    high_critical_incidents=("severity", lambda s: s.isin(["High", "Critical"]).sum()),
    injury_count=("injuries", "sum"),
    unresolved_incidents=("is_unresolved", "sum"),
    avg_incident_resolution_hours=("resolution_hours", "mean"),
).reset_index()

complaints = complaints.copy()
complaints["is_openish"] = complaints["status"].isin(["Open", "Assigned", "In progress"]).astype(int)
complaints["reported_priority_points"] = complaints["priority_reported"].map({"Low": 1, "Normal": 2, "High": 3}).fillna(0)
complaint_agg = complaints.groupby("location_id").agg(
    complaint_count=("complaint_id", "count"),
    openish_complaints=("is_openish", "sum"),
    reported_priority_points=("reported_priority_points", "sum"),
    avg_complaint_days=("days_to_resolution", "mean"),
).reset_index()

df = locations.merge(incident_agg, on="location_id", how="left").merge(complaint_agg, on="location_id", how="left")
numeric = ["incident_count","severity_points","high_critical_incidents","injury_count","unresolved_incidents",
           "avg_incident_resolution_hours","complaint_count","openish_complaints","reported_priority_points","avg_complaint_days"]
df[numeric] = df[numeric].fillna(0)

# Transparent heuristic score. Components are normalized against observed dataset maxima.
def norm(s):
    m = s.max()
    return (s / m * 100) if m and m > 0 else s * 0

condition_points = df["road_condition"].map({"Good": 0, "Fair": 33, "Poor": 67, "Critical": 100}).fillna(0)
ped_points = df["pedestrian_activity"].map({"Low": 0, "Medium": 50, "High": 100}).fillna(0)
context_points = (
    condition_points * 0.50 +
    df["near_school"].astype(float) * 25 +
    df["near_hospital"].astype(float) * 25 +
    ped_points * 0.25
).clip(0, 100)

df["priority_score"] = (
    norm(df["severity_points"]) * 0.30 +
    norm(df["complaint_count"]) * 0.20 +
    norm(df["incident_count"]) * 0.15 +
    norm(df["high_critical_incidents"]) * 0.15 +
    norm(df["openish_complaints"]) * 0.10 +
    context_points * 0.10
).round(1)

def explain(row):
    reasons = []
    if row["road_condition"] == "Critical": reasons.append("road condition marked Critical")
    elif row["road_condition"] == "Poor": reasons.append("road condition marked Poor")
    if row["high_critical_incidents"] > 0: reasons.append(f'{int(row["high_critical_incidents"])} high/critical incident(s)')
    if row["injury_count"] > 0: reasons.append(f'{int(row["injury_count"])} reported injury count')
    if row["complaint_count"] > 0: reasons.append(f'{int(row["complaint_count"])} complaint(s)')
    if row["openish_complaints"] > 0: reasons.append(f'{int(row["openish_complaints"])} complaint(s) open/in progress/assigned')
    if row["near_school"] == 1: reasons.append("near a school (dataset flag)")
    if row["near_hospital"] == 1: reasons.append("near a hospital (dataset flag)")
    if row["pedestrian_activity"] == "High": reasons.append("high pedestrian activity")
    return "; ".join(reasons[:5]) if reasons else "Lower observed counts/context score; review alongside local knowledge"

df["evidence"] = df.apply(explain, axis=1)
df["priority_band"] = pd.cut(df["priority_score"], bins=[-0.1, 25, 50, 75, 100.1],
                             labels=["Monitor", "Moderate", "High", "Urgent"])

with st.sidebar:
    st.header("Filters")
    zones = ["All"] + sorted(df["zone"].dropna().unique().tolist())
    selected_zone = st.selectbox("Zone", zones)
    bands = ["All", "Urgent", "High", "Moderate", "Monitor"]
    selected_band = st.selectbox("Priority band", bands)
    top_n = st.slider("Locations to show", 10, 100, 25, 5)
    st.markdown("---")
    st.caption("Priority weights are a transparent demonstration heuristic, not a validated public-safety model.")

filtered = df.copy()
if selected_zone != "All":
    filtered = filtered[filtered["zone"] == selected_zone]
if selected_band != "All":
    filtered = filtered[filtered["priority_band"].astype(str) == selected_band]
ranked = filtered.sort_values(["priority_score", "high_critical_incidents", "complaint_count"], ascending=False)
top = ranked.head(top_n)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Locations", f"{len(df):,}")
c2.metric("Incidents", f"{len(incidents):,}")
c3.metric("Public complaints", f"{len(complaints):,}")
c4.metric("Unresolved incidents", f'{int((incidents["resolved"] == 0).sum()):,}')

st.info("Synthetic educational dataset: coordinates and reports are simulated. This dashboard supports a demo workflow only and must not be used for actual city safety decisions.")

tab1, tab2, tab3, tab4 = st.tabs(["📌 Priority queue", "🗺️ Location map", "📈 Trends", "⏱️ Resolution"])

with tab1:
    st.subheader("Locations recommended for review")
    st.write("Each score is accompanied by evidence. A higher score means more signals in this heuristic—not a probability of danger.")
    display_cols = ["location_id", "zone", "priority_score", "priority_band", "road_condition",
                    "incident_count", "high_critical_incidents", "injury_count", "complaint_count",
                    "openish_complaints", "evidence"]
    st.dataframe(top[display_cols], use_container_width=True, hide_index=True)
    if not top.empty:
        chosen_id = st.selectbox("Inspect a location", top["location_id"].tolist())
        row = df.loc[df["location_id"] == chosen_id].iloc[0]
        st.markdown(f"### {chosen_id} — score {row['priority_score']}/100 ({row['priority_band']})")
        st.write("**Why it appears here:**", row["evidence"])
        st.write(f"**Context:** {row['road_type']} road, {row['road_condition']} condition, pedestrian activity {row['pedestrian_activity']}, near school: {'Yes' if row['near_school'] else 'No'}, near hospital: {'Yes' if row['near_hospital'] else 'No'}.")
    st.download_button("Download ranked locations CSV", ranked.to_csv(index=False).encode("utf-8"), "civicpriority_ranked_locations.csv", "text/csv")

with tab2:
    st.subheader("Map of locations")
    st.caption("Map points are synthetic coordinates provided with the educational dataset.")
    if not top.empty:
        fig_map = px.scatter_map(
            top,
            lat="latitude",
            lon="longitude",
            color="priority_score",
            size="priority_score",
            hover_name="location_id",
            hover_data=[
                "zone",
                "road_condition",
                "incident_count",
                "complaint_count",
                "priority_band",
             ],
             color_continuous_scale="YlOrRd",
             zoom=8,
             height=560,
             map_style="open-street-map",
          )
          st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No locations match the selected filters.")

with tab3:
    st.subheader("Incident and complaint trends")
    incidents["month"] = incidents["timestamp"].dt.to_period("M").astype(str)
    complaints["month"] = complaints["created_at"].dt.to_period("M").astype(str)
    it = incidents.groupby("month").size().rename("Incidents")
    ct = complaints.groupby("month").size().rename("Complaints")
    trends = pd.concat([it, ct], axis=1).fillna(0).reset_index().rename(columns={"index": "Month", "month": "Month"})
    trends = trends.sort_values("Month")
    fig = px.line(trends, x="Month", y=["Incidents", "Complaints"], markers=True, title="Monthly reported events")
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        cat = incidents["incident_type"].value_counts().rename_axis("Incident type").reset_index(name="Count")
        st.plotly_chart(px.bar(cat, x="Incident type", y="Count", title="Incidents by type"), use_container_width=True)
    with right:
        cat2 = complaints["category"].value_counts().rename_axis("Complaint category").reset_index(name="Count")
        st.plotly_chart(px.bar(cat2, x="Complaint category", y="Count", title="Complaints by category"), use_container_width=True)

with tab4:
    st.subheader("Resolution performance")
    a, b = st.columns(2)
    with a:
        resolved_pct = (incidents["resolved"].eq(1).mean() * 100)
        st.metric("Incidents marked resolved", f"{resolved_pct:.1f}%")
        st.metric("Mean incident resolution time", f'{incidents["resolution_hours"].mean():.1f} hours')
        st.plotly_chart(px.box(incidents, x="severity", y="resolution_hours", title="Incident resolution hours by severity"), use_container_width=True)
    with b:
        st.metric("Mean complaint resolution days", f'{complaints["days_to_resolution"].mean():.1f} days')
        st.plotly_chart(px.box(complaints, x="status", y="days_to_resolution", title="Complaint resolution days by status"), use_container_width=True)
    st.caption("Check how the source dataset defines resolution times and status categories before interpreting these as operational service-level measures.")

with st.expander("How the priority score works"):
    st.markdown("""
    This prototype uses a **transparent weighted heuristic**, not a trained machine-learning model:
    - 30%: total incident severity points (Low=1, Medium=2, High=3, Critical=4), normalized within this dataset
    - 20%: complaint count, normalized
    - 15%: incident count, normalized
    - 15%: count of High/Critical incidents, normalized
    - 10%: count of complaints currently Open, Assigned, or In progress, normalized
    - 10%: contextual signals from road condition, nearby school/hospital flags, and pedestrian activity

    The weights are provisional and should be reviewed with domain experts. A score is a queueing aid, not proof of danger or a probability. Normalization is relative to the supplied dataset, so scores may change when data changes.
    """)
st.caption("Built for a hackathon demonstration. Validate data definitions, review false positives, and involve local domain experts before any real deployment.")
