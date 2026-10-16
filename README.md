# CivicPriority AI — Hackathon Prototype

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.

Keep `app.py`, `locations.csv`, `incidents.csv`, and `public_complaints.csv` in the same folder.

## What it includes
- Location ranking and evidence explanations
- Zone and priority-band filters
- Synthetic-coordinate map
- Monthly incident/complaint trends
- Resolution metrics and charts
- Downloadable ranked CSV

## Important limitations
- This is a rule-based, weighted heuristic, not a trained ML model.
- Weights are provisional demonstration choices, not validated civic-risk thresholds.
- Dataset is synthetic and coordinates are simulated; do not use for actual public-safety decisions.
- Review the score weights and status definitions with domain experts before real use.
