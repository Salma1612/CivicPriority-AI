# 🏙️ CivicPriority AI
### AI-Powered Civic Complaint Prioritization and Resolution Support

CivicPriority AI is a civic technology project designed to help analyze citizen complaints, understand civic incident patterns, and support data-driven prioritization. It combines a Streamlit application with data exploration, analytics, and machine learning capabilities.

## 🚀 Live Demo

**Try the application:** [CivicPriority AI – Hackaholics](https://civicpriority-ai-hackaholics.streamlit.app/)

## 📸 Application Screenshots

_Add screenshots of the application in the sections below._

### 1. Application Overview
<!-- Add Overview screenshot here -->
<img width="959" height="386" alt="image" src="https://github.com/user-attachments/assets/f093c6e2-4f39-40b1-9b6f-5b43070964e8" />

![CivicPriority AI Overview](images/overview.png)

### 2. Incident Priority Prediction
<!-- Add Prediction page screenshot here -->
<img width="778" height="413" alt="image" src="https://github.com/user-attachments/assets/bdf64a7c-680b-4a1e-9341-603dd286b25c" />

![Incident Priority Prediction](images/predict.png)

### 3. Data Explorer
<!-- Add Explorer page screenshot here -->
<img width="469" height="410" alt="image" src="https://github.com/user-attachments/assets/c3402c50-5a30-496d-a1eb-14bade50788a" />

![Data Explorer](images/explorer.png)

### 4. Analytics Dashboard
<!-- Add Analytics screenshot here -->
<img width="467" height="250" alt="image" src="https://github.com/user-attachments/assets/6b252ce2-30c5-4a8b-811a-824fe7b59c15" />

![Analytics Dashboard](images/analytics.png)

### 5. Model Performance
<!-- Add Model Performance screenshot here -->
<img width="243" height="346" alt="image" src="https://github.com/user-attachments/assets/27221f95-27f7-46c9-a33f-7d1f70f198d0" />

![Model Performance](images/model-performance.png)

### 6. Location Insights
<!-- Add Location page screenshot here -->
<img width="242" height="380" alt="image" src="https://github.com/user-attachments/assets/4d4a4066-f352-4a15-8320-5d5f82e72d0b" />

![Location Insights](images/location.png)

### 7. Data and Model Information
<!-- Add Data & Model Info screenshot here -->
<img width="959" height="353" alt="image" src="https://github.com/user-attachments/assets/34406c9d-45a2-42d8-a391-d1447bdae71a" />

![Data and Model Information](images/data-model-info.png)

> **Note:** Create an `images` folder in the repository and upload your screenshots using the filenames shown above. The images will appear automatically in the README once the files are committed.

## ✨ Key Features

- **Interactive dashboard:** Explore civic incident data through a Streamlit interface.
- **Incident prediction:** Enter incident details to request a model-based prediction when the trained model is available.
- **Data exploration:** Inspect records and explore incident attributes.
- **Analytics:** Review incident patterns and summary statistics.
- **Model evaluation:** Examine available model performance metrics and feature importance.
- **Location insights:** Explore geographic patterns using the available location data.
- **Model information:** Review dataset and model details.

## 🧠 Machine Learning

The project includes a machine learning workflow for predicting incident resolution time.

The workflow covers:
- Data preparation and feature processing
- Model training and evaluation
- Comparison with baseline predictions
- Feature importance analysis
- Integration of the trained model with the Streamlit application

**Evaluation note:** The model is experimental. Initial held-out test results did not consistently outperform the median baseline, so its predictions should be treated as exploratory rather than as a reliable estimate of actual resolution times.

## 🛠️ Technology Stack

- **Language:** Python
- **Application:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Plotly
- **Model persistence:** Joblib

## 📂 Project Structure

```text
CivicPriority-AI/
├── app.py
├── ml_app.py
├── ml_components.py
├── ml_utils.py
├── ml_train.py
├── train_model.py
├── predict.py
├── requirements.txt
├── incidents.csv
├── locations.csv
├── public_complaints.csv
├── models/
│   ├── resolution_model.joblib
│   ├── model_metrics.json
│   ├── feature_importance.csv
│   └── test_predictions.csv
├── images/
│   ├── overview.png
│   ├── predict.png
│   ├── explorer.png
│   ├── analytics.png
│   ├── model-performance.png
│   ├── location.png
│   └── data-model-info.png
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Salma1612/CivicPriority-AI.git
cd CivicPriority-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the ML application

```bash
streamlit run ml_app.py
```

To run the original application instead:

```bash
streamlit run app.py
```

## ⚠️ Limitations

- Model performance depends on the quality and predictive value of the available data.
- Location insights depend on the accuracy of the supplied coordinates.
- Model predictions are estimates and should not replace real-world assessment.
- The machine learning model is a prototype and requires further validation before operational use.

## 🔮 Future Improvements

- Improve model performance through additional data and feature engineering.
- Validate predictions on larger, representative datasets.
- Enhance geographic analysis and incident hotspot visualization.
- Add monitoring for prediction quality and data drift.
- Improve model deployment and dependency compatibility.

## 👩‍💻 Author

**Shaik Salma**
**SK Muneer Ahamed**

GitHub: [Salma1612](https://github.com/Salma1612)

---

*Built to explore how data science and machine learning can support civic incident analysis and better-informed public service decisions.*
