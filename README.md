

<div align="center">

<h1>🛂 AI-Enabled Visa Status Prediction<br/>& Processing Time Estimator</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-2.0.3-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Flask-3.0.0-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/Deployed-Render-purple?style=for-the-badge&logo=render" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p><b>A full-stack AI web application that predicts H-1B visa processing times using XGBoost trained on 3.2M+ historical records — built as part of the Infosys Springboard Virtual Internship.</b></p>

<p>
  <a href="https://ai-visa-prediction-processing-time-rohan.onrender.com">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20App-brightgreen?style=for-the-badge" alt="Live Demo" />
  </a>
  &nbsp;
  <a href="https://drive.google.com/file/d/1p4hH_vcX3ZeKZgFjOA-cthwVXEycGRfr/view?usp=sharing">
    <img src="https://img.shields.io/badge/🎬%20Demo%20Video-Watch-red?style=for-the-badge&logo=google-drive" alt="Demo Video" />
  </a>
</p>

</div>



-----

## 📋 Table of Contents

  * [Project Overview](https://www.google.com/search?q=%23-project-overview)
  * [Architecture](https://www.google.com/search?q=%23%EF%B8%8F-system-architecture)
  * [Milestones](https://www.google.com/search?q=%23-milestones--timeline)
      * [Data Collection & Preprocessing](https://www.google.com/search?q=%23milestone-1--data-collection--preprocessing-weeks-12)
      * [EDA & Feature Engineering](https://www.google.com/search?q=%23milestone-2--eda--feature-engineering-weeks-34)
      * [Predictive Modeling](https://www.google.com/search?q=%23milestone-3--predictive-modeling-weeks-56)
      * [Web App & Deployment](https://www.google.com/search?q=%23milestone-4--web-app--deployment-weeks-78)
  * [Application Features](https://www.google.com/search?q=%23-application-features)
  * [Model Details](https://www.google.com/search?q=%23-model-details)
  * [API Reference](https://www.google.com/search?q=%23-api-reference)
  * [Installation & Setup](https://www.google.com/search?q=%23-installation--setup)
  * [Project Structure](https://www.google.com/search?q=%23-project-structure)

-----

## 🎯 Project Overview

Visa applicants — especially those filing H-1B applications — face long waiting times with little clarity on when a decision will come. This project builds a machine learning-powered estimator that predicts how many days a visa case will take to process, based on **3.2M+ historical records** from the US Department of Labor.

By analysing applicant job title, employer details, prevailing wage, work location, and case year, the XGBoost model delivers data-backed estimates with confidence intervals, improving transparency for applicants and agencies alike.

> **Note:** Developed as part of the Infosys Springboard Virtual Internship.

-----

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                         USER / CLIENT                            │
│         Web Browser  ·  API Consumer  ·  Enterprise Platform     │
└───────────────────────────────┬──────────────────────────────────┘
                                │  HTTP / REST
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FRONTEND  (index.html)                        │
│                                                                  │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────┐ ┌─────────────┐ │
│  │  Time       │ │  Analytics   │ │ Trends   │ │ API Docs    │ │
│  │  Estimator  │ │  Dashboard   │ │ Explorer │ │ Panel       │ │
│  └─────────────┘ └──────────────┘ └──────────┘ └─────────────┘ │
└───────────────────────────────┬──────────────────────────────────┘
                                │  JSON POST/GET
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    BACKEND  (app.py · Flask 3.0)                 │
│                                                                  │
│   /v1/predict    /v1/batch-predict    /v1/trends    /v1/health   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Input Encoding Layer                          │  │
│  │  20 features · sklearn LabelEncoders (label_encoders.pkl)  │  │
│  │  UNKNOWN-class fallback for unseen labels                  │  │
│  │  High-cardinality fields validated before use              │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │  Encoded feature vector (float64)  │
│  ┌──────────────────────────▼─────────────────────────────────┐  │
│  │           XGBoost Regression Model                         │  │
│  │  xgboost_visa_model.pkl  (zlib-compressed)                 │  │
│  │  Trained on h1b_kaggle_original.csv · 3.2M records         │  │
│  │  tree_method=hist · device=cuda · n_estimators=150         │  │
│  │  MAE = 6.1 days  ·  R² = 0.84                              │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │  raw_pred (days)                   │
│  ┌──────────────────────────▼─────────────────────────────────┐  │
│  │          Post-processing & Confidence Layer                 │  │
│  │  • Fallback heuristic if raw_pred < 5 (OOD guard)          │  │
│  │  • P10 / P50 / P90 confidence interval                     │  │
│  │  • Risk flag detection                                      │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
└─────────────────────────────┼────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│            DEPLOYMENT  ·  Render  ·  Gunicorn 21.2               │
│   https://ai-visa-prediction-processing-time-rohan.onrender.com  │
└──────────────────────────────────────────────────────────────────┘
```

-----

## 🗓️ Milestones & Timeline

The project is structured across 4 milestones (8 weeks total), each with a concrete input, deliverable, and output.

### Milestone 1 — Data Collection & Preprocessing (Weeks 1–2)

  * **Input dataset:** `h1b_kaggle_original.csv`
  * **Output dataset:** `h1b_cleaned.csv`
  * **Script:** `Milestone1.py`

| Step | Detail |
| :--- | :--- |
| **Column normalisation** | Stripped, uppercased, spaces → underscores |
| **Deduplication** | `drop_duplicates()` |
| **Target variable** | `PROCESSING_DAYS` = `(DECISION_DATE - SUBMIT_DATE).dt.days` |
| **Outlier filter** | Kept rows where 0 ≤ `PROCESSING_DAYS` ≤ 730 |
| **Missing numerics** | Imputed with median |
| **Missing categoricals**| Imputed with mode (fallback: "UNKNOWN") |
| **Wage cleaning** | Removed `$`/`,`, clipped at 99th percentile |
| **Full-time flag** | Mapped Y/Yes/True → 1, N/No/False → 0 |
| **Categorical encoding**| LabelEncoder for low-cardinality; frequency encoding for cols \> 50 unique values |
| **Binary target** | `CASE_STATUS_BINARY`: Certified/CW → 1, Denied/Withdrawn → 0 |
| **Date features** | `SUBMIT_MONTH`, `SUBMIT_YEAR`, `SUBMIT_DAY` (weekday), `SUBMIT_QTR` |

### Milestone 2 — EDA & Feature Engineering (Weeks 3–4)

  * **Input dataset:** `h1b_cleaned.csv`
  * **Output dataset:** `h1b_features.csv`
  * **Script:** `Milestone2.py`

**Engineered Features:**

  * `SEASONAL_INDEX`: monthly\_avg\_days / global\_avg\_days per submit month
  * `EMPLOYER_AVG_DAYS`: Per-employer mean processing time from historical data
  * `EMPLOYER_VOLUME`: Number of cases filed by each employer
  * `WAGE_BUCKET`: Wage binned into 5 bands (0–40k / 40–70k / 70–100k / 100–150k / 150k+)
  * `LOG_WAGE`: `log1p(PREVAILING_WAGE)` — reduces right skew

### Milestone 3 — Predictive Modeling (Weeks 5–6)

  * **Input dataset:** `h1b_kaggle_original.csv`
  * **Notebook:** `xgboost_fast_gpu_fixed.ipynb` (run on Google Colab T4 GPU)
  * **Outputs:** `xgboost_visa_model (1).pkl` · `label_encoders (3).pkl` · `features (1).pkl`

> **Note on Feature Engineering:** XGBoost's gradient-boosted trees automatically learn non-linear feature interactions, so manually engineered features from Milestone 2 add little value. Using the original dataset preserves all raw columns without having to selectively drop engineered ones, producing better generalisation.

**Model Performance:**

  * **MAE:** 6.1 days
  * **R²:** 0.84
  * **Features Used (20):** `case_year`, `case_status`, `emp_city`, `emp_state`, `emp_zip`, `emp_country`, `job_title`, `soc_code`, `soc_name`, `full_time_position`, `prevailing_wage`, `pw_unit`, `pw_level`, `wage_from`, `wage_to`, `wage_unit`, `work_city`, `work_state`, `emp_h1b_dependent`, `emp_willful_violator`

### Milestone 4 — Web App & Deployment (Weeks 7–8)

  * **Files:** `app.py` · `index.html` · `Procfile` · `requirements.txt`
  * **Live URL:** [https://ai-visa-prediction-processing-time-rohan.onrender.com](https://ai-visa-prediction-processing-time-rohan.onrender.com)

The Flask backend serves a single-page frontend across five sections: Time Estimator, Analytics Dashboard, Processing Trends, Past Cases, and API Docs Panel. Deployed as a Gunicorn WSGI server (`web: gunicorn app:app`) on Render.

-----

## ✨ Application Features

  * **⚡ Autofill Random:** One click populates all 20 form fields with a realistic case for instant testing.
  * **Confidence Intervals:** Every prediction includes P10 (optimistic), P50 (median), P90 (conservative).
  * **Risk Flags:** Automatic detection of H1B-dependent employer, willful violator history, low PW level, part-time position, denial risk.
  * **OOD Fallback:** When XGBoost raw output \< 5 days (out-of-distribution), a calibrated heuristic runs instead of clamping, preventing nonsense predictions.
  * **Batch Prediction:** Submit up to 1,000 applications per `/v1/batch-predict` call.
  * **REST API:** CORS-enabled and ready for third-party integrations.

-----

## 🤖 Model Details

The XGBoost model is loaded at startup via zlib decompression in `app.py`:

```python
with open('xgboost_visa_model (1).pkl', 'rb') as f:
    model = pickle.loads(zlib.decompress(f.read()))
```

Categorical encoding uses the same fitted `LabelEncoder` per column that was saved during training (`label_encoders (3).pkl`). Unseen values fall back to the `UNKNOWN` class index. High-cardinality optional fields are only substituted if the updated raw prediction stays ≥ 5 days — ensuring the model stays in its trained distribution.

-----

## 🔌 API Reference

**Base URL:** `https://ai-visa-prediction-processing-time-rohan.onrender.com/v1`

### `POST /v1/predict`

**Request:**

```json
{
  "case_year": 2023,
  "case_status": "C",
  "emp_state": "CA",
  "emp_country": "USA",
  "job_title": "SOFTWARE ENGINEER",
  "soc_code": "15-1132",
  "soc_name": "15-1132",
  "prevailing_wage": 120000,
  "pw_level": "LEVEL III",
  "pw_unit": "Y",
  "wage_from": 110000,
  "wage_to": 130000,
  "wage_unit": "Y",
  "work_state": "CA",
  "full_time_position": true,
  "emp_h1b_dependent": false,
  "emp_willful_violator": false
}
```

**Response:**

```json
{
  "predicted_days": 32,
  "range": { "low": 22, "high": 44 },
  "confidence": 0.91,
  "percentiles": { "p10": 24, "p50": 32, "p90": 46 },
  "risk_flags": [],
  "model_used": "xgboost",
  "model_version": "xgboost_v1.0"
}
```

  * **`POST /v1/batch-predict`**: Submit up to 1,000 applications in one call.
  * **`GET /v1/trends?state=CA&year=2023`**: Monthly average processing times and annual summary.
  * **`GET /v1/health`**: Model load status and feature counts.

-----

## 🚀 Installation & Setup

**Prerequisites:** Python 3.11.9

**1. Clone the Repository**

```bash
git clone https://github.com/rohanreddy-12/AI-Visa-Prediction-Processing-Time.git
cd AI-Visa-Prediction-Processing-Time
```

**2. Virtual Environment Setup**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Start the Server**

```bash
# Ensure your model .pkl files are in the root directory!
python app.py
# Open http://localhost:5000
```

### Retraining (Optional)

```bash
# 1. Clean Data
python Milestone1.py

# 2. Exploratory Data Analysis & Features
python Milestone2.py

# 3. Model Training
# Open Milestone 3/xgboost_fast_gpu_fixed.ipynb in Google Colab (T4 GPU).
# Download the generated .pkl files into your repository root.
```

-----

## 📁 Project Structure

```text
AI-Visa-Prediction-Processing-Time/
│
├── app.py                               # Flask backend API
├── index.html                           # Single-page frontend 
├── requirements.txt                     # Python dependencies
├── Procfile                             # Render deployment
├── .python-version                      # 3.11.9
├── MIT license.txt                      
│
├── xgboost_visa_model.pkl               # Trained XGBoost model (zlib-compressed)
├── label_encoders.pkl                   # Fitted LabelEncoders 
├── features.pkl                         # Exact feature order list 
│
├── part1_data_cleaning.py               # Data Cleaning Pipeline
├── part2_eda_features.py                # EDA & Feature Engineering
│
├── Milestone 3/
│   └── xgboost_fast_gpu_fixed.ipynb     # Model training notebook
│
└── Documents/
    ├── Rohan_Agile_doc.xls
    ├── Rohan_Defect_Tracker.xlsx
    └── Rohan_Unit_Test_Plan.xlsx
```

-----

### 🙏 Acknowledgements

  * **Dataset:** H-1B Non-Immigrant Labour Visa — Kaggle
  * **GPU Training:** Google Colab T4 GPU
  * **Deployment:** Render
  * **Internship:** Infosys Springboard

\<div align="center"\>
Made with ❤️ by Rohan Reddy<br>
⭐ Star the repo if this helped you\!
\</div\>
