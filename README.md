AI-Visa-Prediction-Processing-Time
Python

📝 Description
AI-Visa-Prediction-Processing-Time is an innovative Python-based application that leverages artificial intelligence to predict visa processing durations. By analyzing historical data and various application factors, this tool provides prospective travelers and immigration professionals with data-driven estimates, reducing uncertainty and streamlining international travel planning.

🛠️ Tech Stack
🐍 Python





Predict H-1B visa processing time using Machine Learning (3.2M+ records)

🚀 [Live App](https://ai-visa-prediction-processing-time-rohan-p0uq.onrender.com)
🎬 [Demo Video](https://drive.google.com/file/d/14674a_8mZg4HyCXiDaZHxZrbjKV-CdKz/view?usp=drive_link)




📦 Key Dependencies

xgboost: 2.0.3

flask: 3.0.0

flask-cors: 4.0.0

numpy: 1.26.2

scikit-learn: 1.4.0

gunicorn: 21.2.0

pandas: 2.2.0




-----

## 📋 Table of Contents

  * Project Overview
  * Architecture
  * Milestones
      * Data Collection & Preprocessing
      * EDA & Feature Engineering
      * Predictive Modeling
      * Web App & Deployment
  * Application Features
  * Datasets
  * API Reference
  * Installation & Setup
  * Project Structure
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
  * **Live URL:** [https://ai-visa-prediction-processing-time-rohan.onrender.com](https://ai-visa-prediction-processing-time-rohan-p0uq.onrender.com)

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

## 📂 Datasets

| Dataset | Source | Role in Project |
|---|---|---|
| `h1b_kaggle_original.csv` | [Kaggle — H-1B Non-Immigrant Labour Visa](https://www.kaggle.com/datasets/thedevastator/h-1b-non-immigrant-labour-visa) | Milestone 1 input & Milestone 3 training input — XGBoost does not require many features so original dataset was used directly instead of dropping columns from `h1b_features.csv` |
| `h1b_cleaned.csv` | [Google Drive](https://drive.google.com/file/d/1qYOUJrZ4kmXsKajm3vPsQi4acnxEyqfo/view?usp=drive_link) | Milestone 1 output — Milestone 2 input |
| `h1b_features.csv` | [Google Drive](https://drive.google.com/file/d/12XJjIe9uWv0vys0EJW2YnDrvS9w62LXH/view?usp=drive_link) | Milestone 2 output — not used for model training (see note below) |

> **Note:** For Milestone 3, we used `h1b_kaggle_original.csv` directly from Kaggle instead of `h1b_features.csv`. Since XGBoost does not require many pre-engineered features, rather than dropping columns from `h1b_features.csv`, we went back to the original dataset. Google Colab's T4 GPU was used for somewhat faster training of the model.

> Place all three datasets in the repo root before running any milestone script.

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

### 🙏 Acknowledgements

  * **Dataset:** H-1B Non-Immigrant Labour Visa — Kaggle
  * **GPU Training:** Google Colab T4 GPU
  * **Deployment:** Render
  * **Internship:** Infosys Springboard

<div align="center">
Made with ❤️ by <a href="https://github.com/rohanreddy-12">Rohan Reddy</a>
⭐ Star the repo if this helped you!
</div>
