# 🚀 AI-Enabled Visa Status Prediction & Processing Time Estimator

## 📌 Project Overview

Visa applicants often face uncertainty regarding processing timelines. This project leverages **machine learning** and historical visa data to predict **processing times** and provide insights into patterns affecting delays.

The system analyzes factors such as:

* Applicant country
* Visa type
* Processing office
* Seasonal trends

👉 The goal is to **improve transparency and decision-making** for applicants.

---

## 🎯 Key Features

✅ Predict visa processing time using ML
✅ Dynamic data visualizations (auto-updated per input)
✅ Trend analysis across countries, visa types, and offices
✅ Interactive web interface (Flask + Tailwind)
✅ Production deployment on cloud

---

## 🛠️ Tech Stack

### 👨‍💻 Backend

* Python 3.x
* Flask

### 📊 Data Science

* pandas
* numpy
* scikit-learn

### 📈 Visualization

* matplotlib
* seaborn

### 🎨 Frontend

* HTML
* Tailwind CSS

### ☁️ Deployment

* Render

---

## 📁 Project Structure

```
.
├── app.py                 # Flask app
├── requirements.txt
├── artifacts/             # Model & encoders
├── data/
│   └── processed_data.csv
├── src/
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── templates/
│   └── index.html
├── outputs/               # EDA graphs
├── README.md
```

---

## 📊 Exploratory Data Analysis (EDA)

### 📈 Processing Time Distribution

![Distribution](outputs/processing_time_dist.png)

### 🌍 Processing Time by Country

![Country](outputs/country_boxplot.png)

### 🏢 Processing Office Trends

![Office](outputs/office_barplot.png)

### ❄️ Seasonal Trends

![Season](outputs/season_barplot.png)

### 🔥 Correlation Heatmap

![Heatmap](outputs/correlation_heatmap.png)

---

## 🧠 Machine Learning Model

We trained multiple regression models:

| Model             | MAE  | RMSE  | R²   |
| ----------------- | ---- | ----- | ---- |
| Linear Regression | 7.47 | 12.07 | 0.52 |
| Random Forest     | 8.14 | 12.73 | 0.46 |
| Gradient Boosting | 7.56 | 12.13 | 0.51 |

👉 **Best Model:** Linear Regression




## 🚀 Deployment (Render)

1. Push code to GitHub
2. Go to Render
3. Create Web Service
4. Use:

```bash
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

---

## ⚠️ Challenges Faced

* Handling missing inputs (month validation)
* Fixing dataset column mismatches
* Debugging deployment issues (404, routing)
* Managing unseen categorical values in prediction

---

## 🧠 Future Improvements

* 🔥 Use real-time visa datasets
* 📊 Add interactive charts (Plotly)
* 🤖 Improve model accuracy (XGBoost)
* 🌐 Add user authentication
* 📱 Make mobile-friendly UI

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Rohan Reddy**

