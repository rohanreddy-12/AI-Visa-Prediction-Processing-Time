# ==========================================
# Milestone 3 - Predictive Modeling
# Dataset: h1b_compressed_smart.csv
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# ---------------------------------
# 1 Load Dataset
# ---------------------------------
df = pd.read_csv("h1b_compressed_smart.csv")
print("Dataset Shape:", df.shape)

# ---------------------------------
# 2 Select Features
# ---------------------------------
features = [
    "emp_state", "work_state", "job_title", "soc_name",
    "full_time_position", "prevailing_wage", "pw_level",
    "emp_h1b_dependent", "case_year"
]
target = "processing_time"
df = df[features + [target]].dropna()

# ---------------------------------
# 3 Encode Categorical Columns
# ---------------------------------
label_encoders = {}
for col in features:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ---------------------------------
# 4 Split Dataset
# ---------------------------------
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# ---------------------------------
# 5 Train Baseline Models
# ---------------------------------
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# ---------------------------------
# 6 Evaluation Function
# ---------------------------------
def evaluate_model(model, name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("\nModel:", name)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# ---------------------------------
# 7 Evaluate Baseline Models
# ---------------------------------
evaluate_model(lr, "Linear Regression")
evaluate_model(rf, "Random Forest")
evaluate_model(gb, "Gradient Boosting")

# ---------------------------------
# 8 Hyperparameter Tuning
# ---------------------------------
print("\nTuning Random Forest...")
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("Best RF Parameters:", rf_grid.best_params_)
evaluate_model(best_rf, "Tuned Random Forest")

# ---------------------------------
# 9 Save Best Model & Encoders
# ---------------------------------
joblib.dump(best_rf, "visa_processing_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl") # ADDED THIS LINE
print("\nModel saved as visa_processing_model.pkl")
print("Encoders saved as label_encoders.pkl")

# ---------------------------------
# 10 Test Prediction
# ---------------------------------
sample = X_test.iloc[0:1]
prediction = best_rf.predict(sample)
print("\nPredicted Processing Time:", prediction[0], "days")
