from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelBundle:
    model: object
    lower_model: object
    upper_model: object
    feature_columns: list
    metrics: dict


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numerical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical),
            ("num", numerical_pipe, numerical),
        ],
        remainder="drop",
    )


def get_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=250, random_state=random_state, n_jobs=-1, min_samples_leaf=2
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_select_model(X_train, y_train, X_valid, y_valid, random_state: int = 42):
    preprocessor = build_preprocessor(X_train)
    models = get_models(random_state=random_state)

    results = {}
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)
        results[name] = evaluate(y_valid, pred)
        fitted[name] = pipe

    best_name = sorted(results.keys(), key=lambda k: (results[k]["mae"], results[k]["rmse"]))[0]
    best_model = fitted[best_name]

    # Quantile models for interval estimation
    lower = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=random_state)),
    ])
    upper = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=random_state)),
    ])
    lower.fit(X_train, y_train)
    upper.fit(X_train, y_train)

    bundle = ModelBundle(
        model=best_model,
        lower_model=lower,
        upper_model=upper,
        feature_columns=list(X_train.columns),
        metrics=results,
    )
    return bundle, best_name


def save_bundle(bundle: ModelBundle, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.model, out_dir / "best_model.joblib")
    joblib.dump(bundle.lower_model, out_dir / "lower_model.joblib")
    joblib.dump(bundle.upper_model, out_dir / "upper_model.joblib")
    joblib.dump(bundle.feature_columns, out_dir / "feature_columns.joblib")
    joblib.dump(bundle.metrics, out_dir / "metrics.joblib")


def load_bundle(model_dir: str | Path):
    model_dir = Path(model_dir)
    return {
        "model": joblib.load(model_dir / "best_model.joblib"),
        "lower_model": joblib.load(model_dir / "lower_model.joblib"),
        "upper_model": joblib.load(model_dir / "upper_model.joblib"),
        "feature_columns": joblib.load(model_dir / "feature_columns.joblib"),
        "metrics": joblib.load(model_dir / "metrics.joblib"),
    }


def predict_with_interval(bundle, input_df: pd.DataFrame) -> dict:
    pred = float(bundle["model"].predict(input_df)[0])
    lower = float(bundle["lower_model"].predict(input_df)[0])
    upper = float(bundle["upper_model"].predict(input_df)[0])

    # Keep interval sensible
    low = max(1, min(lower, pred))
    high = max(low, max(upper, pred))
    return {
        "predicted_days": round(pred, 1),
        "lower_days": round(low, 1),
        "upper_days": round(high, 1),
        "range_text": f"{int(round(low))}–{int(round(high))} days",
    }