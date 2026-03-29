from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


COUNTRIES = [
    "India", "China", "Canada", "Mexico", "Brazil", "Nigeria", "Philippines",
    "Pakistan", "Bangladesh", "UK", "Germany", "France", "Australia", "Kenya"
]

VISA_TYPES = [
    "Tourist", "Student", "Work", "Business", "Dependent", "Transit"
]

OFFICES = [
    "New Delhi", "Mumbai", "Chennai", "London", "Ottawa", "Sydney", "Dubai",
    "Berlin", "Paris", "Singapore", "Johannesburg"
]

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}

def season_for_month(month: int) -> str:
    return SEASONS[month]

def month_to_quarter(month: int) -> str:
    return f"Q{((month - 1) // 3) + 1}"

def generate_synthetic_visa_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic dataset with realistic patterns:
    - work/student applications take longer
    - some offices have higher backlog
    - holidays / peak seasons increase processing time
    """
    rng = np.random.default_rng(seed)

    submission_dates = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2021-01-01").value // 10**9,
            pd.Timestamp("2025-12-31").value // 10**9,
            size=n_rows,
        ),
        unit="s",
    ).normalize()

    countries = rng.choice(COUNTRIES, size=n_rows, replace=True)
    visa_types = rng.choice(VISA_TYPES, size=n_rows, replace=True, p=[0.35, 0.20, 0.16, 0.12, 0.12, 0.05])
    offices = rng.choice(OFFICES, size=n_rows, replace=True)

    month = submission_dates.month
    year = submission_dates.year
    season = [season_for_month(m) for m in month]

    # Base days by visa type
    visa_base = {
        "Tourist": 18,
        "Transit": 8,
        "Business": 20,
        "Student": 35,
        "Work": 45,
        "Dependent": 28,
    }
    office_backlog = {
        "New Delhi": 10,
        "Mumbai": 8,
        "Chennai": 7,
        "London": 6,
        "Ottawa": 5,
        "Sydney": 4,
        "Dubai": 9,
        "Berlin": 5,
        "Paris": 6,
        "Singapore": 4,
        "Johannesburg": 7,
    }
    country_factor = {
        "India": 6, "China": 7, "Canada": 2, "Mexico": 5, "Brazil": 4, "Nigeria": 8,
        "Philippines": 7, "Pakistan": 8, "Bangladesh": 9, "UK": 3, "Germany": 2,
        "France": 3, "Australia": 2, "Kenya": 6
    }
    season_factor = {"Winter": 6, "Spring": 2, "Summer": 4, "Autumn": 1}
    year_trend = {2021: 8, 2022: 5, 2023: 2, 2024: -1, 2025: -2}

    days = []
    for i in range(n_rows):
        base = visa_base[visa_types[i]]
        days_i = (
            base
            + office_backlog[offices[i]]
            + country_factor[countries[i]]
            + season_factor[season[i]]
            + year_trend[int(year[i])]
            + rng.normal(0, 6)
        )

        # Peak months near summer / year-end add variability
        if month[i] in [6, 7, 8, 12]:
            days_i += rng.normal(4, 3)

        # Faster processing for simple cases
        if visa_types[i] in ["Transit", "Tourist"] and countries[i] in ["Canada", "Germany", "France", "Australia"]:
            days_i -= rng.normal(3, 2)

        # Rare long-tail delays
        if rng.random() < 0.05:
            days_i += rng.integers(20, 60)

        days.append(max(1, int(round(days_i))))

    decision_dates = submission_dates + pd.to_timedelta(days, unit="D")

    df = pd.DataFrame({
        "application_id": [f"VISA-{i+1:07d}" for i in range(n_rows)],
        "submission_date": submission_dates,
        "decision_date": decision_dates,
        "applicant_country": countries,
        "visa_type": visa_types,
        "processing_office": offices,
        "submission_month": month.astype(int),
        "submission_year": year.astype(int),
        "season": season,
        "processing_time_days": days,
    })

    # Inject a small amount of missingness
    for col in ["applicant_country", "visa_type", "processing_office"]:
        mask = rng.random(n_rows) < 0.02
        df.loc[mask, col] = None

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["submission_date"] = pd.to_datetime(out["submission_date"], errors="coerce")
    out["decision_date"] = pd.to_datetime(out["decision_date"], errors="coerce")
    out["submission_month"] = out["submission_date"].dt.month
    out["submission_year"] = out["submission_date"].dt.year
    out["submission_dayofweek"] = out["submission_date"].dt.dayofweek
    out["is_peak_month"] = out["submission_month"].isin([6, 7, 8, 12]).astype(int)
    out["season"] = out["submission_month"].map(season_for_month)
    out["quarter"] = out["submission_month"].map(month_to_quarter)
    out["target_days"] = (out["decision_date"] - out["submission_date"]).dt.days
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    categorical_cols = ["applicant_country", "visa_type", "processing_office", "season", "quarter"]
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str)

    numeric_cols = [c for c in out.columns if c.startswith("submission_") or c.startswith("is_")]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def get_feature_frame(df: pd.DataFrame):
    feature_cols = [
        "applicant_country",
        "visa_type",
        "processing_office",
        "submission_month",
        "submission_year",
        "submission_dayofweek",
        "is_peak_month",
        "season",
        "quarter",
    ]
    target_col = "target_days"
    X = df[feature_cols].copy()
    y = df[target_col].astype(float).copy()
    return X, y, feature_cols


def train_validation_split(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2):
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, random_state=seed)