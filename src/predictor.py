from __future__ import annotations

from datetime import date

import pandas as pd

from src.model import load_bundle, predict_with_interval


def build_input(
    applicant_country: str,
    visa_type: str,
    processing_office: str,
    submission_date: str | date,
):
    submission_date = pd.to_datetime(submission_date)
    month = int(submission_date.month)
    year = int(submission_date.year)
    dayofweek = int(submission_date.dayofweek)
    is_peak_month = int(month in [6, 7, 8, 12])

    season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}[month]
    quarter = f"Q{((month - 1) // 3) + 1}"

    return pd.DataFrame([{
        "applicant_country": applicant_country,
        "visa_type": visa_type,
        "processing_office": processing_office,
        "submission_month": month,
        "submission_year": year,
        "submission_dayofweek": dayofweek,
        "is_peak_month": is_peak_month,
        "season": season,
        "quarter": quarter,
    }])


class VisaEstimator:
    def __init__(self, model_dir: str = "artifacts"):
        self.bundle = load_bundle(model_dir)

    def predict(self, applicant_country, visa_type, processing_office, submission_date):
        x = build_input(applicant_country, visa_type, processing_office, submission_date)
        return predict_with_interval(self.bundle, x)