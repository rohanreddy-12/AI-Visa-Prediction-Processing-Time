from __future__ import annotations

from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predictor import VisaEstimator


app = FastAPI(title="Visa Processing Time API", version="1.0.0")
estimator = VisaEstimator("artifacts")


class PredictionRequest(BaseModel):
    applicant_country: str = Field(...)
    visa_type: str = Field(...)
    processing_office: str = Field(...)
    submission_date: date = Field(...)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictionRequest):
    result = estimator.predict(
        req.applicant_country,
        req.visa_type,
        req.processing_office,
        req.submission_date,
    )
    return result