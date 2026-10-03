"""Disease-risk prediction API (FastAPI).

Run from the week_4 folder:
    uvicorn app.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""
from pathlib import Path
from typing import List, Literal
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
model = joblib.load(MODEL_DIR / "disease_model.joblib")
meta = json.load(open(MODEL_DIR / "model_metadata.json"))
FEATURES = meta["features"]

app = FastAPI(title="Disease Prediction API", version="1.0.0",
              description="Predicts disease risk from basic health measurements. Educational project - not medical advice.")


class Patient(BaseModel):
    Age: int = Field(..., ge=1, le=120, examples=[56])
    Gender: Literal["Male", "Female"]
    BMI: float = Field(..., ge=10, le=70, examples=[26.6])
    Blood_Pressure_mmHg: float = Field(..., ge=60, le=260, examples=[140])
    Cholesterol_mg_dL: float = Field(..., ge=80, le=500, examples=[210])
    Glucose_mg_dL: float = Field(..., ge=40, le=500, examples=[110])
    Heart_Rate_bpm: float = Field(..., ge=30, le=220, examples=[72])
    Smoking: Literal["Yes", "No"]
    Exercise_Level: Literal["Low", "Moderate", "High"]


class Prediction(BaseModel):
    prediction: Literal["Disease", "No Disease"]
    disease_probability: float
    risk_level: Literal["Low", "Medium", "High"]


def _risk(p: float) -> str:
    return "Low" if p < 0.3 else "Medium" if p < 0.6 else "High"


def _predict(rows: List[Patient]) -> List[Prediction]:
    X = pd.DataFrame([r.model_dump() for r in rows])[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    return [Prediction(prediction="Disease" if p >= 0.5 else "No Disease",
                       disease_probability=round(float(p), 4), risk_level=_risk(float(p))) for p in proba]


@app.get("/")
def root():
    return {"message": "Disease Prediction API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model"]}


@app.get("/model-info")
def model_info():
    return meta


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    return _predict([patient])[0]


@app.post("/predict-batch", response_model=List[Prediction])
def predict_batch(patients: List[Patient]):
    if not patients:
        raise HTTPException(status_code=400, detail="Send at least one record.")
    if len(patients) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 records per request.")
    return _predict(patients)
