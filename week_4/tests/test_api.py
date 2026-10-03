from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
HIGH = {"Age": 70, "Gender": "Male", "BMI": 33, "Blood_Pressure_mmHg": 170, "Cholesterol_mg_dL": 260,
        "Glucose_mg_dL": 150, "Heart_Rate_bpm": 80, "Smoking": "Yes", "Exercise_Level": "Low"}
LOW = {"Age": 25, "Gender": "Female", "BMI": 22, "Blood_Pressure_mmHg": 110, "Cholesterol_mg_dL": 170,
       "Glucose_mg_dL": 90, "Heart_Rate_bpm": 68, "Smoking": "No", "Exercise_Level": "High"}


def test_health():
    r = client.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_ranks_risk_sensibly():
    hi, lo = client.post("/predict", json=HIGH).json(), client.post("/predict", json=LOW).json()
    assert hi["disease_probability"] > lo["disease_probability"]
    assert hi["prediction"] == "Disease" and lo["prediction"] == "No Disease"


def test_batch():
    r = client.post("/predict-batch", json=[HIGH, LOW]); assert r.status_code == 200 and len(r.json()) == 2


def test_validation_error():
    bad = dict(LOW, Age=-5); assert client.post("/predict", json=bad).status_code == 422
    bad = dict(LOW, Gender="Other"); assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict-batch", json=[]).status_code == 400
