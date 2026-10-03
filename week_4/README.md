# Week 4 - AI Project Deployment & Capstone

End-to-end disease-risk prediction service: data preprocessing -> model training -> serialization (joblib) -> REST API (FastAPI) -> tests -> Docker.

## Run
```
pip install -r week_4/requirements.txt
python week_4/src/train_model.py          # from repo root: trains + saves models/disease_model.joblib
cd week_4
uvicorn app.main:app --reload             # API on http://127.0.0.1:8000  (Swagger UI at /docs)
python -m pytest -q tests                 # 4 API tests
```
Docker: `cd week_4 && docker build -t disease-api . && docker run -p 8000:8000 disease-api`

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | service status |
| GET | `/model-info` | features + hold-out metrics |
| POST | `/predict` | one patient -> prediction, probability, risk level |
| POST | `/predict-batch` | up to 500 patients |

Example:
```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"Age":70,"Gender":"Male","BMI":33,"Blood_Pressure_mmHg":170,"Cholesterol_mg_dL":260,"Glucose_mg_dL":150,"Heart_Rate_bpm":80,"Smoking":"Yes","Exercise_Level":"Low"}'
```
Report: `docs/Week_4_Report.docx`. *Educational project - not medical advice.*
