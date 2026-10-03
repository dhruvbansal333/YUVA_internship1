"""Week 4 - train the final model, evaluate it, and serialize it with joblib.

Run from the repository root:
    python week_4/src/train_model.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

WEEK = Path(__file__).resolve().parents[1]
REPO = WEEK.parent
RS = 42

FEATURES = ["Age", "Gender", "BMI", "Blood_Pressure_mmHg", "Cholesterol_mg_dL",
            "Glucose_mg_dL", "Heart_Rate_bpm", "Smoking", "Exercise_Level"]
NUM = ["Age", "BMI", "Blood_Pressure_mmHg", "Cholesterol_mg_dL", "Glucose_mg_dL", "Heart_Rate_bpm"]
CAT = ["Gender", "Smoking", "Exercise_Level"]


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])
    # hyper-parameters come from the Week 3 GridSearchCV
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=1,
                                class_weight="balanced_subsample", random_state=RS)
    return Pipeline([("prep", pre), ("model", rf)])


def main():
    df = pd.read_excel(REPO / "data" / "raw" / "Disease_Prediction_Health_Dataset.xlsx", sheet_name="Health_Data")
    X, y = df[FEATURES], (df["Disease"] == "Disease").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

    pipe = build_pipeline()
    cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), scoring="f1")
    pipe.fit(X_tr, y_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    metrics = {"accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
               "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, proba),
               "cv_f1_mean": cv_f1.mean(), "cv_f1_std": cv_f1.std(),
               "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
               "train_rows": int(len(X_tr)), "test_rows": int(len(X_te))}
    metrics = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()}

    # refit on ALL data for the deployed model (hold-out numbers above remain the honest estimate)
    final = build_pipeline().fit(X, y)
    (WEEK / "models").mkdir(exist_ok=True)
    joblib.dump(final, WEEK / "models" / "disease_model.joblib")
    json.dump({"features": FEATURES, "metrics_holdout": metrics, "model": "RandomForestClassifier",
               "target": "Disease (1) / No Disease (0)"},
              open(WEEK / "models" / "model_metadata.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["No Disease", "Disease"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Final model - hold-out confusion matrix")
    (WEEK / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(WEEK / "outputs" / "figures" / "w4_final_confusion_matrix.png", dpi=130); plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
