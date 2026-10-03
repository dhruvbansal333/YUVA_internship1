# Week 2 - Supervised Machine Learning Models

Builds regression and classification models with Scikit-learn on the Disease Prediction Health Dataset (same data as Week 1).

| Task | Target | Models |
|---|---|---|
| Classification | `Disease` | Logistic Regression, Decision Tree, Random Forest, KNN |
| Regression | `Cholesterol_mg_dL` | Linear Regression, Random Forest Regressor (+ mean baseline) |

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC (classification); MAE, RMSE, R2 (regression). Train/test split happens before preprocessing is fitted (no leakage); `class_weight="balanced"` handles the 77/23 class imbalance.

```
python week_2/src/week2_models.py        # run from repo root, writes week_2/outputs/
jupyter notebook week_2/notebooks/Week_2_Supervised_Models.ipynb
```
Report: `docs/Week_2_Report.docx`
