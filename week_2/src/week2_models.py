# %% [markdown]
# # Week 2 - Supervised Machine Learning Models
# **Dataset:** Disease Prediction Health Dataset (3,000 rows, 10 columns)
#
# * **Classification task:** predict `Disease` (Disease / No Disease) with Logistic Regression, Decision Tree, Random Forest and KNN.
# * **Regression task:** predict `Cholesterol_mg_dL` with Linear Regression and Random Forest Regressor.
#
# Preprocessing follows the Week 1 pipeline (median imputation + scaling, one-hot encoding). The train/test split is done *before* fitting any transformer to avoid data leakage.

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             mean_absolute_error, mean_squared_error, r2_score)

try:
    WEEK = Path(__file__).resolve().parents[1]
except NameError:                       # running inside a notebook
    WEEK = Path.cwd().parent
REPO = WEEK.parent
FIG = WEEK / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RS = 42
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load data

# %%
df = pd.read_excel(REPO / "data" / "raw" / "Disease_Prediction_Health_Dataset.xlsx", sheet_name="Health_Data")
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Classification: predicting `Disease`

# %%
X = df.drop(columns=["Disease"])
y = (df["Disease"] == "Disease").astype(int)
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

def make_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
print("Train:", X_train.shape, "Test:", X_test.shape, "| Disease rate (train/test):", y_train.mean().round(3), y_test.mean().round(3))

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RS),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
}

clf_rows, fitted = [], {}
for name, model in clf_models.items():
    pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    fitted[name] = (pipe, pred, proba)
    clf_rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred),
                     "Recall": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred),
                     "ROC_AUC": roc_auc_score(y_test, proba)})
clf_results = pd.DataFrame(clf_rows).round(4)
clf_results

# %% [markdown]
# ### Confusion matrices

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, (_, pred, _)) in zip(axes, fitted.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["No Disease", "Disease"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout(); plt.savefig(FIG / "w2_confusion_matrices.png", dpi=130); plt.close()

# %% [markdown]
# ### Model comparison chart

# %%
ax = clf_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].plot(kind="bar", figsize=(10, 5), rot=15)
ax.set_title("Classification model comparison (test set)"); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(FIG / "w2_classification_comparison.png", dpi=130); plt.close()

# %% [markdown]
# ### Feature importance (Random Forest) and logistic-regression coefficients

# %%
rf_pipe = fitted["Random Forest"][0]
names = rf_pipe.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names).sort_values()
imp.plot(kind="barh", figsize=(7, 5), title="Random Forest feature importance")
plt.tight_layout(); plt.savefig(FIG / "w2_rf_feature_importance.png", dpi=130); plt.close()

lr_pipe = fitted["Logistic Regression"][0]
coef = pd.Series(lr_pipe.named_steps["model"].coef_[0], index=names).sort_values()
print(coef.round(3))

# %% [markdown]
# ## 3. Regression: predicting `Cholesterol_mg_dL`
# Cholesterol is a continuous clinical measurement, so it is a natural regression target. All other columns, including `Disease`, are used as inputs.

# %%
Xr = df.drop(columns=["Cholesterol_mg_dL"])
yr = df["Cholesterol_mg_dL"]
Xr = Xr.assign(Disease=(Xr["Disease"] == "Disease").astype(int))
num_r = Xr.select_dtypes(include=np.number).columns.tolist()
cat_r = Xr.select_dtypes(exclude=np.number).columns.tolist()
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=RS)

reg_models = {"Linear Regression": LinearRegression(),
              "Random Forest Regressor": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=RS)}
reg_rows, reg_fitted = [], {}
for name, model in reg_models.items():
    pipe = Pipeline([("prep", make_preprocessor(num_r, cat_r)), ("model", model)])
    pipe.fit(Xr_tr, yr_tr)
    pred = pipe.predict(Xr_te)
    reg_fitted[name] = pred
    reg_rows.append({"Model": name, "MAE": mean_absolute_error(yr_te, pred),
                     "RMSE": np.sqrt(mean_squared_error(yr_te, pred)), "R2": r2_score(yr_te, pred)})
baseline = np.full_like(yr_te, yr_tr.mean(), dtype=float)
reg_rows.append({"Model": "Baseline (predict mean)", "MAE": mean_absolute_error(yr_te, baseline),
                 "RMSE": np.sqrt(mean_squared_error(yr_te, baseline)), "R2": r2_score(yr_te, baseline)})
reg_results = pd.DataFrame(reg_rows).round(4)
reg_results

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, pred) in zip(axes, reg_fitted.items()):
    ax.scatter(yr_te, pred, s=8, alpha=0.5)
    lims = [yr_te.min(), yr_te.max()]
    ax.plot(lims, lims, "r--"); ax.set_title(name); ax.set_xlabel("Actual cholesterol"); ax.set_ylabel("Predicted")
plt.tight_layout(); plt.savefig(FIG / "w2_regression_actual_vs_pred.png", dpi=130); plt.close()

# %% [markdown]
# ## 4. Save results

# %%
clf_results.to_csv(WEEK / "outputs" / "classification_results.csv", index=False)
reg_results.to_csv(WEEK / "outputs" / "regression_results.csv", index=False)
json.dump({"classification": clf_results.to_dict("records"),
           "regression": reg_results.to_dict("records"),
           "test_size": int(len(y_test)), "train_size": int(len(y_train)),
           "rf_importance": imp.sort_values(ascending=False).round(4).to_dict(),
           "lr_coef": coef.round(3).to_dict(),
           "cm": {n: confusion_matrix(y_test, p).tolist() for n, (_, p, _) in fitted.items()}},
          open(WEEK / "outputs" / "week2_metrics.json", "w"), indent=2)
print(clf_results.to_string(index=False)); print(); print(reg_results.to_string(index=False))
