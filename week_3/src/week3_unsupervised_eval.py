# %% [markdown]
# # Week 3 - Unsupervised Learning & Model Evaluation
# **Dataset:** Disease Prediction Health Dataset
#
# Part A: clustering (K-Means, Hierarchical) and dimensionality reduction (PCA) on the health measurements.
# Part B: model evaluation (cross-validation, confusion matrix, precision / recall / F1, ROC-AUC) and hyperparameter tuning with GridSearchCV.

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, adjusted_rand_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, roc_auc_score, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_validate, GridSearchCV)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    WEEK = Path(__file__).resolve().parents[1]
except NameError:
    WEEK = Path.cwd().parent
REPO = WEEK.parent
FIG = WEEK / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RS = 42
sns.set_theme(style="whitegrid")
results = {}

df = pd.read_excel(REPO / "data" / "raw" / "Disease_Prediction_Health_Dataset.xlsx", sheet_name="Health_Data")
y = (df["Disease"] == "Disease").astype(int)
X = df.drop(columns=["Disease"])
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])

# %% [markdown]
# ## Part A - Unsupervised learning
# The target (`Disease`) is **not** used for clustering; it is only used afterwards to see whether the clusters line up with disease status.
#
# ### A1. Prepare data (scaled numeric + one-hot categorical)

# %%
Z = make_preprocessor().fit_transform(X)
print("Matrix for clustering:", Z.shape)

# %% [markdown]
# ### A2. K-Means: choose k with the elbow method and silhouette score

# %%
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(Z)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Z, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, "o-", color="darkorange"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(FIG / "w3_kmeans_elbow_silhouette.png", dpi=130); plt.close()
best_k_sil = int(list(ks)[int(np.argmax(sil))])
print({k: round(s, 4) for k, s in zip(ks, sil)}, "-> best silhouette k =", best_k_sil)

# %% [markdown]
# ### A3. Fit final K-Means and profile the clusters
# The silhouette score is highest for very small k, but k=3 is a more useful segmentation and still has a reasonable silhouette, so k=3 is used for profiling.

# %%
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=RS).fit(Z)
df["KMeans_Cluster"] = km.labels_
km_sil = silhouette_score(Z, km.labels_)
profile = df.groupby("KMeans_Cluster").agg(
    Size=("Age", "size"), Age=("Age", "mean"), BMI=("BMI", "mean"), BP=("Blood_Pressure_mmHg", "mean"),
    Chol=("Cholesterol_mg_dL", "mean"), Glucose=("Glucose_mg_dL", "mean"), HR=("Heart_Rate_bpm", "mean"),
    Disease_rate=("Disease", lambda s: (s == "Disease").mean()),
    Smoker_share=("Smoking", lambda s: (s == "Yes").mean())).round(3)
print("K-Means silhouette (k=3):", round(km_sil, 4)); print(profile)
ari_km = adjusted_rand_score(y, km.labels_)

# %% [markdown]
# ### A4. Hierarchical clustering (Ward linkage) and dendrogram

# %%
rng = np.random.RandomState(RS)
idx = rng.choice(len(Z), 300, replace=False)       # dendrogram on a 300-row sample for readability
link = linkage(Z[idx], method="ward")
plt.figure(figsize=(11, 4.5))
dendrogram(link, truncate_mode="lastp", p=24, no_labels=True, color_threshold=0.7 * max(link[:, 2]))
plt.title("Hierarchical clustering dendrogram (Ward, 300-row sample)"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig(FIG / "w3_dendrogram.png", dpi=130); plt.close()

agg = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Z)
df["Hier_Cluster"] = agg.labels_
agg_sil = silhouette_score(Z, agg.labels_)
print("Hierarchical silhouette (k=3):", round(agg_sil, 4))
print("Agreement K-Means vs Hierarchical (ARI):", round(adjusted_rand_score(km.labels_, agg.labels_), 4))
ari_agg = adjusted_rand_score(y, agg.labels_)

# %% [markdown]
# ### A5. Dimensionality reduction with PCA

# %%
pca_full = PCA().fit(Z)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95) + 1)
pca2 = PCA(n_components=2, random_state=RS)
P = pca2.fit_transform(Z)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(range(1, len(cum) + 1), cum, "o-"); ax[0].axhline(0.95, ls="--", c="r")
ax[0].set_title(f"PCA cumulative variance ({n95} comps reach 95%)"); ax[0].set_xlabel("Components")
sc = ax[1].scatter(P[:, 0], P[:, 1], c=df["KMeans_Cluster"], cmap="viridis", s=8, alpha=0.7)
ax[1].set_title("PCA projection coloured by K-Means cluster"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
ax[2].scatter(P[:, 0], P[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.6)
ax[2].set_title("PCA projection coloured by true Disease label"); ax[2].set_xlabel("PC1")
plt.tight_layout(); plt.savefig(FIG / "w3_pca.png", dpi=130); plt.close()
print("Variance explained by PC1, PC2:", pca2.explained_variance_ratio_.round(3), "| comps for 95%:", n95)

results["clustering"] = {
    "kmeans_sil_by_k": {int(k): round(float(s), 4) for k, s in zip(ks, sil)},
    "best_k_by_silhouette": best_k_sil, "chosen_k": K,
    "kmeans_silhouette": round(float(km_sil), 4), "hier_silhouette": round(float(agg_sil), 4),
    "kmeans_vs_disease_ARI": round(float(ari_km), 4), "hier_vs_disease_ARI": round(float(ari_agg), 4),
    "kmeans_profile": profile.reset_index().to_dict("records"),
    "pca_var_ratio_2": [round(float(v), 4) for v in pca2.explained_variance_ratio_], "pca_n_for_95": n95}

# %% [markdown]
# ## Part B - Model evaluation & tuning
# ### B1. Hold-out split and 5-fold stratified cross-validation

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RS),
}
cv_rows = []
for name, m in base_models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", m)])
    cv = cross_validate(pipe, X_train, y_train, cv=skf, scoring=scoring)
    row = {"Model": name}
    for s in scoring:
        row[s] = f"{cv['test_' + s].mean():.3f} +/- {cv['test_' + s].std():.3f}"
    cv_rows.append(row)
cv_table = pd.DataFrame(cv_rows)
print(cv_table.to_string(index=False))

# %% [markdown]
# ### B2. Hyperparameter tuning with GridSearchCV (optimising F1 for the minority 'Disease' class)

# %%
grids = {
    "Logistic Regression": (Pipeline([("prep", make_preprocessor()),
                                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS))]),
                            {"model__C": [0.01, 0.1, 1, 10, 100]}),
    "Random Forest": (Pipeline([("prep", make_preprocessor()),
                                ("model", RandomForestClassifier(class_weight="balanced_subsample", random_state=RS))]),
                      {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, 8, None],
                       "model__min_samples_leaf": [1, 5, 10]}),
}
tuned, tuning_info = {}, {}
for name, (pipe, grid) in grids.items():
    gs = GridSearchCV(pipe, grid, cv=skf, scoring="f1", n_jobs=-1).fit(X_train, y_train)
    tuned[name] = gs.best_estimator_
    tuning_info[name] = {"best_params": {k.replace("model__", ""): (v if v is not None else "None") for k, v in gs.best_params_.items()},
                         "best_cv_f1": round(float(gs.best_score_), 4), "n_combinations": len(gs.cv_results_["params"])}
    print(name, tuning_info[name])

# %% [markdown]
# ### B3. Final evaluation on the untouched test set

# %%
eval_rows, probs, preds = [], {}, {}
for name in base_models:
    base = Pipeline([("prep", make_preprocessor()), ("model", base_models[name])]).fit(X_train, y_train)
    for label, est in [(name + " (default)", base), (name + " (tuned)", tuned[name])]:
        p = est.predict(X_test); pr = est.predict_proba(X_test)[:, 1]
        probs[label], preds[label] = pr, p
        eval_rows.append({"Model": label, "Accuracy": accuracy_score(y_test, p), "Precision": precision_score(y_test, p),
                          "Recall": recall_score(y_test, p), "F1": f1_score(y_test, p), "ROC_AUC": roc_auc_score(y_test, pr)})
eval_table = pd.DataFrame(eval_rows).round(4)
print(eval_table.to_string(index=False))

best_name = eval_table.sort_values("F1", ascending=False).iloc[0]["Model"]
print("\nBest on test F1:", best_name)
print(classification_report(y_test, preds[best_name], target_names=["No Disease", "Disease"]))

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ConfusionMatrixDisplay(confusion_matrix(y_test, preds[best_name]), display_labels=["No Disease", "Disease"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title(f"Confusion matrix - {best_name}")
for label, pr in probs.items():
    fpr, tpr, _ = roc_curve(y_test, pr)
    ax[1].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test, pr):.3f})")
ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set_title("ROC curves"); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate"); ax[1].legend(fontsize=7)
for label, pr in probs.items():
    pcs, rcs, _ = precision_recall_curve(y_test, pr)
    ax[2].plot(rcs, pcs, label=label)
ax[2].set_title("Precision-recall curves"); ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "w3_evaluation.png", dpi=130); plt.close()

results.update({
    "cv_table": cv_table.to_dict("records"), "tuning": tuning_info,
    "test_eval": eval_table.to_dict("records"), "best_model": best_name,
    "best_cm": confusion_matrix(y_test, preds[best_name]).tolist(),
    "best_report": classification_report(y_test, preds[best_name], target_names=["No Disease", "Disease"], output_dict=True)})
json.dump(results, open(WEEK / "outputs" / "week3_metrics.json", "w"), indent=2, default=str)
df.to_csv(WEEK / "outputs" / "clustered_dataset.csv", index=False)
eval_table.to_csv(WEEK / "outputs" / "model_evaluation.csv", index=False)
print("saved")
