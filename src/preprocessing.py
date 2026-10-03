"""
Week 1 - ML Data Preprocessing
Run from the repository root:
    python src/preprocessing.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw" / "Disease_Prediction_Health_Dataset.xlsx"
OUTPUT_FILE = ROOT / "data" / "processed" / "cleaned_dataset.csv"


def main():
    df = pd.read_excel(INPUT_FILE, sheet_name="Health_Data")

    print("Original shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    X = df.drop(columns=["Disease"])
    y = (df["Disease"] == "Disease").astype(int)

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )),
            ]), categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=min(8, X_processed.shape[1])
    )
    X_selected = selector.fit_transform(X_processed, y)
    selected_features = feature_names[selector.get_support()]

    result = pd.DataFrame(X_selected, columns=selected_features)
    result["Disease"] = y

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nSelected features:")
    for feature in selected_features:
        print(" -", feature)

    print("\nProcessed shape:", result.shape)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
