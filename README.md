# YuvaIntern Week 1 - Python for Machine Learning & Data Preprocessing

This repository contains the Week 1 submission for the **AI Pioneers Internship (Machine Learning)**.

## Task

**Week 1 Task: Python for Machine Learning & Data Preprocessing**

The objective is to demonstrate:

- Python fundamentals for ML
- NumPy and Pandas
- Dataset loading and inspection
- Data cleaning
- Missing-value checking and handling
- Duplicate detection
- Exploratory Data Analysis (EDA)
- Categorical-variable encoding
- Numerical feature scaling/normalization
- Feature selection
- Export of a cleaned/processed dataset
- Documentation of preprocessing steps

## Dataset

The project uses the **Disease Prediction Health Dataset**.

### Dataset summary

- Rows: 3,000
- Columns: 10
- Target: `Disease`
- Numerical features:
  - `Age`
  - `BMI`
  - `Blood_Pressure_mmHg`
  - `Cholesterol_mg_dL`
  - `Glucose_mg_dL`
  - `Heart_Rate_bpm`
- Categorical features:
  - `Gender`
  - `Smoking`
  - `Exercise_Level`

The current dataset contains **no missing values** and **no duplicate rows**. The preprocessing pipeline still includes imputation steps so that it remains robust to missing values if they occur in future data.

## Preprocessing workflow

```text
Raw Excel Dataset
       |
       v
Load with Pandas
       |
       v
Inspect structure, types and statistics
       |
       v
Check missing values and duplicates
       |
       v
Separate features and target
       |
       v
Identify numerical/categorical features
       |
       v
EDA and visualization
       |
       v
Train/Test Split
       |
       v
Numerical: Imputation + StandardScaler
Categorical: Imputation + OneHotEncoder
       |
       v
Mutual Information Feature Selection
       |
       v
Export processed CSV
```

## Project structure

```text
YuvaIntern-Week1-ML-Data-Preprocessing/
│
├── data/
│   ├── raw/
│   │   └── Disease_Prediction_Health_Dataset.xlsx
│   └── processed/
│       └── cleaned_dataset.csv
│
├── notebooks/
│   └── Week_1_Data_Preprocessing.ipynb
│
├── src/
│   └── preprocessing.py
│
├── outputs/
│   └── figures/
│
├── docs/
│   └── Week_1_Report.docx
│
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd YuvaIntern-Week1-ML-Data-Preprocessing
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the preprocessing script

```bash
python src/preprocessing.py
```

### 5. Open the notebook

```bash
jupyter notebook
```

Then open:

`notebooks/Week_1_Data_Preprocessing.ipynb`

## Key techniques used

### Missing values
Missing values are checked explicitly. The current dataset has none. The pipeline uses median imputation for numerical columns and most-frequent imputation for categorical columns as a robust safeguard.

### Categorical encoding
Categorical variables are converted to numerical representations using `OneHotEncoder`.

### Normalization / scaling
Numerical features are standardized using `StandardScaler`.

### Feature selection
`SelectKBest` with mutual information is used to identify the most informative transformed features with respect to the target.

### Data leakage prevention
For model-development workflow, the train/test split is performed before fitting preprocessing transformations. The preprocessing objects are fitted on the training data and then applied to the test data.

## Deliverables

- Jupyter Notebook showing the complete workflow
- Python preprocessing script
- Raw dataset
- Processed dataset
- Project README
- Week 1 Word report

## Internship submission

Use the GitHub URL of this repository in the YuvaIntern Week 1 submission form and upload the Word report from the `docs` folder.
