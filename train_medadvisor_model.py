import os

# ============================================================
# Limit CPU threads
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATASET_PATH = (
    "datasets/"
    "medadvisor_cardiovascular_synthetic_20000_v2.csv"
)

MODEL_DIR = "ml/models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "medadvisor_cardiovascular_model.pkl"
)

COMPARISON_PATH = os.path.join(
    MODEL_DIR,
    "medadvisor_model_comparison.csv"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "medadvisor_model_metrics.txt"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print()
print("=" * 70)
print("MedAdvisorAI - Cardiovascular Risk Prediction")
print("V2 Model Training")
print("=" * 70)

print()
print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print()
print("Dataset loaded successfully.")

print("Rows    :", len(df))
print("Columns :", len(df))


# ============================================================
# 2. BASIC DATA VALIDATION
# ============================================================

print()
print("=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

print()
print("Missing values:")

print(
    df.isnull().sum()
)

print()
print(
    "Total missing values:",
    df.isnull().sum().sum()
)

print()
print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print()
print("Target distribution:")

print(
    df["cardiovascular_risk"]
    .value_counts()
    .sort_index()
)

print()
print("Target percentage:")

print(
    df["cardiovascular_risk"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ============================================================
# 3. PARSE BLOOD PRESSURE
# ============================================================

print()
print("Parsing blood pressure...")

bp_split = (
    df["blood_pressure"]
    .astype(str)
    .str.split("/", expand=True)
)

df["systolic_bp"] = pd.to_numeric(
    bp_split[0],
    errors="coerce"
)

df["diastolic_bp"] = pd.to_numeric(
    bp_split[1],
    errors="coerce"
)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print()
print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# BMI
# ------------------------------------------------------------

df["bmi"] = (
    df["weight_kg"]
    /
    (df["height_cm"] / 100) ** 2
)


# ------------------------------------------------------------
# Pulse Pressure
# ------------------------------------------------------------

df["pulse_pressure"] = (
    df["systolic_bp"]
    -
    df["diastolic_bp"]
)


# ------------------------------------------------------------
# Mean Arterial Pressure
# ------------------------------------------------------------

df["mean_arterial_pressure"] = (
    df["diastolic_bp"]
    +
    (
        df["systolic_bp"]
        -
        df["diastolic_bp"]
    ) / 3
)


# ------------------------------------------------------------
# BMI Category
# ------------------------------------------------------------

df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[
        0,
        18.5,
        25,
        30,
        100
    ],
    labels=[
        "Underweight",
        "Normal",
        "Overweight",
        "Obese"
    ],
    include_lowest=True
)


# ------------------------------------------------------------
# Blood Pressure Category
# ------------------------------------------------------------

def classify_bp(systolic):

    if systolic < 120:
        return "Normal"

    elif systolic < 130:
        return "Elevated"

    elif systolic < 140:
        return "High_Stage_1"

    elif systolic < 180:
        return "High_Stage_2"

    else:
        return "Very_High"


df["bp_category"] = (
    df["systolic_bp"]
    .apply(classify_bp)
)


# ------------------------------------------------------------
# Cholesterol Category
# ------------------------------------------------------------

def classify_cholesterol(value):

    if value < 200:
        return "Normal"

    elif value < 240:
        return "Borderline"

    else:
        return "High"


df["cholesterol_category"] = (
    df["cholesterol_mg_dl"]
    .apply(classify_cholesterol)
)


# ------------------------------------------------------------
# Blood Sugar Category
# ------------------------------------------------------------

def classify_glucose(value):

    if value < 100:
        return "Normal"

    elif value < 126:
        return "Prediabetic"

    else:
        return "High"


df["glucose_category"] = (
    df["blood_sugar_mg_dl"]
    .apply(classify_glucose)
)


# ------------------------------------------------------------
# HbA1c Category
# ------------------------------------------------------------

def classify_hba1c(value):

    if value < 5.7:
        return "Normal"

    elif value < 6.5:
        return "Prediabetic"

    else:
        return "Diabetic"


df["hba1c_category"] = (
    df["hba1c_percent"]
    .apply(classify_hba1c)
)


# ------------------------------------------------------------
# Heart Rate Category
# ------------------------------------------------------------

def classify_heart_rate(value):

    if value < 60:
        return "Low"

    elif value <= 100:
        return "Normal"

    else:
        return "High"


df["heart_rate_category"] = (
    df["heart_rate_bpm"]
    .apply(classify_heart_rate)
)


# ------------------------------------------------------------
# Cholesterol / BMI interaction
# ------------------------------------------------------------

df["cholesterol_bmi_ratio"] = (
    df["cholesterol_mg_dl"]
    /
    df["bmi"]
)


# ------------------------------------------------------------
# Cholesterol / Blood Pressure interaction
# ------------------------------------------------------------

df["cholesterol_bp_interaction"] = (
    df["cholesterol_mg_dl"]
    *
    df["systolic_bp"]
    /
    10000
)


# ------------------------------------------------------------
# Glucose / HbA1c ratio
# ------------------------------------------------------------

df["glucose_hba1c_ratio"] = (
    df["blood_sugar_mg_dl"]
    /
    df["hba1c_percent"]
)


# ------------------------------------------------------------
# Glucose - HbA1c relationship
# ------------------------------------------------------------

df["glucose_hba1c_difference"] = (
    df["blood_sugar_mg_dl"]
    -
    (
        df["hba1c_percent"]
        * 18
    )
)


# ------------------------------------------------------------
# BMI risk flag
# ------------------------------------------------------------

df["obesity_flag"] = (
    df["bmi"] >= 30
).astype(int)


# ------------------------------------------------------------
# High BP flag
# ------------------------------------------------------------

df["high_bp_flag"] = (
    df["systolic_bp"] >= 140
).astype(int)


# ------------------------------------------------------------
# High cholesterol flag
# ------------------------------------------------------------

df["high_cholesterol_flag"] = (
    df["cholesterol_mg_dl"] >= 240
).astype(int)


# ------------------------------------------------------------
# High glucose flag
# ------------------------------------------------------------

df["high_glucose_flag"] = (
    df["blood_sugar_mg_dl"] >= 126
).astype(int)


# ------------------------------------------------------------
# High HbA1c flag
# ------------------------------------------------------------

df["high_hba1c_flag"] = (
    df["hba1c_percent"] >= 6.5
).astype(int)


# ------------------------------------------------------------
# Abnormal ECG flag
# ------------------------------------------------------------

df["abnormal_ecg_flag"] = (
    df["ecg_result"]
    .astype(str)
    .str.lower()
    .eq("abnormal")
    .astype(int)
)


# ------------------------------------------------------------
# Chest pain risk flag
# ------------------------------------------------------------

df["chest_pain_flag"] = (
    df["chest_pain_type"]
    .astype(str)
    .str.lower()
    .ne("no chest pain")
    .astype(int)
)


# ------------------------------------------------------------
# Smoking flag
# ------------------------------------------------------------

df["smoking_flag"] = (
    df["smoking"]
    .astype(str)
    .str.lower()
    .eq("yes")
    .astype(int)
)


# ------------------------------------------------------------
# Diabetes flag
# ------------------------------------------------------------

df["diabetes_flag"] = (
    df["diabetes"]
    .astype(str)
    .str.lower()
    .eq("yes")
    .astype(int)
)


# ------------------------------------------------------------
# Important interactions
# ------------------------------------------------------------

df["diabetes_high_glucose"] = (
    df["diabetes_flag"]
    *
    df["high_glucose_flag"]
)


df["high_bp_abnormal_ecg"] = (
    df["high_bp_flag"]
    *
    df["abnormal_ecg_flag"]
)


df["obesity_diabetes"] = (
    df["obesity_flag"]
    *
    df["diabetes_flag"]
)


df["smoking_high_bp"] = (
    df["smoking_flag"]
    *
    df["high_bp_flag"]
)


df["cholesterol_smoking"] = (
    df["high_cholesterol_flag"]
    *
    df["smoking_flag"]
)


# ------------------------------------------------------------
# Hemoglobin deviation
# ------------------------------------------------------------

df["hemoglobin_deviation"] = (
    np.abs(
        df["hemoglobin_g_dl"]
        -
        14
    )
)


# ------------------------------------------------------------
# Heart rate deviation
# ------------------------------------------------------------

df["heart_rate_deviation"] = (
    np.abs(
        df["heart_rate_bpm"]
        -
        75
    )
)


print()
print(
    "Feature engineering completed."
)


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

TARGET = "cardiovascular_risk"


numeric_features = [

    "height_cm",
    "weight_kg",

    "systolic_bp",
    "diastolic_bp",

    "cholesterol_mg_dl",
    "blood_sugar_mg_dl",

    "hemoglobin_g_dl",
    "hba1c_percent",

    "heart_rate_bpm",

    "bmi",
    "pulse_pressure",

    "mean_arterial_pressure",

    "cholesterol_bmi_ratio",

    "cholesterol_bp_interaction",

    "glucose_hba1c_ratio",

    "glucose_hba1c_difference",

    "obesity_flag",
    "high_bp_flag",

    "high_cholesterol_flag",
    "high_glucose_flag",

    "high_hba1c_flag",
    "abnormal_ecg_flag",

    "chest_pain_flag",
    "smoking_flag",
    "diabetes_flag",

    "diabetes_high_glucose",
    "high_bp_abnormal_ecg",

    "obesity_diabetes",
    "smoking_high_bp",

    "cholesterol_smoking",

    "hemoglobin_deviation",
    "heart_rate_deviation"
]


categorical_features = [

    "ecg_result",
    "chest_pain_type",
    "smoking",
    "diabetes",

    "bmi_category",
    "bp_category",

    "cholesterol_category",
    "glucose_category",

    "hba1c_category",
    "heart_rate_category"
]


feature_columns = (
    numeric_features
    +
    categorical_features
)


X = df[
    feature_columns
]

y = df[
    TARGET
]


print()
print("=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print()
print(
    "Numeric features    :",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)

print(
    "Total ML features   :",
    len(feature_columns)
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print()
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print()
print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records :",
    len(X_test)
)


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]
)


categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )

    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )

    ],

    remainder="drop"
)


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(

            max_iter=2000,

            C=1.0,

            class_weight="balanced",

            random_state=RANDOM_STATE
        ),


    "Random Forest":

        RandomForestClassifier(

            n_estimators=250,

            max_depth=14,

            min_samples_leaf=2,

            class_weight="balanced",

            max_features="sqrt",

            n_jobs=1,

            random_state=RANDOM_STATE
        ),


    "Extra Trees":

        ExtraTreesClassifier(

            n_estimators=250,

            max_depth=16,

            min_samples_leaf=2,

            class_weight="balanced",

            max_features="sqrt",

            n_jobs=1,

            random_state=RANDOM_STATE
        ),


    "Gradient Boosting":

        GradientBoostingClassifier(

            n_estimators=200,

            learning_rate=0.05,

            max_depth=3,

            min_samples_leaf=3,

            subsample=0.9,

            random_state=RANDOM_STATE
        ),


    "Hist Gradient Boosting":

        HistGradientBoostingClassifier(

            max_iter=200,

            learning_rate=0.05,

            max_leaf_nodes=15,

            min_samples_leaf=20,

            l2_regularization=0.5,

            random_state=RANDOM_STATE
        )

}


# ============================================================
# 9. CREATE PIPELINES
# ============================================================

pipelines = {}


for model_name, model in models.items():

    pipelines[model_name] = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]
    )


# ============================================================
# 10. CROSS VALIDATION
# ============================================================

print()
print("=" * 70)
print("3-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)


cv = StratifiedKFold(

    n_splits=3,

    shuffle=True,

    random_state=RANDOM_STATE
)


scoring = {

    "accuracy":
        "accuracy",

    "precision":
        "precision",

    "recall":
        "recall",

    "f1":
        "f1",

    "roc_auc":
        "roc_auc"

}


cv_results = []


for model_name, pipeline in pipelines.items():

    print()
    print("-" * 70)

    print(
        "Cross-validating:",
        model_name
    )

    results = cross_validate(

        pipeline,

        X_train,

        y_train,

        cv=cv,

        scoring=scoring,

        n_jobs=1,

        return_train_score=False
    )


    accuracy = np.mean(
        results["test_accuracy"]
    )

    precision = np.mean(
        results["test_precision"]
    )

    recall = np.mean(
        results["test_recall"]
    )

    f1 = np.mean(
        results["test_f1"]
    )

    roc_auc = np.mean(
        results["test_roc_auc"]
    )


    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    cv_results.append({

        "Model": model_name,

        "CV_Accuracy": accuracy,

        "CV_Precision": precision,

        "CV_Recall": recall,

        "CV_F1": f1,

        "CV_ROC_AUC": roc_auc

    })


# ============================================================
# 11. CROSS-VALIDATION COMPARISON
# ============================================================

cv_results_df = pd.DataFrame(
    cv_results
)


cv_results_df = cv_results_df.sort_values(
    by="CV_F1",
    ascending=False
)


print()
print("=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    cv_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = (
    cv_results_df
    .iloc[0]["Model"]
)


best_pipeline = pipelines[
    best_model_name
]


print()
print("=" * 70)

print(
    "BEST MODEL:",
    best_model_name
)

print("=" * 70)


# ============================================================
# 13. TRAIN BEST MODEL
# ============================================================

print()
print(
    "Training final model..."
)

best_pipeline.fit(
    X_train,
    y_train
)

print(
    "Final model training completed."
)


# ============================================================
# 14. TEST SET PREDICTION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)


y_pred = (
    best_pipeline
    .predict(X_test)
)


# Probability for positive class

if hasattr(
    best_pipeline,
    "predict_proba"
):

    y_probability = (
        best_pipeline
        .predict_proba(X_test)[:, 1]
    )

else:

    y_probability = (
        best_pipeline
        .decision_function(X_test)
    )


# ============================================================
# 15. CALCULATE METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    y_pred
)


test_precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


test_recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


test_f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


test_roc_auc = roc_auc_score(
    y_test,
    y_probability
)


cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 16. PRINT FINAL RESULTS
# ============================================================

print()

print(
    f"Accuracy : {test_accuracy:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall   : {test_recall:.4f}"
)

print(
    f"F1 Score : {test_f1:.4f}"
)

print(
    f"ROC-AUC  : {test_roc_auc:.4f}"
)


print()
print("Confusion Matrix:")

print(
    cm
)


# ============================================================
# 17. PERCENTAGE FORMAT
# ============================================================

print()
print("=" * 70)
print("FINAL METRICS IN PERCENTAGE")
print("=" * 70)

print(
    f"Accuracy : {test_accuracy * 100:.2f}%"
)

print(
    f"Precision: {test_precision * 100:.2f}%"
)

print(
    f"Recall   : {test_recall * 100:.2f}%"
)

print(
    f"F1 Score : {test_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {test_roc_auc * 100:.2f}%"
)


# ============================================================
# 18. SAVE BEST MODEL
# ============================================================

print()
print(
    "Saving trained model..."
)


joblib.dump(
    best_pipeline,
    BEST_MODEL_PATH
)


print(
    "Model saved to:"
)

print(
    BEST_MODEL_PATH
)


# ============================================================
# 19. SAVE MODEL COMPARISON
# ============================================================

final_comparison = cv_results_df.copy()


final_comparison[
    "Test_Accuracy"
] = np.nan

final_comparison[
    "Test_Precision"
] = np.nan

final_comparison[
    "Test_Recall"
] = np.nan

final_comparison[
    "Test_F1"
] = np.nan

final_comparison[
    "Test_ROC_AUC"
] = np.nan


best_index = final_comparison[
    "Model"
] == best_model_name


final_comparison.loc[
    best_index,
    "Test_Accuracy"
] = test_accuracy


final_comparison.loc[
    best_index,
    "Test_Precision"
] = test_precision


final_comparison.loc[
    best_index,
    "Test_Recall"
] = test_recall


final_comparison.loc[
    best_index,
    "Test_F1"
] = test_f1


final_comparison.loc[
    best_index,
    "Test_ROC_AUC"
] = test_roc_auc


final_comparison.to_csv(
    COMPARISON_PATH,
    index=False
)


print()
print(
    "Model comparison saved to:"
)

print(
    COMPARISON_PATH
)


# ============================================================
# 20. SAVE METRICS REPORT
# ============================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MedAdvisorAI Cardiovascular Risk Model\n"
    )

    file.write(
        "========================================\n\n"
    )

    file.write(
        "Dataset:\n"
    )

    file.write(
        f"{DATASET_PATH}\n\n"
    )

    file.write(
        f"Total records: {len(df)}\n"
    )

    file.write(
        f"Training records: {len(X_train)}\n"
    )

    file.write(
        f"Testing records: {len(X_test)}\n\n"
    )

    file.write(
        f"Best model: {best_model_name}\n\n"
    )

    file.write(
        "Final Test Metrics\n"
    )

    file.write(
        "------------------\n"
    )

    file.write(
        f"Accuracy : {test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)\n"
    )

    file.write(
        f"Precision: {test_precision:.4f} "
        f"({test_precision * 100:.2f}%)\n"
    )

    file.write(
        f"Recall   : {test_recall:.4f} "
        f"({test_recall * 100:.2f}%)\n"
    )

    file.write(
        f"F1 Score : {test_f1:.4f} "
        f"({test_f1 * 100:.2f}%)\n"
    )

    file.write(
        f"ROC-AUC  : {test_roc_auc:.4f} "
        f"({test_roc_auc * 100:.2f}%)\n"
    )

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\n"
    )

    file.write(
        "IMPORTANT:\n"
    )

    file.write(
        "This model was trained and evaluated on a "
        "synthetic dataset.\n"
    )

    file.write(
        "The metrics represent synthetic-data "
        "benchmark performance and must not be "
        "interpreted as clinical validation.\n"
    )


print()
print(
    "Metrics report saved to:"
)

print(
    METRICS_PATH
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print(
    "Best Model:",
    best_model_name
)

print(
    f"Accuracy : {test_accuracy * 100:.2f}%"
)

print(
    f"Precision: {test_precision * 100:.2f}%"
)

print(
    f"Recall   : {test_recall * 100:.2f}%"
)

print(
    f"F1 Score : {test_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {test_roc_auc * 100:.2f}%"
)

print()
print(
    "Saved model:"
)

print(
    BEST_MODEL_PATH
)

print()
print("=" * 70)
