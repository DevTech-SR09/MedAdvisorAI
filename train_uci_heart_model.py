# ================================================================
# MEDADVISORAI - UCI HEART DISEASE ML MODEL
# UCI Heart Disease Dataset - Cleveland
# ================================================================

import os

# Keep numerical libraries lightweight
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ================================================================
# CONFIGURATION
# ================================================================

DATASET_PATH = "datasets/processed.cleveland.data"
MODEL_DIR = "ml/models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# ================================================================
# CREATE MODEL DIRECTORY
# ================================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("MEDADVISORAI - UCI HEART DISEASE ML MODEL")
print("=" * 80)

print()
print("Dataset: UCI Heart Disease - Cleveland")
print("Source : UCI Machine Learning Repository")
print()


# ================================================================
# UCI COLUMN NAMES
# ================================================================

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]


# ================================================================
# LOAD DATASET
# ================================================================

print("Loading dataset...")
print("-" * 80)

try:
    df = pd.read_csv(
        DATASET_PATH,
        header=None,
        names=COLUMN_NAMES,
        na_values="?"
    )
except FileNotFoundError:
    print()
    print("ERROR: Dataset file was not found.")
    print()
    print("Expected location:")
    print(DATASET_PATH)
    print()
    print("Make sure processed.cleveland.data is inside:")
    print("D:\\MedAdvisorAI\\datasets\\")
    raise SystemExit


print("Dataset loaded successfully.")


# ================================================================
# BASIC DATASET INFORMATION
# ================================================================

print()
print("Original Dataset")
print("-" * 80)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ================================================================
# DATA PREVIEW
# ================================================================

print()
print("First 5 Records")
print("-" * 80)

print(df.head())


# ================================================================
# DATA TYPES
# ================================================================

print()
print("Converting columns to numeric...")
print("-" * 80)

for column in COLUMN_NAMES:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ================================================================
# MISSING VALUE CHECK
# ================================================================

print()
print("Missing Values")
print("-" * 80)

missing_values = df.isnull().sum()

print(missing_values)

print()
print(f"Total missing values: {missing_values.sum()}")


# ================================================================
# DUPLICATE CHECK
# ================================================================

print()
print("Duplicate Records")
print("-" * 80)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows after duplicate removal: {len(df)}")


# ================================================================
# TARGET PROCESSING
# ================================================================

print()
print("Processing target variable...")
print("-" * 80)

print("Original target values:")
print(df["target"].value_counts().sort_index())


# UCI target:
#
# 0 = absence of heart disease
# 1,2,3,4 = presence of heart disease
#
# Convert to binary classification:
#
# 0 -> 0
# 1,2,3,4 -> 1

df["target"] = np.where(
    df["target"] > 0,
    1,
    0
)


print()
print("Binary target distribution:")
print(df["target"].value_counts().sort_index())


print()
print("Target percentage:")
print(
    (df["target"].value_counts(normalize=True)
     .sort_index() * 100)
     .round(2)
)


# ================================================================
# FEATURES AND TARGET
# ================================================================

X = df.drop(columns=["target"])
y = df["target"]


# ================================================================
# FEATURE GROUPS
# ================================================================

# Continuous/numerical variables
NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]


# Categorical/encoded clinical variables
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


print()
print("Features")
print("-" * 80)

print(f"Total features: {len(X.columns)}")
print(f"Numeric features: {len(NUMERIC_FEATURES)}")
print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")


# ================================================================
# TRAIN / TEST SPLIT
# ================================================================

print()
print("=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print()
print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ================================================================
# PREPROCESSING FOR LINEAR / SVM MODELS
# ================================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
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
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# ================================================================
# MODELS
# ================================================================

models = {

    "Logistic Regression": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    ),


    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1
                )
            )
        ]
    ),


    "Gradient Boosting": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE
                )
            )
        ]
    ),


    "Support Vector Machine": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )
}


# ================================================================
# CROSS-VALIDATION
# ================================================================

print()
print("=" * 80)
print("5-FOLD CROSS-VALIDATION")
print("=" * 80)

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


results = []


for model_name, model in models.items():

    print()
    print("-" * 80)
    print(f"Evaluating: {model_name}")
    print("-" * 80)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ],
        n_jobs=1
    )

    cv_accuracy = scores["test_accuracy"].mean()
    cv_precision = scores["test_precision"].mean()
    cv_recall = scores["test_recall"].mean()
    cv_f1 = scores["test_f1"].mean()
    cv_auc = scores["test_roc_auc"].mean()

    print(
        f"CV Accuracy : {cv_accuracy:.4f}"
    )

    print(
        f"CV Precision: {cv_precision:.4f}"
    )

    print(
        f"CV Recall   : {cv_recall:.4f}"
    )

    print(
        f"CV F1       : {cv_f1:.4f}"
    )

    print(
        f"CV ROC-AUC  : {cv_auc:.4f}"
    )


    results.append(
        {
            "Model": model_name,
            "CV Accuracy": cv_accuracy,
            "CV Precision": cv_precision,
            "CV Recall": cv_recall,
            "CV F1": cv_f1,
            "CV ROC-AUC": cv_auc
        }
    )


# ================================================================
# MODEL COMPARISON
# ================================================================

results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    by="CV F1",
    ascending=False
).reset_index(drop=True)


print()
print("=" * 80)
print("MODEL COMPARISON - CROSS VALIDATION")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# SELECT BEST MODEL
# ================================================================

best_model_name = results_df.iloc[0]["Model"]

print()
print("=" * 80)
print("BEST MODEL")
print("=" * 80)

print(f"Selected model: {best_model_name}")

print(
    "Selection criterion: Highest mean 5-fold CV F1 score"
)


# ================================================================
# TRAIN BEST MODEL ON TRAINING DATA
# ================================================================

print()
print("=" * 80)
print("TRAINING FINAL MODEL")
print("=" * 80)

best_model = models[best_model_name]

best_model.fit(
    X_train,
    y_train
)

print("Final model trained successfully.")


# ================================================================
# TEST SET EVALUATION
# ================================================================

print()
print("=" * 80)
print("FINAL HOLD-OUT TEST RESULTS")
print("=" * 80)

y_pred = best_model.predict(X_test)

y_probability = best_model.predict_proba(X_test)[:, 1]


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

test_auc = roc_auc_score(
    y_test,
    y_probability
)


print()
print(f"Accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Precision: {test_precision:.4f} ({test_precision * 100:.2f}%)")
print(f"Recall   : {test_recall:.4f} ({test_recall * 100:.2f}%)")
print(f"F1 Score : {test_f1:.4f} ({test_f1 * 100:.2f}%)")
print(f"ROC-AUC  : {test_auc:.4f}")


# ================================================================
# CONFUSION MATRIX
# ================================================================

print()
print("Confusion Matrix")
print("-" * 80)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print()
print("Classification Report")
print("-" * 80)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Heart Disease",
            "Heart Disease"
        ],
        zero_division=0
    )
)


# ================================================================
# SAVE MODEL
# ================================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "uci_heart_disease_model.pkl"
)

joblib.dump(
    best_model,
    final_model_path
)


print()
print("Model saved:")
print(final_model_path)


# ================================================================
# SAVE MODEL COMPARISON
# ================================================================

comparison_path = os.path.join(
    MODEL_DIR,
    "uci_heart_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)


print()
print("Model comparison saved:")
print(comparison_path)


# ================================================================
# SAVE FINAL METRICS
# ================================================================

metrics_path = os.path.join(
    MODEL_DIR,
    "uci_heart_final_metrics.txt"
)


with open(metrics_path, "w") as file:

    file.write(
        "MEDADVISORAI - UCI HEART DISEASE MODEL\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Dataset: UCI Heart Disease - Cleveland\n"
    )

    file.write(
        f"Best Model: {best_model_name}\n\n"
    )

    file.write(
        "Hold-out Test Metrics\n"
    )

    file.write(
        "-" * 60 + "\n"
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
        f"ROC-AUC  : {test_auc:.4f}\n"
    )

    file.write("\nConfusion Matrix\n")
    file.write(str(cm))
    file.write("\n")


print()
print("Final metrics saved:")
print(metrics_path)


# ================================================================
# FINISHED
# ================================================================

print()
print("=" * 80)
print("UCI HEART DISEASE MODEL TRAINING COMPLETED")
print("=" * 80)

print()
print("Generated files:")
print()

print(f"1. {final_model_path}")
print(f"2. {comparison_path}")
print(f"3. {metrics_path}")

print()
print("You can now use the saved model inside Django.")
print()
