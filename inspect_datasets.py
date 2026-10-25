import pandas as pd

# ==========================================================
# DATASET 1: Cardiovascular Disease Dataset
# ==========================================================

cardio_file = "datasets/cardio_train.csv"

print("\n" + "=" * 70)
print("CARDIOVASCULAR DATASET")
print("=" * 70)

# IMPORTANT:
# cardio_train.csv uses semicolon (;) as the separator
cardio = pd.read_csv(cardio_file, sep=";")

print("\nNumber of rows:", cardio.shape[0])
print("Number of columns:", cardio.shape[1])

print("\nColumn names:")
for column in cardio.columns:
    print(" -", column)

print("\nFirst 5 rows:")
print(cardio.head())

print("\nMissing values:")
print(cardio.isnull().sum())

print("\nData types:")
print(cardio.dtypes)


# ==========================================================
# DATASET 2: Diabetes Risk Prediction Dataset
# ==========================================================

diabetes_file = "datasets/diabetes_risk_prediction_dataset.csv"

print("\n" + "=" * 70)
print("DIABETES RISK DATASET")
print("=" * 70)

diabetes = pd.read_csv(diabetes_file)

print("\nNumber of rows:", diabetes.shape[0])
print("Number of columns:", diabetes.shape[1])

print("\nColumn names:")
for column in diabetes.columns:
    print(" -", column)

print("\nFirst 5 rows:")
print(diabetes.head())

print("\nMissing values:")
print(diabetes.isnull().sum())

print("\nData types:")
print(diabetes.dtypes)


# ==========================================================
# DATASET SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print("\nCardiovascular Dataset:")
print("Rows    :", cardio.shape[0])
print("Columns :", cardio.shape[1])

print("\nDiabetes Risk Dataset:")
print("Rows    :", diabetes.shape[0])
print("Columns :", diabetes.shape[1])

print("\nDataset inspection completed successfully.")
