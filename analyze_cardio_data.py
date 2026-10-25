import pandas as pd

# ==========================================================
# CARDIOVASCULAR DATASET ANALYSIS
# ==========================================================

DATASET_PATH = "datasets/cardio_train.csv"

print("=" * 70)
print("CARDIOVASCULAR DATASET ANALYSIS")
print("=" * 70)

# Load dataset
data = pd.read_csv(DATASET_PATH, sep=";")

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(list(data.columns))


# ==========================================================
# 1. MISSING VALUES
# ==========================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(data.isnull().sum())


# ==========================================================
# 2. DUPLICATE RECORDS
# ==========================================================

print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

duplicates = data.duplicated().sum()

print("Duplicate rows:", duplicates)


# ==========================================================
# 3. TARGET DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(data["cardio"].value_counts())

print("\nTarget percentages:")
print(data["cardio"].value_counts(normalize=True) * 100)


# ==========================================================
# 4. BASIC STATISTICS
# ==========================================================

print("\n" + "=" * 70)
print("BASIC STATISTICS")
print("=" * 70)

print(data.describe().T)


# ==========================================================
# 5. CHECK INVALID VALUES
# ==========================================================

print("\n" + "=" * 70)
print("POTENTIAL INVALID VALUES")
print("=" * 70)

print("\nAge in days:")
print("Minimum:", data["age"].min())
print("Maximum:", data["age"].max())

print("\nHeight:")
print("Minimum:", data["height"].min())
print("Maximum:", data["height"].max())

print("\nWeight:")
print("Minimum:", data["weight"].min())
print("Maximum:", data["weight"].max())

print("\nSystolic BP:")
print("Minimum:", data["ap_hi"].min())
print("Maximum:", data["ap_hi"].max())

print("\nDiastolic BP:")
print("Minimum:", data["ap_lo"].min())
print("Maximum:", data["ap_lo"].max())


# ==========================================================
# 6. CATEGORICAL VALUES
# ==========================================================

print("\n" + "=" * 70)
print("CATEGORICAL VALUE DISTRIBUTION")
print("=" * 70)

categorical_columns = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio"
]

for column in categorical_columns:

    print("\n" + column)

    print(data[column].value_counts())


# ==========================================================
# 7. BMI CALCULATION
# ==========================================================

print("\n" + "=" * 70)
print("BMI ANALYSIS")
print("=" * 70)

data["height_m"] = data["height"] / 100

data["BMI"] = data["weight"] / (
    data["height_m"] ** 2
)

print(data["BMI"].describe())

print("\nBMI values below 10:")
print((data["BMI"] < 10).sum())

print("\nBMI values above 60:")
print((data["BMI"] > 60).sum())


# ==========================================================
# 8. EXTREME BLOOD PRESSURE VALUES
# ==========================================================

print("\n" + "=" * 70)
print("EXTREME BLOOD PRESSURE VALUES")
print("=" * 70)

print(
    "Systolic BP < 70:",
    (data["ap_hi"] < 70).sum()
)

print(
    "Systolic BP > 250:",
    (data["ap_hi"] > 250).sum()
)

print(
    "Diastolic BP < 40:",
    (data["ap_lo"] < 40).sum()
)

print(
    "Diastolic BP > 150:",
    (data["ap_lo"] > 150).sum()
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)
