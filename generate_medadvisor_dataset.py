import os
import numpy as np
import pandas as pd


# ============================================================
# MedAdvisorAI
# Synthetic Cardiovascular Dataset V2
# ============================================================

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42
NUM_RECORDS = 20000

OUTPUT_PATH = os.path.join(
    "datasets",
    "medadvisor_cardiovascular_synthetic_20000_v2.csv"
)

np.random.seed(RANDOM_SEED)


# ============================================================
# 1. Generate Height
# ============================================================

height_cm = np.random.normal(
    loc=168,
    scale=9,
    size=NUM_RECORDS
)

height_cm = np.clip(
    height_cm,
    145,
    195
)


# ============================================================
# 2. Generate BMI
# ============================================================

bmi = np.random.normal(
    loc=26.5,
    scale=5.0,
    size=NUM_RECORDS
)

bmi = np.clip(
    bmi,
    17,
    42
)


# ============================================================
# 3. Generate Weight
# ============================================================

weight_kg = bmi * ((height_cm / 100) ** 2)

weight_kg += np.random.normal(
    loc=0,
    scale=1.5,
    size=NUM_RECORDS
)

weight_kg = np.clip(
    weight_kg,
    45,
    140
)


# ============================================================
# 4. Generate Blood Pressure
# ============================================================

systolic_bp = (
    105
    + (bmi - 21) * 2.2
    + np.random.normal(0, 10, NUM_RECORDS)
)

systolic_bp = np.clip(
    systolic_bp,
    90,
    190
)


diastolic_bp = (
    65
    + (systolic_bp - 110) * 0.35
    + np.random.normal(0, 5, NUM_RECORDS)
)

diastolic_bp = np.clip(
    diastolic_bp,
    55,
    115
)


blood_pressure = [
    f"{int(round(sys))}/{int(round(dia))}"
    for sys, dia in zip(
        systolic_bp,
        diastolic_bp
    )
]


# ============================================================
# 5. Generate Diabetes
# ============================================================

diabetes_probability = (
    0.08
    + 0.025 * np.maximum(bmi - 25, 0)
)

diabetes_probability = np.clip(
    diabetes_probability,
    0.05,
    0.55
)


diabetes_binary = (
    np.random.random(NUM_RECORDS)
    < diabetes_probability
)


diabetes = np.where(
    diabetes_binary,
    "Yes",
    "No"
)


# ============================================================
# 6. Generate Blood Sugar
# ============================================================

blood_sugar_mg_dl = (
    88
    + np.random.normal(
        0,
        10,
        NUM_RECORDS
    )
    + diabetes_binary
    * np.random.normal(
        35,
        8,
        NUM_RECORDS
    )
    + np.maximum(
        bmi - 25,
        0
    ) * 1.2
)

blood_sugar_mg_dl = np.clip(
    blood_sugar_mg_dl,
    65,
    260
)


# ============================================================
# 7. Generate HbA1c
# ============================================================

hba1c_percent = (
    4.8
    + (blood_sugar_mg_dl - 90)
    * 0.018
    + diabetes_binary * 0.7
    + np.random.normal(
        0,
        0.25,
        NUM_RECORDS
    )
)

hba1c_percent = np.clip(
    hba1c_percent,
    3.8,
    12
)


# ============================================================
# 8. Generate Cholesterol
# ============================================================

cholesterol_mg_dl = (
    155
    + (bmi - 22) * 4
    + (systolic_bp - 110) * 0.45
    + np.random.normal(
        0,
        18,
        NUM_RECORDS
    )
)

cholesterol_mg_dl = np.clip(
    cholesterol_mg_dl,
    110,
    320
)


# ============================================================
# 9. Generate Hemoglobin
# ============================================================

hemoglobin_g_dl = np.random.normal(
    loc=14.2,
    scale=1.3,
    size=NUM_RECORDS
)

hemoglobin_g_dl = np.clip(
    hemoglobin_g_dl,
    9,
    18
)


# ============================================================
# 10. Generate Heart Rate
# ============================================================

heart_rate_bpm = (
    72
    + (systolic_bp - 120) * 0.08
    + np.random.normal(
        0,
        8,
        NUM_RECORDS
    )
)

heart_rate_bpm = np.clip(
    heart_rate_bpm,
    50,
    120
)


# ============================================================
# 11. Generate Smoking
# ============================================================

smoking_probability = 0.18

smoking_binary = (
    np.random.random(NUM_RECORDS)
    < smoking_probability
)

smoking = np.where(
    smoking_binary,
    "Yes",
    "No"
)


# ============================================================
# 12. Generate ECG Result
# ============================================================

ecg_abnormal_probability = (
    0.04
    + 0.003
    * np.maximum(
        systolic_bp - 130,
        0
    )
    + 0.08 * diabetes_binary
    + 0.04 * smoking_binary
)

ecg_abnormal_probability = np.clip(
    ecg_abnormal_probability,
    0.03,
    0.45
)


ecg_abnormal = (
    np.random.random(NUM_RECORDS)
    < ecg_abnormal_probability
)


ecg_result = np.where(
    ecg_abnormal,
    "Abnormal",
    "Normal"
)


# ============================================================
# 13. Generate Chest Pain Type
# ============================================================

latent_risk = (

    (systolic_bp - 120) / 20

    + (cholesterol_mg_dl - 180) / 50

    + (blood_sugar_mg_dl - 100) / 35

    + (hba1c_percent - 5.5) / 1.5

    + (bmi - 25) / 5

    + diabetes_binary * 1.5

    + smoking_binary * 1.2

    + ecg_abnormal * 2
)


chest_pain_type = np.empty(
    NUM_RECORDS,
    dtype=object
)


chest_pain_type[
    latent_risk < 0.5
] = "No Chest Pain"


chest_pain_type[
    (latent_risk >= 0.5)
    & (latent_risk < 2)
] = "Mild"


chest_pain_type[
    (latent_risk >= 2)
    & (latent_risk < 4)
] = "Moderate"


chest_pain_type[
    latent_risk >= 4
] = "Severe"


# ============================================================
# 14. Generate Cardiovascular Risk Score
# ============================================================

risk_score = np.zeros(
    NUM_RECORDS
)


# ------------------------------------------------------------
# BMI contribution
# ------------------------------------------------------------

bmi_risk = np.where(
    bmi < 25,
    0,
    np.minimum(
        (bmi - 25) * 2.0,
        25
    )
)

risk_score += bmi_risk


# ------------------------------------------------------------
# Blood Pressure contribution
# ------------------------------------------------------------

bp_risk = np.where(
    systolic_bp < 120,
    0,
    np.minimum(
        (systolic_bp - 120) * 0.55,
        30
    )
)

risk_score += bp_risk


# ------------------------------------------------------------
# Cholesterol contribution
# ------------------------------------------------------------

cholesterol_risk = np.where(
    cholesterol_mg_dl < 180,
    0,
    np.minimum(
        (cholesterol_mg_dl - 180) * 0.15,
        20
    )
)

risk_score += cholesterol_risk


# ------------------------------------------------------------
# Blood Sugar contribution
# ------------------------------------------------------------

sugar_risk = np.where(
    blood_sugar_mg_dl < 100,
    0,
    np.minimum(
        (blood_sugar_mg_dl - 100) * 0.18,
        20
    )
)

risk_score += sugar_risk


# ------------------------------------------------------------
# HbA1c contribution
# ------------------------------------------------------------

hba1c_risk = np.where(
    hba1c_percent < 5.7,
    0,
    np.minimum(
        (hba1c_percent - 5.7) * 6,
        20
    )
)

risk_score += hba1c_risk


# ------------------------------------------------------------
# Heart Rate contribution
# ------------------------------------------------------------

heart_rate_risk = np.where(
    heart_rate_bpm < 60,

    (60 - heart_rate_bpm) * 0.15,

    np.where(
        heart_rate_bpm > 90,
        (heart_rate_bpm - 90) * 0.20,
        0
    )
)

risk_score += heart_rate_risk


# ------------------------------------------------------------
# ECG contribution
# ------------------------------------------------------------

risk_score += (
    ecg_abnormal * 18
)


# ------------------------------------------------------------
# Chest Pain contribution
# ------------------------------------------------------------

chest_pain_scores = {
    "No Chest Pain": 0,
    "Mild": 5,
    "Moderate": 12,
    "Severe": 20
}


chest_pain_risk = np.array([
    chest_pain_scores[value]
    for value in chest_pain_type
])


risk_score += chest_pain_risk


# ------------------------------------------------------------
# Smoking contribution
# ------------------------------------------------------------

risk_score += (
    smoking_binary * 10
)


# ------------------------------------------------------------
# Diabetes contribution
# ------------------------------------------------------------

risk_score += (
    diabetes_binary * 15
)


# ------------------------------------------------------------
# Hemoglobin contribution
# ------------------------------------------------------------

hemoglobin_risk = np.where(
    hemoglobin_g_dl < 12,
    (12 - hemoglobin_g_dl) * 2,
    0
)

risk_score += hemoglobin_risk


# ============================================================
# 15. Risk Interactions
# ============================================================

# Diabetes + high blood sugar

risk_score += np.where(
    diabetes_binary
    & (blood_sugar_mg_dl >= 126),
    8,
    0
)


# High blood pressure + abnormal ECG

risk_score += np.where(
    (systolic_bp >= 140)
    & ecg_abnormal,
    10,
    0
)


# High cholesterol + smoking

risk_score += np.where(
    (cholesterol_mg_dl >= 240)
    & smoking_binary,
    8,
    0
)


# Obesity + diabetes

risk_score += np.where(
    (bmi >= 30)
    & diabetes_binary,
    8,
    0
)


# ============================================================
# 16. Add Small Controlled Noise
# ============================================================

risk_score += np.random.normal(
    loc=0,
    scale=2.0,
    size=NUM_RECORDS
)


# ============================================================
# 17. Convert Risk Score into Target
# ============================================================

threshold = np.percentile(
    risk_score,
    52
)


cardiovascular_risk = (
    risk_score >= threshold
).astype(int)


# ============================================================
# 18. Create Final Dataset
# ============================================================

df = pd.DataFrame({

    "height_cm":
        np.round(
            height_cm,
            1
        ),

    "weight_kg":
        np.round(
            weight_kg,
            1
        ),

    "blood_pressure":
        blood_pressure,

    "cholesterol_mg_dl":
        np.round(
            cholesterol_mg_dl,
            1
        ),

    "blood_sugar_mg_dl":
        np.round(
            blood_sugar_mg_dl,
            1
        ),

    "hemoglobin_g_dl":
        np.round(
            hemoglobin_g_dl,
            1
        ),

    "hba1c_percent":
        np.round(
            hba1c_percent,
            2
        ),

    "heart_rate_bpm":
        np.round(
            heart_rate_bpm
        ).astype(int),

    "ecg_result":
        ecg_result,

    "chest_pain_type":
        chest_pain_type,

    "smoking":
        smoking,

    "diabetes":
        diabetes,

    "cardiovascular_risk":
        cardiovascular_risk
})


# ============================================================
# 19. Dataset Validation
# ============================================================

print()
print("=" * 60)
print("MedAdvisorAI V2 Dataset Generation")
print("=" * 60)

print()
print("Dataset Shape:")
print(df.shape)

print()
print("Columns:")
print(list(df.columns))

print()
print("Missing Values:")
print(df.isnull().sum())

print()
print("Total Missing Values:")
print(df.isnull().sum().sum())

print()
print("Duplicate Rows:")
print(df.duplicated().sum())

print()
print("Target Distribution:")
print(
    df["cardiovascular_risk"].value_counts()
    .sort_index()
)

print()
print("Target Percentage:")
print(
    df["cardiovascular_risk"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print()
print("Chest Pain Distribution:")
print(
    df["chest_pain_type"].value_counts()
)

print()
print("ECG Distribution:")
print(
    df["ecg_result"].value_counts()
)

print()
print("Smoking Distribution:")
print(
    df["smoking"].value_counts()
)

print()
print("Diabetes Distribution:")
print(
    df["diabetes"].value_counts()
)


# ============================================================
# 20. Save Dataset
# ============================================================

os.makedirs(
    "datasets",
    exist_ok=True
)


df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 21. Final Message
# ============================================================

print()
print("=" * 60)
print("Dataset generated successfully!")
print("=" * 60)

print()
print("Output file:")
print(
    os.path.abspath(
        OUTPUT_PATH
    )
)

print()
print("Records:", len(df))
print("Columns:", len(df.columns))

print()
print("IMPORTANT:")
print(
    "This is a synthetic dataset for "
    "prototype/model-development purposes."
)

print(
    "It must not be interpreted as "
    "clinical validation or medical accuracy."
)

print()
print("=" * 60)
