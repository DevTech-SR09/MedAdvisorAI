
import os

import joblib
import numpy as np
import pandas as pd


# ============================================================
# MedAdvisorAI ML Predictor
# ============================================================
# Project root: D:\MedAdvisorAI

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# Trained Model Path
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "medadvisor_cardiovascular_model.pkl"
)


# ============================================================
# Load Trained Model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# Feature Engineering
# ============================================================

def create_features(
    height_cm,
    weight_kg,
    blood_pressure,
    cholesterol_mg_dl,
    blood_sugar_mg_dl,
    hemoglobin_g_dl,
    hba1c_percent,
    heart_rate_bpm,
    ecg_result,
    chest_pain_type,
    smoking,
    diabetes
):
    """
    Convert the 12 health-assessment inputs into the
    42 features expected by the saved ML pipeline.

    ECG categories are normalized to match the training
    dataset, which contains Normal and Abnormal.
    """

    # --------------------------------------------------------
    # Convert numeric values
    # --------------------------------------------------------

    height_cm = float(height_cm)
    weight_kg = float(weight_kg)
    cholesterol_mg_dl = float(cholesterol_mg_dl)
    blood_sugar_mg_dl = float(blood_sugar_mg_dl)
    hemoglobin_g_dl = float(hemoglobin_g_dl)
    hba1c_percent = float(hba1c_percent)
    heart_rate_bpm = float(heart_rate_bpm)

    if height_cm <= 0:
        raise ValueError("Height must be greater than zero.")

    if weight_kg <= 0:
        raise ValueError("Weight must be greater than zero.")

    if hba1c_percent <= 0:
        raise ValueError("HbA1c must be greater than zero.")

    # --------------------------------------------------------
    # Parse blood pressure
    # --------------------------------------------------------

    bp_parts = str(blood_pressure).strip().split("/")

    if len(bp_parts) != 2:
        raise ValueError(
            "Blood pressure must be in the format "
            "systolic/diastolic, for example 120/80."
        )

    try:
        systolic_bp = float(bp_parts[0].strip())
        diastolic_bp = float(bp_parts[1].strip())
    except ValueError as error:
        raise ValueError(
            "Blood pressure must contain numeric values, "
            "for example 120/80."
        ) from error

    # --------------------------------------------------------
    # Normalize categorical inputs
    # --------------------------------------------------------

    raw_ecg = str(ecg_result).strip()
    normalized_ecg = raw_ecg.lower()

    # The training dataset contains only Normal and Abnormal.
    # Map the form's abnormal ECG options to Abnormal.
    if normalized_ecg == "normal":
        ecg_category = "Normal"
    elif normalized_ecg in {
        "abnormal",
        "st-t abnormality",
        "left ventricular hypertrophy"
    }:
        ecg_category = "Abnormal"
    else:
        raise ValueError(
            "Unsupported ECG category. Choose Normal, "
            "Abnormal, ST-T Abnormality, or "
            "Left Ventricular Hypertrophy."
        )

    chest_pain_type = str(chest_pain_type).strip()
    smoking = str(smoking).strip()
    diabetes = str(diabetes).strip()

    # --------------------------------------------------------
    # Normalize smoking and diabetes values
    # --------------------------------------------------------

    smoking_value = smoking.lower()
    diabetes_value = diabetes.lower()

    if smoking_value not in {"yes", "no"}:
        raise ValueError("Smoking must be Yes or No.")

    if diabetes_value not in {"yes", "no"}:
        raise ValueError("Diabetes must be Yes or No.")

    smoking = smoking_value.capitalize()
    diabetes = diabetes_value.capitalize()

    # ========================================================
    # Derived Features
    # ========================================================

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    bmi = weight_kg / ((height_cm / 100) ** 2)

    # --------------------------------------------------------
    # Blood pressure measurements
    # --------------------------------------------------------

    pulse_pressure = systolic_bp - diastolic_bp

    mean_arterial_pressure = (
        diastolic_bp
        + (systolic_bp - diastolic_bp) / 3
    )

    # --------------------------------------------------------
    # BMI Category
    # --------------------------------------------------------

    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # --------------------------------------------------------
    # Blood Pressure Category
    # --------------------------------------------------------

    if systolic_bp < 120:
        bp_category = "Normal"
    elif systolic_bp < 130:
        bp_category = "Elevated"
    elif systolic_bp < 140:
        bp_category = "High_Stage_1"
    elif systolic_bp < 180:
        bp_category = "High_Stage_2"
    else:
        bp_category = "Very_High"

    # --------------------------------------------------------
    # Cholesterol Category
    # --------------------------------------------------------

    if cholesterol_mg_dl < 200:
        cholesterol_category = "Normal"
    elif cholesterol_mg_dl < 240:
        cholesterol_category = "Borderline"
    else:
        cholesterol_category = "High"

    # --------------------------------------------------------
    # Blood Sugar Category
    # --------------------------------------------------------

    if blood_sugar_mg_dl < 100:
        glucose_category = "Normal"
    elif blood_sugar_mg_dl < 126:
        glucose_category = "Prediabetic"
    else:
        glucose_category = "High"

    # --------------------------------------------------------
    # HbA1c Category
    # --------------------------------------------------------

    if hba1c_percent < 5.7:
        hba1c_category = "Normal"
    elif hba1c_percent < 6.5:
        hba1c_category = "Prediabetic"
    else:
        hba1c_category = "Diabetic"

    # --------------------------------------------------------
    # Heart Rate Category
    # --------------------------------------------------------

    if heart_rate_bpm < 60:
        heart_rate_category = "Low"
    elif heart_rate_bpm <= 100:
        heart_rate_category = "Normal"
    else:
        heart_rate_category = "High"

    # ========================================================
    # Binary Flags
    # ========================================================

    obesity_flag = int(bmi >= 30)

    high_bp_flag = int(systolic_bp >= 140)

    high_cholesterol_flag = int(
        cholesterol_mg_dl >= 240
    )

    high_glucose_flag = int(
        blood_sugar_mg_dl >= 126
    )

    high_hba1c_flag = int(
        hba1c_percent >= 6.5
    )

    # ECG flag now uses the normalized ECG category.
    abnormal_ecg_flag = int(
        ecg_category == "Abnormal"
    )

    chest_pain_flag = int(
        chest_pain_type.lower() != "no chest pain"
    )

    smoking_flag = int(
        smoking.lower() == "yes"
    )

    diabetes_flag = int(
        diabetes.lower() == "yes"
    )

    # ========================================================
    # Ratios and Interaction Features
    # ========================================================

    cholesterol_bmi_ratio = (
        cholesterol_mg_dl / bmi
    )

    cholesterol_bp_interaction = (
        cholesterol_mg_dl * systolic_bp / 10000
    )

    glucose_hba1c_ratio = (
        blood_sugar_mg_dl / hba1c_percent
    )

    glucose_hba1c_difference = (
        blood_sugar_mg_dl - (hba1c_percent * 18)
    )

    diabetes_high_glucose = (
        diabetes_flag * high_glucose_flag
    )

    high_bp_abnormal_ecg = (
        high_bp_flag * abnormal_ecg_flag
    )

    obesity_diabetes = (
        obesity_flag * diabetes_flag
    )

    smoking_high_bp = (
        smoking_flag * high_bp_flag
    )

    cholesterol_smoking = (
        high_cholesterol_flag * smoking_flag
    )

    hemoglobin_deviation = abs(
        hemoglobin_g_dl - 14
    )

    heart_rate_deviation = abs(
        heart_rate_bpm - 75
    )

    # ========================================================
    # Construct the 42 Model Features
    # ========================================================

    data = {
        # ----------------------------------------------------
        # Numeric Features: 32
        # ----------------------------------------------------

        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol_mg_dl": cholesterol_mg_dl,
        "blood_sugar_mg_dl": blood_sugar_mg_dl,
        "hemoglobin_g_dl": hemoglobin_g_dl,
        "hba1c_percent": hba1c_percent,
        "heart_rate_bpm": heart_rate_bpm,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "mean_arterial_pressure": mean_arterial_pressure,
        "cholesterol_bmi_ratio": cholesterol_bmi_ratio,
        "cholesterol_bp_interaction": cholesterol_bp_interaction,
        "glucose_hba1c_ratio": glucose_hba1c_ratio,
        "glucose_hba1c_difference": glucose_hba1c_difference,
        "obesity_flag": obesity_flag,
        "high_bp_flag": high_bp_flag,
        "high_cholesterol_flag": high_cholesterol_flag,
        "high_glucose_flag": high_glucose_flag,
        "high_hba1c_flag": high_hba1c_flag,
        "abnormal_ecg_flag": abnormal_ecg_flag,
        "chest_pain_flag": chest_pain_flag,
        "smoking_flag": smoking_flag,
        "diabetes_flag": diabetes_flag,
        "diabetes_high_glucose": diabetes_high_glucose,
        "high_bp_abnormal_ecg": high_bp_abnormal_ecg,
        "obesity_diabetes": obesity_diabetes,
        "smoking_high_bp": smoking_high_bp,
        "cholesterol_smoking": cholesterol_smoking,
        "hemoglobin_deviation": hemoglobin_deviation,
        "heart_rate_deviation": heart_rate_deviation,

        # ----------------------------------------------------
        # Categorical Features: 10
        # ----------------------------------------------------

        "ecg_result": ecg_category,
        "chest_pain_type": chest_pain_type,
        "smoking": smoking,
        "diabetes": diabetes,
        "bmi_category": bmi_category,
        "bp_category": bp_category,
        "cholesterol_category": cholesterol_category,
        "glucose_category": glucose_category,
        "hba1c_category": hba1c_category,
        "heart_rate_category": heart_rate_category
    }

    features = pd.DataFrame([data])

    # Preserve the exact feature order expected by the model.
    expected_features = getattr(
        model, "feature_names_in_", None
    )

    if expected_features is not None:
        features = features.reindex(
            columns=list(expected_features)
        )

    return features


# ============================================================
# Cardiovascular Risk Prediction
# ============================================================

def predict_cardiovascular_risk(
    height_cm,
    weight_kg,
    blood_pressure,
    cholesterol_mg_dl,
    blood_sugar_mg_dl,
    hemoglobin_g_dl,
    hba1c_percent,
    heart_rate_bpm,
    ecg_result,
    chest_pain_type,
    smoking,
    diabetes
):
    """
    Generate a model prediction and return its class,
    probability, risk band, and recommendation.

    This is a project model output, not a medical diagnosis.
    """

    # --------------------------------------------------------
    # Create Model Features
    # --------------------------------------------------------

    features = create_features(
        height_cm=height_cm,
        weight_kg=weight_kg,
        blood_pressure=blood_pressure,
        cholesterol_mg_dl=cholesterol_mg_dl,
        blood_sugar_mg_dl=blood_sugar_mg_dl,
        hemoglobin_g_dl=hemoglobin_g_dl,
        hba1c_percent=hba1c_percent,
        heart_rate_bpm=heart_rate_bpm,
        ecg_result=ecg_result,
        chest_pain_type=chest_pain_type,
        smoking=smoking,
        diabetes=diabetes
    )

    # --------------------------------------------------------
    # Predict Class
    # --------------------------------------------------------

    prediction = int(
        model.predict(features)[0]
    )

    # --------------------------------------------------------
    # Get Probability for Class 1
    # --------------------------------------------------------

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The saved model does not contain class 1."
        )

    class_1_index = classes.index(1)

    probability = float(
        model.predict_proba(features)[0][class_1_index]
    )

    probability_percent = round(
        probability * 100,
        2
    )

    # ========================================================
    # Risk Bands
    # ========================================================
    # These thresholds are application display rules.
    # They are not clinically validated risk thresholds.

    if probability < 0.30:
        risk_level = "LOW RISK"
        risk_color = "#16A34A"

        recommendation = (
            "The model estimates a lower probability for "
            "class 1 based on the entered parameters. "
            "Maintain healthy lifestyle habits and consult "
            "a qualified healthcare professional for "
            "personalized medical advice."
        )

    elif probability < 0.60:
        risk_level = "MODERATE RISK"
        risk_color = "#F59E0B"

        recommendation = (
            "The model estimates an intermediate probability "
            "for class 1 based on the entered parameters. "
            "Consider discussing your health measurements "
            "with a qualified healthcare professional."
        )

    else:
        risk_level = "HIGH RISK"
        risk_color = "#DC2626"

        recommendation = (
            "The model estimates a higher probability for "
            "class 1 based on the entered parameters. "
            "Consult a qualified healthcare professional "
            "for appropriate evaluation. This result is "
            "not a confirmed medical diagnosis."
        )

    # ========================================================
    # Return Prediction Information
    # ========================================================

    return {
        "prediction": prediction,
        "probability": probability,
        "probability_percent": probability_percent,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation
    }
