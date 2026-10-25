# ============================================================
# MedAdvisorAI - Personalized Recommendation Agents
# ============================================================
#
# This module contains specialized recommendation agents.
#
# Each agent analyzes different health parameters and produces
# personalized educational recommendations.
#
# IMPORTANT:
# This is an academic prototype.
# It does NOT diagnose diseases or prescribe medication.
# ============================================================


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_bmi(health):
    """
    Calculate BMI from height and weight.
    """

    try:
        height_m = float(health.height) / 100

        if height_m <= 0:
            return 0

        return float(health.weight) / (height_m ** 2)

    except (TypeError, ValueError):
        return 0


def get_systolic_bp(health):
    """
    Extract systolic blood pressure.

    Example:
        120/80 -> 120
        150/95 -> 150
    """

    try:
        return float(
            str(health.blood_pressure)
            .split("/")[0]
            .strip()
        )

    except (
        TypeError,
        ValueError,
        AttributeError,
        IndexError
    ):
        return 0


def get_diastolic_bp(health):
    """
    Extract diastolic blood pressure.

    Example:
        120/80 -> 80
        150/95 -> 95
    """

    try:
        return float(
            str(health.blood_pressure)
            .split("/")[1]
            .strip()
        )

    except (
        TypeError,
        ValueError,
        AttributeError,
        IndexError
    ):
        return 0


def normalize(value):
    """
    Convert text values to lowercase for reliable comparison.
    """

    if value is None:
        return ""

    return str(value).strip().lower()


# ============================================================
# 1. DIET AGENT
# ============================================================

def diet_agent(health, risk_level):
    """
    Personalized diet recommendation agent.

    Parameters considered:
        - BMI
        - Cholesterol
        - Blood Sugar
        - HbA1c
        - Diabetes
        - Risk Level
    """

    recommendations = []

    bmi = calculate_bmi(health)

    cholesterol = float(
        health.cholesterol or 0
    )

    blood_sugar = float(
        health.blood_sugar or 0
    )

    hba1c = float(
        health.hba1c or 0
    )

    diabetes = normalize(
        health.diabetes
    )

    # --------------------------------------------------------
    # BMI-based recommendation
    # --------------------------------------------------------

    if bmi >= 30:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Focus on portion-controlled meals containing vegetables, "
            "whole grains, fruits, and appropriate protein sources."
        )

    elif bmi >= 25:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Consider balanced portions and include more vegetables, "
            "whole grains, fruits, and lean protein sources."
        )

    else:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Your BMI does not trigger a weight-related diet warning "
            "in this prototype. Continue maintaining a balanced diet."
        )

    # --------------------------------------------------------
    # Cholesterol-based recommendation
    # --------------------------------------------------------

    if cholesterol >= 240:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "The value is high according to this prototype's rules. "
            "Limit foods high in saturated and trans fats and discuss "
            "the result with a healthcare professional."
        )

    elif cholesterol >= 200:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "Consider reducing saturated and trans fats and increasing "
            "fiber-rich foods."
        )

    else:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "Continue a heart-healthy diet containing vegetables, "
            "fruits, whole grains, and appropriate protein sources."
        )

    # --------------------------------------------------------
    # Blood sugar recommendation
    # --------------------------------------------------------

    if blood_sugar >= 126:

        recommendations.append(
            f"Your blood sugar value is {blood_sugar:.0f} mg/dL. "
            "Consider controlling refined carbohydrates and added sugars "
            "and discuss the result with a healthcare professional."
        )

    elif blood_sugar >= 100:

        recommendations.append(
            f"Your blood sugar value is {blood_sugar:.0f} mg/dL. "
            "Consider limiting excess added sugars and refined carbohydrates "
            "while maintaining balanced meals."
        )

    else:

        recommendations.append(
            f"Your blood sugar value is {blood_sugar:.0f} mg/dL. "
            "Continue balanced meals and avoid excessive added sugars."
        )

    # --------------------------------------------------------
    # HbA1c recommendation
    # --------------------------------------------------------

    if hba1c >= 6.5:

        recommendations.append(
            f"Your HbA1c is {hba1c:.1f}%. "
            "Discuss the result and an appropriate dietary plan "
            "with a qualified healthcare professional."
        )

    elif hba1c >= 5.7:

        recommendations.append(
            f"Your HbA1c is {hba1c:.1f}%. "
            "Consider reducing excess added sugars and refined "
            "carbohydrates and maintaining regular physical activity."
        )

    else:

        recommendations.append(
            f"Your HbA1c is {hba1c:.1f}%. "
            "Continue maintaining balanced meals and a consistent "
            "healthy eating routine."
        )

    # --------------------------------------------------------
    # Diabetes recommendation
    # --------------------------------------------------------

    if diabetes == "yes":

        recommendations.append(
            "Diabetes is marked as present. "
            "Maintain consistent meal patterns and discuss an "
            "individualized dietary plan with a healthcare professional."
        )

    else:

        recommendations.append(
            "Diabetes is marked as absent. "
            "Continue maintaining balanced meals and regular health monitoring."
        )

    return {
        "agent": "Diet Agent",
        "category": "Diet",
        "recommendations": recommendations
    }


# ============================================================
# 2. EXERCISE AGENT
# ============================================================

def exercise_agent(health, risk_level):
    """
    Personalized exercise recommendation agent.

    Parameters considered:
        - BMI
        - Blood Pressure
        - Heart Rate
        - Risk Level
    """

    recommendations = []

    bmi = calculate_bmi(health)

    systolic = get_systolic_bp(
        health
    )

    diastolic = get_diastolic_bp(
        health
    )

    heart_rate = float(
        health.heart_rate or 0
    )

    # --------------------------------------------------------
    # Blood pressure
    # --------------------------------------------------------

    if systolic >= 140 or diastolic >= 90:

        recommendations.append(
            f"Your entered blood pressure is "
            f"{health.blood_pressure}. "
            "Consider discussing appropriate exercise intensity "
            "with a healthcare professional before vigorous activity."
        )

    elif systolic >= 130 or diastolic >= 80:

        recommendations.append(
            f"Your entered blood pressure is "
            f"{health.blood_pressure}. "
            "Consider regular moderate physical activity with gradual "
            "progression according to your fitness level."
        )

    else:

        recommendations.append(
            f"Your entered blood pressure is "
            f"{health.blood_pressure}. "
            "Continue regular physical activity appropriate for your "
            "fitness level."
        )

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    if bmi >= 30:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Consider starting with low-impact activities such as "
            "walking and gradually increasing activity."
        )

    elif bmi >= 25:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Regular walking, cycling, or other moderate activities "
            "can be incorporated gradually."
        )

    else:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Maintain regular physical activity appropriate for your "
            "current fitness level."
        )

    # --------------------------------------------------------
    # Heart rate
    # --------------------------------------------------------

    if heart_rate >= 100:

        recommendations.append(
            f"Your entered heart rate is {heart_rate:.0f} BPM. "
            "If this represents a resting measurement and remains "
            "consistently elevated, discuss it with a healthcare "
            "professional before increasing exercise intensity."
        )

    elif heart_rate < 60:

        recommendations.append(
            f"Your entered heart rate is {heart_rate:.0f} BPM. "
            "If this is a resting measurement and you have symptoms, "
            "discuss it with a healthcare professional."
        )

    else:

        recommendations.append(
            f"Your entered heart rate is {heart_rate:.0f} BPM. "
            "Continue exercising at an intensity appropriate for "
            "your fitness level."
        )

    # --------------------------------------------------------
    # ML risk level
    # --------------------------------------------------------

    if risk_level == "HIGH RISK":

        recommendations.append(
            "The ML assessment is classified as HIGH RISK. "
            "Discuss an appropriate exercise plan with a healthcare "
            "professional before beginning vigorous activity."
        )

    elif risk_level == "MODERATE RISK":

        recommendations.append(
            "The ML assessment is classified as MODERATE RISK. "
            "Consider gradual physical activity and regular health monitoring."
        )

    else:

        recommendations.append(
            "The ML assessment is classified as LOW RISK. "
            "Continue regular physical activity appropriate for your fitness level."
        )

    return {
        "agent": "Exercise Agent",
        "category": "Exercise",
        "recommendations": recommendations
    }


# ============================================================
# 3. MEDICATION AWARENESS AGENT
# ============================================================

def medication_agent(health, risk_level):
    """
    Medication-awareness agent.

    This agent does NOT prescribe medications or dosages.
    It only identifies parameters that may warrant discussion
    with a qualified healthcare professional.
    """

    recommendations = []

    systolic = get_systolic_bp(
        health
    )

    cholesterol = float(
        health.cholesterol or 0
    )

    blood_sugar = float(
        health.blood_sugar or 0
    )

    hba1c = float(
        health.hba1c or 0
    )

    diabetes = normalize(
        health.diabetes
    )

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    if diabetes == "yes":

        recommendations.append(
            "Diabetes is marked as present. "
            "Discuss diabetes management and any appropriate treatment "
            "options with a qualified healthcare professional."
        )

    else:

        recommendations.append(
            "Diabetes is marked as absent. "
            "No diabetes-specific medication guidance is triggered."
        )

    # --------------------------------------------------------
    # Glucose / HbA1c
    # --------------------------------------------------------

    if blood_sugar >= 126 or hba1c >= 6.5:

        recommendations.append(
            f"Your glucose-related values are elevated "
            f"(blood sugar: {blood_sugar:.0f} mg/dL, "
            f"HbA1c: {hba1c:.1f}%). "
            "Discuss these results with a healthcare professional."
        )

    else:

        recommendations.append(
            "Your glucose-related values do not trigger "
            "medication-awareness guidance in this prototype."
        )

    # --------------------------------------------------------
    # Cholesterol
    # --------------------------------------------------------

    if cholesterol >= 240:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "Discuss the result with a healthcare professional "
            "to determine whether further evaluation or treatment "
            "is appropriate."
        )

    elif cholesterol >= 200:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "Consider discussing the result with a healthcare "
            "professional for further evaluation."
        )

    else:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "No cholesterol-specific medication-awareness warning "
            "is triggered by this prototype."
        )

    # --------------------------------------------------------
    # Blood pressure
    # --------------------------------------------------------

    if systolic >= 140:

        recommendations.append(
            f"Your entered systolic blood pressure is "
            f"{systolic:.0f} mmHg. "
            "Discuss repeated elevated readings with a healthcare "
            "professional before making medication decisions."
        )

    else:

        recommendations.append(
            f"Your entered systolic blood pressure is "
            f"{systolic:.0f} mmHg. "
            "No blood-pressure medication-awareness warning is "
            "triggered by this prototype."
        )

    # --------------------------------------------------------
    # Overall risk
    # --------------------------------------------------------

    if risk_level == "HIGH RISK":

        recommendations.append(
            "The ML model classified the assessment as HIGH RISK. "
            "A qualified healthcare professional should review the "
            "complete health profile before treatment decisions are made."
        )

    elif risk_level == "MODERATE RISK":

        recommendations.append(
            "The ML model classified the assessment as MODERATE RISK. "
            "Consider discussing modifiable risk factors with a healthcare professional."
        )

    else:

        recommendations.append(
            "The ML model classified the assessment as LOW RISK. "
            "Continue routine health monitoring."
        )

    return {
        "agent": "Medication Awareness Agent",
        "category": "Medication",
        "recommendations": recommendations
    }


# ============================================================
# 4. LIFESTYLE AGENT
# ============================================================

def lifestyle_agent(health, risk_level):
    """
    Personalized lifestyle recommendation agent.

    Parameters considered:
        - Smoking
        - BMI
        - Diabetes
        - Risk Level
    """

    recommendations = []

    smoking = normalize(
        health.smoking
    )

    diabetes = normalize(
        health.diabetes
    )

    bmi = calculate_bmi(
        health
    )

    # --------------------------------------------------------
    # Smoking
    # --------------------------------------------------------

    if smoking == "yes":

        recommendations.append(
            "Smoking is marked as present. "
            "Consider a structured smoking-cessation plan and "
            "seek professional support if needed."
        )

    else:

        recommendations.append(
            "Smoking is marked as absent. "
            "Continue avoiding tobacco and tobacco smoke exposure."
        )

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    if bmi >= 30:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Focus on sustainable lifestyle changes involving "
            "nutrition, physical activity, sleep, and routine monitoring."
        )

    elif bmi >= 25:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Regular physical activity and balanced eating habits "
            "can support long-term health."
        )

    else:

        recommendations.append(
            f"Your BMI is {bmi:.1f}. "
            "Continue maintaining your current healthy lifestyle habits."
        )

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    if diabetes == "yes":

        recommendations.append(
            "Diabetes is marked as present. "
            "Maintain a consistent daily routine and monitor relevant "
            "health parameters according to professional guidance."
        )

    else:

        recommendations.append(
            "Diabetes is marked as absent. "
            "Continue regular health monitoring and healthy daily habits."
        )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if risk_level == "HIGH RISK":

        recommendations.append(
            "The ML assessment is classified as HIGH RISK. "
            "Prioritize regular health monitoring and discuss the "
            "overall result with a healthcare professional."
        )

    elif risk_level == "MODERATE RISK":

        recommendations.append(
            "The ML assessment is classified as MODERATE RISK. "
            "Focus on sustainable lifestyle improvements and routine monitoring."
        )

    else:

        recommendations.append(
            "The ML assessment is classified as LOW RISK. "
            "Continue maintaining healthy daily habits."
        )

    return {
        "agent": "Lifestyle Agent",
        "category": "Lifestyle",
        "recommendations": recommendations
    }


# ============================================================
# 5. CARDIOVASCULAR RISK AGENT
# ============================================================

def cardiovascular_agent(health, risk_level):
    """
    Personalized cardiovascular recommendation agent.

    Parameters considered:
        - ECG
        - Chest Pain
        - Blood Pressure
        - Cholesterol
        - Smoking
        - Diabetes
        - Risk Level
    """

    recommendations = []

    ecg = normalize(
        health.ecg_result
    )

    chest_pain = normalize(
        health.chest_pain_type
    )

    systolic = get_systolic_bp(
        health
    )

    cholesterol = float(
        health.cholesterol or 0
    )

    smoking = normalize(
        health.smoking
    )

    diabetes = normalize(
        health.diabetes
    )

    # --------------------------------------------------------
    # ECG
    # --------------------------------------------------------

    if ecg == "abnormal":

        recommendations.append(
            "The ECG result is marked as abnormal. "
            "This result should be reviewed by a qualified "
            "healthcare professional."
        )

    else:

        recommendations.append(
            "The ECG result is marked as normal. "
            "Continue routine health monitoring."
        )

    # --------------------------------------------------------
    # Chest pain
    # --------------------------------------------------------

    if chest_pain in [
        "moderate",
        "severe"
    ]:

        recommendations.append(
            f"Chest pain is recorded as "
            f"{health.chest_pain_type}. "
            "This should receive appropriate professional medical "
            "evaluation, particularly if symptoms are new, persistent, "
            "or worsening."
        )

    elif chest_pain == "mild":

        recommendations.append(
            "Mild chest pain is recorded. "
            "If it is persistent, recurrent, or associated with "
            "other symptoms, seek appropriate medical evaluation."
        )

    else:

        recommendations.append(
            "No chest pain is recorded in the assessment."
        )

    # --------------------------------------------------------
    # Blood pressure
    # --------------------------------------------------------

    if systolic >= 140:

        recommendations.append(
            f"Your entered systolic blood pressure is "
            f"{systolic:.0f} mmHg. "
            "Discuss repeated elevated readings with a healthcare professional."
        )

    elif systolic >= 130:

        recommendations.append(
            f"Your entered systolic blood pressure is "
            f"{systolic:.0f} mmHg. "
            "Continue monitoring blood pressure and maintain "
            "heart-healthy lifestyle habits."
        )

    else:

        recommendations.append(
            f"Your entered systolic blood pressure is "
            f"{systolic:.0f} mmHg. "
            "Continue routine blood-pressure monitoring."
        )

    # --------------------------------------------------------
    # Cholesterol
    # --------------------------------------------------------

    if cholesterol >= 240:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "This is high according to the prototype rules. "
            "Discuss the result with a healthcare professional."
        )

    elif cholesterol >= 200:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "Consider heart-healthy dietary habits and discuss "
            "the result with a healthcare professional."
        )

    else:

        recommendations.append(
            f"Your cholesterol value is {cholesterol:.0f} mg/dL. "
            "Continue maintaining heart-healthy dietary habits."
        )

    # --------------------------------------------------------
    # Smoking
    # --------------------------------------------------------

    if smoking == "yes":

        recommendations.append(
            "Smoking is marked as present and may contribute to "
            "cardiovascular risk. Consider a smoking-cessation plan."
        )

    else:

        recommendations.append(
            "Smoking is marked as absent. "
            "Continue avoiding tobacco exposure."
        )

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    if diabetes == "yes":

        recommendations.append(
            "Diabetes is marked as present. "
            "Maintain appropriate diabetes management and discuss "
            "cardiovascular risk factors with a healthcare professional."
        )

    else:

        recommendations.append(
            "Diabetes is marked as absent in the assessment."
        )

    # --------------------------------------------------------
    # ML risk
    # --------------------------------------------------------

    if risk_level == "HIGH RISK":

        recommendations.append(
            "The ML model classified this assessment as HIGH RISK. "
            "Discuss the complete cardiovascular risk assessment "
            "with a qualified healthcare professional."
        )

    elif risk_level == "MODERATE RISK":

        recommendations.append(
            "The ML model classified this assessment as MODERATE RISK. "
            "Consider improving modifiable lifestyle factors and "
            "maintaining regular health monitoring."
        )

    else:

        recommendations.append(
            "The ML model classified this assessment as LOW RISK. "
            "Continue healthy lifestyle practices and routine monitoring."
        )

    return {
        "agent": "Cardiovascular Risk Agent",
        "category": "Cardiovascular",
        "recommendations": recommendations
    }
    