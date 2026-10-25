
from io import BytesIO
from xml.sax.saxutils import escape

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.utils import timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

from .models import PatientProfile, HealthAssessment

from ml.predictor import predict_cardiovascular_risk
from ml.explainability import explain_prediction
from ml.recommendation_coordinator import generate_recommendations


# ============================================================
# Shared helpers
# ============================================================

def get_patient_inputs(health):
    """Convert a saved health assessment to the ML input format."""
    return {
        "height_cm": health.height,
        "weight_kg": health.weight,
        "blood_pressure": health.blood_pressure,
        "cholesterol_mg_dl": health.cholesterol,
        "blood_sugar_mg_dl": health.blood_sugar,
        "hemoglobin_g_dl": health.hemoglobin,
        "hba1c_percent": health.hba1c,
        "heart_rate_bpm": health.heart_rate,
        "ecg_result": health.ecg_result,
        "chest_pain_type": health.chest_pain_type,
        "smoking": health.smoking,
        "diabetes": health.diabetes,
    }


def get_patient_records(request):
    """Return the logged-in user's profile and health assessment."""
    profile = PatientProfile.objects.filter(
        user=request.user
    ).first()

    if profile is None:
        return None, None

    health = HealthAssessment.objects.filter(
        patient=profile
    ).first()

    return profile, health


def get_bmi(health):
    """Calculate BMI when valid height and weight are available."""
    if health.height is None or health.weight is None:
        return None

    if health.height <= 0 or health.weight <= 0:
        return None

    height_m = health.height / 100
    return round(health.weight / (height_m ** 2), 1)


def pdf_text(value):
    """Escape values before placing them in ReportLab paragraphs."""
    if value is None or str(value).strip() == "":
        return "Not provided"

    return escape(str(value))


def add_pdf_section(story, styles, title, rows):
    """Add a titled two-column section to the PDF."""
    story.append(Paragraph(title, styles["SectionHeading"]))

    data = [
        [
            Paragraph(f"<b>{pdf_text(label)}</b>", styles["BodyText"]),
            Paragraph(pdf_text(value), styles["BodyText"]),
        ]
        for label, value in rows
    ]

    if not data:
        data = [[Paragraph("No information available.", styles["BodyText"]), ""]]

    table = Table(
        data,
        colWidths=[52 * mm, 118 * mm],
        hAlign="LEFT",
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3FB")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DCE8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(table)
    story.append(Spacer(1, 5 * mm))


# ============================================================
# Patient Profile
# ============================================================

@login_required
def profile(request):
    patient_profile = PatientProfile.objects.filter(
        user=request.user
    ).first()

    if request.method == "POST":
        if patient_profile is None:
            patient_profile = PatientProfile(user=request.user)

        patient_profile.full_name = request.POST.get("full_name")
        patient_profile.age = request.POST.get("age")
        patient_profile.gender = request.POST.get("gender")
        patient_profile.phone = request.POST.get("phone")
        patient_profile.blood_group = request.POST.get("blood_group")
        patient_profile.date_of_birth = (
            request.POST.get("date_of_birth") or None
        )
        patient_profile.address = request.POST.get("address")

        patient_profile.save()
        return redirect("health_form")

    return render(
        request,
        "patients/profile.html",
        {"profile": patient_profile},
    )


# ============================================================
# Health Assessment Form
# ============================================================

@login_required
def health_form(request):
    patient_profile = PatientProfile.objects.filter(
        user=request.user
    ).first()

    if patient_profile is None:
        return redirect("profile")

    health = HealthAssessment.objects.filter(
        patient=patient_profile
    ).first()

    if request.method == "POST":
        if health is None:
            health = HealthAssessment(patient=patient_profile)

        try:
            health.height = float(request.POST.get("height"))
            health.weight = float(request.POST.get("weight"))
            health.blood_pressure = request.POST.get("blood_pressure", "")
            health.heart_rate = int(request.POST.get("heart_rate"))
            health.cholesterol = float(request.POST.get("cholesterol"))
            health.blood_sugar = float(request.POST.get("blood_sugar"))
            health.hemoglobin = float(request.POST.get("hb"))
            health.hba1c = float(request.POST.get("hba1c"))
            health.ecg_result = request.POST.get("ecg", "")
            health.chest_pain_type = request.POST.get("chest_pain", "")
            health.smoking = request.POST.get("smoking", "")
            health.diabetes = request.POST.get("diabetes", "")

            health.save()
            return redirect("prediction")

        except (TypeError, ValueError):
            return render(
                request,
                "patients/health_form.html",
                {
                    "profile": patient_profile,
                    "health": health,
                    "error": "Please enter valid health values.",
                },
            )

    return render(
        request,
        "patients/health_form.html",
        {
            "profile": patient_profile,
            "health": health,
        },
    )


# ============================================================
# PAGE 1: Cardiovascular Prediction
# ============================================================

@login_required
def prediction(request):
    patient_profile, health = get_patient_records(request)

    if patient_profile is None:
        return redirect("profile")

    if health is None:
        return redirect("health_form")

    try:
        result = predict_cardiovascular_risk(
            **get_patient_inputs(health)
        )

    except Exception as error:
        print("Prediction error:", repr(error))

        return render(
            request,
            "patients/prediction.html",
            {
                "profile": patient_profile,
                "health": health,
                "error": (
                    "Unable to generate the prediction. "
                    "Please check the server terminal."
                ),
            },
        )

    context = {
        "profile": patient_profile,
        "health": health,
        "bmi": get_bmi(health),
        "prediction": result["prediction"],
        "probability": result["probability"],
        "probability_percent": result["probability_percent"],
        "risk_level": result["risk_level"],
        "risk_color": result["risk_color"],
        "recommendation": result["recommendation"],
    }

    return render(
        request,
        "patients/prediction.html",
        context,
    )


# ============================================================
# PAGE 2: Personalized Health Recommendations
# ============================================================

@login_required
def recommendations_page(request):
    patient_profile, health = get_patient_records(request)

    if patient_profile is None:
        return redirect("profile")

    if health is None:
        return redirect("health_form")

    result = None
    recommendations = []
    total_recommendations = 0
    recommendation_error = None

    try:
        result = predict_cardiovascular_risk(
            **get_patient_inputs(health)
        )

        recommendation_result = generate_recommendations(
            health=health,
            risk_level=result["risk_level"],
        )

        recommendations = recommendation_result.get(
            "recommendations", []
        )

        total_recommendations = recommendation_result.get(
            "total_recommendations", 0
        )

    except Exception as error:
        print("Recommendation engine error:", repr(error))

        recommendation_error = (
            "Unable to generate personalized recommendations."
        )

    context = {
        "profile": patient_profile,
        "health": health,
        "risk_level": (
            result["risk_level"] if result else None
        ),
        "recommendations": recommendations,
        "total_recommendations": total_recommendations,
        "recommendation_error": recommendation_error,
    }

    return render(
        request,
        "patients/recommendations.html",
        context,
    )


# ============================================================
# PAGE 3: Explainable AI (SHAP)
# ============================================================

@login_required
def xai_page(request):
    patient_profile, health = get_patient_records(request)

    if patient_profile is None:
        return redirect("profile")

    if health is None:
        return redirect("health_form")

    xai_result = None
    xai_error = None

    try:
        inputs = get_patient_inputs(health)

        xai_result = explain_prediction(
            **inputs,
            top_n=100,
        )

        display_labels = {
            "hba1c_percent": "HbA1c level",
            "bmi": "Body mass index (BMI)",
            "systolic_bp": "Systolic blood pressure",
            "diastolic_bp": "Diastolic blood pressure",
            "cholesterol_mg_dl": "Total cholesterol",
            "blood_sugar_mg_dl": "Blood glucose",
            "chest_pain_flag": "Chest pain indicator",
            "smoking_flag": "Smoking indicator",
            "diabetes_flag": "Diabetes indicator",
            "abnormal_ecg_flag": "ECG abnormality indicator",
            "mean_arterial_pressure": "Mean arterial pressure",
            "glucose_hba1c_ratio": "Glucose-to-HbA1c ratio",
            "glucose_hba1c_difference": "Glucose-HbA1c difference",
        }

        features = xai_result.get("top_features", [])

        for feature in features:
            original_name = feature.get("feature", "Unknown feature")

            normalized_name = (
                original_name
                .replace("numeric__", "")
                .replace("categorical__", "")
                .replace("num__", "")
                .replace("cat__", "")
            )

            lookup_name = normalized_name.lower()

            feature["display_name"] = display_labels.get(
                lookup_name,
                normalized_name.replace("_", " ").title(),
            )

            contribution = float(feature.get("contribution", 0))

            if contribution > 0:
                feature["bar_color"] = "#DC2626"
                feature["direction_label"] = (
                    "Pushes class 1 model output higher"
                )
            elif contribution < 0:
                feature["bar_color"] = "#16A34A"
                feature["direction_label"] = (
                    "Pushes class 1 model output lower"
                )
            else:
                feature["bar_color"] = "#94A3B8"
                feature["direction_label"] = "Zero contribution"

        max_contribution = max(
            (
                float(feature.get("absolute_contribution", 0))
                for feature in features
            ),
            default=0,
        )

        for feature in features:
            absolute_value = float(
                feature.get("absolute_contribution", 0)
            )

            feature["bar_width"] = (
                round(absolute_value / max_contribution * 100, 2)
                if max_contribution > 0
                else 0
            )

        xai_result["hba1c_features"] = [
            feature for feature in features
            if "hba1c" in feature.get("feature", "").lower()
            or "glucose_hba1c" in feature.get("feature", "").lower()
        ]

        xai_result["ecg_features"] = [
            feature for feature in features
            if "ecg" in feature.get("feature", "").lower()
        ]

    except Exception as error:
        print("XAI error:", repr(error))

        xai_result = None
        xai_error = (
            "The SHAP explanation could not be generated. "
            "Please check the server terminal."
        )

    return render(
        request,
        "patients/xai.html",
        {
            "profile": patient_profile,
            "health": health,
            "xai_result": xai_result,
            "xai_error": xai_error,
        },
    )


# ============================================================
# Download Complete Patient PDF Report
# ============================================================

@login_required
def download_pdf_report(request):
    """Generate a PDF from the logged-in patient's own records."""

    profile, health = get_patient_records(request)

    if profile is None:
        return redirect("profile")

    if health is None:
        return redirect("health_form")

    # Generate the existing prediction.
    try:
        inputs = get_patient_inputs(health)
        result = predict_cardiovascular_risk(**inputs)

    except Exception as error:
        print("PDF prediction error:", repr(error))
        return HttpResponse(
            "Unable to generate the prediction for this report. "
            "Please check the server terminal.",
            status=500,
            content_type="text/plain",
        )

    # Generate personalized recommendations.
    recommendations = []
    try:
        recommendation_result = generate_recommendations(
            health=health,
            risk_level=result["risk_level"],
        )
        recommendations = recommendation_result.get(
            "recommendations", []
        ) or []

    except Exception as error:
        print("PDF recommendation error:", repr(error))

    # Generate SHAP explanations.
    xai_features = []
    try:
        xai_result = explain_prediction(**inputs, top_n=100)
        xai_features = xai_result.get("top_features", []) or []

        # Match the friendly feature names used by the XAI page.
        display_labels = {
            "hba1c_percent": "HbA1c level",
            "bmi": "Body mass index (BMI)",
            "systolic_bp": "Systolic blood pressure",
            "diastolic_bp": "Diastolic blood pressure",
            "cholesterol_mg_dl": "Total cholesterol",
            "blood_sugar_mg_dl": "Blood glucose",
            "chest_pain_flag": "Chest pain indicator",
            "smoking_flag": "Smoking indicator",
            "diabetes_flag": "Diabetes indicator",
            "abnormal_ecg_flag": "ECG abnormality indicator",
            "mean_arterial_pressure": "Mean arterial pressure",
            "glucose_hba1c_ratio": "Glucose-to-HbA1c ratio",
            "glucose_hba1c_difference": "Glucose-HbA1c difference",
        }

        for feature in xai_features:
            original = feature.get("feature", "Unknown feature")
            normalized = (
                original.replace("numeric__", "")
                .replace("categorical__", "")
                .replace("num__", "")
                .replace("cat__", "")
            )
            feature["display_name"] = display_labels.get(
                normalized.lower(),
                normalized.replace("_", " ").title(),
            )

    except Exception as error:
        print("PDF XAI error:", repr(error))
        xai_features = []

    # Create PDF in memory.
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="MedAdvisorAI Patient Health Report",
        author="MedAdvisorAI",
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1746A2"),
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1746A2"),
        spaceBefore=12,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
    ))

    story = []

    story.append(Paragraph("MedAdvisorAI", styles["ReportTitle"]))
    story.append(Paragraph(
        "Patient Health Assessment Report",
        styles["Heading2"],
    ))
    story.append(Paragraph(
        "Generated on: " + pdf_text(
            timezone.localtime(timezone.now()).strftime(
                "%d %B %Y, %I:%M %p"
            )
        ),
        styles["BodyText"],
    ))
    story.append(Spacer(1, 5 * mm))

    # 1. Patient profile.
    add_pdf_section(story, styles, "1. Patient Profile", [
        ("Full name", profile.full_name),
        ("Age", profile.age),
        ("Gender", profile.gender),
        ("Date of birth", profile.date_of_birth),
        ("Phone", profile.phone),
        ("Blood group", profile.blood_group),
        ("Address", profile.address),
    ])

    # 2. Health assessment.
    add_pdf_section(story, styles, "2. Health Assessment", [
        ("Height (cm)", health.height),
        ("Weight (kg)", health.weight),
        ("BMI", get_bmi(health)),
        ("Blood pressure", health.blood_pressure),
        ("Cholesterol (mg/dL)", health.cholesterol),
        ("Blood sugar (mg/dL)", health.blood_sugar),
        ("Hemoglobin (g/dL)", health.hemoglobin),
        ("HbA1c (%)", health.hba1c),
        ("Heart rate (bpm)", health.heart_rate),
        ("ECG result", health.ecg_result),
        ("Chest pain type", health.chest_pain_type),
        ("Smoking", health.smoking),
        ("Diabetes", health.diabetes),
        ("Assessment created", health.created_at),
    ])

    # 3. Prediction results.
    add_pdf_section(story, styles, "3. Cardiovascular Risk Prediction", [
        ("Predicted class", result.get("prediction")),
        (
            "Class 1 probability",
            (
                f'{float(result["probability_percent"]):.2f}%'
                if result.get("probability_percent") is not None
                else "Not available"
            ),
        ),
        ("Model risk category", result.get("risk_level")),
        ("AI risk assessment", result.get("recommendation")),
    ])

    # 4. Recommendations.
    story.append(Paragraph(
        "4. Personalized Health Recommendations",
        styles["SectionHeading"],
    ))

    if recommendations:
        for index, item in enumerate(recommendations, start=1):
            if isinstance(item, dict):
                title = (
                    item.get("title")
                    or item.get("category")
                    or f"Recommendation {index}"
                )
                detail = (
                    item.get("description")
                    or item.get("recommendation")
                    or item.get("message")
                    or ""
                )

                recommendation_text = f"<b>{pdf_text(title)}</b>"

                if detail:
                    recommendation_text += (
                        "<br/>" + pdf_text(detail)
                    )
            else:
                recommendation_text = pdf_text(item)

            story.append(Paragraph(
                f"{index}. {recommendation_text}",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 3 * mm))
    else:
        story.append(Paragraph(
            "No personalized recommendations were available "
            "when this report was generated.",
            styles["BodyText"],
        ))

    # 5. Explainable AI.
    story.append(Paragraph(
        "5. Explainable AI (SHAP)",
        styles["SectionHeading"],
    ))

    if xai_features:
        xai_data = [[
            Paragraph("<b>Feature</b>", styles["BodyText"]),
            Paragraph("<b>Contribution</b>", styles["BodyText"]),
            Paragraph("<b>Interpretation</b>", styles["BodyText"]),
        ]]

        for feature in xai_features:
            feature_name = feature.get(
                "display_name",
                feature.get("feature", "Unknown feature"),
            )
            contribution = feature.get("contribution", 0)

            try:
                contribution_value = float(contribution)
                contribution_text = f"{contribution_value:.6f}"

                if contribution_value > 0:
                    direction = "Pushes class 1 output higher"
                elif contribution_value < 0:
                    direction = "Pushes class 1 output lower"
                else:
                    direction = "Zero contribution"

            except (TypeError, ValueError):
                contribution_text = str(contribution)
                direction = "Interpretation unavailable"

            xai_data.append([
                Paragraph(pdf_text(feature_name), styles["BodyText"]),
                Paragraph(pdf_text(contribution_text), styles["BodyText"]),
                Paragraph(pdf_text(direction), styles["BodyText"]),
            ])

        xai_table = Table(
            xai_data,
            colWidths=[55 * mm, 35 * mm, 80 * mm],
            repeatRows=1,
        )

        xai_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3FB")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DCE8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))

        story.append(xai_table)
    else:
        story.append(Paragraph(
            "SHAP explanations were unavailable when this report "
            "was generated. No feature contributions are included.",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 8 * mm))

    story.append(KeepTogether([
        Paragraph("Medical Disclaimer", styles["Heading3"]),
        Paragraph(
            "MedAdvisorAI is an academic prototype. Its predictions "
            "are not a medical diagnosis, prescription, or validated "
            "clinical risk estimate. Model probability is not necessarily "
            "the patient's clinical probability of cardiovascular disease. "
            "Consult a qualified healthcare professional for medical advice.",
            styles["SmallNote"],
        ),
    ]))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    response = HttpResponse(
        pdf_bytes,
        content_type="application/pdf",
    )
    response["Content-Disposition"] = (
        'attachment; filename="MedAdvisorAI_Health_Report.pdf"'
    )
    response["X-Content-Type-Options"] = "nosniff"

    return response
