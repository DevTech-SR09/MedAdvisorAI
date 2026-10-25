from django.contrib import admin
from .models import PatientProfile, HealthAssessment


@admin.register(PatientProfile)
class PatientProfileAdmin(admin.ModelAdmin):
    list_display = (
        "full_name",
        "age",
        "gender",
        "blood_group",
        "phone",
    )
    search_fields = ("full_name", "phone")


@admin.register(HealthAssessment)
class HealthAssessmentAdmin(admin.ModelAdmin):
    list_display = (
        "patient",
        "blood_pressure",
        "cholesterol",
        "heart_rate",
        "hba1c",
        "diabetes",
    )
    search_fields = ("patient__full_name",)
    