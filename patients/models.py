from django.db import models
from django.contrib.auth.models import User


# ==========================================
# Patient Profile (Step 1)
# ==========================================
class PatientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    full_name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=10)
    phone = models.CharField(max_length=15)
    blood_group = models.CharField(max_length=5)
    date_of_birth = models.DateField()
    address = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.full_name


# ==========================================
# Health Assessment (Step 2)
# ==========================================
class HealthAssessment(models.Model):
    patient = models.OneToOneField(
        PatientProfile,
        on_delete=models.CASCADE
    )

    # Basic Measurements
    height = models.FloatField(null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)
    blood_pressure = models.CharField(max_length=10, blank=True)

    # Laboratory Parameters
    cholesterol = models.FloatField(null=True, blank=True)
    blood_sugar = models.FloatField(null=True, blank=True)
    hemoglobin = models.FloatField(null=True, blank=True)
    hba1c = models.FloatField(null=True, blank=True)

    # Cardiac Parameters
    heart_rate = models.PositiveIntegerField(null=True, blank=True)
    ecg_result = models.CharField(max_length=50, blank=True)
    chest_pain_type = models.CharField(max_length=50, blank=True)

    # Medical History
    smoking = models.CharField(max_length=5, blank=True)
    diabetes = models.CharField(max_length=5, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.patient.full_name} - Assessment"
    