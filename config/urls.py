from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),

    # Accounts (Home, Login, Signup)
    path("", include("accounts.urls")),

    # Patients (Profile, Health Form, Prediction)
    path("", include("patients.urls")),
]
