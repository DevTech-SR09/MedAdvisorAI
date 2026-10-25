
from django.urls import path
from . import views

urlpatterns = [
    path("profile/", views.profile, name="profile"),
    path("health-form/", views.health_form, name="health_form"),
    path("prediction/", views.prediction, name="prediction"),

    path(
        "prediction/download-pdf/",
        views.download_pdf_report,
        name="download_pdf_report",
    ),

    path(
        "recommendations/",
        views.recommendations_page,
        name="recommendations",
    ),
    path("xai/", views.xai_page, name="xai"),
]
