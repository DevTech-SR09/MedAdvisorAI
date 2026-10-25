
from ml.predictor import create_features, model

features = create_features(
    height_cm=170,
    weight_kg=70,
    blood_pressure="120/80",
    cholesterol_mg_dl=190,
    blood_sugar_mg_dl=95,
    hemoglobin_g_dl=14,
    hba1c_percent=5.4,
    heart_rate_bpm=75,
    ecg_result="Normal",
    chest_pain_type="No Chest Pain",
    smoking="No",
    diabetes="No",
)

expected = list(model.feature_names_in_)
actual = list(features.columns)

print("Expected feature count:", len(expected))
print("Actual feature count:", len(actual))
print("Missing features:", sorted(set(expected) - set(actual)))
print("Unexpected features:", sorted(set(actual) - set(expected)))
print("Feature order matches:", expected == actual)

try:
    prediction = model.predict(features)
    print("Prediction:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        print("Probabilities:", probabilities)

    print("Prediction test passed.")
except Exception as error:
    print("Prediction test failed:", type(error).__name__, error)
