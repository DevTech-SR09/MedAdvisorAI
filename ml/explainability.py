
"""
MedAdvisorAI - Explainable AI (XAI)
-----------------------------------
Generates local SHAP explanations for the existing
cardiovascular risk prediction model.

Features:
1. Reuses the saved model without retraining.
2. Uses the existing feature engineering function.
3. Calculates SHAP contributions.
4. Displays the top-ranked features.
5. Always includes HbA1c and ECG-related features.
6. Returns prediction and explanation metadata.

Important:
SHAP contributions describe model output, not medical causality.
This model is not a substitute for clinical assessment.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from ml.predictor import create_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "medadvisor_cardiovascular_model.pkl"
)


# Features to include even when they are outside the top-N ranking.
PRIORITY_FEATURE_PATTERNS = (
    "hba1c",
    "abnormal_ecg",
    "ecg_result",
    "ecg_",
)


def load_xai_components():
    """Load the saved pipeline and initialize SHAP."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    pipeline = joblib.load(MODEL_PATH)

    if not hasattr(pipeline, "named_steps"):
        raise TypeError(
            "The saved model is not a scikit-learn pipeline."
        )

    required_steps = {"preprocessor", "model"}

    if not required_steps.issubset(
        pipeline.named_steps.keys()
    ):
        raise ValueError(
            "Expected pipeline steps: preprocessor and model."
        )

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    # Explain the fitted classifier after preprocessing.
    explainer = shap.TreeExplainer(classifier)

    return (
        pipeline,
        preprocessor,
        explainer,
        feature_names,
    )


def extract_class_contributions(
    shap_values,
    class_index,
    feature_count,
):
    """
    Extract contributions for a single prediction.

    Supports common SHAP formats:
    - List of arrays, one array per class.
    - 1D array: features.
    - 2D array: samples x features.
    - 3D array: samples x features x outputs.

    The result must contain one contribution per feature.
    """

    if isinstance(shap_values, list):
        if class_index >= len(shap_values):
            raise ValueError(
                "Requested class is missing from SHAP output."
            )

        values = np.asarray(shap_values[class_index])

    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            # Typical layout: samples x features x outputs.
            if values.shape[1] == feature_count:
                if class_index >= values.shape[2]:
                    raise ValueError(
                        "Requested class is missing from SHAP output."
                    )

                values = values[0, :, class_index]

            # Alternative layout: outputs x samples x features.
            elif values.shape[2] == feature_count:
                if class_index >= values.shape[0]:
                    raise ValueError(
                        "Requested class is missing from SHAP output."
                    )

                values = values[class_index, 0, :]

            else:
                raise ValueError(
                    "Cannot identify the feature axis in "
                    f"SHAP output shape {values.shape}."
                )

        elif values.ndim == 2:
            # One row for the single patient's input.
            values = values[0, :]

        elif values.ndim != 1:
            raise ValueError(
                f"Unsupported SHAP output shape: {values.shape}"
            )

    values = np.asarray(values).reshape(-1)

    if len(values) != feature_count:
        raise ValueError(
            "SHAP contribution count does not match feature count. "
            f"Got {len(values)}, expected {feature_count}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "SHAP returned non-finite contribution values."
        )

    return values


def select_explanation_features(
    contribution_df,
    top_n=10,
):
    """
    Select the top-N features and always include matching
    HbA1c and ECG-related features when present.

    No contribution values are modified.
    """

    top_features = contribution_df.head(top_n).copy()

    names_lower = contribution_df["feature"].str.lower()

    priority_mask = names_lower.apply(
        lambda name: any(
            pattern in name
            for pattern in PRIORITY_FEATURE_PATTERNS
        )
    )

    priority_features = contribution_df[
        priority_mask
    ].copy()

    combined = pd.concat(
        [top_features, priority_features],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["feature"]
    )

    combined = combined.sort_values(
        "absolute_contribution",
        ascending=False,
    )

    selected_names = set(combined["feature"])

    # Report missing categories rather than pretending
    # the model contains features it does not have.
    all_names = contribution_df["feature"].str.lower()

    hba1c_present = all_names.str.contains(
        "hba1c", regex=False
    ).any()

    ecg_present = all_names.str.contains(
        "ecg", regex=False
    ).any()

    return combined, {
        "hba1c_feature_found": bool(hba1c_present),
        "ecg_feature_found": bool(ecg_present),
        "selected_feature_count": len(selected_names),
        "requested_top_n": top_n,
    }


def explain_prediction(
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
    diabetes,
    top_n=10,
):
    """
    Generate a local SHAP explanation for one prediction.

    Input arguments match ml.predictor.create_features().

    Returns:
        prediction
        class_1_probability
        probability_percent
        top_features
        explanation_metadata
    """

    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or top_n < 1
    ):
        raise ValueError(
            "top_n must be a positive integer."
        )

    (
        pipeline,
        preprocessor,
        explainer,
        feature_names,
    ) = load_xai_components()

    # 1. Use the project's existing feature engineering.
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
        diabetes=diabetes,
    )

    # 2. Transform the inputs using the fitted preprocessor.
    transformed = preprocessor.transform(features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(
        transformed,
        dtype=float,
    )

    if transformed.shape[0] != 1:
        raise ValueError(
            "Expected exactly one input record."
        )

    if transformed.shape[1] != len(feature_names):
        raise ValueError(
            "Transformed feature count does not match "
            "the preprocessor's feature names."
        )

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
    )

    # 3. Generate prediction using the original pipeline.
    prediction = int(
        pipeline.predict(features)[0]
    )

    classes = list(pipeline.classes_)

    if 1 not in classes:
        raise ValueError(
            "The trained model does not contain class 1."
        )

    class_1_index = classes.index(1)

    probabilities = pipeline.predict_proba(features)[0]

    class_1_probability = float(
        probabilities[class_1_index]
    )

    # 4. Calculate SHAP contributions.
    shap_values = explainer.shap_values(
        transformed_df
    )

    contributions = extract_class_contributions(
        shap_values=shap_values,
        class_index=class_1_index,
        feature_count=len(feature_names),
    )

    # 5. Build a table of actual SHAP contributions.
    contribution_df = pd.DataFrame({
        "feature": [str(name) for name in feature_names],
        "contribution": contributions,
    })

    contribution_df["absolute_contribution"] = (
        contribution_df["contribution"].abs()
    )

    contribution_df["direction"] = np.select(
        [
            contribution_df["contribution"] > 0,
            contribution_df["contribution"] < 0,
        ],
        [
            "pushes class 1 output higher",
            "pushes class 1 output lower",
        ],
        default="no contribution",
    )

    contribution_df = contribution_df.sort_values(
        "absolute_contribution",
        ascending=False,
    )

    # 6. Include top-N features plus HbA1c/ECG features.
    selected_features, selection_metadata = (
        select_explanation_features(
            contribution_df,
            top_n=top_n,
        )
    )

    top_features_records = []

    for record in selected_features.to_dict(
        orient="records"
    ):
        top_features_records.append({
            "feature": str(record["feature"]),
            "contribution": float(
                record["contribution"]
            ),
            "absolute_contribution": float(
                record["absolute_contribution"]
            ),
            "direction": str(record["direction"]),
        })

    # 7. Return results for the test script or Django view.
    return {
        "prediction": prediction,
        "class_1_probability": class_1_probability,
        "probability_percent": round(
            class_1_probability * 100,
            2,
        ),
        "top_features": top_features_records,
        "explanation_metadata": {
            "method": "SHAP TreeExplainer",
            "shap_version": shap.__version__,
            "classifier": type(
                pipeline.named_steps["model"]
            ).__name__,
            "original_feature_count": int(
                features.shape[1]
            ),
            "transformed_feature_count": int(
                len(feature_names)
            ),
            "explained_class": 1,
            "output_scale": (
                "Not yet verified; do not interpret SHAP values "
                "as probability percentage points."
            ),
            **selection_metadata,
        },
    }


if __name__ == "__main__":
    (
        pipeline,
        preprocessor,
        explainer,
        feature_names,
    ) = load_xai_components()

    print("=" * 60)
    print("MedAdvisorAI XAI Module")
    print("=" * 60)

    print("SHAP version:", shap.__version__)
    print(
        "Classifier:",
        type(pipeline.named_steps["model"]).__name__,
    )
    print(
        "Original input features:",
        len(pipeline.feature_names_in_),
    )
    print(
        "Transformed features:",
        len(feature_names),
    )
    print("SHAP explainer initialized successfully.")
    print("HbA1c/ECG priority feature selection enabled.")
