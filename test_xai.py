
from colorama import Fore, Style, init

from ml.explainability import explain_prediction
from ml.predictor import predict_cardiovascular_risk


# Initialize colored terminal output
init(autoreset=True)


def print_heading(title):
    print()
    print(Fore.CYAN + Style.BRIGHT + "=" * 65)
    print(Fore.CYAN + Style.BRIGHT + title)
    print(Fore.CYAN + Style.BRIGHT + "=" * 65)


def main():
    # Synthetic sample input for software testing only.
    # This is not a real patient's medical information.
    patient = {
        "height_cm": 170,
        "weight_kg": 75,
        "blood_pressure": "120/80",
        "cholesterol_mg_dl": 200,
        "blood_sugar_mg_dl": 100,
        "hemoglobin_g_dl": 14,
        "hba1c_percent": 5.5,
        "heart_rate_bpm": 75,
        "ecg_result": "Normal",
        "chest_pain_type": "No Chest Pain",
        "smoking": "No",
        "diabetes": "No",
    }

    print_heading("MedAdvisorAI - Explainable AI Test")

    try:
        # Step 1: Existing prediction
        print(Fore.YELLOW + "\n[1] Running existing model...")

        prediction = predict_cardiovascular_risk(**patient)

        print(Fore.GREEN + Style.BRIGHT + "\nPrediction result:")
        print(Fore.WHITE + str(prediction))

        # Step 2: Generate explanation
        print(Fore.YELLOW + "\n[2] Generating SHAP explanation...")

        explanation = explain_prediction(
            **patient,
            top_n=10,
        )

        print(Fore.GREEN + Style.BRIGHT + "\nSHAP explanation generated!")

        print_heading("Prediction Summary")

        print(
            Fore.WHITE + "Predicted class: "
            + Fore.MAGENTA
            + str(explanation["prediction"])
        )

        print(
            Fore.WHITE + "Class 1 probability: "
            + Fore.YELLOW
            + f'{explanation["probability_percent"]:.2f}%'
        )

        print_heading("Top 10 Influential Features")

        for index, feature in enumerate(
            explanation["top_features"],
            start=1,
        ):
            value = feature["contribution"]

            # Positive and negative contributions use different colors.
            if value > 0:
                color = Fore.RED
            elif value < 0:
                color = Fore.GREEN
            else:
                color = Fore.WHITE

            print(
                color
                + f'{index:02d}. {feature["feature"]}: '
                + f'{value:+.6f}'
            )

            print(
                Fore.WHITE
                + "    Direction: "
                + color
                + feature["direction"]
            )

        print_heading("Explanation Metadata")

        metadata = explanation["explanation_metadata"]

        for key, value in metadata.items():
            print(
                Fore.CYAN + f"{key}: "
                + Fore.WHITE + str(value)
            )

        print_heading("Validation Checks")

        # Check that XAI and the existing model agree on the class.
        if explanation["prediction"] == prediction["prediction"]:
            print(Fore.GREEN + "PASS: Predicted classes match.")
        else:
            print(Fore.RED + "FAIL: Predicted classes do not match.")

        # Compare probability values when the predictor exposes them.
        if "probability" in prediction:
            difference = abs(
                float(explanation["class_1_probability"])
                - float(prediction["probability"])
            )

            if difference < 1e-6:
                print(
                    Fore.GREEN
                    + "PASS: Predicted probabilities match."
                )
            else:
                print(
                    Fore.RED
                    + "WARNING: Probability values differ by "
                    + f"{difference:.8f}."
                )

        print(
            Fore.YELLOW
            + "\nNote: SHAP contribution colors describe model output "
            "direction, not whether a feature is medically good or bad."
        )

        print(
            Fore.YELLOW
            + "This synthetic test is not a clinical assessment."
        )

        print(Fore.GREEN + Style.BRIGHT + "\nXAI test finished.")

    except Exception as error:
        print_heading("XAI Test Error")
        print(Fore.RED + Style.BRIGHT + type(error).__name__)
        print(Fore.RED + str(error))
        print(
            Fore.YELLOW
            + "\nPlease share the complete traceback if the test fails."
        )
        raise


if __name__ == "__main__":
    main()
