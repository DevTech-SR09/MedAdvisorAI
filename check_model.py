
import os
import sys
import joblib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "ml",
    "models",
    "medadvisor_cardiovascular_model.pkl"
)


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def inspect_features(name, component):
    """Display feature information exposed by a model or transformer."""

    print(f"\nComponent: {name}")

    if hasattr(component, "n_features_in_"):
        print("Input feature count:", component.n_features_in_)

    if hasattr(component, "feature_names_in_"):
        features = list(component.feature_names_in_)

        print("Input feature names:")
        for index, feature in enumerate(features, start=1):
            print(f"  {index:02d}. {feature}")

    elif not hasattr(component, "n_features_in_"):
        print("Feature metadata is not exposed by this component.")


def main():
    print_section("MedAdvisorAI - ML Model Diagnostic")

    print("Python version:", sys.version.split()[0])
    print("Project directory:", PROJECT_DIR)
    print("Model path:", MODEL_PATH)

    if not os.path.isfile(MODEL_PATH):
        print("\nERROR: Saved model file was not found.")
        print("Expected location:")
        print(MODEL_PATH)
        print("\nCheck the filename and folder location.")
        return

    print("\nSaved model file exists.")

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        print("\nERROR: Could not load the saved model.")
        print("Error type:", type(error).__name__)
        print("Details:", error)
        return

    print_section("Loaded Model")

    print("Model class:", type(model).__name__)
    print("Model module:", type(model).__module__)

    inspect_features("Saved model", model)

    if hasattr(model, "named_steps"):
        print_section("Pipeline Steps")

        for step_name, step in model.named_steps.items():
            print(f"\nStep name: {step_name}")
            print("Step type:", type(step).__name__)

            inspect_features(step_name, step)

            # Inspect fitted transformers inside a ColumnTransformer.
            if hasattr(step, "transformers_"):
                print("Configured transformers:")

                for transformer_name, transformer, columns in (
                    step.transformers_
                ):
                    print(f"\n  Transformer: {transformer_name}")
                    print(
                        "  Type:",
                        type(transformer).__name__
                    )

                    if isinstance(columns, (list, tuple)):
                        print("  Selected columns:", list(columns))
                    else:
                        print("  Column selector:", columns)

                    if transformer not in ("drop", "passthrough"):
                        if hasattr(transformer, "feature_names_in_"):
                            print(
                                "  Recorded input features:",
                                list(transformer.feature_names_in_)
                            )

    if hasattr(model, "get_params"):
        print_section("Model Configuration")

        params = model.get_params(deep=False)

        for name, value in params.items():
            print(f"{name}: {value}")

    print_section("Diagnostic Completed")

    print(
        "This script only inspects the saved model. "
        "It does not retrain or modify it."
    )


if __name__ == "__main__":
    main()
