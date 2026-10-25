import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(
    BASE_DIR,
    "datasets",
    "medadvisor_cardiovascular_synthetic_20000.csv"
)

model_path = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "medadvisor_cardiovascular_model.pkl"
)

df = pd.read_csv(dataset_path)
model = joblib.load(model_path)

print("=" * 55)
print("FULL DATASET LABEL INSPECTION")
print("=" * 55)

print("\nComplete ECG distribution:")
print(df["ecg_result"].value_counts(dropna=False))

print("\nComplete cardiovascular risk distribution:")
print(df["cardiovascular_risk"].value_counts(dropna=False).sort_index())

print("\nSaved model class order:")
print(model.classes_)

print("\nProbability column mapping:")
for index, class_label in enumerate(model.classes_):
    print(f"Column {index}: class {class_label}")

print("\nDiagnostic complete.")
