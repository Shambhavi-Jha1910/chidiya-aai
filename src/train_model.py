from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


FEATURE_FILE = Path("features.csv")
MODEL_FOLDER = Path("models")

MODEL_FOLDER.mkdir(exist_ok=True)


# Load features
df = pd.read_csv(FEATURE_FILE)

feature_columns = [
    column for column in df.columns
    if column.startswith("FEATURE_")
]


# Separate datasets
train_data = df[df["split"] == "train"]
validation_data = df[df["split"] == "validation"]
test_data = df[df["split"] == "test"]


X_train = train_data[feature_columns]
y_train = train_data["species"]

X_val = validation_data[feature_columns]
y_val = validation_data["species"]

X_test = test_data[feature_columns]
y_test = test_data["species"]


print("\n" + "=" * 60)
print("CHIDIYA AAI - IMPROVED BIRD CLASSIFIER")
print("=" * 60)

print(f"\nFeatures per sample: {len(feature_columns)}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")


# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# SVM classifier
model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)


# Train
print("\nTraining SVM model...")

model.fit(X_train_scaled, y_train)

print("Training complete!")


# Validation
val_predictions = model.predict(X_val_scaled)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print("\nValidation Accuracy:")
print(f"{val_accuracy * 100:.2f}%")


# Test
test_predictions = model.predict(X_test_scaled)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\nTest Accuracy:")
print(f"{test_accuracy * 100:.2f}%")


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# Confusion matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# Save model
joblib.dump(
    model,
    MODEL_FOLDER / "bird_classifier.pkl"
)

joblib.dump(
    scaler,
    MODEL_FOLDER / "scaler.pkl"
)


print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE 🐦🤖")
print("=" * 60)

print("\nSaved:")
print("models/bird_classifier.pkl")
print("models/scaler.pkl")