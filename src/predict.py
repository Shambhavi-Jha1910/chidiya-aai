from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

SAMPLE_RATE = 22050
N_MFCC = 40

MODEL_PATH = Path("models/bird_classifier.pkl")
SCALER_PATH = Path("models/scaler.pkl")


def extract_features(audio_file):

    y, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfcc)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std
    ])

    return features


def predict_bird(audio_file):

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = extract_features(audio_file)
    features = features.reshape(1, -1)

    feature_columns = scaler.feature_names_in_

    features_df = pd.DataFrame(
        features,
        columns=feature_columns
    )

    features_scaled = scaler.transform(features_df)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    classes = model.classes_
    confidence = np.max(probabilities) * 100

    print("\n" + "=" * 60)
    print("                 CHIDIYA AAI")
    print("=" * 60)

    print("\nBIRD DETECTION RESULT")
    print("-" * 60)

    print(f"Species    : {prediction.upper()}")
    print(f"Confidence : {confidence:.2f}%")

    print("\nClass Probabilities")
    print("-" * 60)

    for species, probability in zip(classes, probabilities):
        print(
            f"{species:<18} "
            f"{probability * 100:>6.2f}%"
        )

    print("-" * 60)

    if confidence >= 70:
        print("Status     : Strong classification")
    elif confidence >= 50:
        print("Status     : Moderate classification")
    else:
        print("Status     : Low-confidence classification")

    print("=" * 60)


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python src/predict.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    predict_bird(audio_file)