from pathlib import Path
import sys

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


def detect(audio_file):

    print("\n" + "=" * 60)
    print("CHIDIYA AAI - UNKNOWN DETECTION")
    print("=" * 60)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    audio, _ = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    rms = float(
        np.sqrt(
            np.mean(audio ** 2)
        )
    )

    features = extract_features(audio_file)
    features = features.reshape(1, -1)

    feature_columns = scaler.feature_names_in_

    features_df = pd.DataFrame(
        features,
        columns=feature_columns
    )

    features_scaled = scaler.transform(features_df)

    prediction = model.predict(features_scaled)[0]

    probabilities = model.predict_proba(
        features_scaled
    )[0]

    confidence = np.max(probabilities) * 100

    print("\nAudio:", audio_file)

    print("\nAudio signal:")
    print("RMS        :", f"{rms:.6f}")

    print("\nClass probabilities:")

    for species, probability in zip(
        model.classes_,
        probabilities
    ):
        print(
            f"{species:<18} "
            f"{probability * 100:.2f}%"
        )

    print("\n" + "-" * 60)

    if rms < 0.001:
        result = "UNKNOWN"
        status = "Very low signal - likely silence or inactive audio"

    elif confidence >= 70:
        result = prediction.upper()
        status = "Known bird - strong classification"

    else:
        result = "UNKNOWN"
        status = "Unknown or uncertain audio"

    print("RESULT     :", result)
    print("CONFIDENCE :", f"{confidence:.2f}%")
    print("STATUS     :", status)

    print("-" * 60)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python src\\unknown_detector.py "
            "<audio_file>"
        )
        raise SystemExit(1)

    audio_file = sys.argv[1]

    detect(audio_file)