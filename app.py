
from pathlib import Path
from io import BytesIO

import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st


SAMPLE_RATE = 22050
N_MFCC = 40

MODEL_PATH = Path("models/bird_classifier.pkl")
SCALER_PATH = Path("models/scaler.pkl")


st.set_page_config(
    page_title="Chidiya Aai",
    layout="centered"
)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def extract_features(audio_file):
    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfcc)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1)
    ])

    return features


def predict_bird(audio_file, model, scaler):
    features = extract_features(audio_file).reshape(1, -1)

    if hasattr(scaler, "feature_names_in_"):
        features = pd.DataFrame(
            features,
            columns=scaler.feature_names_in_
        )

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index]) * 100

    return prediction, confidence, model.classes_, probabilities


# ---------------- PAGE DESIGN ----------------

st.title("Chidiya Aai")
st.subheader("AI-Powered Bird Vocalization Monitoring System")

st.write(
    "Recognize birds. Understand their calls. "
    "Interpret their world."
)

st.divider()

st.markdown("### Bird Identification")

st.write(
    "Upload an audio recording to identify the most likely "
    "bird species using our trained machine learning model."
)

st.info(
    "Currently supported species: House Sparrow, Indian Robin, "
    "and Myna."
)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "flac", "ogg"]
)

if uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()

    st.audio(audio_bytes)

    if st.button("Identify Bird", type="primary"):
        try:
            model, scaler = load_model()

            prediction, confidence, classes, probabilities = (
                predict_bird(
                    BytesIO(audio_bytes),
                    model,
                    scaler
                )
            )

            readable_name = prediction.replace("_", " ").title()

            st.divider()
            st.markdown("### Analysis Result")

            st.success(f"Predicted Species: {readable_name}")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.markdown("#### Class Probability Scores")

            for species, probability in zip(classes, probabilities):
                name = species.replace("_", " ").title()

                st.write(f"**{name}** — {probability * 100:.2f}%")
                st.progress(float(probability))

            st.caption(
                "Confidence is the model's estimated score, "
                "not a guarantee of correct identification."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

st.divider()

st.markdown("### About the Project")

st.write(
    "Chidiya Aai is a prototype for bird vocalization "
    "recognition using audio feature extraction and machine learning."
)

st.write(
    "The current model classifies recordings into three trained "
    "species. Reliable recognition of other sounds and interpretation "
    "of different bird call types require further development."
)
