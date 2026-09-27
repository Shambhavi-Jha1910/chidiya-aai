from pathlib import Path

import librosa
import numpy as np
import pandas as pd


DATASET_FOLDER = Path("dataset")
OUTPUT_FILE = Path("features.csv")

SAMPLE_RATE = 22050
N_MFCC = 40


def extract_features(audio_file):

    y, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # MFCC delta
    delta = librosa.feature.delta(mfcc)

    # Mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    # Combine all features
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std
    ])

    return features


rows = []

for split in ["train", "validation", "test"]:

    split_folder = DATASET_FOLDER / split

    for species_folder in split_folder.iterdir():

        if not species_folder.is_dir():
            continue

        species = species_folder.name

        for audio_file in species_folder.glob("*.wav"):

            print(f"Processing: {audio_file}")

            features = extract_features(audio_file)

            row = {
                "split": split,
                "species": species,
                "filename": audio_file.name
            }

            for i, value in enumerate(features, start=1):
                row[f"FEATURE_{i}"] = value

            rows.append(row)


df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)


print("\n" + "=" * 60)
print("IMPROVED FEATURE EXTRACTION COMPLETE 🐦")
print("=" * 60)

print(f"Total samples: {len(df)}")
print(f"Features per sample: {len(df.columns) - 3}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nSamples by split:")
print(df["split"].value_counts())

print("\nSamples by species:")
print(df["species"].value_counts())