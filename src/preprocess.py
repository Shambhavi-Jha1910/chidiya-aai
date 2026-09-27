import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def process_audio(audio_path):
    audio_path = Path(audio_path)

    print(f"Loading: {audio_path}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y) / sr:.2f} seconds")

    # Remove leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    print(f"Trimmed duration: {len(y_trimmed) / sr:.2f} seconds")

    # Create Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y_trimmed,
        sr=sr,
        n_mels=128,
        fmax=sr // 2
    )

    # Convert to decibels
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Create output folder
    output_dir = Path("spectrograms")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{audio_path.stem}_mel.png"

    # Plot spectrogram
    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram - {audio_path.stem}")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved spectrogram: {output_path}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/preprocess.py <audio_file>")
        sys.exit(1)

    process_audio(sys.argv[1])