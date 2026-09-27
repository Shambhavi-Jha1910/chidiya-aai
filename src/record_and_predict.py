from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np

from predict import predict_bird


SAMPLE_RATE = 22050
DURATION = 5
OUTPUT_FILE = Path("audio/live_recording.wav")

AMPLIFICATION = 3.0


def record_audio():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("                 CHIDIYA AAI")
    print("        AI BIRD VOCALIZATION MONITOR")
    print("=" * 60)

    print("\nListening for bird sounds...")
    print(f"Recording for {DURATION} seconds...\n")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    volume_before = np.max(np.abs(audio))

    audio = audio * AMPLIFICATION
    audio = np.clip(audio, -1.0, 1.0)

    volume_after = np.max(np.abs(audio))

    sf.write(
        OUTPUT_FILE,
        audio,
        SAMPLE_RATE
    )

    print("Recording complete.")
    print(f"Audio saved: {OUTPUT_FILE}")

    print("\nProcessing audio...")
    print("Extracting acoustic features...")
    print("Running AI classification...")

    return volume_before, volume_after


if __name__ == "__main__":

    volume_before, volume_after = record_audio()

    predict_bird(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("              CLASSIFICATION COMPLETE")
    print("=" * 60)