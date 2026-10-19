import sys
from pathlib import Path

import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 22050
DURATION = 5
INPUT_DEVICE = 1


def record_audio(category, filename):

    output_folder = Path("audio") / "unknown" / category
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / filename

    print("Using microphone:", INPUT_DEVICE)
    print("Recording starts in 2 seconds...")
    sd.sleep(2000)

    print("Recording now...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE
    )

    sd.wait()

    sf.write(
        output_file,
        audio,
        SAMPLE_RATE
    )

    print("Recording saved:", output_file)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(
            "Usage: python src\\record_unknown.py "
            "<category> <filename>"
        )
        raise SystemExit(1)

    category = sys.argv[1]
    filename = sys.argv[2]

    record_audio(category, filename)