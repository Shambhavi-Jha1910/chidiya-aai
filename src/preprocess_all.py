from pathlib import Path
import subprocess
import sys

# Folder containing all bird recordings
raw_folder = Path("audio/raw")

# Supported audio formats
audio_extensions = {".wav", ".mp3", ".m4a", ".flac"}

# Find every audio file inside species folders
audio_files = [
    file for file in raw_folder.rglob("*")
    if file.suffix.lower() in audio_extensions
]

print(f"\nFound {len(audio_files)} audio recordings.\n")

# Process every recording
for i, audio_file in enumerate(audio_files, start=1):

    print("=" * 60)
    print(f"[{i}/{len(audio_files)}] Processing: {audio_file}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "src/preprocess.py", str(audio_file)],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"ERROR processing: {audio_file}")

print("\n" + "=" * 60)
print("ALL AUDIO PROCESSING COMPLETED! 🐦")
print("=" * 60)