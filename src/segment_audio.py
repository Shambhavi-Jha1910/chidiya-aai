from pathlib import Path
import librosa
import soundfile as sf

# Settings
RAW_FOLDER = Path("audio/raw")
OUTPUT_FOLDER = Path("audio/clips")

SAMPLE_RATE = 22050
CLIP_DURATION = 5
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

# Create output folder
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

audio_files = [
    file for file in RAW_FOLDER.rglob("*")
    if file.suffix.lower() in AUDIO_EXTENSIONS
]

print(f"Found {len(audio_files)} recordings.\n")

total_clips = 0

for audio_file in audio_files:

    # Species = parent folder name
    species = audio_file.parent.name

    # Create species output folder
    species_folder = OUTPUT_FOLDER / species
    species_folder.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {audio_file}")

    # Load audio
    y, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Remove silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Create 5-second clips
    clip_number = 1

    for start in range(0, len(y) - CLIP_SAMPLES + 1, CLIP_SAMPLES):

        clip = y[start:start + CLIP_SAMPLES]

        output_name = (
            f"{audio_file.stem}_clip_{clip_number:02d}.wav"
        )

        output_path = species_folder / output_name

        sf.write(output_path, clip, SAMPLE_RATE)

        print(f"  Saved: {output_path}")

        clip_number += 1
        total_clips += 1

print("\n" + "=" * 60)
print(f"TOTAL 5-SECOND CLIPS CREATED: {total_clips}")
print("=" * 60)