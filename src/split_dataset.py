from pathlib import Path
import random
import shutil

CLIPS_FOLDER = Path("audio/clips")
DATASET_FOLDER = Path("dataset")

random.seed(42)

# 5 recordings → train
# 1 recording → validation
# 1 recording → test
TRAIN_COUNT = 5

species_folders = [
    folder for folder in CLIPS_FOLDER.iterdir()
    if folder.is_dir()
]

for species_folder in species_folders:

    species = species_folder.name

    # Find all clips
    clips = list(species_folder.glob("*.wav"))

    # Get original recording names
    # Example:
    # myna_03_clip_01.wav → myna_03
    recording_names = sorted(
        set(
            "_".join(clip.stem.split("_")[:-2])
            for clip in clips
        )
    )

    random.shuffle(recording_names)

    train_recordings = recording_names[:TRAIN_COUNT]
    validation_recordings = recording_names[TRAIN_COUNT:TRAIN_COUNT + 1]
    test_recordings = recording_names[TRAIN_COUNT + 1:]

    print(f"\n{species.upper()}")
    print("Train:", train_recordings)
    print("Validation:", validation_recordings)
    print("Test:", test_recordings)

    for split_name, recordings in [
        ("train", train_recordings),
        ("validation", validation_recordings),
        ("test", test_recordings)
    ]:

        output_folder = DATASET_FOLDER / split_name / species
        output_folder.mkdir(parents=True, exist_ok=True)

        for recording in recordings:

            matching_clips = [
                clip for clip in clips
                if clip.stem.startswith(recording + "_clip_")
            ]

            for clip in matching_clips:
                shutil.copy2(
                    clip,
                    output_folder / clip.name
                )

print("\n" + "=" * 60)
print("DATASET SPLIT COMPLETE 🐦")
print("=" * 60)