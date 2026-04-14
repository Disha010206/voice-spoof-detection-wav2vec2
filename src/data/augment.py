import os
import shutil
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, AirAbsorption
from tqdm import tqdm

def get_augment_pipeline():
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
        AirAbsorption(p=0.4)
    ])

def augment_train_data(input_base, output_base, num_augments=2):
    train_dir = os.path.join(input_base, "train")
    augment_pipeline = get_augment_pipeline()

    for label in ["bonafide", "spoof"]:
        input_dir = os.path.join(train_dir, label)
        output_dir = os.path.join(output_base, "train", label)

        os.makedirs(output_dir, exist_ok=True)

        print(f"\nAugmenting {label}...")
        for file in tqdm(os.listdir(input_dir)):
            if file.endswith(".flac"):
                path = os.path.join(input_dir, file)
                audio, sr = sf.read(path)

                for i in range(num_augments):
                    augmented = augment_pipeline(samples=audio, sample_rate=sr)
                    new_name = file.replace(".flac", f"_aug{i+1}.flac")
                    sf.write(os.path.join(output_dir, new_name), augmented, sr)

def copy_original_train(input_base, output_base):
    train_dir = os.path.join(input_base, "train")

    for label in ["bonafide", "spoof"]:
        src = os.path.join(train_dir, label)
        dst = os.path.join(output_base, "train", label)

        os.makedirs(dst, exist_ok=True)

        for file in os.listdir(src):
            if file.endswith(".flac"):
                shutil.copy(os.path.join(src, file), os.path.join(dst, file))

    print("Original training files copied!")

def copy_test_data(input_base, output_base):
    test_dir = os.path.join(input_base, "test")

    for label in ["bonafide", "spoof"]:
        src = os.path.join(test_dir, label)
        dst = os.path.join(output_base, "test", label)

        os.makedirs(dst, exist_ok=True)

        print(f"\nCopying TEST files for {label}...")
        for file in os.listdir(src):
            if file.endswith(".flac"):
                shutil.copy(os.path.join(src, file), os.path.join(dst, file))

    print("Test set copied (no augmentation).")

def run_augmentation(input_base, output_base):
    print("Starting data augmentation pipeline...")

    augment_train_data(input_base, output_base)
    copy_original_train(input_base, output_base)
    copy_test_data(input_base, output_base)

    print("\nALL DONE")

if __name__ == "__main__":
    INPUT_BASE = "data/asvspoof_preprocessed"
    OUTPUT_BASE = "data/asvspoof_augmented"

    run_augmentation(INPUT_BASE, OUTPUT_BASE)