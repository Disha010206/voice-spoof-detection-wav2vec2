import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base-960h"
AUDIO_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_wav2vec2():
    print("Loading Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return processor, model

def extract_embeddings(path, processor, model):
    audio, sr = sf.read(path)

    if sr != AUDIO_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SR)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    inputs = processor(
        audio,
        sampling_rate=AUDIO_SR,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(inputs.input_values.to(DEVICE)).last_hidden_state

    return outputs.squeeze(0).cpu().numpy()

def process_dataset(input_dir, output_dir):
    processor, model = load_wav2vec2()

    metadata_rows = []

    splits = ["train", "test"]
    labels = ["bonafide", "spoof"]

    for split in splits:
        for label in labels:
            in_dir = os.path.join(input_dir, split, label)
            out_dir = os.path.join(output_dir, split, label)

            os.makedirs(out_dir, exist_ok=True)

            print(f"\nExtracting → {split}/{label}")

            for file in tqdm(os.listdir(in_dir)):
                if file.endswith(".wav"):
                    file_path = os.path.join(in_dir, file)

                    emb = extract_embeddings(file_path, processor, model)

                    out_path = os.path.join(
                        out_dir,
                        file.replace(".wav", "_emb.npy")
                    )

                    np.save(out_path, emb)

                    metadata_rows.append([
                        split,
                        label,
                        file,
                        out_path,
                        emb.shape[0]
                    ])

    df = pd.DataFrame(
        metadata_rows,
        columns=["split", "label", "filename", "embedding_path", "timesteps"]
    )

    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print("\n Embedding extraction complete!")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    INPUT_DIR = "data/asvspoof_augmented"
    OUTPUT_DIR = "data/asvspoof_embeddings"

    process_dataset(INPUT_DIR, OUTPUT_DIR)