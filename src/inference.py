import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from src.models.cnn_lstm import CNNLSTMSpoofDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/best_cnn_lstm_model.pth"
WAV2VEC_MODEL = "facebook/wav2vec2-base-960h"

SAMPLE_RATE = 16000
MAX_LEN = 400

def load_models():
    print("Loading Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
    wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL).to(DEVICE)
    wav2vec2.eval()

    print("Loading CNN+LSTM...")
    model = CNNLSTMSpoofDetector().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return processor, wav2vec2, model

def load_audio(path):
    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio

def get_embedding(audio, processor, wav2vec2):
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        emb = wav2vec2(inputs.input_values.to(DEVICE)).last_hidden_state

    T = emb.shape[1]

    if T > MAX_LEN:
        emb = emb[:, :MAX_LEN, :]
    else:
        pad = torch.zeros((1, MAX_LEN - T, emb.shape[2]), device=DEVICE)
        emb = torch.cat([emb, pad], dim=1)

    return emb

def predict(audio_path):
    processor, wav2vec2, model = load_models()

    audio = load_audio(audio_path)
    emb = get_embedding(audio, processor, wav2vec2)

    with torch.no_grad():
        logits = model(emb)
        prob_spoof = torch.sigmoid(logits).item()

    return {
        "bonafide": float(1 - prob_spoof),
        "spoof": float(prob_spoof)
    }

if __name__ == "__main__":
    test_audio = "sample.wav"  
    result = predict(test_audio)

    print("\nPrediction:")
    print(result)