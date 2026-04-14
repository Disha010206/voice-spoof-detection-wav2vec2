import gradio as gr
import torch
import numpy as np
import librosa

from src.inference import load_models, get_embedding

processor, wav2vec2, model = load_models()

def predict_spoof(audio):
    if audio is None:
        return {"Error": 1.0}

    sr, data = audio

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != 16000:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)

    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))

    emb = get_embedding(data, processor, wav2vec2)

    with torch.no_grad():
        logits = model(emb)
        prob_spoof = torch.sigmoid(logits).item()

    return {
        "Bonafide (Real Voice)": float(1 - prob_spoof),
        "Spoof (AI Generated)": float(prob_spoof)
    }


app = gr.Interface(
    fn=predict_spoof,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="numpy",
        label="Upload or Record Audio"
    ),
    outputs=gr.Label(num_top_classes=2),
    title="🎙 Voice Spoof Detection",
    description="Detect whether a voice is real or AI-generated using Wav2Vec2 + CNN-BiLSTM."
)


if __name__ == "__main__":
    app.launch(share=True)