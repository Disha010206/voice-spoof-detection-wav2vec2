# Voice Spoof Detection using Wav2Vec2 + CNN-BiLSTM

A deep learning-based system for detecting spoofed (AI-generated) speech using Wav2Vec2 feature extraction and a CNN-BiLSTM classifier.

---

## Overview

With the rise of AI-generated voices and deepfake audio, detecting spoofed speech has become critical for security systems such as voice authentication.

This project builds an **end-to-end voice spoof detection pipeline** using:

* **Wav2Vec2 (Facebook AI)** for feature extraction
* **CNN + BiLSTM** for temporal pattern learning
* **Gradio UI** for real-time inference

---

## Architecture

```text
Audio → Augmentation → Wav2Vec2 → Embeddings → CNN + BiLSTM → Classification
```

### Key Components

* **Feature Extraction**: Pretrained Wav2Vec2 model
* **Temporal Modeling**: Bidirectional LSTM
* **Spatial Feature Learning**: 1D CNN layers
* **Binary Classification**: Real vs Spoof

---

## Project Structure

```
voice-spoof-detection-wav2vec2/
│
├── src/
│   ├── data/
│   │   ├── augment.py
│   │   ├── embeddings.py
│   │
│   ├── models/
│   │   └── cnn_lstm.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── app/
│   └── gradio_app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/your-username/voice-spoof-detection-wav2vec2.git
cd voice-spoof-detection-wav2vec2
pip install -r requirements.txt
```

---

## Dataset

This project uses the **ASVspoof dataset**.

> Dataset is not included due to size.

Expected structure:

```
data/
└── asvspoof_preprocessed/
    ├── train/
    │   ├── bonafide/
    │   └── spoof/
    └── test/
        ├── bonafide/
        └── spoof/
```

---

## Pipeline

### 1. Data Augmentation

```bash
python src/data/augment.py
```

### 2. Feature Extraction (Wav2Vec2)

```bash
python src/data/embeddings.py
```

### 3. Model Training

```bash
python src/train.py
```

### 4. Evaluation

```bash
python src/evaluate.py
```

### 5. Inference

```bash
python src/inference.py
```

Example output:

```json
{
  "bonafide": 0.82,
  "spoof": 0.18
}
```

---

## Demo (Gradio)

```bash
python app/gradio_app.py
```

Upload or record audio to get real-time predictions.

---

## Results

The model was evaluated on a held-out test set:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 92.50% |
| Precision | 92.36% |
| Recall    | 92.67% |
| F1 Score  | 92.51% |

### Confusion Matrix

```
[[277  23]
 [ 22 278]]
```

### Observations

* The model correctly classifies **555 out of 600 samples**
* Balanced precision and recall indicate strong generalization
* Low false positives and false negatives across both classes

---

## Key Features

* Transfer learning using Wav2Vec2
* Sequence modeling with BiLSTM
* End-to-end pipeline from raw audio to prediction
* Real-time inference using Gradio
* Clean and modular code structure

---

## Notes

* Dataset and trained models are not included due to size constraints
* GPU is recommended for training
* Inference works on CPU

---

## Future Improvements

* Transformer-based classifier
* REST API deployment
* Noise robustness improvements
* Real-time streaming inference

---

## 👤 Author

**C Disha**
AIML Student

---

## Support

If you found this useful, consider giving it a star 
