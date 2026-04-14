import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torch

from src.models.cnn_lstm import CNNLSTMSpoofDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMB_DIR = "data/asvspoof_embeddings"
MODEL_PATH = "models/best_cnn_lstm_model.pth"

MAX_LEN = 400
BATCH_SIZE = 32

class TestEmbeddingDataset(Dataset):
    def __init__(self, df, max_len=400):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = np.load(row["embedding_path"]) 

        if emb.shape[0] > self.max_len:
            emb = emb[:self.max_len, :]
        else:
            pad_len = self.max_len - emb.shape[0]
            emb = np.vstack([emb, np.zeros((pad_len, emb.shape[1]))])

        return (
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(row["label"], dtype=torch.float32)
        )

def load_test_data():
    metadata = pd.read_csv(os.path.join(EMB_DIR, "metadata.csv"))

    metadata["label"] = metadata["label"].map({"bonafide": 0, "spoof": 1})

    test_df = metadata[metadata["split"] == "test"]

    test_loader = DataLoader(
        TestEmbeddingDataset(test_df, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return test_loader

def evaluate():
    test_loader = load_test_data()

    model = CNNLSTMSpoofDetector().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    labels, preds = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits = model(xb)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).int()

            labels.extend(yb.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())

    labels = np.array(labels).astype(int)
    preds = np.array(preds).astype(int)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    print("\nTEST RESULTS ")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)

if __name__ == "__main__":
    evaluate()