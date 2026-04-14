import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.cnn_lstm import CNNLSTMSpoofDetector


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMB_DIR = "data/asvspoof_embeddings"
MODEL_SAVE_PATH = "models/best_cnn_lstm_model.pth"

MAX_LEN = 400
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4


class EmbeddingDataset(Dataset):
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


def load_data():
    metadata = pd.read_csv(os.path.join(EMB_DIR, "metadata.csv"))

    metadata["label"] = metadata["label"].map({"bonafide": 0, "spoof": 1})

    train_df, val_df = train_test_split(
        metadata,
        test_size=0.2,
        stratify=metadata["label"],
        random_state=42
    )

    train_loader = DataLoader(
        EmbeddingDataset(train_df, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        EmbeddingDataset(val_df, MAX_LEN),
        batch_size=BATCH_SIZE
    )

    return train_loader, val_loader


def train():
    train_loader, val_loader = load_data()

    model = CNNLSTMSpoofDetector().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # TRAIN
        model.train()
        train_correct, train_total = 0, 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(xb)

            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            train_correct += (preds == yb.int()).sum().item()
            train_total += len(yb)

        train_acc = train_correct / train_total

        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                outputs = model(xb)
                preds = (torch.sigmoid(outputs) > 0.5).int()

                val_correct += (preds == yb.int()).sum().item()
                val_total += len(yb)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Saved best model!")


if __name__ == "__main__":
    train()