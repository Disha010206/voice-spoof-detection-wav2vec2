import torch
import torch.nn as nn


class CNNLSTMSpoofDetector(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))

        x = x.transpose(1, 2)

        x, _ = self.lstm(x)

        x = x.mean(dim=1)

        return self.fc(x).squeeze(1)