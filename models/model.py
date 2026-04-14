import torch
import torch.nn as nn
import torch.nn.init as init
from models.layers import ResidualBlock, AttentionLayer

class CANIDS(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        self.attention = AttentionLayer(hidden_dim * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            ResidualBlock(64),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            ResidualBlock(32),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended = self.attention(lstm_out)
        return self.fc(attended)