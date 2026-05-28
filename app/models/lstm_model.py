import torch
import torch.nn as nn


class DualLSTMModel(nn.Module):
    """
    Dual-channel LSTM: 20-day (short-term) + 60-day (medium-term) branches.
    forward() returns raw logits — apply torch.sigmoid() for probabilities.

    Keras 포팅 수정:
    - Keras LSTM(dropout=0.3)는 입력→은닉 연결에 dropout 적용
    - PyTorch는 LSTM 앞에 input_dropout을 별도로 붙여서 동일 효과 재현
    - BN → LayerNorm: 시퀀스 데이터에서 배치 크기에 독립적으로 정규화
    """

    def __init__(self, num_features: int, dropout_lstm: float = 0.3, dropout_dense: float = 0.2):
        super().__init__()

        # --- Channel 1: 20-day ---
        self.input_drop_20  = nn.Dropout(dropout_lstm)          # 입력 dropout (Keras dropout=0.3 재현)
        self.lstm_20_1      = nn.LSTM(num_features, 32, batch_first=True)
        self.lstm_20_2      = nn.LSTM(32, 32, batch_first=True)
        self.out_drop_20    = nn.Dropout(dropout_lstm)

        # --- Channel 2: 60-day ---
        self.input_drop_60  = nn.Dropout(dropout_lstm)
        self.lstm_60_1      = nn.LSTM(num_features, 64, batch_first=True)
        self.lstm_60_2      = nn.LSTM(64, 64, batch_first=True)
        self.lstm_60_3      = nn.LSTM(64, 32, batch_first=True)
        self.out_drop_60    = nn.Dropout(dropout_lstm)

        # --- Dense head (32+32=64) ---
        # LayerNorm: 배치 크기와 무관하게 피처 축 정규화 → 시퀀스 모델에 적합
        self.fc1   = nn.Linear(64, 48)
        self.ln1   = nn.LayerNorm(48)
        self.drop1 = nn.Dropout(dropout_dense + 0.1)   # 0.3

        self.fc2   = nn.Linear(48, 16)
        self.ln2   = nn.LayerNorm(16)
        self.drop2 = nn.Dropout(dropout_dense)          # 0.2

        self.out = nn.Linear(16, 1)

    def forward(self, x20: torch.Tensor, x60: torch.Tensor) -> torch.Tensor:
        # Channel 1
        h = self.input_drop_20(x20)             # 입력 dropout
        h, _ = self.lstm_20_1(h)
        h, _ = self.lstm_20_2(h)
        out20 = self.out_drop_20(h[:, -1, :])   # 마지막 타임스텝

        # Channel 2
        h = self.input_drop_60(x60)
        h, _ = self.lstm_60_1(h)
        h, _ = self.lstm_60_2(h)
        h, _ = self.lstm_60_3(h)
        out60 = self.out_drop_60(h[:, -1, :])

        merged = torch.cat([out20, out60], dim=1)   # (B, 64)

        x = self.drop1(torch.relu(self.ln1(self.fc1(merged))))
        x = self.drop2(torch.relu(self.ln2(self.fc2(x))))
        return self.out(x).squeeze(1)   # raw logits
