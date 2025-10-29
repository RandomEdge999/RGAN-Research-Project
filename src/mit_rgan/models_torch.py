import torch
import torch.nn as nn


class LSTMStack(nn.Module):
    def __init__(self, input_size: int, units: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=units,
            num_layers=max(1, num_layers),
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        if out.dim() == 3:
            out = out[:, -1, :]
        return out


class Generator(nn.Module):
    def __init__(self, L: int, H: int, n_in: int, units: int, num_layers: int, dropout: float):
        super().__init__()
        self.stack = LSTMStack(input_size=n_in, units=units, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(units, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stack(x)
        y = self.fc(h)
        return y.view(y.size(0), -1, 1)


class Discriminator(nn.Module):
    def __init__(self, L_plus_H: int, units: int, num_layers: int, dropout: float):
        super().__init__()
        self.stack = LSTMStack(input_size=1, units=units, num_layers=num_layers, dropout=dropout)
        self.out = nn.Sequential(nn.Linear(units, 1), nn.Sigmoid())

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        h = self.stack(x_concat)
        p = self.out(h)
        return p


def build_generator(L: int, H: int, n_in: int = 1, units: int = 64, dropout: float = 0.0, num_layers: int = 1, **kwargs) -> nn.Module:
    return Generator(L, H, n_in, units, num_layers, dropout)


def build_discriminator(L: int, H: int, units: int = 64, dropout: float = 0.0, num_layers: int = 1, **kwargs) -> nn.Module:
    return Discriminator(L + H, units, num_layers, dropout)


