import torch
import torch.nn as nn
from typing import Optional


_ACTIVATIONS = {
    None: nn.Identity,
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _get_activation(name: Optional[str]) -> nn.Module:
    factory = _ACTIVATIONS.get(name, nn.Identity)
    return factory()


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
    def __init__(
        self,
        L: int,
        H: int,
        n_in: int,
        units: int,
        num_layers: int,
        dropout: float,
        dense_activation: Optional[str] = None,
    ):
        super().__init__()
        self.stack = LSTMStack(input_size=n_in, units=units, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(units, H)
        self.out_activation = _get_activation(dense_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stack(x)
        y = self.fc(h)
        y = self.out_activation(y)
        return y.view(y.size(0), -1, 1)


class Discriminator(nn.Module):
    def __init__(
        self,
        L_plus_H: int,
        units: int,
        num_layers: int,
        dropout: float,
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.stack = LSTMStack(input_size=1, units=units, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(units, 1)
        self.out_activation = _get_activation(activation or "sigmoid")

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        h = self.stack(x_concat)
        p = self.out(h)
        return self.out_activation(p)


def build_generator(
    L: int,
    H: int,
    n_in: int = 1,
    units: int = 64,
    dropout: float = 0.0,
    num_layers: int = 1,
    dense_activation: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    return Generator(L, H, n_in, units, num_layers, dropout, dense_activation=dense_activation)


def build_discriminator(
    L: int,
    H: int,
    units: int = 64,
    dropout: float = 0.0,
    num_layers: int = 1,
    activation: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    return Discriminator(L + H, units, num_layers, dropout, activation=activation)


