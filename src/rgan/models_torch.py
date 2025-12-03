import torch
import torch.nn as nn
from typing import Optional, Dict, Any

_ACTIVATIONS: Dict[Optional[str], Any] = {
    None: nn.Identity,
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _get_activation(name: Optional[str]) -> nn.Module:
    """Retrieves a PyTorch activation module by name.

    Args:
        name: The name of the activation function (e.g., 'relu', 'tanh').
              If None or 'linear', returns nn.Identity().

    Returns:
        An instantiated PyTorch module (e.g., nn.ReLU()).
    """
    factory = _ACTIVATIONS.get(name, nn.Identity)
    return factory()


class LSTMStack(nn.Module):
    """A stacked LSTM module with optional Layer Normalization.

    This module encapsulates a multi-layer LSTM followed by an optional
    LayerNorm step, commonly used in deep sequence models.

    Attributes:
        lstm (nn.LSTM): The core LSTM module.
        ln (nn.Module): LayerNorm or Identity module.
    """

    def __init__(
        self,
        input_size: int,
        units: int,
        num_layers: int,
        dropout: float,
        layer_norm: bool = False,
    ):
        """Initializes the LSTMStack.

        Args:
            input_size: Number of expected features in the input `x`.
            units: Number of features in the hidden state `h`.
            num_layers: Number of recurrent layers.
            dropout: If non-zero, introduces a Dropout layer on the outputs of each
                     LSTM layer except the last layer.
            layer_norm: If True, applies LayerNorm to the output.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=units,
            num_layers=max(1, num_layers),
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(units) if layer_norm else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using orthogonal initialization for better convergence."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Initialize forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Output tensor of shape (batch, units).
            Returns only the hidden state of the last time step.
        """
        out, _ = self.lstm(x)
        # Select the last time step's output
        if out.dim() == 3:
            out = out[:, -1, :]
        return self.ln(out)


class Generator(nn.Module):
    """The RGAN Generator Network.

    Transforms a noise sequence (and optional covariates) into a synthetic
    time series sequence.

    Attributes:
        stack (LSTMStack): The recurrent component processing the input.
        fc (nn.Linear): The final dense layer mapping hidden state to output.
        out_activation (nn.Module): The final activation function.
    """

    def __init__(
        self,
        L: int,
        H: int,
        n_in: int,
        units: int,
        num_layers: int,
        dropout: float,
        dense_activation: Optional[str] = None,
        layer_norm: bool = False,
    ):
        """Initializes the Generator.

        Args:
            L: Input sequence length (not directly used by architecture but kept for config consistency).
            H: Output horizon (size of the output vector).
            n_in: Number of input features (noise dimension + covariates).
            units: Hidden units in LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            dense_activation: Activation for the final dense layer.
            layer_norm: Whether to use LayerNorm in the LSTM stack.
        """
        super().__init__()
        self.stack = LSTMStack(
            input_size=n_in,
            units=units,
            num_layers=num_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.fc = nn.Linear(units, H)
        self.out_activation = _get_activation(dense_activation)
        self._init_weights()

    def _init_weights(self):
        """Initializes the linear layer weights."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates a sequence from input noise.

        Args:
            x: Input noise tensor of shape (batch, seq_len, n_in).

        Returns:
            Generated sequence tensor of shape (batch, H, 1).
        """
        h = self.stack(x)
        y = self.fc(h)
        y = self.out_activation(y)
        # Reshape to (batch, H, 1) to match expected time-series format
        return y.view(y.size(0), -1, 1)


class Discriminator(nn.Module):
    """The RGAN Discriminator Network.

    Classifies a time series sequence as real or fake.

    Attributes:
        stack (LSTMStack): The recurrent component processing the input.
        out (nn.Module): The final linear layer (optionally spectral normalized).
        out_activation (nn.Module): The final activation function (e.g., Sigmoid).
    """

    def __init__(
        self,
        L_plus_H: int,
        units: int,
        num_layers: int,
        dropout: float,
        activation: Optional[str] = None,
        layer_norm: bool = False,
        use_spectral_norm: bool = False,
    ):
        """Initializes the Discriminator.

        Args:
            L_plus_H: Total length of the input sequence (Lookback + Horizon).
                      (Note: LSTM handles variable length, but this is conceptually the input dim if flattened).
            units: Hidden units in LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            activation: Output activation function (default: 'sigmoid').
            layer_norm: Whether to use LayerNorm in the LSTM stack.
            use_spectral_norm: Whether to apply spectral normalization to the final layer.
        """
        super().__init__()
        # Input size is 1 because we process a univariate time series sequence
        self.stack = LSTMStack(
            input_size=1,
            units=units,
            num_layers=num_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.out = nn.Linear(units, 1)
        if use_spectral_norm:
            self.out = nn.utils.spectral_norm(self.out)
        self.out_activation = _get_activation(activation or "sigmoid")
        self._init_weights()

    def _init_weights(self):
        """Initializes the linear layer weights."""
        # Spectral norm renames 'weight' to 'weight_orig'.
        # We should initialize the underlying parameter.
        if hasattr(self.out, 'weight_orig'):
             nn.init.xavier_uniform_(self.out.weight_orig)
        elif hasattr(self.out, 'weight'):
             nn.init.xavier_uniform_(self.out.weight)
        
        if hasattr(self.out, 'bias') and self.out.bias is not None:
             nn.init.zeros_(self.out.bias)

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        """Classifies the input sequence.

        Args:
            x_concat: Concatenated input sequence (Real/Fake) of shape (batch, L+H, 1).

        Returns:
            Classification score tensor of shape (batch, 1).
        """
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
    layer_norm: bool = False,
    **kwargs,
) -> Generator:
    """Factory function to build a Generator instance.

    Args:
        L: Input sequence length.
        H: Output horizon.
        n_in: Number of input features.
        units: Hidden units.
        dropout: Dropout rate.
        num_layers: Number of layers.
        dense_activation: Output activation.
        layer_norm: Use layer normalization.
        **kwargs: Ignored extra arguments.

    Returns:
        An instantiated Generator module.
    """
    return Generator(
        L,
        H,
        n_in,
        units,
        num_layers,
        dropout,
        dense_activation=dense_activation,
        layer_norm=layer_norm,
    )


def build_discriminator(
    L: int,
    H: int,
    units: int = 64,
    dropout: float = 0.0,
    num_layers: int = 1,
    activation: Optional[str] = None,
    layer_norm: bool = False,
    use_spectral_norm: bool = False,
    **kwargs,
) -> Discriminator:
    """Factory function to build a Discriminator instance.

    Args:
        L: Input sequence length.
        H: Output horizon.
        units: Hidden units.
        dropout: Dropout rate.
        num_layers: Number of layers.
        activation: Output activation.
        layer_norm: Use layer normalization.
        use_spectral_norm: Use spectral normalization.
        **kwargs: Ignored extra arguments.

    Returns:
        An instantiated Discriminator module.
    """
    return Discriminator(
        L + H,
        units,
        num_layers,
        dropout,
        activation=activation,
        layer_norm=layer_norm,
        use_spectral_norm=use_spectral_norm,
    )


