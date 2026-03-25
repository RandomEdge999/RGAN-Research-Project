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
        noise_dim: int = 0,
    ):
        """Initializes the Generator.

        Args:
            L: Input sequence length (not directly used by architecture but kept for config consistency).
            H: Output horizon (size of the output vector).
            n_in: Number of input features (real data features).
            units: Hidden units in LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            dense_activation: Activation for the final dense layer.
            layer_norm: Whether to use LayerNorm in the LSTM stack.
            noise_dim: Dimension of latent noise vector z concatenated with input
                       at each timestep (RCGAN-style). 0 disables noise input.
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.n_in = n_in
        self.stack = LSTMStack(
            input_size=n_in + noise_dim,
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

    def forward(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """Generates a forecast from input sequence and optional latent noise.

        Args:
            x: Input time-series tensor of shape (batch, seq_len, n_in).
            z: Optional latent noise tensor of shape (batch, seq_len, noise_dim).
               When noise_dim > 0 and z is None, zeros are used (deterministic mode
               for evaluation/metrics).

        Returns:
            Generated sequence tensor of shape (batch, H, 1).
        """
        if self.noise_dim > 0:
            if z is not None:
                x = torch.cat([x, z], dim=-1)
            else:
                # Deterministic mode: zero noise for consistent eval predictions
                zeros = torch.zeros(
                    x.size(0), x.size(1), self.noise_dim,
                    device=x.device, dtype=x.dtype,
                )
                x = torch.cat([x, zeros], dim=-1)
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


class _Chomp1d(nn.Module):
    """Removes right-side padding introduced by causal convolutions."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = max(0, int(chomp_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size].contiguous()


class ResidualTCNBlock(nn.Module):
    """A causal residual Conv1d block for sequence critics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        layer_norm: bool = False,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp1 = _Chomp1d(padding)
        self.norm1 = nn.GroupNorm(1, out_channels) if layer_norm else nn.Identity()
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp2 = _Chomp1d(padding)
        self.norm2 = nn.GroupNorm(1, out_channels) if layer_norm else nn.Identity()
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.out_act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for module in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(self.residual, nn.Conv1d):
            nn.init.xavier_uniform_(self.residual.weight)
            if self.residual.bias is not None:
                nn.init.zeros_(self.residual.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)
        return self.out_act(out + residual)


class TCNDiscriminator(nn.Module):
    """A causal residual TCN critic for WGAN-style sequence discrimination."""

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
        super().__init__()
        del L_plus_H  # Sequence length is handled dynamically by Conv1d blocks.
        kernel_size = 3
        blocks = []
        in_channels = 1
        for block_idx in range(max(1, num_layers)):
            dilation = 2 ** block_idx
            blocks.append(
                ResidualTCNBlock(
                    in_channels=in_channels,
                    out_channels=units,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
            )
            in_channels = units
        self.stack = nn.Sequential(*blocks)
        self.out = nn.Linear(units, 1)
        if use_spectral_norm:
            self.out = nn.utils.spectral_norm(self.out)
        self.out_activation = _get_activation(activation or "sigmoid")
        self._init_weights()

    def _init_weights(self):
        if hasattr(self.out, "weight_orig"):
            nn.init.xavier_uniform_(self.out.weight_orig)
        elif hasattr(self.out, "weight"):
            nn.init.xavier_uniform_(self.out.weight)
        if hasattr(self.out, "bias") and self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        x = x_concat.transpose(1, 2)
        h = self.stack(x)
        last_timestep = h[:, :, -1]
        p = self.out(last_timestep)
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
    noise_dim: int = 0,
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
        noise_dim: Latent noise dimension (0 = deterministic, >0 = stochastic).
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
        noise_dim=noise_dim,
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
    critic_arch: str = "tcn",
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
    critic_arch = (critic_arch or "tcn").lower()
    if critic_arch == "lstm":
        return Discriminator(
            L + H,
            units,
            num_layers,
            dropout,
            activation=activation,
            layer_norm=layer_norm,
            use_spectral_norm=use_spectral_norm,
        )
    if critic_arch == "tcn":
        return TCNDiscriminator(
            L + H,
            units,
            num_layers,
            dropout,
            activation=activation,
            layer_norm=layer_norm,
            use_spectral_norm=use_spectral_norm,
        )
    raise ValueError(f"Unsupported critic_arch='{critic_arch}'. Expected 'lstm' or 'tcn'.")

