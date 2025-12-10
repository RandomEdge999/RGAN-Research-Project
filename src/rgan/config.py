from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for the RGAN model architecture.

    Attributes:
        L: Input sequence length (lookback).
        H: Output sequence length (horizon).
        n_features: Number of input features.
        units_g: Number of hidden units in the Generator LSTM.
        units_d: Number of hidden units in the Discriminator LSTM.
        g_layers: Number of stacked LSTM layers in the Generator.
        d_layers: Number of stacked LSTM layers in the Discriminator.
        dropout: Dropout probability (0.0 to 1.0).
        g_dense_activation: Activation function for Generator's dense output layer (e.g., 'relu', 'tanh').
        d_activation: Activation function for Discriminator's output (e.g., 'sigmoid').
        use_spectral_norm: Whether to apply spectral normalization to the Discriminator.
    """
    L: int
    H: int
    n_features: int = 1
    units_g: int = 64
    units_d: int = 64
    g_layers: int = 1
    d_layers: int = 1
    dropout: float = 0.0
    g_dense_activation: Optional[str] = None
    d_activation: str = "sigmoid"
    use_spectral_norm: bool = False

@dataclass
class TrainConfig:
    """Configuration for the training process.

    Attributes:
        epochs: Total number of training epochs.
        batch_size: Batch size for training.
        lr_g: Learning rate for the Generator.
        lr_d: Learning rate for the Discriminator.
        lambda_reg: Regularization strength (for supervised loss).
        gan_variant: GAN loss variant ('standard' or 'wgan-gp').
        d_steps: Number of Discriminator updates per Generator update.
        g_steps: Number of Generator updates per Discriminator update.
        wgan_gp_lambda: Gradient penalty coefficient for WGAN-GP.
        label_smooth: Label smoothing factor (for standard GAN).
        grad_clip: Maximum gradient norm for clipping.
        patience: Early stopping patience.
        device: Device to train on ('cpu' or 'cuda').
        num_workers: Number of DataLoader workers.
        seed: Random seed for reproducibility.
        
        # Model Hyperparameters (required for LSTM and convenience)
        L: Input sequence length (lookback).
        H: Output sequence length (horizon).
        units_g: Number of hidden units in the Generator/LSTM.
        units_d: Number of hidden units in the Discriminator.
        g_layers: Number of stacked LSTM layers in the Generator.
        d_layers: Number of stacked LSTM layers in the Discriminator.
        dropout: Dropout probability.
        g_dense_activation: Activation for Generator's dense layer.
        d_activation: Activation for Discriminator.
    """
    epochs: int = 100
    batch_size: int = 64
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    lambda_reg: float = 0.1
    gan_variant: str = "standard"
    d_steps: int = 1
    g_steps: int = 1
    wgan_gp_lambda: float = 10.0
    label_smooth: float = 0.9
    grad_clip: float = 5.0
    patience: int = 10
    device: str = "cpu"
    num_workers: int = 0
    seed: int = 42
    
    # Model Params
    L: int = 24
    H: int = 12
    units_g: int = 64
    units_d: int = 64
    g_layers: int = 1
    d_layers: int = 1
    dropout: float = 0.0
    g_dense_activation: Optional[str] = None
    d_activation: str = "sigmoid"
    
    # Advanced Training Options
    supervised_warmup_epochs: int = 0
    lambda_reg_start: Optional[float] = None
    lambda_reg_end: Optional[float] = None
    lambda_reg_warmup_epochs: int = 1
    adv_weight: float = 1.0
    instance_noise_std: float = 0.0
    instance_noise_decay: float = 0.95
    ema_decay: float = 0.0
    use_logits: bool = False
    track_discriminator_outputs: bool = True
    amp: bool = True
    strict_device: bool = False
    wgan_clip_value: float = 0.01
    
    def __post_init__(self):
        if self.lambda_reg_start is None:
            self.lambda_reg_start = self.lambda_reg
        if self.lambda_reg_end is None:
            self.lambda_reg_end = self.lambda_reg

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        csv_path: Path to the input CSV file.
        target_col: Name of the target column.
        time_col: Name of the time column (optional).
        resample: Resampling frequency (e.g., '1H').
        agg: Aggregation method for resampling (e.g., 'mean', 'last').
        val_fraction: Fraction of data to use for validation.
        test_fraction: Fraction of data to use for testing (if not using split index).
        train_ratio: Ratio of data to use for training (legacy support).
    """
    csv_path: str
    target_col: str = "auto"
    time_col: str = "auto"
    resample: str = ""
    agg: str = "last"
    val_fraction: float = 0.1
    train_ratio: float = 0.8
