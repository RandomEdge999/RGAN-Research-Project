"""TimeGAN: Time-series Generative Adversarial Network.

From "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019).
Four-component system:
  1. Embedder:   real data -> latent space
  2. Recovery:   latent space -> data space (autoencoder with embedder)
  3. Generator:  noise -> latent space (generates synthetic latent sequences)
  4. Discriminator: latent sequences -> real/fake classification

Three training phases:
  Phase 1: Autoencoder (embedder + recovery) — reconstruction loss
  Phase 2: Supervised (generator) — next-step prediction in latent space
  Phase 3: Joint (all four) — adversarial + supervised + reconstruction

This is the standard GAN baseline for time-series generation. Used to compare
against RGAN's synthetic data quality and augmentation effectiveness.

Interface for generation:
    generate(n_samples, seq_len) -> (n_samples, seq_len, 1) synthetic sequences
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .logging_utils import get_console, epoch_progress, update_epoch, log_info, log_step, log_debug
from .metrics import error_stats


# ── Component Networks ──────────────────────────────────────────────

class _RNNBlock(nn.Module):
    """Shared GRU-based building block for all TimeGAN components."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 1, activation: str = "sigmoid"):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid() if activation == "sigmoid" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.activation(self.fc(h))


class TimeGAN(nn.Module):
    """Full TimeGAN model with all four components."""

    def __init__(
        self,
        feature_dim: int = 1,
        hidden_dim: int = 24,
        latent_dim: int = 24,
        n_layers: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Embedder: data space -> latent space
        self.embedder = _RNNBlock(feature_dim, hidden_dim, latent_dim,
                                  n_layers, activation="sigmoid")
        # Recovery: latent space -> data space
        self.recovery = _RNNBlock(latent_dim, hidden_dim, feature_dim,
                                  n_layers, activation="sigmoid")
        # Generator: noise -> latent space
        self.generator = _RNNBlock(feature_dim, hidden_dim, latent_dim,
                                   n_layers, activation="sigmoid")
        # Supervisor: latent -> next latent (supervised loss)
        self.supervisor = _RNNBlock(latent_dim, hidden_dim, latent_dim,
                                    n_layers, activation="sigmoid")
        # Discriminator: latent -> real/fake
        self.discriminator = _RNNBlock(latent_dim, hidden_dim, 1,
                                       n_layers, activation="none")

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)

    def _recover(self, h: torch.Tensor) -> torch.Tensor:
        return self.recovery(h)

    def _generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def _supervise(self, h: torch.Tensor) -> torch.Tensor:
        return self.supervisor(h)

    def _discriminate(self, h: torch.Tensor) -> torch.Tensor:
        return self.discriminator(h)

    def forward_autoencoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed then recover. Returns (recovered, embedding)."""
        h = self._embed(x)
        x_hat = self._recover(h)
        return x_hat, h

    def forward_supervisor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed then supervise. Returns (h_hat_supervise, h_real)."""
        h = self._embed(x)
        h_hat = self._supervise(h)
        return h_hat, h

    def forward_generator(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic latent, supervise it, recover to data space.
        Returns (x_hat, h_hat_supervise, h_hat)."""
        h_hat = self._generate(z)
        h_hat_s = self._supervise(h_hat)
        x_hat = self._recover(h_hat_s)
        return x_hat, h_hat_s, h_hat

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int,
                 device: torch.device | str = "cpu") -> np.ndarray:
        """Generate synthetic time series sequences.

        Args:
            n_samples: Number of sequences to generate.
            seq_len: Length of each sequence.
            device: Device to run on.

        Returns:
            Numpy array of shape (n_samples, seq_len, feature_dim).
        """
        self.eval()
        device = torch.device(device)
        z = torch.randn(n_samples, seq_len, self.feature_dim, device=device)
        h_hat = self._generate(z)
        h_hat_s = self._supervise(h_hat)
        x_hat = self._recover(h_hat_s)
        return x_hat.cpu().numpy()


# ── Training ────────────────────────────────────────────────────────

def _random_noise(batch_size: int, seq_len: int, dim: int,
                  device: torch.device) -> torch.Tensor:
    """Generate random noise for the generator."""
    return torch.randn(batch_size, seq_len, dim, device=device)


def train_timegan(
    real_data: np.ndarray,
    hidden_dim: int = 24,
    latent_dim: int = 24,
    n_layers: int = 1,
    epochs_ae: int = 50,
    epochs_sup: int = 50,
    epochs_joint: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "auto",
    gamma: float = 1.0,
) -> Dict[str, Any]:
    """Train TimeGAN on real time series data.

    Args:
        real_data: Array of shape (n_samples, seq_len, n_features).
                   For univariate: (n_samples, seq_len, 1).
        hidden_dim: GRU hidden dimension.
        latent_dim: Latent embedding dimension.
        n_layers: Number of GRU layers.
        epochs_ae: Epochs for autoencoder phase.
        epochs_sup: Epochs for supervised phase.
        epochs_joint: Epochs for joint adversarial phase.
        batch_size: Training batch size.
        lr: Learning rate.
        device: "auto", "cpu", "cuda", or "mps".
        gamma: Weight for supervised loss in joint phase.

    Returns:
        Dict with model, history, and generated samples.
    """
    console = get_console()

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    n_samples, seq_len, feature_dim = real_data.shape
    log_info(f"[TimeGAN] Training on {n_samples} sequences, seq_len={seq_len}, features={feature_dim}")

    # Normalize to [0, 1] for sigmoid activations
    data_min = real_data.min(axis=(0, 1), keepdims=True)
    data_max = real_data.max(axis=(0, 1), keepdims=True)
    data_range = data_max - data_min + 1e-7
    real_normed = (real_data - data_min) / data_range

    model = TimeGAN(feature_dim, hidden_dim, latent_dim, n_layers).to(dev)

    # Optimizers for different phases
    opt_ae = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=lr,
    )
    opt_sup = torch.optim.Adam(model.supervisor.parameters(), lr=lr)
    opt_g = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()),
        lr=lr,
    )
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    history = {"ae_loss": [], "sup_loss": [], "g_loss": [], "d_loss": []}
    steps_per_epoch = max(1, n_samples // batch_size)

    def _get_batch() -> torch.Tensor:
        idx = np.random.randint(0, n_samples, size=batch_size)
        return torch.from_numpy(real_normed[idx].astype(np.float32)).to(dev)

    # ── Phase 1: Autoencoder ──────────────────────────────────────
    log_info("[TimeGAN] Phase 1: Autoencoder training")
    with epoch_progress(epochs_ae, description="TimeGAN-AE") as (progress, task_id):
        for epoch in range(1, epochs_ae + 1):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                x = _get_batch()
                x_hat, _ = model.forward_autoencoder(x)
                loss = 10.0 * torch.sqrt(mse(x_hat, x))
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()
                epoch_loss += loss.item()
            avg = epoch_loss / steps_per_epoch
            history["ae_loss"].append(avg)
            update_epoch(progress, task_id, epoch, epochs_ae, {"AE Loss": avg})

    # ── Phase 2: Supervised ───────────────────────────────────────
    log_info("[TimeGAN] Phase 2: Supervised training")
    with epoch_progress(epochs_sup, description="TimeGAN-Sup") as (progress, task_id):
        for epoch in range(1, epochs_sup + 1):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                x = _get_batch()
                h_hat, h_real = model.forward_supervisor(x)
                # Supervised: predict next step from current
                loss = mse(h_hat[:, :-1, :], h_real[:, 1:, :])
                opt_sup.zero_grad()
                loss.backward()
                opt_sup.step()
                epoch_loss += loss.item()
            avg = epoch_loss / steps_per_epoch
            history["sup_loss"].append(avg)
            update_epoch(progress, task_id, epoch, epochs_sup, {"Sup Loss": avg})

    # ── Phase 3: Joint training ───────────────────────────────────
    log_info("[TimeGAN] Phase 3: Joint adversarial training")
    total_joint = epochs_joint
    with epoch_progress(total_joint, description="TimeGAN-Joint") as (progress, task_id):
        for epoch in range(1, total_joint + 1):
            g_loss_epoch = 0.0
            d_loss_epoch = 0.0

            for _ in range(steps_per_epoch):
                # ── Train Generator (2 steps per D step) ──
                for _ in range(2):
                    x = _get_batch()
                    z = _random_noise(batch_size, seq_len, feature_dim, dev)

                    # Forward passes
                    h_real = model._embed(x)
                    x_fake, h_fake_s, h_fake = model.forward_generator(z)

                    # Adversarial loss
                    d_fake = model._discriminate(h_fake)
                    g_adv = bce(d_fake, torch.ones_like(d_fake))

                    # Supervised loss (latent)
                    g_sup = mse(h_fake_s[:, :-1, :], h_fake[:, 1:, :].detach())

                    # Moment matching loss (mean + variance)
                    g_mean = torch.abs(h_real.mean(dim=0) - h_fake.mean(dim=0)).mean()
                    g_var = torch.abs(h_real.var(dim=0) - h_fake.var(dim=0)).mean()

                    g_loss = g_adv + gamma * g_sup + 100.0 * (g_mean + g_var)

                    opt_g.zero_grad()
                    g_loss.backward()
                    opt_g.step()

                    g_loss_epoch += g_loss.item()

                # ── Train Discriminator ──
                x = _get_batch()
                z = _random_noise(batch_size, seq_len, feature_dim, dev)

                h_real = model._embed(x).detach()
                h_fake = model._generate(z).detach()

                d_real = model._discriminate(h_real)
                d_fake = model._discriminate(h_fake)

                d_loss = bce(d_real, torch.ones_like(d_real)) + \
                         bce(d_fake, torch.zeros_like(d_fake))

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                d_loss_epoch += d_loss.item()

            g_avg = g_loss_epoch / (steps_per_epoch * 2)
            d_avg = d_loss_epoch / steps_per_epoch
            history["g_loss"].append(g_avg)
            history["d_loss"].append(d_avg)
            update_epoch(progress, task_id, epoch, total_joint,
                         {"G": g_avg, "D": d_avg})

    log_info("[TimeGAN] Training complete.")

    # Generate a batch of synthetic data for evaluation
    synthetic = model.generate(n_samples, seq_len, device=dev)
    # Denormalize
    synthetic = synthetic * data_range + data_min

    return {
        "model": model,
        "history": history,
        "synthetic_data": synthetic,
        "data_min": data_min,
        "data_max": data_max,
    }
