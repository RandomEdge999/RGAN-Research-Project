"""Autoformer: Decomposition Transformers with Auto-Correlation.

From "Autoformer: Decomposition Transformers with Auto-Correlation for
Long-Term Series Forecasting" (Wu et al., NeurIPS 2021). Uses series
decomposition blocks and auto-correlation attention instead of standard
self-attention.

Interface:
    input:  (batch, L, 1)
    output: (batch, H, 1)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch
from .metrics import error_stats


# ── Series Decomposition ────────────────────────────────────────────

class MovingAvg(nn.Module):
    """Moving average block for series decomposition."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        # AvgPool1d expects (batch, channels, seq_len)
        out = self.avg(x.unsqueeze(1)).squeeze(1)
        return out


class SeriesDecomp(nn.Module):
    """Decompose time series into trend and seasonal components."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (batch, seq_len)
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ── Auto-Correlation Mechanism ──────────────────────────────────────

class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism: period-based attention.

    Instead of point-wise dot-product attention, this computes correlations
    via FFT, selects top-k periods, and aggregates by rolling.
    """

    def __init__(self, d_model: int, n_heads: int, top_k: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, L, _ = q.shape
        H = self.n_heads
        D = self.d_head

        Q = self.q_proj(q).reshape(B, L, H, D).permute(0, 2, 1, 3)  # (B, H, L, D)
        K = self.k_proj(k).reshape(B, L, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(v).reshape(B, L, H, D).permute(0, 2, 1, 3)

        # Auto-correlation via FFT
        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)
        corr = torch.fft.irfft(Q_fft * K_fft.conj(), n=L, dim=2)  # (B, H, L, D)

        # Average correlation over d_head dimension
        corr_mean = corr.mean(dim=-1)  # (B, H, L)

        # Top-k time delays
        top_k = min(self.top_k, L)
        weights, delays = torch.topk(corr_mean, top_k, dim=-1)  # (B, H, top_k)
        weights = torch.softmax(weights, dim=-1)

        # Aggregate V by rolling (vectorized — no Python loops over B/H)
        out = torch.zeros_like(V)
        # Build index tensor: (L,) base indices
        idx_base = torch.arange(L, device=V.device)  # (L,)
        for i in range(top_k):
            delay = delays[:, :, i]  # (B, H)
            w = weights[:, :, i].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            # Compute rolled indices for all B,H simultaneously
            # delay: (B, H) -> (B, H, 1), idx_base: (L,) -> broadcast to (B, H, L)
            rolled_idx = (idx_base.view(1, 1, L) - delay.unsqueeze(-1)) % L  # (B, H, L)
            # Gather V along the time dimension
            rolled_idx_exp = rolled_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, H, L, D)
            rolled_V = torch.gather(V, 2, rolled_idx_exp)  # (B, H, L, D)
            out = out + w * rolled_V

        out = out.permute(0, 2, 1, 3).reshape(B, L, H * D)  # (B, L, d_model)
        return self.out_proj(out)


# ── Autoformer Encoder Layer ────────────────────────────────────────

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, decomp_kernel: int = 25):
        super().__init__()
        self.auto_corr = AutoCorrelation(d_model, n_heads)
        self.decomp1 = SeriesDecomp(decomp_kernel)
        self.decomp2 = SeriesDecomp(decomp_kernel)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        B, L, D = x.shape

        # Auto-correlation + decomposition
        attn_out = self.auto_corr(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # Decompose: keep seasonal, discard trend
        seasonal_flat = x.reshape(B * L, D)
        # Apply decomp on each feature dim across the sequence
        x_2d = x.mean(dim=-1)  # (B, L) for decomposition
        seasonal, _ = self.decomp1(x_2d)
        # Scale x by seasonal ratio
        scale = (seasonal / (x_2d + 1e-8)).unsqueeze(-1)
        x = x * (1 + scale * 0.1)  # soft decomposition influence

        # Feed-forward + decomposition
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)

        return x


# ── Autoformer Model ────────────────────────────────────────────────

class Autoformer(nn.Module):
    """Autoformer for univariate time series forecasting."""

    def __init__(
        self,
        L: int,
        H: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        decomp_kernel: int = 25,
    ):
        super().__init__()
        self.L = L
        self.H = H

        # Value embedding
        self.value_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, L, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, dropout, decomp_kernel)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Projection head
        self.head = nn.Sequential(
            nn.Linear(d_model * L, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, H),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, n_features)
        B = x.size(0)
        x_1d = x[:, :, 0]  # (batch, L) — target column

        # Instance normalization
        mean = x_1d.mean(dim=-1, keepdim=True)
        std = x_1d.std(dim=-1, keepdim=True) + 1e-5
        x_norm = (x_1d - mean) / std

        # Embed
        z = self.value_embed(x_norm.unsqueeze(-1))  # (B, L, d_model)
        z = z + self.pos_embed
        z = self.dropout(z)

        # Encoder layers
        for layer in self.layers:
            z = layer(z)

        z = self.norm(z)

        # Flatten and project
        z = z.reshape(B, -1)  # (B, L * d_model)
        y = self.head(z)  # (B, H)

        # De-normalize
        y = y * std + mean

        return y.unsqueeze(-1)  # (B, H, 1)


# ── Training ────────────────────────────────────────────────────────

def _predict_in_batches(
    model: nn.Module, data: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    if len(data) == 0:
        return np.empty((0, model.H, 1), dtype=np.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            xb = torch.from_numpy(data[start:start + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_autoformer(
    config: TrainConfig,
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "Autoformer",
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
    decomp_kernel: int = 25,
) -> Dict[str, Any]:
    """Train an Autoformer model."""
    console = get_console()

    preferred = config.device
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)

    model = Autoformer(
        L=config.L, H=config.H,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, dropout=dropout, decomp_kernel=decomp_kernel,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr_g, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    hist = {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": []}
    best_val, best_state = float("inf"), None
    bad_epochs = 0
    batch_size = max(1, config.batch_size)

    Xtr, Ytr = data_splits["Xtr"], data_splits["Ytr"]
    Xte, Yte = data_splits["Xte"], data_splits["Yte"]
    Xval, Yval = data_splits["Xval"], data_splits["Yval"]

    N = len(Xtr)
    steps = max(1, int(np.ceil(N / batch_size)))

    with epoch_progress(config.epochs, description=tag) as (progress, task_id):
        for epoch in range(1, config.epochs + 1):
            perm = np.random.permutation(N)
            model.train()
            for s in range(steps):
                b0 = s * batch_size
                b1 = min((s + 1) * batch_size, N)
                idx = perm[b0:b1]
                Xb = torch.from_numpy(Xtr[idx]).to(device)
                Yb = torch.from_numpy(Ytr[idx]).to(device)
                opt.zero_grad()
                pred = model(Xb)
                loss = loss_fn(pred, Yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()

            eval_bs = max(1, min(batch_size, 1024))
            tr = _predict_in_batches(model, Xtr, device, eval_bs)
            te = _predict_in_batches(model, Xte, device, eval_bs)
            va = _predict_in_batches(model, Xval, device, eval_bs)

            tr_rmse = float(np.sqrt(np.mean((tr.reshape(-1) - Ytr.reshape(-1)) ** 2)))
            te_rmse = float(np.sqrt(np.mean((te.reshape(-1) - Yte.reshape(-1)) ** 2)))
            va_rmse = float(np.sqrt(np.mean((va.reshape(-1) - Yval.reshape(-1)) ** 2)))

            hist["epoch"].append(epoch)
            hist["train_rmse"].append(tr_rmse)
            hist["test_rmse"].append(te_rmse)
            hist["val_rmse"].append(va_rmse)

            update_epoch(progress, task_id, epoch, config.epochs,
                         {"Train": tr_rmse, "Val": va_rmse, "Test": te_rmse})

            if va_rmse < best_val - 1e-7:
                best_val = va_rmse
                best_state = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= config.patience:
                    console.log(f"[{tag}] Early stopping at epoch {epoch}.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_bs = max(1, min(config.batch_size, 1024))
    tr = _predict_in_batches(model, Xtr, device, eval_bs)
    te = _predict_in_batches(model, Xte, device, eval_bs)
    train_stats = error_stats(Ytr.reshape(-1), tr.reshape(-1))
    test_stats = error_stats(Yte.reshape(-1), te.reshape(-1))

    return {
        "model": model,
        "history": hist,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "pred_train": tr,
        "pred_test": te,
    }
