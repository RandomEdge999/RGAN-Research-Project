"""PatchTST: A Time Series is Worth 64 Words.

From "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
(Nie et al., ICLR 2023). Patches the input time series into sub-sequences,
applies a channel-independent Transformer encoder, and projects to the forecast.

Interface:
    input:  (batch, L, 1)
    output: (batch, H, 1)
"""

from __future__ import annotations

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch, log_info, log_step, log_debug
from .metrics import error_stats


class PatchEmbedding(nn.Module):
    """Split time series into patches and project to d_model."""

    def __init__(self, L: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Number of patches
        self.n_patches = (L - patch_len) // stride + 1
        # Linear projection from patch to d_model
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L)
        # Unfold into patches: (batch, n_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Project: (batch, n_patches, d_model)
        return self.proj(patches)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patch tokens."""

    def __init__(self, n_patches: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


class PatchTST(nn.Module):
    """PatchTST for univariate time series forecasting."""

    def __init__(
        self,
        L: int,
        H: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.L = L
        self.H = H

        # Patch embedding
        self.patch_embed = PatchEmbedding(L, patch_len, stride, d_model)
        n_patches = self.patch_embed.n_patches

        # Positional encoding
        self.pos_enc = PositionalEncoding(n_patches, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Flatten all patch representations and project to H
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * n_patches),
            nn.Linear(d_model * n_patches, H),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, 1)
        x_1d = x[:, :, 0]                                 # (batch, L) — target column

        # Instance normalization (RevIN-lite)
        mean = x_1d.mean(dim=-1, keepdim=True)
        std = x_1d.std(dim=-1, keepdim=True) + 1e-5
        x_norm = (x_1d - mean) / std

        # Patch + position embed
        z = self.patch_embed(x_norm)                    # (batch, n_patches, d_model)
        z = self.pos_enc(z)
        z = self.dropout(z)

        # Transformer encoder
        z = self.encoder(z)                             # (batch, n_patches, d_model)

        # Flatten and project
        z = z.reshape(z.size(0), -1)                    # (batch, n_patches * d_model)
        y = self.head(z)                                # (batch, H)

        # De-normalize
        y = y * std + mean

        return y.unsqueeze(-1)                          # (batch, H, 1)


def _predict_in_batches(
    model: nn.Module, data: np.ndarray, device: torch.device, batch_size: int,
    use_amp: bool = False,
) -> np.ndarray:
    if len(data) == 0:
        return np.empty((0, model.H, 1), dtype=np.float32)
    preds = []
    model.eval()
    from .rgan_torch import AMP as _AMP
    with torch.no_grad(), _AMP.autocast(device.type, enabled=use_amp):
        for start in range(0, len(data), batch_size):
            xb = torch.from_numpy(data[start:start + batch_size]).to(device)
            preds.append(model(xb).float().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _make_loader(X, Y, batch_size, shuffle=True, device=None):
    """Create a DataLoader with pin_memory when using CUDA."""
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    pin = (device is not None and device.type == 'cuda')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin, num_workers=0, drop_last=False)


def train_patchtst(
    config: TrainConfig,
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "PatchTST",
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 256,
    dropout: float = 0.1,
) -> Dict[str, Any]:
    """Train a PatchTST model."""
    console = get_console()

    preferred = config.device
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)
    use_amp = config.amp and device.type == 'cuda'

    model = PatchTST(
        L=config.L, H=config.H,
        patch_len=patch_len, stride=stride,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, dropout=dropout,
    ).to(device)

    lr = min(config.lr_g, 1e-4)  # Transformers need lower LR than GAN default
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    from .rgan_torch import AMP
    scaler = AMP.make_scaler(enabled=use_amp)

    hist = {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": []}
    best_val, best_state = float("inf"), None
    bad_epochs = 0
    batch_size = max(1, config.batch_size)

    Xtr, Ytr = data_splits["Xtr"], data_splits["Ytr"]
    Xte, Yte = data_splits["Xte"], data_splits["Yte"]
    Xval, Yval = data_splits["Xval"], data_splits["Yval"]

    train_loader = _make_loader(Xtr, Ytr, batch_size, shuffle=True, device=device)

    N = len(Xtr)

    with epoch_progress(config.epochs, description=tag) as (progress, task_id):
        for epoch in range(1, config.epochs + 1):
            model.train()
            nan_break = False
            for Xb, Yb in train_loader:
                Xb = Xb.to(device, non_blocking=True)
                Yb = Yb.to(device, non_blocking=True)
                opt.zero_grad()
                with AMP.autocast(device.type, enabled=use_amp):
                    pred = model(Xb)
                    loss = loss_fn(pred, Yb)
                if torch.isnan(loss):
                    log_info(f"[{tag}] NaN loss at epoch {epoch}, stopping training.")
                    nan_break = True
                    break
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(opt)
                scaler.update()
            if nan_break:
                break

            eval_bs = max(1, min(batch_size, 1024))
            tr = _predict_in_batches(model, Xtr, device, eval_bs, use_amp=use_amp)
            te = _predict_in_batches(model, Xte, device, eval_bs, use_amp=use_amp)
            va = _predict_in_batches(model, Xval, device, eval_bs, use_amp=use_amp)

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
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= config.patience:
                    log_info(f"[{tag}] Early stopping at epoch {epoch}. best_val={best_val:.6f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_bs = max(1, min(config.batch_size, 1024))
    tr = _predict_in_batches(model, Xtr, device, eval_bs, use_amp=use_amp)
    te = _predict_in_batches(model, Xte, device, eval_bs, use_amp=use_amp)
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
