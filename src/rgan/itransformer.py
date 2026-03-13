"""iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.

From "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
(Liu et al., ICLR 2024). Inverts the standard Transformer by applying attention
across variates (channels) instead of across time steps. For univariate forecasting,
each time step is treated as a "variate token" with a 1-dimensional feature that
gets embedded to d_model.

Interface:
    input:  (batch, L, 1)
    output: (batch, H, 1)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch, log_info, log_step, log_debug
from .metrics import error_stats


class iTransformer(nn.Module):
    """Inverted Transformer for univariate time series forecasting.

    In the original multivariate setting, each variate's full time series is a token.
    For univariate: we embed the full lookback window as a single token per variate,
    then project to the forecast horizon. With 1 variate, we use a slightly adapted
    approach: segment the lookback into n_segments tokens, each covering L//n_segments
    time steps, apply attention across segments, then project.
    """

    def __init__(
        self,
        L: int,
        H: int,
        n_segments: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.L = L
        self.H = H
        self.n_segments = min(n_segments, L)

        # Each segment covers seg_len time steps
        self.seg_len = L // self.n_segments
        # Adjust n_segments if L isn't evenly divisible
        self.effective_len = self.seg_len * self.n_segments

        # Embed each segment (seg_len values -> d_model)
        self.token_embed = nn.Linear(self.seg_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_segments, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder (attention across segments/variates)
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

        # Projection head: each segment token predicts part of the horizon
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * self.n_segments),
            nn.Linear(d_model * self.n_segments, H),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, 1)
        B = x.size(0)
        x_1d = x[:, :, 0]                                 # (batch, L) — target column

        # Instance normalization
        mean = x_1d.mean(dim=-1, keepdim=True)
        std = x_1d.std(dim=-1, keepdim=True) + 1e-5
        x_norm = (x_1d - mean) / std

        # Take last effective_len values (drop leading values if L not divisible)
        x_trim = x_norm[:, -self.effective_len:]        # (batch, effective_len)

        # Reshape into segments: (batch, n_segments, seg_len)
        x_segs = x_trim.reshape(B, self.n_segments, self.seg_len)

        # Embed segments to tokens
        tokens = self.token_embed(x_segs)               # (batch, n_segments, d_model)
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)

        # Transformer: attention across segments
        z = self.encoder(tokens)                        # (batch, n_segments, d_model)

        # Flatten and project to horizon
        z = z.reshape(B, -1)                            # (batch, n_segments * d_model)
        y = self.head(z)                                # (batch, H)

        # De-normalize
        y = y * std + mean

        return y.unsqueeze(-1)                          # (batch, H, 1)


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


def train_itransformer(
    config: TrainConfig,
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "iTransformer",
    n_segments: int = 8,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
) -> Dict[str, Any]:
    """Train an iTransformer model."""
    console = get_console()

    preferred = config.device
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)

    model = iTransformer(
        L=config.L, H=config.H,
        n_segments=n_segments, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, dropout=dropout,
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
                    log_info(f"[{tag}] Early stopping at epoch {epoch}. best_val={best_val:.6f}")
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
