"""FITS: Frequency Interpolation for Time Series forecasting.

From "FITS: Modeling Time Series with 10k Parameters" (Xu et al., ICLR 2024 Spotlight).
Operates entirely in the frequency domain: rFFT -> low-pass filter -> complex linear
interpolation -> iFFT. Only ~10k parameters regardless of sequence length.

Interface matches the project convention:
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


class FITS(nn.Module):
    """Frequency Interpolation Time Series model."""

    def __init__(self, L: int, H: int, cut_freq: int = 0):
        super().__init__()
        self.L = L
        self.H = H

        # Number of frequency components after rFFT
        n_freq_in = L // 2 + 1
        n_freq_out = (L + H) // 2 + 1

        # Auto-select cut frequency: keep low-frequency components
        if cut_freq <= 0:
            cut_freq = max(1, n_freq_in // 2)
        self.cut_freq = min(cut_freq, n_freq_in)

        # Complex-valued linear layer for frequency interpolation
        # Maps cut_freq input frequencies to n_freq_out output frequencies
        self.freq_linear = nn.Linear(self.cut_freq, n_freq_out, dtype=torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, 1)
        x_1d = x[:, :, 0]                                 # (batch, L) — target column

        # Normalize
        mean = x_1d.mean(dim=-1, keepdim=True)
        x_norm = x_1d - mean

        # Forward FFT
        x_freq = torch.fft.rfft(x_norm, dim=-1)        # (batch, L//2+1) complex

        # Low-pass filter: keep only cut_freq components
        x_cut = x_freq[:, :self.cut_freq]               # (batch, cut_freq) complex

        # Interpolate frequencies to output length
        # freq_linear expects (batch, cut_freq) complex -> (batch, n_freq_out) complex
        y_freq = self.freq_linear(x_cut)                # (batch, n_freq_out)

        # Inverse FFT to get time domain signal of length L+H
        y_full = torch.fft.irfft(y_freq, n=self.L + self.H, dim=-1)  # (batch, L+H)

        # Take only the forecast horizon (last H values)
        y = y_full[:, -self.H:]                         # (batch, H)

        # Add back mean
        y = y + mean

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


def train_fits(
    config: TrainConfig,
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "FITS",
    cut_freq: int = 0,
) -> Dict[str, Any]:
    """Train a FITS model.

    Args:
        config: Training configuration.
        data_splits: Dict with Xtr, Ytr, Xval, Yval, Xte, Yte.
        results_dir: Output directory (kept for API consistency).
        tag: Display tag for progress bar.
        cut_freq: Number of low-frequency components to keep (0 = auto).

    Returns:
        Dict with model, history, train_stats, test_stats, pred_train, pred_test.
    """
    console = get_console()

    preferred = config.device
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)

    model = FITS(config.L, config.H, cut_freq=cut_freq).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr_g)
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
                # clip_grad_value_ doesn't support complex tensors; use norm clipping
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
