"""DLinear and NLinear baselines for time-series forecasting.

From "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023).
These dead-simple linear models outperform many Transformers on standard benchmarks.
If RGAN can't beat DLinear, the GAN component adds no value.

Both models follow the same interface as LSTMSupervised:
    input:  (batch, L, 1)  — lookback window
    output: (batch, H, 1)  — forecast horizon
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch, log_info, log_step, log_debug
from .metrics import error_stats


class DLinear(nn.Module):
    """Decomposition-Linear: trend + remainder via moving average.

    Decomposes the input into trend (moving average) and remainder,
    then applies a separate linear layer to each component.
    """

    def __init__(self, L: int, H: int, kernel_size: int = 25):
        super().__init__()
        self.L = L
        self.H = H
        self.kernel_size = kernel_size

        # Moving average for decomposition
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=(kernel_size - 1) // 2, count_include_pad=False)

        # Separate linear projections for trend and remainder
        self.linear_trend = nn.Linear(L, H)
        self.linear_remainder = nn.Linear(L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, 1)
        # Decompose into trend and remainder
        x_1d = x[:, :, 0]                            # (batch, L) — target column
        trend = self.avg(x_1d.unsqueeze(1))        # (batch, 1, L)
        trend = trend.squeeze(1)                    # (batch, L)
        remainder = x_1d - trend                    # (batch, L)

        # Project each component
        y_trend = self.linear_trend(trend)          # (batch, H)
        y_remainder = self.linear_remainder(remainder)  # (batch, H)

        y = y_trend + y_remainder                   # (batch, H)
        return y.unsqueeze(-1)                      # (batch, H, 1)


class NLinear(nn.Module):
    """Normalized-Linear: subtract last value, project, add back.

    Handles distribution shift by normalizing with the last value
    of the lookback window before applying a linear projection.
    """

    def __init__(self, L: int, H: int):
        super().__init__()
        self.linear = nn.Linear(L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, 1)
        x_1d = x[:, :, 0]                             # (batch, L) — target column
        last = x_1d[:, -1:]                         # (batch, 1)
        x_norm = x_1d - last                        # (batch, L)
        y = self.linear(x_norm) + last              # (batch, H)
        return y.unsqueeze(-1)                      # (batch, H, 1)


def _predict_in_batches(
    model: nn.Module, data: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    """Run model inference without exhausting GPU memory."""
    if len(data) == 0:
        return np.empty((0, model.linear.out_features if hasattr(model, 'linear') else data.shape[1], 1), dtype=np.float32)

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            xb = torch.from_numpy(data[start:start + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_linear_baseline(
    config: TrainConfig,
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    model_type: str = "dlinear",
    tag: str | None = None,
) -> Dict[str, Any]:
    """Train a DLinear or NLinear baseline.

    Args:
        config: Training configuration (uses L, H, epochs, batch_size, lr_g,
                patience, grad_clip, device).
        data_splits: Dict with keys Xtr, Ytr, Xval, Yval, Xte, Yte.
        results_dir: Directory for outputs (unused but kept for API consistency).
        model_type: "dlinear" or "nlinear".
        tag: Display tag for progress bar.

    Returns:
        Dict with model, history, train_stats, test_stats, pred_train, pred_test.
    """
    console = get_console()
    if tag is None:
        tag = model_type.upper()

    preferred = config.device
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)

    L, H = config.L, config.H

    if model_type == "nlinear":
        model = NLinear(L, H).to(device)
    else:
        model = DLinear(L, H).to(device)

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
                torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
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
