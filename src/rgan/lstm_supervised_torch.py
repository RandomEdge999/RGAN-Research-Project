import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch, log_info, log_step, log_warn, log_error, log_shape, log_debug
from .metrics import error_stats


def _is_cuda_launch_failure(err: RuntimeError) -> bool:
    msg = str(err)
    return "cuda" in msg.lower() and "unspecified launch failure" in msg.lower()


class LSTMSupervised(nn.Module):
    """Simple LSTM model for supervised time series forecasting."""

    def __init__(self, L: int, H: int, n_in: int, units: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_in, hidden_size=units, batch_first=True)
        self.fc = nn.Linear(units, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y = self.fc(h)
        return y.view(y.size(0), -1, 1)


def _predict_in_batches(
    model: nn.Module, data: np.ndarray, device: torch.device, batch_size: int,
    use_amp: bool = False,
) -> np.ndarray:
    """Run model inference on ``data`` without exhausting GPU memory."""

    if len(data) == 0:
        out_dim = (
            getattr(model, "fc", None).out_features
            if hasattr(model, "fc")
            else data.shape[1]
        )
        return np.empty((0, out_dim, 1), dtype=np.float32)

    preds = []
    model.eval()
    from .rgan_torch import AMP
    with torch.no_grad(), AMP.autocast(device.type, enabled=use_amp):
        for start in range(0, len(data), batch_size):
            stop = start + batch_size
            xb = torch.from_numpy(data[start:stop]).to(device)
            preds.append(model(xb).float().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _make_loader(X, Y, batch_size, shuffle=True, device=None):
    """Create a DataLoader with pin_memory when using CUDA."""
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    pin = (device is not None and device.type == 'cuda')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin, num_workers=0, drop_last=False)


def train_lstm_supervised_torch(
    config: TrainConfig,
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "lstm",
) -> Dict[str, Any]:
    """Train a supervised LSTM baseline."""
    console = get_console()
    log_info(f"train_lstm_supervised_torch called: epochs={config.epochs}, device={config.device}")

    # Determine device preference
    preferred = config.device
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    log_step(f"LSTM device preference resolved to: {preferred}")

    def _train_on_device(device: torch.device) -> Dict[str, Any]:
        L, H = config.L, config.H
        n_in = data_splits["Xtr"].shape[-1]
        use_amp = config.amp and device.type == 'cuda'
        log_step(f"Building LSTM: L={L}, H={H}, n_in={n_in}, units={config.units_g}")

        # Note: 'units_g' is used as the hidden size for the LSTM
        model = LSTMSupervised(L, H, n_in=n_in, units=config.units_g).to(device)
        log_step(f"LSTM model params: {sum(p.numel() for p in model.parameters())}")
        log_shape("Xtr", data_splits["Xtr"])
        log_shape("Ytr", data_splits["Ytr"])
        opt = torch.optim.Adam(model.parameters(), lr=config.lr_g)
        loss_fn = torch.nn.MSELoss()
        from .rgan_torch import AMP
        scaler = AMP.make_scaler(enabled=use_amp)

        hist = {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": []}
        best_val, best_state = float("inf"), None
        patience, bad_epochs = config.patience, 0
        batch_size = max(1, config.batch_size)

        Xtr = data_splits["Xtr"]
        Ytr = data_splits["Ytr"]
        Xte = data_splits["Xte"]
        Yte = data_splits["Yte"]
        Xval = data_splits["Xval"]
        Yval = data_splits["Yval"]

        train_loader = _make_loader(Xtr, Ytr, batch_size, shuffle=True, device=device)

        N = len(Xtr)

        nan_break = False
        with epoch_progress(config.epochs, description="LSTM (Torch)") as (progress, task_id):
            for epoch in range(1, config.epochs + 1):
                model.train()
                for Xb, Yb in train_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    Yb = Yb.to(device, non_blocking=True)
                    opt.zero_grad()
                    with AMP.autocast(device.type, enabled=use_amp):
                        pred = model(Xb)
                        loss = loss_fn(pred, Yb)
                    if torch.isnan(loss):
                        log_info(f"[LSTM] NaN loss at epoch {epoch}, stopping training.")
                        nan_break = True
                        break
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(opt)
                    scaler.update()

                if nan_break:
                    break

                model.eval()
                eval_batch_size = max(1, min(batch_size, 1024))
                tr = _predict_in_batches(model, Xtr, device, eval_batch_size, use_amp=use_amp)
                te = _predict_in_batches(model, Xte, device, eval_batch_size, use_amp=use_amp)
                va = _predict_in_batches(model, Xval, device, eval_batch_size, use_amp=use_amp)
                
                tr_rmse = float(np.sqrt(np.mean((tr.reshape(-1) - Ytr.reshape(-1)) ** 2)))
                te_rmse = float(np.sqrt(np.mean((te.reshape(-1) - Yte.reshape(-1)) ** 2)))
                va_rmse = float(np.sqrt(np.mean((va.reshape(-1) - Yval.reshape(-1)) ** 2)))

                hist["epoch"].append(epoch)
                hist["train_rmse"].append(tr_rmse)
                hist["test_rmse"].append(te_rmse)
                hist["val_rmse"].append(va_rmse)
                
                update_epoch(
                    progress,
                    task_id,
                    epoch,
                    config.epochs,
                    {"Train": tr_rmse, "Val": va_rmse, "Test": te_rmse},
                )

                if va_rmse < best_val - 1e-7:
                    best_val = va_rmse
                    best_state = copy.deepcopy(model.state_dict())
                    bad_epochs = 0
                    log_debug(f"LSTM epoch {epoch}: new best val_rmse={va_rmse:.6f}")
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        log_info(f"LSTM early stopping at epoch {epoch}. best_val={best_val:.6f}")
                        break

        if best_state is not None:
            log_step(f"Loading LSTM best state (best_val={best_val:.6f})")
            model.load_state_dict(best_state)
        else:
            log_warn("No LSTM best state saved")

        eval_batch_size = max(1, min(config.batch_size, 1024))
        log_step(f"Computing final LSTM metrics (eval_batch_size={eval_batch_size})")
        tr = _predict_in_batches(model, Xtr, device, eval_batch_size, use_amp=use_amp)
        te = _predict_in_batches(model, Xte, device, eval_batch_size, use_amp=use_amp)
        train_stats = error_stats(Ytr.reshape(-1), tr.reshape(-1))
        test_stats = error_stats(Yte.reshape(-1), te.reshape(-1))
        log_step(f"LSTM final: train_rmse={train_stats.get('rmse', 'N/A'):.6f}, test_rmse={test_stats.get('rmse', 'N/A'):.6f}")

        return {
            "model": model,
            "history": hist,
            "train_stats": train_stats,
            "test_stats": test_stats,
            "pred_train": tr,
            "pred_test": te,
        }

    try:
        return _train_on_device(torch.device(preferred))
    except RuntimeError as err:
        if (
            isinstance(preferred, str)
            and preferred != "cpu"
            and _is_cuda_launch_failure(err)
        ):
            log_error(f"LSTM CUDA failure detected: {err}")
            log_warn("Retrying LSTM training on CPU...")
            torch.cuda.empty_cache()
            return _train_on_device(torch.device("cpu"))
        raise
