import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch
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
    model: nn.Module, data: np.ndarray, device: torch.device, batch_size: int
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
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = start + batch_size
            xb = torch.from_numpy(data[start:stop]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_lstm_supervised_torch(
    config: TrainConfig,
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "lstm",
) -> Dict[str, Any]:
    """Train a supervised LSTM baseline."""
    console = get_console()

    # Determine device preference
    preferred = config.device
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"

    def _train_on_device(device: torch.device) -> Dict[str, Any]:
        L, H = config.L, config.H
        n_in = data_splits["Xtr"].shape[-1]
        
        # Note: 'units_g' is used as the hidden size for the LSTM
        model = LSTMSupervised(L, H, n_in=n_in, units=config.units_g).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr_g)
        loss_fn = torch.nn.MSELoss()

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

        N = len(Xtr)
        steps = max(1, int(np.ceil(N / batch_size)))

        with epoch_progress(config.epochs, description="LSTM (Torch)") as (progress, task_id):
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

                model.eval()
                eval_batch_size = max(1, min(batch_size, 1024))
                tr = _predict_in_batches(model, Xtr, device, eval_batch_size)
                te = _predict_in_batches(model, Xte, device, eval_batch_size)
                va = _predict_in_batches(model, Xval, device, eval_batch_size)
                
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
                    best_state = model.state_dict()
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        console.log(f"[LSTM Torch] Early stopping at epoch {epoch}.")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        eval_batch_size = max(1, min(config.batch_size, 1024))
        tr = _predict_in_batches(model, Xtr, device, eval_batch_size)
        te = _predict_in_batches(model, Xte, device, eval_batch_size)
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

    try:
        return _train_on_device(torch.device(preferred))
    except RuntimeError as err:
        if (
            isinstance(preferred, str)
            and preferred != "cpu"
            and _is_cuda_launch_failure(err)
        ):
            console.log("[LSTM Torch] CUDA failure detected; retrying on CPU.")
            torch.cuda.empty_cache()
            return _train_on_device(torch.device("cpu"))
        raise
