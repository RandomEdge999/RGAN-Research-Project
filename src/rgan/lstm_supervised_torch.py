import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from .logging_utils import get_console, epoch_progress, update_epoch


def _error_stats(y_true: np.ndarray, y_pred: np.ndarray):
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "mse": mse, "bias": bias}


class LSTMSupervised(nn.Module):
    def __init__(self, L: int, H: int, n_in: int, units: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_in, hidden_size=units, batch_first=True)
        self.fc = nn.Linear(units, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y = self.fc(h)
        return y.view(y.size(0), -1, 1)


def _predict_in_batches(model: nn.Module, data: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """Run model inference on ``data`` without exhausting GPU memory."""

    if len(data) == 0:
        out_dim = getattr(model, "fc", None).out_features if hasattr(model, "fc") else data.shape[1]
        return np.empty((0, out_dim, 1), dtype=np.float32)

    preds = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = start + batch_size
            xb = torch.from_numpy(data[start:stop]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_lstm_supervised_torch(config: Dict, data_splits, results_dir: str, tag="lstm"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, H = config["L"], config["H"]
    n_in = data_splits["Xtr"].shape[-1]
    model = LSTMSupervised(L, H, n_in=n_in, units=config["units_g"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lrG"]) 
    loss_fn = torch.nn.MSELoss()

    hist = {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": []}
    best_val, best_state = float("inf"), None
    patience, bad_epochs = config["patience"], 0
    batch_size = config["batch_size"]

    Xtr = data_splits["Xtr"]; Ytr = data_splits["Ytr"]
    Xte = data_splits["Xte"]; Yte = data_splits["Yte"]
    Xval = data_splits["Xval"]; Yval = data_splits["Yval"]

    N = len(Xtr)
    steps = max(1, int(np.ceil(N / batch_size)))

    console = get_console()
    with epoch_progress(config["epochs"], description="LSTM (Torch)") as (progress, task_id):
        for epoch in range(1, config["epochs"] + 1):
            perm = np.random.permutation(N)
            model.train()
            for s in range(steps):
                b0 = s * batch_size; b1 = min((s + 1) * batch_size, N)
                idx = perm[b0:b1]
                Xb = torch.from_numpy(Xtr[idx]).to(device)
                Yb = torch.from_numpy(Ytr[idx]).to(device)
                opt.zero_grad()
                pred = model(Xb)
                loss = loss_fn(pred, Yb)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), config["grad_clip"])
                opt.step()

            model.eval()
            eval_batch_size = max(1, min(batch_size, 1024))
            tr = _predict_in_batches(model, Xtr, device, eval_batch_size)
            te = _predict_in_batches(model, Xte, device, eval_batch_size)
            va = _predict_in_batches(model, Xval, device, eval_batch_size)
            tr_rmse = float(np.sqrt(np.mean((tr.reshape(-1) - Ytr.reshape(-1)) ** 2)))
            te_rmse = float(np.sqrt(np.mean((te.reshape(-1) - Yte.reshape(-1)) ** 2)))
            va_rmse = float(np.sqrt(np.mean((va.reshape(-1) - Yval.reshape(-1)) ** 2)))

            hist["epoch"].append(epoch); hist["train_rmse"].append(tr_rmse); hist["test_rmse"].append(te_rmse); hist["val_rmse"].append(va_rmse)
            update_epoch(progress, task_id, epoch, config["epochs"], {"Train": tr_rmse, "Val": va_rmse, "Test": te_rmse})

            if va_rmse < best_val - 1e-7:
                best_val = va_rmse; best_state = model.state_dict(); bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    console.log(f"[LSTM Torch] Early stopping at epoch {epoch}."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_batch_size = max(1, min(batch_size, 1024))
    tr = _predict_in_batches(model, Xtr, device, eval_batch_size)
    te = _predict_in_batches(model, Xte, device, eval_batch_size)
    train_stats = _error_stats(Ytr.reshape(-1), tr.reshape(-1))
    test_stats = _error_stats(Yte.reshape(-1), te.reshape(-1))
    return {"model": model, "history": hist, "train_stats": train_stats, "test_stats": test_stats}



