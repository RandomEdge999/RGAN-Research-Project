import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple


def _error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "mse": mse, "bias": bias}


@torch.no_grad()
def compute_metrics(G, X, Y, batch_size: int = 1024) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute evaluation metrics for a trained generator.

    The original implementation forwarded the entire evaluation set through
    the generator in a single batch.  When tuning with large datasets this
    caused the GPU to exhaust its memory (see the CUDA OOM in
    ``run_experiment.py``).  To keep memory usage bounded we now process the
    inputs in configurable mini-batches while accumulating the predictions on
    the CPU.  This mirrors what we do during training but without gradient
    tracking.
    """

    G.eval()
    device = next(G.parameters()).device
    dtype = next(G.parameters()).dtype

    n_samples = len(X)
    batch_size = max(1, int(batch_size))
    Yp = np.empty_like(Y)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        Xb = torch.from_numpy(X[start:end]).to(device=device, dtype=dtype)
        Yb = G(Xb).detach().cpu().numpy()
        Yp[start:end] = Yb

    stats = _error_stats(Y.reshape(-1), Yp.reshape(-1))
    return stats, Yp


def train_rgan_torch(config: Dict, models, data_splits, results_dir: str, tag: str = "rgan") -> Dict:
    G, D = models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device); D.to(device)

    Xtr, Ytr = data_splits["Xtr"], data_splits["Ytr"]
    Xval, Yval = data_splits["Xval"], data_splits["Yval"]
    Xte,  Yte  = data_splits["Xte"],  data_splits["Yte"]

    optG = torch.optim.Adam(G.parameters(), lr=config["lrG"])
    optD = torch.optim.Adam(D.parameters(), lr=config["lrD"])
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience, bad_epochs = config["patience"], 0
    batch_size = config["batch_size"]

    hist = {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": [],
            "D_loss": [], "G_loss": [], "G_adv": [], "G_reg": []}

    N = len(Xtr)
    steps = max(1, int(np.ceil(N / batch_size)))

    for epoch in range(1, config["epochs"] + 1):
        perm = np.random.permutation(N)
        D_losses, G_losses, G_advs, G_regs = [], [], [], []
        for s in range(steps):
            b0 = s * batch_size; b1 = min((s + 1) * batch_size, N)
            idx = perm[b0:b1]
            Xb = torch.from_numpy(Xtr[idx]).to(device)
            Yb = torch.from_numpy(Ytr[idx]).to(device)

            # Discriminator step
            G.train(); D.train()
            optD.zero_grad()
            Y_fake = G(Xb)
            real_pairs = torch.cat([Xb[..., :1], Yb], dim=1)
            fake_pairs = torch.cat([Xb[..., :1], Y_fake.detach()], dim=1)
            D_real = D(real_pairs)
            D_fake = D(fake_pairs)
            y_real = torch.full_like(D_real, fill_value=config["label_smooth"])
            y_fake = torch.zeros_like(D_fake)
            loss_real = bce(D_real, y_real)
            loss_fake = bce(D_fake, y_fake)
            D_loss = 0.5 * (loss_real + loss_fake)
            D_loss.backward()
            nn.utils.clip_grad_value_(D.parameters(), config["grad_clip"])
            optD.step()

            # Generator step
            optG.zero_grad()
            Y_fake = G(Xb)
            fake_pairs = torch.cat([Xb[..., :1], Y_fake], dim=1)
            D_fake = D(fake_pairs)
            y_adv = torch.ones_like(D_fake)
            adv_loss = bce(D_fake, y_adv)
            reg_loss = mse(Y_fake, Yb)
            G_loss = adv_loss + config["lambda_reg"] * reg_loss
            G_loss.backward()
            nn.utils.clip_grad_value_(G.parameters(), config["grad_clip"])
            optG.step()

            D_losses.append(float(D_loss.detach().cpu()))
            G_losses.append(float(G_loss.detach().cpu()))
            G_advs.append(float(adv_loss.detach().cpu()))
            G_regs.append(float(reg_loss.detach().cpu()))

        eval_bs = config.get("eval_batch_size", config.get("batch_size", 1024))
        tr_stats, _ = compute_metrics(G, Xtr, Ytr, batch_size=eval_bs)
        te_stats, _ = compute_metrics(G, Xte, Yte, batch_size=eval_bs)
        va_stats, _ = compute_metrics(G, Xval, Yval, batch_size=eval_bs)
        va_rmse = va_stats["rmse"]

        hist["epoch"].append(epoch)
        hist["D_loss"].append(np.mean(D_losses)); hist["G_loss"].append(np.mean(G_losses))
        hist["G_adv"].append(np.mean(G_advs));   hist["G_reg"].append(np.mean(G_regs))
        hist["train_rmse"].append(tr_stats["rmse"])
        hist["test_rmse"].append(te_stats["rmse"])
        hist["val_rmse"].append(va_stats["rmse"])

        print(f"[R-GAN Torch] Epoch {epoch:03d} | D {np.mean(D_losses):.4f} | G {np.mean(G_losses):.4f} | "
              f"Train {tr_stats['rmse']:.5f} | Val {va_rmse:.5f} | Test {te_stats['rmse']:.5f}")

        if va_rmse < best_val - 1e-7:
            best_val = va_rmse
            best_state = {"G": G.state_dict()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[R-GAN Torch] Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        G.load_state_dict(best_state["G"])

    eval_bs = config.get("eval_batch_size", config.get("batch_size", 1024))
    train_stats, _ = compute_metrics(G, Xtr, Ytr, batch_size=eval_bs)
    test_stats, Y_pred = compute_metrics(G, Xte, Yte, batch_size=eval_bs)

    return {
        "G": G, "D": D, "history": hist,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "pred_test": Y_pred
    }


