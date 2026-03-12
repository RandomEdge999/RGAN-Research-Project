"""Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.

From "Informer: Beyond Efficient Transformer for Long Sequence Time-Series
Forecasting" (Zhou et al., AAAI 2021 Best Paper). Uses ProbSparse self-attention
to reduce complexity from O(L^2) to O(L log L).

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
from typing import Dict, Any, Optional

from .config import TrainConfig
from .logging_utils import get_console, epoch_progress, update_epoch
from .metrics import error_stats


# ── ProbSparse Attention ────────────────────────────────────────────

class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention: selects top-u queries based on KL-divergence
    from uniform distribution, reducing O(L^2) to O(L log L).
    """

    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, top_n: int):
        """Compute ProbSparse attention scores.

        Q: (B, H, L_Q, D)
        K: (B, H, L_K, D)
        """
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample a subset of keys for the sparsity measure
        K_sample_idx = torch.randint(0, L_K, (sample_k,), device=K.device)
        K_sample = K[:, :, K_sample_idx, :]  # (B, H, sample_k, D)

        # Q_K_sample: (B, H, L_Q, sample_k)
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(D)

        # Sparsity measurement: max - mean along the key dimension
        M = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)  # (B, H, L_Q)

        # Select top-n queries with highest sparsity
        M_top = M.topk(top_n, sorted=False).indices  # (B, H, top_n)

        return M_top

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, L_Q, _ = q.shape
        _, L_K, _ = k.shape
        H = self.n_heads
        D = self.d_head

        Q = self.q_proj(q).reshape(B, L_Q, H, D).permute(0, 2, 1, 3)  # (B, H, L_Q, D)
        K = self.k_proj(k).reshape(B, L_K, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(v).reshape(B, L_K, H, D).permute(0, 2, 1, 3)

        # Determine sampling parameters
        U = max(1, min(self.factor * int(math.ceil(math.log(L_K + 1))), L_K))
        u = max(1, min(self.factor * int(math.ceil(math.log(L_Q + 1))), L_Q))

        # For short sequences, fall back to standard attention
        if L_Q <= u * 2 or L_K <= U:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            out = torch.matmul(attn, V)
        else:
            # ProbSparse: select top-u queries
            top_idx = self._prob_QK(Q, K, sample_k=U, top_n=u)  # (B, H, u)

            # Initial output: mean of V (for non-selected queries)
            V_mean = V.mean(dim=2, keepdim=True).expand_as(V)  # (B, H, L_Q, D)
            out = V_mean.clone()

            # Vectorized sparse attention over all B,H simultaneously
            # top_idx: (B, H, u) — gather selected queries
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, H, u, D)
            Q_sparse = torch.gather(Q, 2, top_idx_exp)  # (B, H, u, D)
            # scores: (B, H, u, L_K)
            scores = torch.matmul(Q_sparse, K.transpose(-2, -1)) / math.sqrt(D)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            # attn_out: (B, H, u, D)
            attn_out = torch.matmul(attn, V)
            # Scatter back into output
            out.scatter_(2, top_idx_exp, attn_out)

        out = out.permute(0, 2, 1, 3).reshape(B, L_Q, H * D)
        return self.out_proj(out)


# ── Informer Encoder Layer ──────────────────────────────────────────

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, factor, dropout)
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
        # Pre-norm architecture
        z = self.norm1(x)
        x = x + self.dropout(self.attn(z, z, z))
        z = self.norm2(x)
        x = x + self.ff(z)
        return x


# ── Distilling Layer (halves sequence length) ──────────────────────

class ConvDistilling(nn.Module):
    """Conv1d + MaxPool to halve the sequence length between encoder layers."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) -> (B, L//2, d_model)
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.pool(self.act(self.norm(self.conv(x))))
        return x.permute(0, 2, 1)  # (B, L//2, d_model)


# ── Informer Model ──────────────────────────────────────────────────

class Informer(nn.Module):
    """Informer for univariate time series forecasting."""

    def __init__(
        self,
        L: int,
        H: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        factor: int = 5,
        dropout: float = 0.1,
        distil: bool = True,
    ):
        super().__init__()
        self.L = L
        self.H = H

        # Value embedding (1 -> d_model)
        self.value_embed = nn.Linear(1, d_model)

        # Sinusoidal positional encoding
        pe = torch.zeros(L, d_model)
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d_model)

        self.dropout = nn.Dropout(dropout)

        # Encoder layers with optional distilling
        self.layers = nn.ModuleList()
        self.distil_layers = nn.ModuleList()
        self.distil = distil

        seq_len = L
        for i in range(n_layers):
            self.layers.append(
                InformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            )
            if distil and i < n_layers - 1:
                self.distil_layers.append(ConvDistilling(d_model))
                seq_len = seq_len // 2

        self.norm = nn.LayerNorm(d_model)

        # Final projection: flatten encoder output -> H
        self.head = nn.Sequential(
            nn.Linear(d_model * seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, H),
        )

        self._final_seq_len = seq_len
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
        z = z + self.pe[:, :self.L, :]
        z = self.dropout(z)

        # Encoder with distilling
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if self.distil and i < len(self.distil_layers):
                z = self.distil_layers[i](z)

        z = self.norm(z)

        # Flatten and project
        z = z.reshape(B, -1)  # (B, final_seq_len * d_model)
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


def train_informer(
    config: TrainConfig,
    data_splits: Dict[str, np.ndarray],
    results_dir: str,
    tag: str = "Informer",
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    factor: int = 5,
    dropout: float = 0.1,
    distil: bool = True,
) -> Dict[str, Any]:
    """Train an Informer model."""
    console = get_console()

    preferred = config.device
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(preferred)

    model = Informer(
        L=config.L, H=config.H,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, factor=factor, dropout=dropout, distil=distil,
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
