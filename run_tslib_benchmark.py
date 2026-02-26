#!/usr/bin/env python3
"""Benchmark RGAN models on standard TSLib datasets.

Run our models (RGAN, LSTM, Naive, ARIMA, Tree Ensemble) on benchmark
datasets used by the time-series research community so results can be
compared with published numbers from papers like PatchTST, iTransformer,
DLinear, etc.

Usage:
    python run_tslib_benchmark.py                       # all 4 datasets, all horizons
    python run_tslib_benchmark.py --datasets ETTh1      # single dataset quick test
    python run_tslib_benchmark.py --skip_classical      # skip slow ARIMA / Tree
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.rgan.tslib_data import (
    TSLIB_DATASETS,
    load_dataset,
    split_and_standardize,
)
from src.rgan.data import make_windows_univariate
from src.rgan.config import TrainConfig
from src.rgan.models_torch import build_generator, build_discriminator
from src.rgan.rgan_torch import train_rgan_torch
from src.rgan.lstm_supervised_torch import train_lstm_supervised_torch
from src.rgan.baselines import naive_baseline, arima_forecast, tree_ensemble_forecast
from src.rgan.metrics import error_stats


# ── helpers ─────────────────────────────────────────────────────────

def _split_windows(X, Y, val_frac=0.1):
    """Split windowed arrays into train / val sets (chronological)."""
    n = len(X)
    n_val = max(1, int(round(val_frac * n)))
    n_train = n - n_val
    return {
        "Xtr": X[:n_train], "Ytr": Y[:n_train],
        "Xval": X[n_train:], "Yval": Y[n_train:],
    }


def _device_str(gpu_id: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{gpu_id}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── single benchmark run ───────────────────────────────────────────

def run_single(
    dataset_name: str,
    seq_len: int,
    pred_len: int,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    """Run all models on one (dataset, pred_len) pair. Return metrics."""

    tag = f"{dataset_name}_{pred_len}"
    sub_dir = Path(args.results_dir) / tag
    sub_dir.mkdir(parents=True, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────
    df, target = load_dataset(dataset_name, data_dir=args.data_dir)
    split = split_and_standardize(df, target, dataset_name)

    X_train, Y_train = make_windows_univariate(split["scaled_train"], target, seq_len, pred_len)
    X_val, Y_val = make_windows_univariate(split["scaled_val"], target, seq_len, pred_len)
    X_test, Y_test = make_windows_univariate(split["scaled_test"], target, seq_len, pred_len)

    print(f"\n{'='*60}")
    print(f"  {dataset_name}  |  seq_len={seq_len}  pred_len={pred_len}")
    print(f"  Windows  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    print(f"{'='*60}")

    if len(X_test) < 10:
        print(f"  WARNING: only {len(X_test)} test windows – results may be unreliable")

    device = _device_str(args.gpu_id)
    results: Dict[str, Dict[str, float]] = {}

    # ── RGAN ────────────────────────────────────────────────────────
    config = TrainConfig(
        L=seq_len, H=pred_len,
        epochs=args.epochs, batch_size=args.batch_size,
        units_g=args.units, units_d=args.units,
        lr_g=args.lr, lr_d=args.lr * 0.5,
        lambda_reg=5.0,
        adv_weight=0.1,
        grad_clip=1.0,
        supervised_warmup_epochs=5,
        patience=args.patience, device=device,
        amp=False,
    )
    G = build_generator(L=seq_len, H=pred_len, n_in=1, units=args.units, num_layers=1, dropout=0.1, layer_norm=True)
    D = build_discriminator(L=seq_len, H=pred_len, units=args.units, num_layers=1, dropout=0.1, activation="sigmoid", layer_norm=True)

    data_dict = {
        "Xtr": X_train, "Ytr": Y_train,
        "Xval": X_val, "Yval": Y_val,
        "Xte": X_test, "Yte": Y_test,
    }

    t0 = time.time()
    rgan_out = train_rgan_torch(config, (G, D), data_dict, str(sub_dir), tag="rgan")
    rgan_pred = rgan_out["pred_test"]
    rgan_stats = error_stats(Y_test.reshape(-1), rgan_pred.reshape(-1))
    results["RGAN"] = {"mse": rgan_stats["mse"], "mae": rgan_stats["mae"]}
    print(f"  RGAN      MSE={rgan_stats['mse']:.6f}  MAE={rgan_stats['mae']:.6f}  ({time.time()-t0:.0f}s)")

    # ── LSTM ────────────────────────────────────────────────────────
    t0 = time.time()
    lstm_out = train_lstm_supervised_torch(config, data_dict, str(sub_dir), tag="lstm")
    lstm_pred = lstm_out["pred_test"]
    lstm_stats = error_stats(Y_test.reshape(-1), lstm_pred.reshape(-1))
    results["LSTM"] = {"mse": lstm_stats["mse"], "mae": lstm_stats["mae"]}
    print(f"  LSTM      MSE={lstm_stats['mse']:.6f}  MAE={lstm_stats['mae']:.6f}  ({time.time()-t0:.0f}s)")

    # ── Naive ───────────────────────────────────────────────────────
    naive_stats, _ = naive_baseline(X_test, Y_test)
    results["Naive"] = {"mse": naive_stats["mse"], "mae": naive_stats["mae"]}
    print(f"  Naive     MSE={naive_stats['mse']:.6f}  MAE={naive_stats['mae']:.6f}")

    # ── Classical baselines ─────────────────────────────────────────
    if not args.skip_classical:
        # ARIMA
        t0 = time.time()
        try:
            arima_stats, _ = arima_forecast(X_train, Y_train, X_test, Y_test)
            results["ARIMA"] = {"mse": arima_stats["mse"], "mae": arima_stats["mae"]}
            print(f"  ARIMA     MSE={arima_stats['mse']:.6f}  MAE={arima_stats['mae']:.6f}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  ARIMA     FAILED: {e}")

        # Tree Ensemble
        t0 = time.time()
        try:
            tree_stats, _ = tree_ensemble_forecast(X_train, Y_train, X_test, Y_test)
            results["TreeEnsemble"] = {"mse": tree_stats["mse"], "mae": tree_stats["mae"]}
            print(f"  Tree      MSE={tree_stats['mse']:.6f}  MAE={tree_stats['mae']:.6f}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  Tree      FAILED: {e}")

    # ── save per-run metrics ────────────────────────────────────────
    run_meta = {
        "dataset": dataset_name,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "results": results,
    }
    with open(sub_dir / "metrics.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return results


# ── aggregate results ───────────────────────────────────────────────

def _build_table(all_results: List[dict]) -> str:
    """Build a CSV string from collected results."""
    import pandas as pd

    rows = []
    for entry in all_results:
        ds = entry["dataset"]
        pl = entry["pred_len"]
        for model, metrics in entry["results"].items():
            rows.append({
                "Dataset": ds,
                "Pred_Len": pl,
                "Model": model,
                "MSE": metrics["mse"],
                "MAE": metrics["mae"],
            })
    return pd.DataFrame(rows)


def _pivot_table(df) -> str:
    """Pivot into TSLib paper format: rows=(Dataset,Pred_Len), cols=models."""
    import pandas as pd

    models = df["Model"].unique()
    pivot_rows = []
    for (ds, pl), grp in df.groupby(["Dataset", "Pred_Len"]):
        row = {"Dataset": ds, "Pred_Len": pl}
        for _, r in grp.iterrows():
            m = r["Model"]
            row[f"{m}_MSE"] = r["MSE"]
            row[f"{m}_MAE"] = r["MAE"]
        pivot_rows.append(row)
    return pd.DataFrame(pivot_rows)


def _save_latex(pivot_df, path: Path):
    """Save a LaTeX-formatted table."""
    with open(path, "w") as f:
        f.write(pivot_df.to_latex(index=False, float_format="%.4f"))
    print(f"  LaTeX table saved to {path}")


# ── main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RGAN models on TSLib datasets"
    )
    parser.add_argument(
        "--datasets", type=str,
        default=",".join(TSLIB_DATASETS.keys()),
        help="Comma-separated dataset names (default: all 4)",
    )
    parser.add_argument("--seq_len", type=int, default=96, help="Lookback window (default: 96)")
    parser.add_argument(
        "--pred_lens", type=str, default="96,192,336,720",
        help="Comma-separated prediction horizons",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per run")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--units", type=int, default=64, help="Hidden units for RGAN/LSTM")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--skip_classical", action="store_true", help="Skip ARIMA and Tree Ensemble")
    parser.add_argument("--results_dir", type=str, default="results/tslib_benchmark")
    parser.add_argument("--data_dir", type=str, default="data/tslib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets = [d.strip() for d in args.datasets.split(",")]
    pred_lens = [int(p.strip()) for p in args.pred_lens.split(",")]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[dict] = []

    for ds in datasets:
        for pl in pred_lens:
            try:
                metrics = run_single(ds, args.seq_len, pl, args)
                all_results.append({
                    "dataset": ds,
                    "pred_len": pl,
                    "results": metrics,
                })
            except Exception as e:
                print(f"\n  SKIPPED {ds} pred_len={pl}: {e}")

    if not all_results:
        print("\nNo results collected.")
        return

    # ── save consolidated tables ────────────────────────────────────
    import pandas as pd

    df = _build_table(all_results)
    csv_path = results_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    pivot = _pivot_table(df)
    pivot_csv = results_dir / "benchmark_pivot.csv"
    pivot.to_csv(pivot_csv, index=False)

    _save_latex(pivot, results_dir / "benchmark_results.tex")

    # ── print summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*70}\n")

    # save full metadata
    meta = {
        "datasets": datasets,
        "seq_len": args.seq_len,
        "pred_lens": pred_lens,
        "epochs": args.epochs,
        "seed": args.seed,
        "results": all_results,
    }
    with open(results_dir / "benchmark_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
