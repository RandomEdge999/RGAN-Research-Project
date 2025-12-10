#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from itertools import combinations
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from contextlib import contextmanager

import numpy as np
from numpy.lib import NumpyVersion


def _ensure_numpy_compat() -> None:
    """Abort early when running with an incompatible NumPy major version."""
    if NumpyVersion(np.__version__) >= NumpyVersion("2.0.0"):
        raise RuntimeError(
            "Detected NumPy %s, but this project requires NumPy < 2. "
            "Install a 1.x release (e.g., pip install 'numpy<2')." % np.__version__
        )


_ensure_numpy_compat()

import torch

from src.rgan.baselines import (
    naive_baseline,
    arima_forecast,
    arma_forecast,
    tree_ensemble_forecast,
    classical_curves_vs_samples,
)
from src.rgan.data import (
    load_csv_series,
    interpolate_and_standardize,
    make_windows_univariate,
    make_windows_with_covariates,
    DataSplit,
)
from src.rgan.plots import (
    plot_single_train_test,
    plot_constant_train_test,
    plot_compare_models_bars,
    plot_classical_curves,
    plot_learning_curves,
    create_error_metrics_table,
)
from src.rgan.tune import tune_rgan
from src.rgan.logging_utils import get_console, print_banner, print_kv_table
from src.rgan.metrics import (
    describe_model,
    error_stats,
    summarise_with_uncertainty,
    diebold_mariano,
)
from src.rgan.config import TrainConfig, ModelConfig


@contextmanager
def log_phase(console, title: str):
    """Lightweight console-timed phase logger to surface progress during long steps."""

    console.print(f"[bold cyan]→ {title}...[/bold cyan]")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        console.print(f"[green]✓ {title} completed[/green] [dim]{elapsed:.1f}s[/dim]")


def describe_resource_heavy_steps(args: argparse.Namespace, console) -> None:
    """Surface optional costly steps so users understand startup/runtime cost."""

    heavy_steps = []

    if not args.skip_classical:
        heavy_steps.append(
            "Classical baselines (ARIMA/ARMA/tree ensemble) fit on the full training set"
            " and can dominate startup on long series. Use --skip_classical to disable."
        )
    if args.tune:
        heavy_steps.append(
            "Hyperparameter sweep (--tune) runs multiple R-GAN fits; expect extended runtime."
        )
    if args.curve_steps > 0:
        heavy_steps.append(
            "Learning-curve resampling (--curve_steps) trains several models on growing subsets."
        )
    noise_levels = parse_noise_levels(str(args.noise_levels))
    if len(noise_levels) > 1:
        heavy_steps.append(
            "Robustness evaluation across multiple noise levels; reduce --noise_levels to a single"
            " value to skip."
        )
    if args.bootstrap_samples > 0:
        heavy_steps.append(
            "Bootstrap uncertainty estimation adds repeated metric computation; set"
            " --bootstrap_samples 0 to omit."
        )

    if heavy_steps:
        console.print("[bold yellow]Resource-heavy components enabled:[/bold yellow]")
        for item in heavy_steps:
            console.print(f" • {item}")
    else:
        console.print("[green]All optional heavy components are disabled for this run.[/green]")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        pass


def split_windows_for_training(
    X: np.ndarray,
    Y: np.ndarray,
    val_fraction: float = 0.1,
    eval_fraction: float = 0.0,
    *,
    minimum_val: int = 1,
    minimum_train: int = 1,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Partition windowed data into train/validation/(optional) evaluation sets."""

    n = len(X)
    if n == 0:
        raise ValueError("Cannot split empty window arrays.")

    indices = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)
    Xw = X[indices]
    Yw = Y[indices]

    n_val = max(minimum_val, int(round(val_fraction * n))) if val_fraction > 0 else 0
    n_eval = max(0, int(round(eval_fraction * n))) if eval_fraction > 0 else 0

    if n_val + n_eval >= n:
        spare = max(0, n - minimum_train)
        n_val = min(max(minimum_val, min(n - minimum_train, n_val)), spare)
        n_eval = max(0, min(n_eval, n - n_val - minimum_train))

    train_end = n - (n_val + n_eval)
    if train_end < minimum_train:
        raise ValueError("Not enough samples to maintain the requested split proportions.")

    val_end = train_end + n_val

    result = {
        "X_train": Xw[:train_end],
        "Y_train": Yw[:train_end],
        "X_val": Xw[train_end:val_end],
        "Y_val": Yw[train_end:val_end],
        "X_eval": Xw[val_end:] if n_eval > 0 else Xw[val_end:val_end],
        "Y_eval": Yw[val_end:] if n_eval > 0 else Yw[val_end:val_end],
    }
    return result


def parse_noise_levels(spec: str) -> np.ndarray:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = float(part)
        except ValueError as exc:
            raise ValueError(f"Invalid noise level '{part}'.") from exc
        if val < 0:
            raise ValueError("Noise levels must be non-negative.")
        values.append(val)
    if 0.0 not in values:
        values.append(0.0)
    return np.array(sorted(set(values)), dtype=float)


def compute_learning_curves(
    args: argparse.Namespace,
    base_config: TrainConfig,
    Xfull_tr: np.ndarray,
    Yfull_tr: np.ndarray,
    Xte: np.ndarray,
    Yte: np.ndarray,
    n_features: int,
) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]]]:
    if args.curve_steps <= 0:
        return [], {}, {}

    total = len(Xfull_tr)
    if total < 2:
        return [], {}, {}

    min_size = max(int(args.curve_min_frac * total), 5)
    if args.curve_steps == 1:
        sizes = np.array([total], dtype=int)
    else:
        sizes = np.linspace(min_size, total, args.curve_steps, dtype=int)
    sizes = np.clip(sizes, 2, total)
    sizes = np.unique(sizes)

    curves_mean = {
        "R-GAN": [],
        "LSTM": [],
        "Tree Ensemble": [],
        "Naïve Baseline": [],
        "ARIMA": [],
        "ARMA": [],
    }
    curves_std = {key: [] for key in curves_mean}
    used_sizes = []

    rng = np.random.default_rng(args.seed)
    naive_test_stats, _ = naive_baseline(Xte, Yte)

    from src.rgan.models_torch import (
        build_generator as build_generator_backend,
        build_discriminator as build_discriminator_backend,
    )
    from src.rgan.rgan_torch import train_rgan_torch as train_rgan_backend
    from src.rgan.lstm_supervised_torch import train_lstm_supervised_torch as train_lstm_backend

    repeats = max(1, getattr(args, "curve_repeats", 1))

    for size in sizes:
        if size < 2:
            continue
        if size > total:
            continue

        rmse_acc = {key: [] for key in curves_mean}

        for rep in range(repeats):
            if size == total:
                selection = np.arange(total)
            else:
                selection = rng.choice(total, size=size, replace=False)

            Xsubset = Xfull_tr[selection]
            Ysubset = Yfull_tr[selection]

            try:
                sub_split = split_windows_for_training(
                    Xsubset,
                    Ysubset,
                    val_fraction=args.val_frac,
                    eval_fraction=0.0,
                    shuffle=True,
                    rng=rng,
                )
            except ValueError:
                continue

            data_splits = {
                "Xtr": sub_split["X_train"],
                "Ytr": sub_split["Y_train"],
                "Xval": sub_split["X_val"],
                "Yval": sub_split["Y_val"],
                "Xte": Xte,
                "Yte": Yte,
            }

            # Create a copy of base_config for this curve point
            # Since TrainConfig is a dataclass, we can use replace or just create new
            import dataclasses
            curve_config = dataclasses.replace(base_config)
            curve_config.epochs = max(1, min(base_config.epochs, args.curve_epochs))
            curve_config.patience = min(curve_config.patience, curve_config.epochs)

            set_seed(args.seed + rep)

            # Build models
            # Note: We need to pass args manually to build functions or update them to take config
            # Assuming build functions still take kwargs
            
            G_curve = build_generator_backend(
                L=base_config.L,
                H=base_config.H,
                n_in=n_features,
                units=curve_config.units_g,
                dropout=curve_config.dropout,
                num_layers=curve_config.g_layers,
                dense_activation=curve_config.g_dense_activation,
                layer_norm=(curve_config.gan_variant == "wgan-gp"),
            )
            D_curve = build_discriminator_backend(
                L=base_config.L,
                H=base_config.H,
                units=curve_config.units_d,
                dropout=curve_config.dropout,
                num_layers=curve_config.d_layers,
                activation=curve_config.d_activation,
                layer_norm=(curve_config.gan_variant == "wgan-gp"),
                use_spectral_norm=(curve_config.gan_variant == "wgan-gp"),
            )
            
            rgan_curve_out = train_rgan_backend(
                curve_config,
                (G_curve, D_curve),
                data_splits,
                str(args.results_dir),
                tag=f"rgan_curve_{size}_{rep}",
            )
            lstm_curve_out = train_lstm_backend(
                curve_config,
                data_splits,
                str(args.results_dir),
                tag=f"lstm_curve_{size}_{rep}",
            )

            rmse_acc["R-GAN"].append(rgan_curve_out["test_stats"]["rmse"])
            rmse_acc["LSTM"].append(lstm_curve_out["test_stats"]["rmse"])

            rmse_acc["Naïve Baseline"].append(naive_test_stats["rmse"])
            
            arima_stats, _ = arima_forecast(
                sub_split["X_train"], sub_split["Y_train"], Xte, Yte
            )
            rmse_acc["ARIMA"].append(arima_stats["rmse"])
            
            arma_stats, _ = arma_forecast(
                sub_split["X_train"], sub_split["Y_train"], Xte, Yte
            )
            rmse_acc["ARMA"].append(arma_stats["rmse"])
            tree_stats, _ = tree_ensemble_forecast(
                sub_split["X_train"],
                sub_split["Y_train"],
                Xte,
                Yte,
                random_state=args.seed + rep,
            )
            rmse_acc["Tree Ensemble"].append(tree_stats["rmse"])

        if not any(rmse_acc[key] for key in rmse_acc):
            continue

        used_sizes.append(int(size))
        for key in curves_mean:
            values = rmse_acc[key]
            if values:
                curves_mean[key].append(float(np.mean(values)))
                curves_std[key].append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            else:
                curves_mean[key].append(float("nan"))
                curves_std[key].append(float("nan"))

    return used_sizes, curves_mean, curves_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="auto")
    ap.add_argument("--time_col", default="auto")
    ap.add_argument("--resample", default="")
    ap.add_argument("--agg", default="last")
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lambda_reg", type=float, default=0.1)
    ap.add_argument(
        "--gan_variant",
        choices=["standard", "wgan", "wgan-gp"],
        default="standard",
        help="Adversarial loss variant (standard BCE or Wasserstein with GP).",
    )
    ap.add_argument("--d_steps", type=int, default=1, help="Number of discriminator updates per batch.")
    ap.add_argument("--g_steps", type=int, default=1, help="Number of generator updates per batch.")
    ap.add_argument("--units_g", type=int, default=64)
    ap.add_argument("--units_d", type=int, default=64)
    ap.add_argument("--g_layers", type=int, default=1)
    ap.add_argument("--d_layers", type=int, default=1)
    ap.add_argument("--g_dense_activation", default="", help="Optional dense activation for the generator (PyTorch activation name).")
    ap.add_argument("--d_activation", default="sigmoid", help="Discriminator activation (PyTorch activation name).")
    ap.add_argument("--lrG", type=float, default=1e-3)
    ap.add_argument("--lrD", type=float, default=1e-3)
    ap.add_argument("--label_smooth", type=float, default=0.9)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument(
        "--wgan_clip_value",
        type=float,
        default=0.01,
        help="Weight-clipping magnitude for vanilla WGAN training.",
    )
    ap.add_argument(
        "--supervised_warmup_epochs",
        type=int,
        default=0,
        help="Train with pure supervised loss for the first N epochs before enabling GAN loss.",
    )
    ap.add_argument(
        "--lambda_reg_start",
        type=float,
        default=None,
        help="Initial lambda_reg value for annealing (defaults to --lambda_reg).",
    )
    ap.add_argument(
        "--lambda_reg_end",
        type=float,
        default=None,
        help="Final lambda_reg value for annealing (defaults to --lambda_reg).",
    )
    ap.add_argument(
        "--lambda_reg_warmup_epochs",
        type=int,
        default=1,
        help="Epochs over which to interpolate between lambda_reg_start and lambda_reg_end.",
    )
    ap.add_argument(
        "--adv_weight",
        type=float,
        default=1.0,
        help="Global weight applied to the adversarial term when combining with the supervised loss.",
    )
    ap.add_argument(
        "--instance_noise_std",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian instance noise added to discriminator inputs (0 to disable).",
    )
    ap.add_argument(
        "--instance_noise_decay",
        type=float,
        default=0.95,
        help="Multiplicative decay applied to instance-noise std each epoch.",
    )
    ap.add_argument(
        "--wgan_gp_lambda",
        type=float,
        default=10.0,
        help="Gradient penalty coefficient when using WGAN-GP.",
    )
    ap.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="EMA decay for generator weights (0 disables EMA).",
    )
    ap.add_argument(
        "--use_logits",
        type=lambda v: str(v).lower() in ["1", "true", "yes"],
        default=False,
        help="Treat discriminator outputs as logits (set when d_activation is linear).",
    )
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1, help="Fraction of training windows reserved for validation during fitting.")
    ap.add_argument(
        "--tune_eval_frac",
        type=float,
        default=0.1,
        help="Additional fraction of the training windows reserved as a tuning-only hold-out set.",
    )
    ap.add_argument("--curve_steps", type=int, default=0)
    ap.add_argument("--curve_min_frac", type=float, default=0.4)
    ap.add_argument("--curve_epochs", type=int, default=40)
    ap.add_argument(
        "--curve_repeats",
        type=int,
        default=3,
        help="Number of resampled runs per learning-curve point for variance estimation.",
    )
    # Torch runtime knobs
    ap.add_argument("--amp", type=lambda v: str(v).lower() in ["1","true","yes"], default=True,
                    help="Enable automatic mixed precision for PyTorch training")
    ap.add_argument("--eval_batch_size", type=int, default=512,
                    help="Evaluation batch size for PyTorch inference")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="Number of workers for PyTorch DataLoader (0 = main process, safest)")
    ap.add_argument("--prefetch_factor", type=int, default=2,
                    help="Prefetch factor for the PyTorch DataLoader")
    ap.add_argument("--persistent_workers", type=lambda v: str(v).lower() in ["1","true","yes"], default=True,
                    help="Keep DataLoader workers alive between epochs")
    ap.add_argument("--pin_memory", type=lambda v: str(v).lower() in ["1","true","yes"], default=True,
                    help="Pin host memory for faster host-to-device copies")
    ap.add_argument(
        "--noise_levels",
        default="0,0.05,0.1",
        help="Comma-separated list of Gaussian noise standard deviations for robustness evaluation (0 always included).",
    )
    ap.add_argument(
        "--bootstrap_samples",
        type=int,
        default=300,
        help="Number of bootstrap resamples for metric uncertainty estimates.",
    )
    ap.add_argument(
        "--skip_classical",
        action="store_true",
        help=(
            "Skip slow classical baselines (ARIMA/ARMA/tree ensemble) and the classical "
            "error curves. Useful for quick smoke tests or when you only care about the "
            "neural models."
        ),
    )
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Run the R-GAN hyperparameter sweep (disabled by default).",
    )
    ap.add_argument(
        "--require_cuda",
        action="store_true",
        help="Exit immediately if CUDA is unavailable (helpful when you expect to use a GPU).",
    )
    ap.add_argument("--tune_csv", default="")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use (default: 0).")
    args = ap.parse_args()

    console = get_console()
    print_banner(console, "RGAN Research Project", "Noise-Resilient Forecasting – Experiment Runner")

    # Device Check & Announcement
    import torch

    def _cuda_build_hint() -> str:
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_version:
            return f"PyTorch CUDA build detected (CUDA {cuda_version})."
        return "PyTorch was installed without CUDA support. Install a CUDA wheel (e.g., torch==2.2.2+cu121)."

    if args.require_cuda:
        if not torch.cuda.is_available():
            console.print("[bold red]CRITICAL ERROR: --require_cuda specified but no CUDA device found.[/bold red]")
            console.print(f"[red]{_cuda_build_hint()}[/red]")
            sys.exit(1)
        
        if args.gpu_id >= torch.cuda.device_count():
            console.print(f"[bold red]CRITICAL ERROR: --gpu_id {args.gpu_id} requested but only {torch.cuda.device_count()} devices available.[/bold red]")
            sys.exit(1)

        torch.cuda.set_device(args.gpu_id)
        device_name = torch.cuda.get_device_name(args.gpu_id)
        console.print(f"[bold green]✓ Strict GPU Mode Active:[/bold green] Using device {args.gpu_id} ({device_name})")
        console.print("[dim]CPU fallback is DISABLED.[/dim]")
    
    elif torch.cuda.is_available():
        # Auto-detect mode (default behavior)
        torch.cuda.set_device(args.gpu_id)
        device_name = torch.cuda.get_device_name(args.gpu_id)
        console.print(f"[bold green]✓ GPU Detected:[/bold green] {device_name} (ID: {args.gpu_id})")
        console.print(f"[dim]Using CUDA for training. Fallback to CPU enabled if CUDA fails.[/dim]")
        console.print(f"[dim]{_cuda_build_hint()}[/dim]")
    else:
        console.print("[bold yellow]! No GPU detected.[/bold yellow] Falling back to CPU.")
        console.print("[dim]Training will be significantly slower.[/dim]")
        console.print(f"[dim]{_cuda_build_hint()}[/dim]")

    # Windows-specific fix: REMOVED as per user request (running on Linux/Mac)
    # if platform.system() == "Windows" and args.num_workers > 0:
    #     console.print(f"[yellow]Windows detected: Overriding num_workers from {args.num_workers} to 0 to prevent deadlocks.[/yellow]")
    #     args.num_workers = 0
    #     args.persistent_workers = False

    if args.tune_csv and not args.tune:
        args.tune = True

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = parse_noise_levels(str(args.noise_levels))

    describe_resource_heavy_steps(args, console)

    try:
        from src.rgan.models_torch import (
            build_generator as build_generator_backend,
            build_discriminator as build_discriminator_backend,
        )
        from src.rgan.rgan_torch import (
            train_rgan_torch as train_rgan_backend,
            compute_metrics as compute_metrics_backend,
        )
        from src.rgan.lstm_supervised_torch import (
            train_lstm_supervised_torch as train_lstm_backend,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError(
                "PyTorch is required but not installed. Please install the 'torch' package to continue."
            ) from exc
        raise

    set_seed(args.seed)

    with log_phase(console, "Load dataset and standardize"):
        df, target_col, time_used = load_csv_series(args.csv, args.target, args.time_col, args.resample, args.agg)
        prep = interpolate_and_standardize(df, target_col, train_ratio=args.train_ratio)
        target_mean = prep["target_mean"]
        target_std = prep["target_std"]

        print_kv_table(console, "Dataset", {
            "CSV": args.csv,
            "Target": target_col,
            "Time Column": time_used or "(none)",
            "Rows": len(df),
            "Train/Test Split": f"{args.train_ratio:.2f}/{1-args.train_ratio:.2f}",
        })

    with log_phase(console, "Create training and test windows"):
        try:
            if prep["covariates"]:
                Xfull_tr, Yfull_tr = make_windows_with_covariates(
                    prep["scaled_train"], target_col, prep["covariates"], args.L, args.H
                )
                Xte, Yte = make_windows_with_covariates(
                    prep["scaled_test"], target_col, prep["covariates"], args.L, args.H
                )
            else:
                Xfull_tr, Yfull_tr = make_windows_univariate(
                    prep["scaled_train"], target_col, args.L, args.H
                )
                Xte, Yte = make_windows_univariate(
                    prep["scaled_test"], target_col, args.L, args.H
                )
        except ValueError as exc:
            raise ValueError(
                "Unable to construct training/test windows with the provided L/H settings. "
                "Consider decreasing --L/--H or ensuring the dataset has sufficient rows."
            ) from exc

    base_splits = split_windows_for_training(
        Xfull_tr,
        Yfull_tr,
        val_fraction=args.val_frac,
        eval_fraction=0.0,
        shuffle=False,
    )
    Xtr, Ytr = base_splits["X_train"], base_splits["Y_train"]
    Xval, Yval = base_splits["X_val"], base_splits["Y_val"]

    training_series_scaled = prep["scaled_train"][target_col].to_numpy(dtype=np.float32)
    training_series_orig = prep["train_df"][target_col].to_numpy(dtype=np.float32)
    horizon = int(Ytr.shape[1])

    # Determine device string strictly
    if args.require_cuda or torch.cuda.is_available():
        device_str = f"cuda:{args.gpu_id}"
    else:
        device_str = "cpu"

    # Construct TrainConfig
    base_config = TrainConfig(
        L=args.L,
        H=args.H,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lambda_reg=args.lambda_reg,
        units_g=args.units_g,
        units_d=args.units_d,
        lr_g=args.lrG,
        lr_d=args.lrD,
        label_smooth=args.label_smooth,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        patience=args.patience,
        g_layers=args.g_layers,
        d_layers=args.d_layers,
        g_dense_activation=args.g_dense_activation if args.g_dense_activation else None,
        d_activation=args.d_activation if args.d_activation else None,
        amp=args.amp,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        device=device_str,
        gan_variant=args.gan_variant,
        d_steps=args.d_steps,
        g_steps=args.g_steps,
        supervised_warmup_epochs=args.supervised_warmup_epochs,
        lambda_reg_start=args.lambda_reg_start,
        lambda_reg_end=args.lambda_reg_end,
        lambda_reg_warmup_epochs=args.lambda_reg_warmup_epochs,
        adv_weight=args.adv_weight,
        instance_noise_std=args.instance_noise_std,
        instance_noise_decay=args.instance_noise_decay,
        wgan_gp_lambda=args.wgan_gp_lambda,
        ema_decay=args.ema_decay,
        use_logits=args.use_logits,
        track_discriminator_outputs=True,
        strict_device=args.require_cuda,
        wgan_clip_value=args.wgan_clip_value,
    )

    print_kv_table(console, "Configuration", {
        "L/H": f"{args.L}/{args.H}",
        "Epochs": args.epochs,
        "Batch Size": args.batch_size,
        "Eval Batch Size": args.eval_batch_size,
        "Units (G/D)": f"{args.units_g}/{args.units_d}",
        "GAN Variant": args.gan_variant,
        "D/G Steps": f"{args.d_steps}/{args.g_steps}",
        "Lambda": args.lambda_reg,
        "Learning Rates (G/D)": f"{args.lrG}/{args.lrD}",
        "Dropout": args.dropout,
        "Layers (G/D)": f"{args.g_layers}/{args.d_layers}",
    })

    used_tune = ""
    best_hp = None
    if args.tune:
        with log_phase(console, "Hyperparameter tuning sweep"):
            if args.tune_csv:
                df_tune, target_tune, _ = load_csv_series(args.tune_csv, args.target, args.time_col, args.resample, args.agg)
                prep_t = interpolate_and_standardize(df_tune, target_tune, train_ratio=args.train_ratio)
                if prep_t["covariates"]:
                    Xtune, Ytune = make_windows_with_covariates(prep_t["scaled_train"], target_tune, prep_t["covariates"], args.L, args.H)
                else:
                    Xtune, Ytune = make_windows_univariate(prep_t["scaled_train"], target_tune, args.L, args.H)
                used_tune = args.tune_csv
            else:
                Xtune, Ytune = Xfull_tr, Yfull_tr
                used_tune = args.csv
            tune_splits = split_windows_for_training(
                Xtune,
                Ytune,
                val_fraction=args.val_frac,
                eval_fraction=args.tune_eval_frac,
                shuffle=False,
            )

            lr_candidates_g = sorted({max(1e-5, args.lrG * factor) for factor in (0.5, 1.0, 1.5)})
            lr_candidates_g.append(max(1e-5, args.lrG))
            lr_candidates_g = sorted(set(lr_candidates_g))
            lr_candidates_d = sorted({max(1e-5, args.lrD * factor) for factor in (0.5, 1.0, 1.5)})
            lr_candidates_d.append(max(1e-5, args.lrD))
            lr_candidates_d = sorted(set(lr_candidates_d))

            dropout_candidates = sorted(
                {round(val, 3) for val in (0.0, args.dropout, min(args.dropout + 0.1, 0.3))}
            )
            lambda_candidates = sorted({0.05, 0.1, 0.2, args.lambda_reg})
            layer_choices = sorted({1, args.g_layers, max(1, args.g_layers - 1), args.g_layers + 1})
            unit_choices_g = sorted({32, 64, 128, args.units_g})
            unit_choices_d = sorted({32, 64, 128, args.units_d})

            dense_candidates = [None]
            if args.g_dense_activation:
                dense_candidates.append(args.g_dense_activation)
            dense_candidates.append("relu")
            dense_candidates = list(dict.fromkeys(dense_candidates))

            disc_act_candidates = [None]
            if args.d_activation:
                disc_act_candidates.append(args.d_activation)
            disc_act_candidates.extend(["sigmoid", "tanh"])
            disc_act_candidates = list(dict.fromkeys(disc_act_candidates))

            hp_grid = {
                "units_g": unit_choices_g,
                "units_d": unit_choices_d,
                "lambda_reg": lambda_candidates,
                "dropout": dropout_candidates,
                "g_layers": layer_choices,
                "d_layers": layer_choices,
                "lr_g": lr_candidates_g, # Changed to match TrainConfig field name
                "lr_d": lr_candidates_d, # Changed to match TrainConfig field name
                "g_dense_activation": dense_candidates,
                "d_activation": disc_act_candidates,
                "epochs_each": [30, 45],
            }

            # Note: tune_rgan expects a dict for base_config to update it easily,
            # but we refactored it to take a dict and convert internally.
            # So we pass a dict representation of base_config.
            # Actually, tune_rgan signature is: base_config: Dict[str, Any]
            # So we should pass base_config.__dict__ or similar.
            # But wait, base_config is a dataclass instance now.
            # I should pass dataclasses.asdict(base_config)
            import dataclasses

            best_hp, df_tune_res = tune_rgan(
                hp_grid,
                dataclasses.asdict(base_config),
                tune_splits,
                str(results_dir),
                seed=args.seed,
            )
            df_tune_res.to_csv(results_dir/"tuning_results.csv", index=False)

            # Update base_config with best_hp
            # Since base_config is a dataclass, we can use replace
            # But best_hp is a dict.
            # We need to filter keys again or just iterate.
            valid_fields = {f.name for f in dataclasses.fields(base_config)}
            updates = {}
            for key, value in best_hp.items():
                if key in {"val_rmse", "seed"}:
                    continue
                if value is None:
                    continue
                if key in valid_fields:
                    updates[key] = value

            base_config = dataclasses.replace(base_config, **updates)

    g_dense_act = base_config.g_dense_activation

    layer_norm = True
    use_spectral_norm = True if base_config.gan_variant.lower() == "wgan-gp" else False

    console.rule("Build Models")
    G = build_generator_backend(
        L=base_config.L,
        H=base_config.H,
        n_in=Xtr.shape[-1],
        units=base_config.units_g,
        dropout=base_config.dropout,
        num_layers=base_config.g_layers,
        dense_activation=g_dense_act,
        layer_norm=layer_norm,
    )
    D = build_discriminator_backend(
        L=base_config.L,
        H=base_config.H,
        units=base_config.units_d,
        dropout=base_config.dropout,
        num_layers=base_config.d_layers,
        activation=base_config.d_activation,
        layer_norm=layer_norm,
        use_spectral_norm=use_spectral_norm,
    )

    with log_phase(console, "Train R-GAN (PyTorch)"):
        rgan_out = train_rgan_backend(
            base_config,
            (G,D),
            {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte},
            str(results_dir),
            tag="rgan"
        )

    with log_phase(console, "Train supervised LSTM baseline"):
        lstm_out = train_lstm_backend(
            base_config,
            {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte},
            str(results_dir),
            tag="lstm"
        )

    rgan_curve_artifacts = plot_single_train_test(
        rgan_out["history"]["epoch"],
        rgan_out["history"]["train_rmse"],
        rgan_out["history"]["test_rmse"],
        "R-GAN: Error vs Epochs",
        results_dir / "rgan_train_test_rmse_vs_epochs.png",
    )
    lstm_curve_artifacts = plot_single_train_test(
        lstm_out["history"]["epoch"],
        lstm_out["history"]["train_rmse"],
        lstm_out["history"]["test_rmse"],
        "LSTM (Supervised): Error vs Epochs",
        results_dir / "lstm_train_test_rmse_vs_epochs.png",
    )

    def _blank_stats() -> Dict[str, float]:
        nan_keys = [
            "rmse",
            "mae",
            "mse",
            "bias",
            "smape",
            "maape",
            "mase",
            "rmse_std",
            "rmse_ci_low",
            "rmse_ci_high",
            "mae_std",
            "mae_ci_low",
            "mae_ci_high",
            "rmse_orig",
            "mae_orig",
            "mse_orig",
            "bias_orig",
            "smape_orig",
            "maape_orig",
            "mase_orig",
            "rmse_orig_std",
            "rmse_orig_ci_low",
            "rmse_orig_ci_high",
            "mae_orig_std",
            "mae_orig_ci_low",
            "mae_orig_ci_high",
        ]
        return {k: float("nan") for k in nan_keys}

    _, naive_train_pred = naive_baseline(Xtr, Ytr)
    _, naive_test_pred = naive_baseline(Xte, Yte)
    naive_train_stats = summarise_with_uncertainty(
        Ytr,
        naive_train_pred,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        original_mean=target_mean,
        original_std=target_std,
        training_series=training_series_scaled,
        training_series_orig=training_series_orig,
    )
    naive_test_stats = summarise_with_uncertainty(
        Yte,
        naive_test_pred,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
        original_mean=target_mean,
        original_std=target_std,
        training_series=training_series_scaled,
        training_series_orig=training_series_orig,
    )
    naive_curve_artifacts = plot_constant_train_test(
        naive_train_stats["rmse"],
        naive_test_stats["rmse"],
        "Naïve Baseline: Error vs Epochs",
        results_dir / "naive_train_test_rmse_vs_epochs.png",
    )

    if args.skip_classical:
        arima_train_stats = _blank_stats()
        arima_test_stats = _blank_stats()
        arima_curve_artifacts = {"static": "", "interactive": ""}
        arma_train_stats = _blank_stats()
        arma_test_stats = _blank_stats()
        arma_curve_artifacts = {"static": "", "interactive": ""}
        tree_train_stats = _blank_stats()
        tree_test_stats = _blank_stats()
        tree_curve_artifacts = {"static": "", "interactive": ""}
        tree_config = {}
    else:
        with log_phase(console, "Train classical baselines (ARIMA/ARMA/Tree ensemble)"):
            # ARIMA implementation
            _, arima_train_pred = arima_forecast(Xtr, Ytr)
            arima_train_stats = summarise_with_uncertainty(
                Ytr,
                arima_train_pred,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
                original_mean=target_mean,
                original_std=target_std,
                training_series=training_series_scaled,
                training_series_orig=training_series_orig,
            )
            _, arima_test_pred = arima_forecast(Xtr, Ytr, Xte, Yte)
            arima_test_stats = summarise_with_uncertainty(
                Yte,
                arima_test_pred,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed + 1,
                original_mean=target_mean,
                original_std=target_std,
                training_series=training_series_scaled,
                training_series_orig=training_series_orig,
            )
            arima_curve_artifacts = plot_constant_train_test(
                arima_train_stats["rmse"],
                arima_test_stats["rmse"],
                "ARIMA: Error vs Epochs",
                results_dir / "arima_train_test_rmse_vs_epochs.png",
            )

            # ARMA implementation
            _, arma_train_pred = arma_forecast(Xtr, Ytr)
            arma_train_stats = summarise_with_uncertainty(
                Ytr,
                arma_train_pred,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
                original_mean=target_mean,
                original_std=target_std,
                training_series=training_series_scaled,
                training_series_orig=training_series_orig,
            )
            _, arma_test_pred = arma_forecast(Xtr, Ytr, Xte, Yte)
            arma_test_stats = summarise_with_uncertainty(
                Yte,
                arma_test_pred,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed + 1,
                original_mean=target_mean,
                original_std=target_std,
                training_series=training_series_scaled,
                training_series_orig=training_series_orig,
            )
            arma_curve_artifacts = plot_constant_train_test(
                arma_train_stats["rmse"],
                arma_test_stats["rmse"],
                "ARMA: Error vs Epochs",
                results_dir / "arma_train_test_rmse_vs_epochs.png",
            )

            _, tree_train_pred, tree_model = tree_ensemble_forecast(
                Xtr,
                Ytr,
                random_state=args.seed,
                return_model=True,
            )
            tree_train_pred = tree_train_pred.astype(np.float32, copy=False)
            tree_train_stats = summarise_with_uncertainty(
                Ytr,
                tree_train_pred,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed + 2,
                original_mean=target_mean,
                original_std=target_std,
                training_series=training_series_scaled,
                training_series_orig=training_series_orig,
            )
            tree_test_pred = tree_model.predict(Xte.reshape(Xte.shape[0], -1)).astype(np.float32)
            tree_test_pred = tree_test_pred.reshape(Xte.shape[0], horizon, 1)
            tree_test_stats = summarise_with_uncertainty(
                Yte,
                tree_test_pred,
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed + 3,
                original_mean=target_mean,
                original_std=target_std,
                training_series=training_series_scaled,
                training_series_orig=training_series_orig,
            )
            tree_curve_artifacts = plot_constant_train_test(
                tree_train_stats["rmse"],
                tree_test_stats["rmse"],
                "Tree Ensemble: Error vs Epochs",
                results_dir / "tree_ensemble_train_test_rmse_vs_epochs.png",
            )
            tree_config = {
                "estimator": "GradientBoostingRegressor",
                "loss": "squared_error",
                "learning_rate": 0.05,
                "n_estimators": 400,
                "subsample": 0.7,
                "max_depth": 3,
                "random_state": args.seed,
            }

    learning_sizes, learning_curve_values, learning_curve_stds = compute_learning_curves(
        args, base_config, Xfull_tr, Yfull_tr, Xte, Yte, Xtr.shape[-1]
    )
    learning_curve_artifacts = None
    if learning_sizes:
        learning_curve_artifacts = plot_learning_curves(
            learning_sizes,
            learning_curve_values,
            results_dir / "ml_error_vs_samples.png",
            curve_stds=learning_curve_stds,
        )

    _, rgan_train_pred = compute_metrics_backend(rgan_out["G"], Xtr, Ytr)
    rgan_train_stats = summarise_with_uncertainty(
        Ytr,
        rgan_train_pred,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        original_mean=target_mean,
        original_std=target_std,
        training_series=training_series_scaled,
        training_series_orig=training_series_orig,
    )
    rgan_test_stats = summarise_with_uncertainty(
        Yte,
        rgan_out["pred_test"],
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
        original_mean=target_mean,
        original_std=target_std,
        training_series=training_series_scaled,
        training_series_orig=training_series_orig,
    )
    rgan_out["train_stats"] = rgan_train_stats
    rgan_out["test_stats"] = rgan_test_stats

    lstm_train_stats = summarise_with_uncertainty(
        Ytr,
        lstm_out["pred_train"],
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        original_mean=target_mean,
        original_std=target_std,
        training_series=training_series_scaled,
        training_series_orig=training_series_orig,
    )
    lstm_test_stats = summarise_with_uncertainty(
        Yte,
        lstm_out["pred_test"],
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
        original_mean=target_mean,
        original_std=target_std,
        training_series=training_series_scaled,
        training_series_orig=training_series_orig,
    )
    lstm_out["train_stats"] = lstm_train_stats
    lstm_out["test_stats"] = lstm_test_stats

    if args.skip_classical:
        class_curve_artifacts = None
        ets_rmse_full = float("nan")
        arima_rmse_full = float("nan")
    else:
        sizes, ets_curve, arima_curve = classical_curves_vs_samples(prep["train_df"][target_col].values, prep["test_df"][target_col].values, min_frac=0.3, steps=6)
        class_curve_artifacts = (
            plot_classical_curves(sizes, ets_curve, arima_curve, results_dir / "classical_error_vs_samples.png")
            if sizes is not None
            else None
        )
        ets_rmse_full = float(np.nan if ets_curve is None else ets_curve[-1])
        arima_rmse_full = float(np.nan if arima_curve is None else arima_curve[-1])

    # Noise robustness across multiple perturbation levels
    noise_results = []
    rng_noise = np.random.default_rng(args.seed + 2048)
    for idx, sd in enumerate(noise_levels):
        if sd == 0.0:
            noise_results.append(
                {
                    "sd": float(sd),
                    "rgan": rgan_test_stats,
                    "lstm": lstm_test_stats,
                    "naive_baseline": naive_test_stats,
                    "arima": arima_test_stats,
                    "arma": arma_test_stats,
                    "tree_ensemble": tree_test_stats,
                }
            )
            continue

        perturbation = rng_noise.normal(0, sd, size=Xte.shape).astype(Xte.dtype)
        Xte_noisy = Xte + perturbation

        _, rgan_noise_pred = compute_metrics_backend(rgan_out["G"], Xte_noisy, Yte)
        rgan_noise_summary = summarise_with_uncertainty(
            Yte,
            rgan_noise_pred,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + idx,
            original_mean=target_mean,
            original_std=target_std,
            training_series=training_series_scaled,
            training_series_orig=training_series_orig,
        )

        import torch

        lstm_model = lstm_out["model"]
        lstm_model.eval()
        device = next(lstm_model.parameters()).device
        with torch.no_grad():
            lstm_pred_noisy = lstm_model(torch.from_numpy(Xte_noisy).to(device)).cpu().numpy()
        lstm_noise_summary = summarise_with_uncertainty(
            Yte,
            lstm_pred_noisy,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + idx,
            original_mean=target_mean,
            original_std=target_std,
            training_series=training_series_scaled,
            training_series_orig=training_series_orig,
        )

        _, naive_noisy_pred = naive_baseline(Xte_noisy, Yte)
        naive_noise_summary = summarise_with_uncertainty(
            Yte,
            naive_noisy_pred,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + idx,
            original_mean=target_mean,
            original_std=target_std,
            training_series=training_series_scaled,
            training_series_orig=training_series_orig,
        )

        _, arima_noisy_pred = arima_forecast(Xtr, Ytr, Xte_noisy, Yte)
        arima_noise_summary = summarise_with_uncertainty(
            Yte,
            arima_noisy_pred,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + idx,
            original_mean=target_mean,
            original_std=target_std,
            training_series=training_series_scaled,
            training_series_orig=training_series_orig,
        )

        _, arma_noisy_pred = arma_forecast(Xtr, Ytr, Xte_noisy, Yte)
        arma_noise_summary = summarise_with_uncertainty(
            Yte,
            arma_noisy_pred,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + idx,
            original_mean=target_mean,
            original_std=target_std,
            training_series=training_series_scaled,
            training_series_orig=training_series_orig,
        )

        tree_noisy_flat = tree_model.predict(Xte_noisy.reshape(Xte_noisy.shape[0], -1)).astype(np.float32)
        tree_noisy_pred = tree_noisy_flat.reshape(Xte_noisy.shape[0], horizon, 1)
        tree_noise_summary = summarise_with_uncertainty(
            Yte,
            tree_noisy_pred,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + idx,
            original_mean=target_mean,
            original_std=target_std,
            training_series=training_series_scaled,
            training_series_orig=training_series_orig,
        )

        noise_results.append(
            {
                "sd": float(sd),
                "rgan": rgan_noise_summary,
                "lstm": lstm_noise_summary,
                "naive_baseline": naive_noise_summary,
                "arima": arima_noise_summary,
                "arma": arma_noise_summary,
                "tree_ensemble": tree_noise_summary,
            }
        )

    test_errors = {
        "R-GAN": rgan_out["test_stats"].get("rmse_orig", rgan_out["test_stats"]["rmse"]),
        "LSTM": lstm_out["test_stats"].get("rmse_orig", lstm_out["test_stats"]["rmse"]),
        "Tree Ensemble": tree_test_stats.get("rmse_orig", tree_test_stats["rmse"]),
        "Naïve Baseline": naive_test_stats.get("rmse_orig", naive_test_stats["rmse"]),
        "ARIMA": arima_test_stats.get("rmse_orig", arima_test_stats["rmse"]),
        "ARMA": arma_test_stats.get("rmse_orig", arma_test_stats["rmse"]),
    }
    train_errors = {
        "R-GAN": rgan_out["train_stats"].get("rmse_orig", rgan_out["train_stats"]["rmse"]),
        "LSTM": lstm_out["train_stats"].get("rmse_orig", lstm_out["train_stats"]["rmse"]),
        "Tree Ensemble": tree_train_stats.get("rmse_orig", tree_train_stats["rmse"]),
        "Naïve Baseline": naive_train_stats.get("rmse_orig", naive_train_stats["rmse"]),
        "ARIMA": arima_train_stats.get("rmse_orig", arima_train_stats["rmse"]),
        "ARMA": arma_train_stats.get("rmse_orig", arma_train_stats["rmse"]),
    }
    model_compare_artifacts = plot_compare_models_bars(
        train_errors,
        test_errors,
        results_dir / "models_test_error.png",
        results_dir / "models_train_error.png",
    )

    # Create comprehensive error metrics table
    model_results = {
        "R-GAN": {
            "train": rgan_out["train_stats"],
            "test": rgan_out["test_stats"]
        },
        "LSTM": {
            "train": lstm_out["train_stats"],
            "test": lstm_out["test_stats"]
        },
        "Tree Ensemble": {
            "train": tree_train_stats,
            "test": tree_test_stats
        },
        "Naïve Baseline": {
            "train": naive_train_stats,
            "test": naive_test_stats
        },
        "ARIMA": {
            "train": arima_train_stats,
            "test": arima_test_stats
        },
        "ARMA": {
            "train": arma_train_stats,
            "test": arma_test_stats
        }
    }
    
    # Generate and save error metrics table
    metrics_table = create_error_metrics_table(model_results, results_dir/"error_metrics_table.csv")
    print("\n" + "="*80)
    print("ERROR METRICS TABLE (RMSE, MSE, BIAS, MAE)")
    print("="*80)
    print(metrics_table.to_string(index=False))
    print("="*80)

    def _safe_float(value):
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(val):
            return None
        return val

    def _fmt_metric(value):
        val = _safe_float(value)
        return "nan" if val is None else f"{val:.6f}"

    print("\n" + "="*80)
    print("ADVANCED METRICS (sMAPE / MAAPE / MASE)")
    print("="*80)
    for model_name, results in model_results.items():
        for split_name in ("train", "test"):
            stats = results.get(split_name)
            if not stats:
                continue
            smape_val = stats.get("smape_orig", stats.get("smape"))
            maape_val = stats.get("maape_orig", stats.get("maape"))
            mase_val = stats.get("mase_orig", stats.get("mase"))
            print(
                f"{model_name:<20} {split_name.title():<5} | "
                f"sMAPE: {_fmt_metric(smape_val)}  "
                f"MAAPE: {_fmt_metric(maape_val)}  "
                f"MASE: {_fmt_metric(mase_val)}"
            )
    print("="*80)

    predictions_test = {
        "R-GAN": rgan_out["pred_test"],
        "LSTM": lstm_out["pred_test"],
        "Tree Ensemble": tree_test_pred,
        "Naïve Baseline": naive_test_pred,
        "ARIMA": arima_test_pred,
        "ARMA": arma_test_pred,
    }
    actual_test_orig_flat = (Yte.reshape(-1) * target_std) + target_mean
    preds_orig_flat = {
        name: (np.asarray(pred).reshape(-1) * target_std) + target_mean for name, pred in predictions_test.items()
    }

    dm_results = {}
    for model_a, model_b in combinations(predictions_test.keys(), 2):
        dm_stat = diebold_mariano(
            actual_test_orig_flat,
            preds_orig_flat[model_a],
            preds_orig_flat[model_b],
        )
        dm_results[f"{model_a} vs {model_b}"] = {
            "stat": _safe_float(dm_stat.get("stat")),
            "pvalue": _safe_float(dm_stat.get("pvalue")),
        }

    print("\n" + "="*80)
    print("DIEBOLD-MARIANO TESTS (ORIGINAL SCALE)")
    print("="*80)
    for pair, res in dm_results.items():
        stat_str = _fmt_metric(res.get("stat"))
        pval_str = _fmt_metric(res.get("pvalue"))
        print(f"{pair:<40} -> stat={stat_str}  pvalue={pval_str}")
    print("="*80)

    def _capture_metrics(stats: Dict[str, float]):
        if not stats:
            return {}
        return {
            "smape": _safe_float(stats.get("smape")),
            "maape": _safe_float(stats.get("maape")),
            "mase": _safe_float(stats.get("mase")),
            "smape_orig": _safe_float(stats.get("smape_orig")),
            "maape_orig": _safe_float(stats.get("maape_orig")),
            "mase_orig": _safe_float(stats.get("mase_orig")),
        }

    advanced_metrics_summary = {
        model_name: {
            split_name: _capture_metrics(stats)
            for split_name, stats in results.items()
            if stats
        }
        for model_name, results in model_results.items()
    }

    rgan_architecture = {
        "generator": describe_model(G),
        "discriminator": describe_model(D),
    }
    lstm_architecture = describe_model(lstm_out["model"])

    learning_curves_serializable = {
        "means": {k: [float(v) for v in vals] for k, vals in learning_curve_values.items()},
        "stds": {
            k: [float(v) for v in learning_curve_stds.get(k, [0.0] * len(learning_curve_values.get(k, [])))]
            for k in learning_curve_values
        },
    }

    packages = {}
    for pkg in ("numpy", "pandas", "torch", "scikit-learn"):
        module_name = pkg.replace("-", "_")
        try:
            module = __import__(module_name)
            packages[pkg] = getattr(module, "__version__", "unknown")
        except ModuleNotFoundError:
            continue

    git_commit = ""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
        ).strip()
    except Exception:
        git_commit = ""

    env_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_commit,
        "seed": args.seed,
    }

    metrics = dict(
        dataset=args.csv,
        tuning_dataset=used_tune,
        time_col_used=time_used,
        target_col=target_col,
        L=args.L,
        H=args.H,
        train_size=int(prep["split"]),
        test_size=len(prep["test_df"]),
        num_train_windows=int(Xfull_tr.shape[0]),
        num_test_windows=int(Xte.shape[0]),
        noise_robustness=noise_results,
        rgan=dict(
            train=rgan_out["train_stats"],
            test=rgan_out["test_stats"],
            curve=rgan_curve_artifacts["static"],
            curve_interactive=rgan_curve_artifacts.get("interactive", ""),
            history=rgan_out["history"],
            architecture=rgan_architecture,
            config=dict(
                units_g=base_config.units_g,
                units_d=base_config.units_d,
                lambda_reg=base_config.lambda_reg,
                lrG=base_config.lr_g,
                lrD=base_config.lr_d,
                dropout=base_config.dropout,
                g_layers=base_config.g_layers,
                d_layers=base_config.d_layers,
                g_dense=(g_dense_act if g_dense_act else "linear"),
                d_activation=base_config.d_activation or "sigmoid",
                gan_variant=base_config.gan_variant,
                d_steps=base_config.d_steps,
                g_steps=base_config.g_steps,
                wgan_gp_lambda=base_config.wgan_gp_lambda,
                wgan_clip_value=base_config.wgan_clip_value,
                use_logits=base_config.use_logits,
                amp=base_config.amp,
                device=base_config.device,
            ),
        ),
        lstm=dict(
            train=lstm_out["train_stats"],
            test=lstm_out["test_stats"],
            curve=lstm_curve_artifacts["static"],
            curve_interactive=lstm_curve_artifacts.get("interactive", ""),
            history=lstm_out["history"],
            architecture=lstm_architecture,
            config=dict(units=base_config.units_g, lr=base_config.lr_g, dropout=base_config.dropout),
        ),
        naive_baseline=dict(
            train=naive_train_stats,
            test=naive_test_stats,
            curve=naive_curve_artifacts["static"],
            curve_interactive=naive_curve_artifacts.get("interactive", ""),
        ),
        arima=dict(
            train=arima_train_stats,
            test=arima_test_stats,
            curve=arima_curve_artifacts["static"],
            curve_interactive=arima_curve_artifacts.get("interactive", ""),
        ),
        arma=dict(
            train=arma_train_stats,
            test=arma_test_stats,
            curve=arma_curve_artifacts["static"],
            curve_interactive=arma_curve_artifacts.get("interactive", ""),
        ),
        tree_ensemble=dict(
            train=tree_train_stats,
            test=tree_test_stats,
            curve=tree_curve_artifacts["static"],
            curve_interactive=tree_curve_artifacts.get("interactive", ""),
            config=tree_config,
        ),
        compare_plots=dict(
            test=model_compare_artifacts["test"]["static"],
            test_interactive=model_compare_artifacts["test"].get("interactive", ""),
            train=model_compare_artifacts["train"]["static"],
            train_interactive=model_compare_artifacts["train"].get("interactive", ""),
            naive_comparison="",
            naive_comparison_interactive="",
        ),
        classical=dict(
            ets_rmse_full=ets_rmse_full,
            arima_rmse_full=arima_rmse_full,
            curves=class_curve_artifacts["static"] if class_curve_artifacts else "",
            curves_interactive=class_curve_artifacts.get("interactive", "") if class_curve_artifacts else "",
        ),
        learning_curves=dict(
            sizes=[int(s) for s in learning_sizes],
            curves=learning_curves_serializable,
            plot=learning_curve_artifacts["static"] if learning_curve_artifacts else "",
            plot_interactive=learning_curve_artifacts.get("interactive", "") if learning_curve_artifacts else "",
        ),
        advanced_metrics=advanced_metrics_summary,
        statistical_tests=dict(diebold_mariano=dm_results),
        tuning=dict(
            enabled=bool(args.tune),
            dataset=used_tune,
            best=best_hp or {},
        ),
        environment=env_info,
        scaling={"target_mean": float(target_mean), "target_std": float(target_std)},
        created=datetime.utcnow().isoformat(),
    )
    # Sanitize metrics to ensure valid JSON (replace NaN/Infinity with None)
    def sanitize_for_json(obj):
        import math
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return obj

    metrics = sanitize_for_json(metrics)

    with open(results_dir/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Auto-update dashboard - REMOVED (Decoupled)
    # The dashboard is now a standalone viewer that loads metrics.json manually.
    console.print(f"[green]Training complete. Results saved to {results_dir}[/green]")
    console.print(f"[dim]To view results, open the dashboard and load: {results_dir/'metrics.json'}[/dim]")

if __name__ == "__main__":
    # make src importable if run directly
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
    main()
