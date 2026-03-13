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
import torch

from rgan.baselines import (
    naive_baseline,
    arima_forecast,
    arma_forecast,
    tree_ensemble_forecast,
)
from rgan.data import (
    load_csv_series,
    interpolate_and_standardize,
    make_windows_univariate,
    make_windows_with_covariates,
    DataSplit,
)
from rgan.plots import (
    plot_single_train_test,
    plot_constant_train_test,
    plot_compare_models_bars,
    plot_predictions,
    create_error_metrics_table,
    create_noise_robustness_table,
    plot_noise_robustness,
)
from rgan.tune import tune_rgan
from rgan.logging_utils import (
    get_console, print_banner, print_kv_table,
    log_info, log_error, log_step, log_var, log_shape, log_debug,
    log_exception, log_warn, setup_log_file, close_log_file,
)
from rgan.metrics import (
    describe_model,
    error_stats,
    summarise_with_uncertainty,
    diebold_mariano,
)
from rgan.config import TrainConfig, ModelConfig


@contextmanager
def log_phase(console, title: str):
    """Timed phase logger — logs start, end, and elapsed time with caller location."""
    log_info(f"PHASE START: {title}")
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        log_error(f"PHASE FAILED: {title} after {elapsed:.1f}s — {type(exc).__name__}: {exc}")
        raise
    else:
        elapsed = time.perf_counter() - start
        log_info(f"PHASE DONE: {title} ({elapsed:.1f}s)")


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
        console.print("Resource-heavy components enabled:")
        for item in heavy_steps:
            console.print(f" * {item}")
    else:
        console.print("All optional heavy components are disabled for this run.")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # benchmark=True auto-tunes kernels for fixed input shapes — significant
            # speedup for LSTM training where every batch has the same dimensions.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="auto")
    ap.add_argument("--time_col", default="auto")
    ap.add_argument("--resample", default="")
    ap.add_argument("--agg", default="last")
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_train_windows", type=int, default=0,
                    help="Limit training windows (0 = use all). Use e.g. 500 for fast smoke tests.")
    ap.add_argument("--lambda_reg", type=float, default=0.5)
    ap.add_argument(
        "--gan_variant",
        choices=["standard", "wgan", "wgan-gp"],
        default="wgan-gp",
        help="Adversarial loss variant (standard BCE or Wasserstein with GP).",
    )
    ap.add_argument("--d_steps", type=int, default=3, help="Number of discriminator updates per batch.")
    ap.add_argument("--g_steps", type=int, default=1, help="Number of generator updates per batch.")
    ap.add_argument("--units_g", type=int, default=128)
    ap.add_argument("--units_d", type=int, default=128)
    ap.add_argument("--g_layers", type=int, default=2)
    ap.add_argument("--d_layers", type=int, default=2)
    ap.add_argument("--g_dense_activation", default="", help="Optional dense activation for the generator (PyTorch activation name).")
    ap.add_argument("--d_activation", default="sigmoid", help="Discriminator activation (PyTorch activation name).")
    ap.add_argument("--lrG", type=float, default=5e-4)
    ap.add_argument("--lrD", type=float, default=5e-4)
    ap.add_argument("--label_smooth", type=float, default=0.9)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument(
        "--wgan_clip_value",
        type=float,
        default=0.01,
        help="Weight-clipping magnitude for vanilla WGAN training.",
    )
    ap.add_argument(
        "--supervised_warmup_epochs",
        type=int,
        default=10,
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
        default=0.999,
        help="EMA decay for generator weights (0 disables EMA).",
    )
    ap.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="L2 weight decay for Adam optimizers (0 disables).",
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
    # Torch runtime knobs
    ap.add_argument("--amp", type=lambda v: str(v).lower() in ["1","true","yes"], default=True,
                    help="Enable automatic mixed precision for PyTorch training")
    ap.add_argument("--eval_batch_size", type=int, default=2048,
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
        default="0,0.01,0.05,0.1,0.2",
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
            "Skip slow classical baselines (ARIMA/ARMA/tree ensemble). Useful for quick "
            "smoke tests or when you only care about the neural models."
        ),
    )
    ap.add_argument(
        "--skip_noise_robustness",
        action="store_true",
        help=(
            "Skip multi-level noise robustness evaluation. Saves significant time on cloud "
            "runs. Clean-data metrics and single-noise-level results are still computed."
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
    ap.add_argument("--results_dir", default="./results/experiment")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use (default: 0).")
    # Checkpoint & Resume
    ap.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Directory for saving training checkpoints. Disabled when not set.",
    )
    ap.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (0 = disabled). Requires --checkpoint_dir.",
    )
    ap.add_argument(
        "--resume_from",
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    ap.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Evaluate every N epochs (1 = every epoch, 5 = every 5th). Reduces overhead on large datasets.",
    )
    args = ap.parse_args()

    # Install a global exception hook so unhandled errors always show full traceback
    # and write to the log file
    import traceback as _tb

    def _global_exception_handler(exc_type, exc_value, exc_tb):
        header = "\n" + "=" * 80
        msg = "UNHANDLED EXCEPTION — full traceback follows"
        footer = "=" * 80
        tb_str = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
        for line in [header, msg, footer, tb_str, footer]:
            print(line, flush=True)
        # Also write to log file so it persists
        from rgan.logging_utils import _write_to_log
        for line in [header, msg, footer, tb_str, footer]:
            _write_to_log(line)
        close_log_file()
        sys.exit(1)

    sys.excepthook = _global_exception_handler

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
            console.print(f"CRITICAL ERROR: --require_cuda specified but no CUDA device found.")
            console.print(_cuda_build_hint())
            sys.exit(1)

        if args.gpu_id >= torch.cuda.device_count():
            console.print(f"CRITICAL ERROR: --gpu_id {args.gpu_id} requested but only {torch.cuda.device_count()} devices available.")
            sys.exit(1)

        torch.cuda.set_device(args.gpu_id)
        device_name = torch.cuda.get_device_name(args.gpu_id)
        console.print(f"Strict GPU Mode Active: Using device {args.gpu_id} ({device_name})")
        console.print("CPU fallback is DISABLED.")

    elif torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device_name = torch.cuda.get_device_name(args.gpu_id)
        console.print(f"GPU Detected: {device_name} (ID: {args.gpu_id})")
        console.print("Using CUDA for training. Fallback to CPU enabled if CUDA fails.")
        console.print(_cuda_build_hint())
    else:
        console.print("WARNING: No GPU detected. Falling back to CPU.")
        console.print("Training will be significantly slower.")
        console.print(_cuda_build_hint())

    # Windows-specific fix: REMOVED as per user request (running on Linux/Mac)
    # if platform.system() == "Windows" and args.num_workers > 0:
    #     console.print(f"[yellow]Windows detected: Overriding num_workers from {args.num_workers} to 0 to prevent deadlocks.[/yellow]")
    #     args.num_workers = 0
    #     args.persistent_workers = False

    if args.tune_csv and not args.tune:
        args.tune = True

    # Auto-generate unique results directory: results/<dataset>_<timestamp>/
    run_start_time = time.time()
    if args.results_dir == "./results/experiment":
        # Default path — auto-generate a unique directory per run
        dataset_stem = Path(args.csv).stem.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / f"{dataset_stem}_{timestamp}"
    else:
        results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create/update a "latest" symlink for convenience
    latest_link = results_dir.parent / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(results_dir.name)
    except OSError:
        pass  # Symlinks may fail on some systems

    console.print(f"Results directory: {results_dir.resolve()}")

    # Set up persistent log file so that everything is captured even on crash
    log_file_path = setup_log_file(str(results_dir))
    log_info(f"Log file created: {log_file_path}")
    log_info(f"Command: {' '.join(sys.argv)}")
    log_info(f"Working directory: {os.getcwd()}")
    log_info(f"Python executable: {sys.executable}")
    log_info(f"Platform: {platform.platform()}")
    log_info(f"PyTorch version: {torch.__version__}")
    log_info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_info(f"CUDA version: {torch.version.cuda}")
        log_info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log_info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    log_info(f"Args: {vars(args)}")

    # Save full run configuration for reproducibility
    run_config = {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "results_dir": str(results_dir.resolve()),
        "started_at": datetime.now().isoformat(),
    }
    with open(results_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    noise_levels = parse_noise_levels(str(args.noise_levels))

    describe_resource_heavy_steps(args, console)

    try:
        from rgan.models_torch import (
            build_generator as build_generator_backend,
            build_discriminator as build_discriminator_backend,
        )
        from rgan.rgan_torch import (
            train_rgan_torch as train_rgan_backend,
            compute_metrics as compute_metrics_backend,
        )
        from rgan.lstm_supervised_torch import (
            train_lstm_supervised_torch as train_lstm_backend,
        )
        from rgan.linear_baselines import train_linear_baseline
        from rgan.fits import train_fits
        from rgan.patchtst import train_patchtst
        from rgan.itransformer import train_itransformer
        from rgan.autoformer import train_autoformer
        from rgan.informer import train_informer
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError(
                "PyTorch is required but not installed. Please install the 'torch' package to continue."
            ) from exc
        raise

    set_seed(args.seed)

    with log_phase(console, "Load dataset and standardize"):
        log_step(f"Loading CSV from: {args.csv}")
        df, target_col, time_used = load_csv_series(args.csv, args.target, args.time_col, args.resample, args.agg)
        log_step(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        log_step(f"Columns: {list(df.columns)}")
        log_step(f"Target column resolved to: {target_col}")
        log_step(f"Time column: {time_used or '(none)'}")
        log_step(f"Interpolating and standardizing with train_ratio={args.train_ratio}")
        prep = interpolate_and_standardize(df, target_col, train_ratio=args.train_ratio)
        target_mean = prep["target_mean"]
        target_std = prep["target_std"]
        log_step(f"Target mean={target_mean:.6f}, std={target_std:.6f}")
        log_step(f"Train split index: {prep['split']}, test rows: {len(prep['test_df'])}")
        log_step(f"Covariates detected: {prep['covariates'] or '(none)'}")

        print_kv_table(console, "Dataset", {
            "CSV": args.csv,
            "Target": target_col,
            "Time Column": time_used or "(none)",
            "Rows": len(df),
            "Train/Test Split": f"{args.train_ratio:.2f}/{1-args.train_ratio:.2f}",
        })

    with log_phase(console, "Create training and test windows"):
        log_step(f"Creating windows with L={args.L}, H={args.H}")
        log_step(f"Using {'multivariate (with covariates)' if prep['covariates'] else 'univariate'} windowing")
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

    log_step(f"Full training windows: X={Xfull_tr.shape}, Y={Yfull_tr.shape}")
    log_step(f"Test windows: X={Xte.shape}, Y={Yte.shape}")

    log_step(f"Splitting training windows: val_fraction={args.val_frac}")
    base_splits = split_windows_for_training(
        Xfull_tr,
        Yfull_tr,
        val_fraction=args.val_frac,
        eval_fraction=0.0,
        shuffle=False,
    )
    Xtr, Ytr = base_splits["X_train"], base_splits["Y_train"]
    Xval, Yval = base_splits["X_val"], base_splits["Y_val"]
    log_step(f"Train: X={Xtr.shape}, Y={Ytr.shape}")
    log_step(f"Val: X={Xval.shape}, Y={Yval.shape}")
    log_step(f"Test: X={Xte.shape}, Y={Yte.shape}")

    # Subsample training windows for fast smoke tests
    if args.max_train_windows > 0 and len(Xtr) > args.max_train_windows:
        rng_sub = np.random.default_rng(args.seed)
        idx = rng_sub.choice(len(Xtr), args.max_train_windows, replace=False)
        Xtr, Ytr = Xtr[idx], Ytr[idx]
        idx_val = rng_sub.choice(len(Xval), min(len(Xval), args.max_train_windows // 5), replace=False)
        Xval, Yval = Xval[idx_val], Yval[idx_val]
        idx_te = rng_sub.choice(len(Xte), min(len(Xte), args.max_train_windows // 2), replace=False)
        Xte, Yte = Xte[idx_te], Yte[idx_te]
        console.log(f"Subsampled to {len(Xtr)} train / {len(Xval)} val / {len(Xte)} test windows")

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
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        eval_every=args.eval_every,
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
    log_step(f"Building Generator: L={base_config.L}, H={base_config.H}, n_in={Xtr.shape[-1]}, units={base_config.units_g}, layers={base_config.g_layers}, dropout={base_config.dropout}, dense_act={g_dense_act}, layer_norm={layer_norm}")
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
    log_step(f"Generator built: {sum(p.numel() for p in G.parameters())} parameters")
    log_step(f"Building Discriminator: units={base_config.units_d}, layers={base_config.d_layers}, activation={base_config.d_activation}, spectral_norm={use_spectral_norm}")
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

    log_step(f"Discriminator built: {sum(p.numel() for p in D.parameters())} parameters")

    # Save all model weights to a models/ subdirectory
    models_dir = results_dir / "models"
    models_dir.mkdir(exist_ok=True)
    log_step(f"Models directory: {models_dir.resolve()}")

    def _save_model(name: str, model_obj):
        """Save a PyTorch model's state_dict to models/<name>.pt"""
        path = models_dir / f"{name}.pt"
        torch.save(model_obj.state_dict(), path)
        return path

    log_step(f"Device for training: {device_str}")
    log_step(f"AMP enabled: {base_config.amp}")
    log_step(f"GAN variant: {base_config.gan_variant}")
    log_step(f"Checkpoint dir: {base_config.checkpoint_dir or '(none)'}")
    log_step(f"Resume from: {base_config.resume_from or '(none)'}")

    # ── Launch classical baselines on CPU in background while GPU trains ──
    import threading

    _classical_results = {}  # filled by background thread

    def _run_classical_baselines():
        """Run CPU-only baselines (naive, ARIMA, ARMA, tree) in a background thread."""
        try:
            log_info("BACKGROUND: Starting classical baselines on CPU (parallel with GPU training)")

            # Naive baseline (always runs)
            log_step("BACKGROUND: Computing naive baseline...")
            _, naive_train_pred_bg = naive_baseline(Xtr, Ytr)
            _, naive_test_pred_bg = naive_baseline(Xte, Yte)
            naive_train_stats_bg = summarise_with_uncertainty(
                Ytr, naive_train_pred_bg,
                n_bootstrap=args.bootstrap_samples, seed=args.seed,
                original_mean=target_mean, original_std=target_std,
                training_series=training_series_scaled, training_series_orig=training_series_orig,
            )
            naive_test_stats_bg = summarise_with_uncertainty(
                Yte, naive_test_pred_bg,
                n_bootstrap=args.bootstrap_samples, seed=args.seed + 1,
                original_mean=target_mean, original_std=target_std,
                training_series=training_series_scaled, training_series_orig=training_series_orig,
            )
            naive_curve_bg = plot_constant_train_test(
                naive_train_stats_bg["rmse"], naive_test_stats_bg["rmse"],
                "Naïve Baseline: Error vs Epochs",
                results_dir / "naive_train_test_rmse_vs_epochs.png",
            )
            _classical_results["naive_train_pred"] = naive_train_pred_bg
            _classical_results["naive_test_pred"] = naive_test_pred_bg
            _classical_results["naive_train_stats"] = naive_train_stats_bg
            _classical_results["naive_test_stats"] = naive_test_stats_bg
            _classical_results["naive_curve_artifacts"] = naive_curve_bg
            log_step(f"BACKGROUND: Naive done. Test RMSE={naive_test_stats_bg.get('rmse', 'N/A')}")

            if args.skip_classical:
                _classical_results["skip_classical"] = True
                log_info("BACKGROUND: Skipping ARIMA/ARMA/Tree (--skip_classical)")
                return

            _classical_results["skip_classical"] = False

            # ARIMA
            log_step("BACKGROUND: Computing ARIMA forecast...")
            _, arima_train_pred_bg = arima_forecast(Xtr, Ytr)
            arima_train_stats_bg = summarise_with_uncertainty(
                Ytr, arima_train_pred_bg,
                n_bootstrap=args.bootstrap_samples, seed=args.seed,
                original_mean=target_mean, original_std=target_std,
                training_series=training_series_scaled, training_series_orig=training_series_orig,
            )
            _, arima_test_pred_bg = arima_forecast(Xtr, Ytr, Xte, Yte)
            arima_test_stats_bg = summarise_with_uncertainty(
                Yte, arima_test_pred_bg,
                n_bootstrap=args.bootstrap_samples, seed=args.seed + 1,
                original_mean=target_mean, original_std=target_std,
                training_series=training_series_scaled, training_series_orig=training_series_orig,
            )
            arima_curve_bg = plot_constant_train_test(
                arima_train_stats_bg["rmse"], arima_test_stats_bg["rmse"],
                "ARIMA: Error vs Epochs",
                results_dir / "arima_train_test_rmse_vs_epochs.png",
            )
            _classical_results["arima_train_stats"] = arima_train_stats_bg
            _classical_results["arima_test_stats"] = arima_test_stats_bg
            _classical_results["arima_curve_artifacts"] = arima_curve_bg
            log_step(f"BACKGROUND: ARIMA done. Test RMSE={arima_test_stats_bg.get('rmse', 'N/A')}")

            # ARMA
            log_step("BACKGROUND: Computing ARMA forecast...")
            _, arma_train_pred_bg = arma_forecast(Xtr, Ytr)
            arma_train_stats_bg = summarise_with_uncertainty(
                Ytr, arma_train_pred_bg,
                n_bootstrap=args.bootstrap_samples, seed=args.seed,
                original_mean=target_mean, original_std=target_std,
                training_series=training_series_scaled, training_series_orig=training_series_orig,
            )
            _, arma_test_pred_bg = arma_forecast(Xtr, Ytr, Xte, Yte)
            arma_test_stats_bg = summarise_with_uncertainty(
                Yte, arma_test_pred_bg,
                n_bootstrap=args.bootstrap_samples, seed=args.seed + 1,
                original_mean=target_mean, original_std=target_std,
                training_series=training_series_scaled, training_series_orig=training_series_orig,
            )
            arma_curve_bg = plot_constant_train_test(
                arma_train_stats_bg["rmse"], arma_test_stats_bg["rmse"],
                "ARMA: Error vs Epochs",
                results_dir / "arma_train_test_rmse_vs_epochs.png",
            )
            _classical_results["arma_train_stats"] = arma_train_stats_bg
            _classical_results["arma_test_stats"] = arma_test_stats_bg
            _classical_results["arma_curve_artifacts"] = arma_curve_bg
            log_step(f"BACKGROUND: ARMA done. Test RMSE={arma_test_stats_bg.get('rmse', 'N/A')}")

            # Tree Ensemble
            log_step("BACKGROUND: Computing Tree Ensemble forecast...")
            _, tree_train_pred_bg, tree_model_bg = tree_ensemble_forecast(
                Xtr, Ytr, random_state=args.seed, return_model=True,
            )
            tree_train_pred_bg = tree_train_pred_bg.astype(np.float32, copy=False)
            tree_train_stats_bg = summarise_with_uncertainty(
                Ytr, tree_train_pred_bg,
                n_bootstrap=args.bootstrap_samples, seed=args.seed + 2,
                original_mean=target_mean, original_std=target_std,
                training_series=training_series_scaled, training_series_orig=training_series_orig,
            )
            tree_test_pred_bg = tree_model_bg.predict(Xte.reshape(Xte.shape[0], -1)).astype(np.float32)
            tree_test_pred_bg = tree_test_pred_bg.reshape(Xte.shape[0], horizon, 1)
            tree_test_stats_bg = summarise_with_uncertainty(
                Yte, tree_test_pred_bg,
                n_bootstrap=args.bootstrap_samples, seed=args.seed + 3,
                original_mean=target_mean, original_std=target_std,
                training_series=training_series_scaled, training_series_orig=training_series_orig,
            )
            tree_curve_bg = plot_constant_train_test(
                tree_train_stats_bg["rmse"], tree_test_stats_bg["rmse"],
                "Tree Ensemble: Error vs Epochs",
                results_dir / "tree_ensemble_train_test_rmse_vs_epochs.png",
            )
            _classical_results["tree_train_stats"] = tree_train_stats_bg
            _classical_results["tree_test_stats"] = tree_test_stats_bg
            _classical_results["tree_curve_artifacts"] = tree_curve_bg
            _classical_results["tree_model"] = tree_model_bg
            _classical_results["tree_config"] = {
                "estimator": "GradientBoostingRegressor",
                "loss": "squared_error",
                "learning_rate": 0.05,
                "n_estimators": 400,
                "subsample": 0.7,
                "max_depth": 3,
                "random_state": args.seed,
            }
            log_step(f"BACKGROUND: Tree Ensemble done. Test RMSE={tree_test_stats_bg.get('rmse', 'N/A')}")

            log_info("BACKGROUND: All classical baselines completed.")
        except Exception as exc:
            log_error(f"BACKGROUND: Classical baselines FAILED — {type(exc).__name__}: {exc}")
            _classical_results["error"] = exc

    classical_thread = threading.Thread(target=_run_classical_baselines, name="classical-baselines", daemon=True)
    classical_thread.start()
    log_info("Classical baselines launched in background thread (running parallel with GPU training)")

    with log_phase(console, "Train R-GAN (PyTorch)"):
        log_step("Calling train_rgan_backend...")
        rgan_out = train_rgan_backend(
            base_config,
            (G,D),
            {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte},
            str(results_dir),
            tag="rgan"
        )

        log_step(f"R-GAN training completed. Train RMSE={rgan_out['train_stats'].get('rmse', 'N/A')}, Test RMSE={rgan_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("rgan_generator", rgan_out["G"])
        _save_model("rgan_discriminator", rgan_out["D"])
        if rgan_out.get("G_ema"):
            _save_model("rgan_generator_ema", rgan_out["G_ema"])
        log_step(f"Saved RGAN models to: {models_dir}")

    with log_phase(console, "Train supervised LSTM baseline"):
        log_step("Calling train_lstm_backend...")
        lstm_out = train_lstm_backend(
            base_config,
            {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte},
            str(results_dir),
            tag="lstm"
        )
        log_step(f"LSTM training completed. Train RMSE={lstm_out['train_stats'].get('rmse', 'N/A')}, Test RMSE={lstm_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("lstm", lstm_out["model"])

    data_dict = {"Xtr": Xtr, "Ytr": Ytr, "Xval": Xval, "Yval": Yval, "Xte": Xte, "Yte": Yte}
    log_step("data_dict prepared for baseline models")

    with log_phase(console, "Train DLinear baseline"):
        log_step("Calling train_linear_baseline(model_type='dlinear')...")
        dlinear_out = train_linear_baseline(
            base_config, data_dict, str(results_dir),
            model_type="dlinear", tag="DLinear"
        )
        log_step(f"DLinear done. Test RMSE={dlinear_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("dlinear", dlinear_out["model"])

    with log_phase(console, "Train NLinear baseline"):
        log_step("Calling train_linear_baseline(model_type='nlinear')...")
        nlinear_out = train_linear_baseline(
            base_config, data_dict, str(results_dir),
            model_type="nlinear", tag="NLinear"
        )
        log_step(f"NLinear done. Test RMSE={nlinear_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("nlinear", nlinear_out["model"])

    with log_phase(console, "Train FITS baseline"):
        log_step("Calling train_fits...")
        fits_out = train_fits(base_config, data_dict, str(results_dir), tag="FITS")
        log_step(f"FITS done. Test RMSE={fits_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("fits", fits_out["model"])

    with log_phase(console, "Train PatchTST baseline"):
        log_step("Calling train_patchtst...")
        patchtst_out = train_patchtst(base_config, data_dict, str(results_dir), tag="PatchTST")
        log_step(f"PatchTST done. Test RMSE={patchtst_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("patchtst", patchtst_out["model"])

    with log_phase(console, "Train iTransformer baseline"):
        log_step("Calling train_itransformer...")
        itransformer_out = train_itransformer(base_config, data_dict, str(results_dir), tag="iTransformer")
        log_step(f"iTransformer done. Test RMSE={itransformer_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("itransformer", itransformer_out["model"])

    with log_phase(console, "Train Autoformer baseline"):
        log_step("Calling train_autoformer...")
        autoformer_out = train_autoformer(base_config, data_dict, str(results_dir), tag="Autoformer")
        log_step(f"Autoformer done. Test RMSE={autoformer_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("autoformer", autoformer_out["model"])

    with log_phase(console, "Train Informer baseline"):
        log_step("Calling train_informer...")
        informer_out = train_informer(base_config, data_dict, str(results_dir), tag="Informer")
        log_step(f"Informer done. Test RMSE={informer_out['test_stats'].get('rmse', 'N/A')}")
        _save_model("informer", informer_out["model"])

    log_step(f"All model weights saved to: {models_dir}")

    rgan_curve_artifacts = plot_single_train_test(
        rgan_out["history"]["epoch"],
        rgan_out["history"]["train_rmse"],
        rgan_out["history"]["test_rmse"],
        "RGAN: Error vs Epochs",
        results_dir / "rgan_train_test_rmse_vs_epochs.png",
    )
    lstm_curve_artifacts = plot_single_train_test(
        lstm_out["history"]["epoch"],
        lstm_out["history"]["train_rmse"],
        lstm_out["history"]["test_rmse"],
        "LSTM (Supervised): Error vs Epochs",
        results_dir / "lstm_train_test_rmse_vs_epochs.png",
    )
    dlinear_curve_artifacts = plot_single_train_test(
        dlinear_out["history"]["epoch"],
        dlinear_out["history"]["train_rmse"],
        dlinear_out["history"]["test_rmse"],
        "DLinear: Error vs Epochs",
        results_dir / "dlinear_train_test_rmse_vs_epochs.png",
    )
    nlinear_curve_artifacts = plot_single_train_test(
        nlinear_out["history"]["epoch"],
        nlinear_out["history"]["train_rmse"],
        nlinear_out["history"]["test_rmse"],
        "NLinear: Error vs Epochs",
        results_dir / "nlinear_train_test_rmse_vs_epochs.png",
    )
    fits_curve_artifacts = plot_single_train_test(
        fits_out["history"]["epoch"],
        fits_out["history"]["train_rmse"],
        fits_out["history"]["test_rmse"],
        "FITS: Error vs Epochs",
        results_dir / "fits_train_test_rmse_vs_epochs.png",
    )
    patchtst_curve_artifacts = plot_single_train_test(
        patchtst_out["history"]["epoch"],
        patchtst_out["history"]["train_rmse"],
        patchtst_out["history"]["test_rmse"],
        "PatchTST: Error vs Epochs",
        results_dir / "patchtst_train_test_rmse_vs_epochs.png",
    )
    itransformer_curve_artifacts = plot_single_train_test(
        itransformer_out["history"]["epoch"],
        itransformer_out["history"]["train_rmse"],
        itransformer_out["history"]["test_rmse"],
        "iTransformer: Error vs Epochs",
        results_dir / "itransformer_train_test_rmse_vs_epochs.png",
    )
    autoformer_curve_artifacts = plot_single_train_test(
        autoformer_out["history"]["epoch"],
        autoformer_out["history"]["train_rmse"],
        autoformer_out["history"]["test_rmse"],
        "Autoformer: Error vs Epochs",
        results_dir / "autoformer_train_test_rmse_vs_epochs.png",
    )
    informer_curve_artifacts = plot_single_train_test(
        informer_out["history"]["epoch"],
        informer_out["history"]["train_rmse"],
        informer_out["history"]["test_rmse"],
        "Informer: Error vs Epochs",
        results_dir / "informer_train_test_rmse_vs_epochs.png",
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

    # ── Wait for classical baselines background thread to finish ────────
    with log_phase(console, "Wait for classical baselines (background thread)"):
        classical_thread.join()
        if "error" in _classical_results:
            raise _classical_results["error"]
        log_info("Classical baselines background thread completed successfully.")

    # Extract results from background thread
    naive_train_stats = _classical_results["naive_train_stats"]
    naive_test_stats = _classical_results["naive_test_stats"]
    naive_curve_artifacts = _classical_results["naive_curve_artifacts"]

    if _classical_results.get("skip_classical", False):
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
        arima_train_stats = _classical_results["arima_train_stats"]
        arima_test_stats = _classical_results["arima_test_stats"]
        arima_curve_artifacts = _classical_results["arima_curve_artifacts"]
        arma_train_stats = _classical_results["arma_train_stats"]
        arma_test_stats = _classical_results["arma_test_stats"]
        arma_curve_artifacts = _classical_results["arma_curve_artifacts"]
        tree_train_stats = _classical_results["tree_train_stats"]
        tree_test_stats = _classical_results["tree_test_stats"]
        tree_curve_artifacts = _classical_results["tree_curve_artifacts"]
        tree_model = _classical_results["tree_model"]
        tree_config = _classical_results["tree_config"]

    # ── Parallel bootstrap uncertainty for all models (train + test) ────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    log_info("Computing RGAN train predictions for bootstrap...")
    _, rgan_train_pred = compute_metrics_backend(rgan_out["G"], Xtr, Ytr)
    log_info("RGAN train predictions computed.")

    train_kwargs = dict(
        n_bootstrap=args.bootstrap_samples, seed=args.seed,
        original_mean=target_mean, original_std=target_std,
        training_series=training_series_scaled, training_series_orig=training_series_orig,
    )
    test_kwargs = dict(
        n_bootstrap=args.bootstrap_samples, seed=args.seed + 1,
        original_mean=target_mean, original_std=target_std,
        training_series=training_series_scaled, training_series_orig=training_series_orig,
    )

    _bootstrap_jobs = [
        ("rgan_train", Ytr, rgan_train_pred, train_kwargs),
        ("rgan_test", Yte, rgan_out["pred_test"], test_kwargs),
        ("lstm_train", Ytr, lstm_out["pred_train"], train_kwargs),
        ("lstm_test", Yte, lstm_out["pred_test"], test_kwargs),
        ("dlinear_train", Ytr, dlinear_out["pred_train"], train_kwargs),
        ("dlinear_test", Yte, dlinear_out["pred_test"], test_kwargs),
        ("nlinear_train", Ytr, nlinear_out["pred_train"], train_kwargs),
        ("nlinear_test", Yte, nlinear_out["pred_test"], test_kwargs),
        ("fits_train", Ytr, fits_out["pred_train"], train_kwargs),
        ("fits_test", Yte, fits_out["pred_test"], test_kwargs),
        ("patchtst_train", Ytr, patchtst_out["pred_train"], train_kwargs),
        ("patchtst_test", Yte, patchtst_out["pred_test"], test_kwargs),
        ("itransformer_train", Ytr, itransformer_out["pred_train"], train_kwargs),
        ("itransformer_test", Yte, itransformer_out["pred_test"], test_kwargs),
        ("autoformer_train", Ytr, autoformer_out["pred_train"], train_kwargs),
        ("autoformer_test", Yte, autoformer_out["pred_test"], test_kwargs),
        ("informer_train", Ytr, informer_out["pred_train"], train_kwargs),
        ("informer_test", Yte, informer_out["pred_test"], test_kwargs),
    ]

    log_info(f"Starting parallel bootstrap ({len(_bootstrap_jobs)} jobs, {args.bootstrap_samples} samples each)")
    _bs_t0 = time.perf_counter()
    _bs_results = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        futures = {
            pool.submit(summarise_with_uncertainty, yt, yp, **kw): label
            for label, yt, yp, kw in _bootstrap_jobs
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                _bs_results[label] = future.result()
                log_info(f"  Bootstrap done: {label} (RMSE={_bs_results[label].get('rmse', 'N/A'):.6f})")
            except Exception as exc:
                log_error(f"  Bootstrap FAILED: {label} — {type(exc).__name__}: {exc}")
                raise
    log_info(f"All bootstrap jobs completed in {time.perf_counter() - _bs_t0:.1f}s")

    rgan_train_stats = _bs_results["rgan_train"]
    rgan_test_stats = _bs_results["rgan_test"]
    rgan_out["train_stats"] = rgan_train_stats
    rgan_out["test_stats"] = rgan_test_stats

    lstm_train_stats = _bs_results["lstm_train"]
    lstm_test_stats = _bs_results["lstm_test"]
    lstm_out["train_stats"] = lstm_train_stats
    lstm_out["test_stats"] = lstm_test_stats

    dlinear_out["train_stats"] = _bs_results["dlinear_train"]
    dlinear_out["test_stats"] = _bs_results["dlinear_test"]

    nlinear_out["train_stats"] = _bs_results["nlinear_train"]
    nlinear_out["test_stats"] = _bs_results["nlinear_test"]

    fits_out["train_stats"] = _bs_results["fits_train"]
    fits_out["test_stats"] = _bs_results["fits_test"]

    patchtst_out["train_stats"] = _bs_results["patchtst_train"]
    patchtst_out["test_stats"] = _bs_results["patchtst_test"]

    itransformer_out["train_stats"] = _bs_results["itransformer_train"]
    itransformer_out["test_stats"] = _bs_results["itransformer_test"]

    autoformer_out["train_stats"] = _bs_results["autoformer_train"]
    autoformer_out["test_stats"] = _bs_results["autoformer_test"]

    informer_train_stats = _bs_results["informer_train"]
    informer_test_stats = _bs_results["informer_test"]
    informer_out["train_stats"] = informer_train_stats
    informer_out["test_stats"] = informer_test_stats

    class_curve_artifacts = None
    ets_rmse_full = float("nan")
    arima_rmse_full = float("nan")

    # ── Noise robustness across multiple perturbation levels ──────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    noise_results = []
    if args.skip_noise_robustness:
        noise_levels = np.array([0.0])
        log_info("Skipping noise robustness evaluation (--skip_noise_robustness)")

    log_info(f"Noise robustness: {len(noise_levels)} levels = {noise_levels.tolist()}")
    log_info(f"  skip_classical={args.skip_classical}, bootstrap_samples={args.bootstrap_samples}")

    def _make_summary_kwargs(seed_offset):
        return dict(
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + seed_offset,
            original_mean=target_mean,
            original_std=target_std,
            training_series=training_series_scaled,
            training_series_orig=training_series_orig,
        )

    def _gpu_predict(model, Xn):
        """Run a torch model on noisy input (must be called from main thread)."""
        model.eval()
        dev = next(model.parameters()).device
        with torch.no_grad():
            return model(torch.from_numpy(Xn).to(dev)).cpu().numpy()

    has_tree = "tree_model" in locals() and tree_model is not None
    log_info(f"  has_tree={has_tree}")

    rng_noise = np.random.default_rng(args.seed + 2048)
    for idx, sd in enumerate(noise_levels):
        noise_t0 = time.perf_counter()
        log_info(f"NOISE LEVEL {idx+1}/{len(noise_levels)}: sd={sd}")

        if sd == 0.0:
            log_info("  sd=0 — reusing clean test stats")
            noise_results.append(
                {
                    "sd": float(sd),
                    "rgan": rgan_test_stats,
                    "lstm": lstm_test_stats,
                    "dlinear": dlinear_out["test_stats"],
                    "nlinear": nlinear_out["test_stats"],
                    "fits": fits_out["test_stats"],
                    "patchtst": patchtst_out["test_stats"],
                    "itransformer": itransformer_out["test_stats"],
                    "autoformer": autoformer_out["test_stats"],
                    "informer": informer_out["test_stats"],
                    "naive_baseline": naive_test_stats,
                    "arima": arima_test_stats,
                    "arma": arma_test_stats,
                    "tree_ensemble": tree_test_stats,
                }
            )
            continue

        log_info(f"  Generating noise perturbation (shape={Xte.shape})...")
        perturbation = rng_noise.normal(0, sd, size=Xte.shape).astype(Xte.dtype)
        Xte_noisy = Xte + perturbation

        # Phase 1: GPU inference (sequential — single GPU, fast)
        log_info("  Phase 1: GPU inference on noisy inputs...")
        _gpu_t0 = time.perf_counter()

        log_info("    Predicting: RGAN...")
        _, rgan_noise_pred = compute_metrics_backend(rgan_out["G"], Xte_noisy, Yte)
        log_info("    Predicting: LSTM...")
        lstm_noise_pred = _gpu_predict(lstm_out["model"], Xte_noisy)
        log_info("    Predicting: DLinear...")
        dlinear_noise_pred = _gpu_predict(dlinear_out["model"], Xte_noisy)
        log_info("    Predicting: NLinear...")
        nlinear_noise_pred = _gpu_predict(nlinear_out["model"], Xte_noisy)
        log_info("    Predicting: FITS...")
        fits_noise_pred = _gpu_predict(fits_out["model"], Xte_noisy)
        log_info("    Predicting: PatchTST...")
        patchtst_noise_pred = _gpu_predict(patchtst_out["model"], Xte_noisy)
        log_info("    Predicting: iTransformer...")
        itransformer_noise_pred = _gpu_predict(itransformer_out["model"], Xte_noisy)
        log_info("    Predicting: Autoformer...")
        autoformer_noise_pred = _gpu_predict(autoformer_out["model"], Xte_noisy)
        log_info("    Predicting: Informer...")
        informer_noise_pred = _gpu_predict(informer_out["model"], Xte_noisy)
        log_info("    Predicting: Naive baseline...")
        _, naive_noise_pred = naive_baseline(Xte_noisy, Yte)

        if not args.skip_classical:
            log_info("    Predicting: ARIMA (classical, may be slow)...")
            _, arima_noise_pred = arima_forecast(Xtr, Ytr, Xte_noisy, Yte)
            log_info("    Predicting: ARMA (classical, may be slow)...")
            _, arma_noise_pred = arma_forecast(Xtr, Ytr, Xte_noisy, Yte)

        if has_tree:
            log_info("    Predicting: Tree ensemble...")
            tree_noisy_flat = tree_model.predict(Xte_noisy.reshape(Xte_noisy.shape[0], -1)).astype(np.float32)
            tree_noise_pred = tree_noisy_flat.reshape(Xte_noisy.shape[0], horizon, 1)

        log_info(f"  Phase 1 done in {time.perf_counter() - _gpu_t0:.1f}s")

        # Phase 2: Bootstrap uncertainty — parallel across all models
        log_info("  Phase 2: Parallel bootstrap uncertainty...")
        _bs2_t0 = time.perf_counter()
        summary_kwargs = _make_summary_kwargs(seed_offset=idx)
        predictions = {
            "rgan": rgan_noise_pred,
            "lstm": lstm_noise_pred,
            "dlinear": dlinear_noise_pred,
            "nlinear": nlinear_noise_pred,
            "fits": fits_noise_pred,
            "patchtst": patchtst_noise_pred,
            "itransformer": itransformer_noise_pred,
            "autoformer": autoformer_noise_pred,
            "informer": informer_noise_pred,
            "naive_baseline": naive_noise_pred,
        }
        if not args.skip_classical:
            predictions["arima"] = arima_noise_pred
            predictions["arma"] = arma_noise_pred
        if has_tree:
            predictions["tree_ensemble"] = tree_noise_pred

        summaries = {}
        with ThreadPoolExecutor(max_workers=min(len(predictions), os.cpu_count() or 4)) as pool:
            futures = {
                pool.submit(summarise_with_uncertainty, Yte, pred, **summary_kwargs): name
                for name, pred in predictions.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    summaries[name] = future.result()
                    log_info(f"    Bootstrap done: {name}")
                except Exception as exc:
                    log_error(f"    Bootstrap FAILED: {name} — {type(exc).__name__}: {exc}")
                    raise

        # Fill in missing keys with blank stats
        for key in ("arima", "arma", "tree_ensemble"):
            if key not in summaries:
                summaries[key] = _blank_stats()

        summaries["sd"] = float(sd)
        log_info(f"  Noise level sd={sd} completed in {time.perf_counter() - noise_t0:.1f}s")
        noise_results.append(summaries)

    # ── Noise Robustness Summary ──────────────────────────────────────
    if len(noise_results) > 1:
        noise_table = create_noise_robustness_table(
            noise_results, out_path=str(results_dir / "noise_robustness_table.csv"),
        )
        print("\n" + "=" * 80)
        print("NOISE ROBUSTNESS TABLE (RMSE at each noise level)")
        print("=" * 80)
        print(noise_table.to_string(index=False))
        print("=" * 80)

        noise_plot_artifacts = plot_noise_robustness(
            noise_results, str(results_dir / "noise_robustness"),
        )
        console.log(f"Noise robustness plot: {noise_plot_artifacts['static']}")

    test_errors = {
        "RGAN": rgan_out["test_stats"].get("rmse_orig", rgan_out["test_stats"]["rmse"]),
        "LSTM": lstm_out["test_stats"].get("rmse_orig", lstm_out["test_stats"]["rmse"]),
        "DLinear": dlinear_out["test_stats"].get("rmse_orig", dlinear_out["test_stats"]["rmse"]),
        "NLinear": nlinear_out["test_stats"].get("rmse_orig", nlinear_out["test_stats"]["rmse"]),
        "FITS": fits_out["test_stats"].get("rmse_orig", fits_out["test_stats"]["rmse"]),
        "PatchTST": patchtst_out["test_stats"].get("rmse_orig", patchtst_out["test_stats"]["rmse"]),
        "iTransformer": itransformer_out["test_stats"].get("rmse_orig", itransformer_out["test_stats"]["rmse"]),
        "Autoformer": autoformer_out["test_stats"].get("rmse_orig", autoformer_out["test_stats"]["rmse"]),
        "Informer": informer_out["test_stats"].get("rmse_orig", informer_out["test_stats"]["rmse"]),
        "Tree Ensemble": tree_test_stats.get("rmse_orig", tree_test_stats["rmse"]),
        "Naïve Baseline": naive_test_stats.get("rmse_orig", naive_test_stats["rmse"]),
        "ARIMA": arima_test_stats.get("rmse_orig", arima_test_stats["rmse"]),
        "ARMA": arma_test_stats.get("rmse_orig", arma_test_stats["rmse"]),
    }
    train_errors = {
        "RGAN": rgan_out["train_stats"].get("rmse_orig", rgan_out["train_stats"]["rmse"]),
        "LSTM": lstm_out["train_stats"].get("rmse_orig", lstm_out["train_stats"]["rmse"]),
        "DLinear": dlinear_out["train_stats"].get("rmse_orig", dlinear_out["train_stats"]["rmse"]),
        "NLinear": nlinear_out["train_stats"].get("rmse_orig", nlinear_out["train_stats"]["rmse"]),
        "FITS": fits_out["train_stats"].get("rmse_orig", fits_out["train_stats"]["rmse"]),
        "PatchTST": patchtst_out["train_stats"].get("rmse_orig", patchtst_out["train_stats"]["rmse"]),
        "iTransformer": itransformer_out["train_stats"].get("rmse_orig", itransformer_out["train_stats"]["rmse"]),
        "Autoformer": autoformer_out["train_stats"].get("rmse_orig", autoformer_out["train_stats"]["rmse"]),
        "Informer": informer_out["train_stats"].get("rmse_orig", informer_out["train_stats"]["rmse"]),
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
        "RGAN": {
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
        },
        "DLinear": {
            "train": dlinear_out["train_stats"],
            "test": dlinear_out["test_stats"]
        },
        "NLinear": {
            "train": nlinear_out["train_stats"],
            "test": nlinear_out["test_stats"]
        },
        "FITS": {
            "train": fits_out["train_stats"],
            "test": fits_out["test_stats"]
        },
        "PatchTST": {
            "train": patchtst_out["train_stats"],
            "test": patchtst_out["test_stats"]
        },
        "iTransformer": {
            "train": itransformer_out["train_stats"],
            "test": itransformer_out["test_stats"]
        },
        "Autoformer": {
            "train": autoformer_out["train_stats"],
            "test": autoformer_out["test_stats"]
        },
        "Informer": {
            "train": informer_out["train_stats"],
            "test": informer_out["test_stats"]
        },
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

    # Plot predictions
    print("Visualizing predictions...")
    
    predictions_dict = {
        "True": Yte,
        "RGAN": rgan_out['pred_test'],
        "LSTM (Supervised)": lstm_out['pred_test'],
        "DLinear": dlinear_out['pred_test'],
        "NLinear": nlinear_out['pred_test'],
    }

    if "tree_test_pred" in locals() and tree_test_pred is not None:
         predictions_dict["Tree Ensemble"] = tree_test_pred
    if "arima_test_pred" in locals() and arima_test_pred is not None:
         predictions_dict["ARIMA"] = arima_test_pred
    if "arma_test_pred" in locals() and arma_test_pred is not None:
         predictions_dict["ARMA"] = arma_test_pred

    plot_predictions(
        predictions_dict,
        save_path=os.path.join(args.results_dir, "predictions_comparison")
    )

    predictions_test = {
        "RGAN": rgan_out["pred_test"],
        "LSTM": lstm_out["pred_test"],
        "DLinear": dlinear_out["pred_test"],
        "NLinear": nlinear_out["pred_test"],
        "FITS": fits_out["pred_test"],
        "PatchTST": patchtst_out["pred_test"],
        "iTransformer": itransformer_out["pred_test"],
        "Autoformer": autoformer_out["pred_test"],
        "Informer": informer_out["pred_test"],
        "Naïve Baseline": naive_test_pred,
    }
    if "tree_test_pred" in locals() and tree_test_pred is not None:
         predictions_test["Tree Ensemble"] = tree_test_pred
    if "arima_test_pred" in locals() and arima_test_pred is not None:
         predictions_test["ARIMA"] = arima_test_pred
    if "arma_test_pred" in locals() and arma_test_pred is not None:
         predictions_test["ARMA"] = arma_test_pred
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
        dlinear=dict(
            train=dlinear_out["train_stats"],
            test=dlinear_out["test_stats"],
            curve=dlinear_curve_artifacts["static"],
            curve_interactive=dlinear_curve_artifacts.get("interactive", ""),
        ),
        nlinear=dict(
            train=nlinear_out["train_stats"],
            test=nlinear_out["test_stats"],
            curve=nlinear_curve_artifacts["static"],
            curve_interactive=nlinear_curve_artifacts.get("interactive", ""),
        ),
        fits=dict(
            train=fits_out["train_stats"],
            test=fits_out["test_stats"],
            curve=fits_curve_artifacts["static"],
            curve_interactive=fits_curve_artifacts.get("interactive", ""),
        ),
        patchtst=dict(
            train=patchtst_out["train_stats"],
            test=patchtst_out["test_stats"],
            curve=patchtst_curve_artifacts["static"],
            curve_interactive=patchtst_curve_artifacts.get("interactive", ""),
        ),
        itransformer=dict(
            train=itransformer_out["train_stats"],
            test=itransformer_out["test_stats"],
            curve=itransformer_curve_artifacts["static"],
            curve_interactive=itransformer_curve_artifacts.get("interactive", ""),
        ),
        autoformer=dict(
            train=autoformer_out["train_stats"],
            test=autoformer_out["test_stats"],
            curve=autoformer_curve_artifacts["static"],
            curve_interactive=autoformer_curve_artifacts.get("interactive", ""),
        ),
        informer=dict(
            train=informer_out["train_stats"],
            test=informer_out["test_stats"],
            curve=informer_curve_artifacts["static"],
            curve_interactive=informer_curve_artifacts.get("interactive", ""),
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
            curves="",
            curves_interactive="",
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

    # ── Run Summary ─────────────────────────────────────────────────
    total_elapsed = time.time() - run_start_time
    hours, remainder = divmod(int(total_elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_str = f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"

    # Build leaderboard sorted by test RMSE (original scale)
    leaderboard = {}
    for name, out in [
        ("RGAN", rgan_out), ("LSTM", lstm_out),
        ("DLinear", dlinear_out), ("NLinear", nlinear_out),
        ("FITS", fits_out), ("PatchTST", patchtst_out),
        ("iTransformer", itransformer_out), ("Autoformer", autoformer_out),
        ("Informer", informer_out),
    ]:
        ts = out.get("test_stats", {})
        rmse = ts.get("rmse_orig", ts.get("rmse"))
        if rmse is not None and not math.isnan(rmse):
            leaderboard[name] = rmse
    for name, stats in [
        ("Naive", naive_test_stats), ("ARIMA", arima_test_stats),
        ("ARMA", arma_test_stats), ("Tree Ensemble", tree_test_stats),
    ]:
        rmse = stats.get("rmse_orig", stats.get("rmse"))
        if rmse is not None and not math.isnan(rmse):
            leaderboard[name] = rmse
    sorted_lb = sorted(leaderboard.items(), key=lambda x: x[1])

    # Save run summary
    summary = {
        "dataset": args.csv,
        "target": target_col,
        "results_dir": str(results_dir.resolve()),
        "elapsed": elapsed_str,
        "elapsed_seconds": round(total_elapsed, 1),
        "epochs": args.epochs,
        "leaderboard": [{"rank": i+1, "model": name, "test_rmse_orig": rmse}
                        for i, (name, rmse) in enumerate(sorted_lb)],
        "best_model": sorted_lb[0][0] if sorted_lb else "N/A",
        "best_rmse": sorted_lb[0][1] if sorted_lb else None,
        "completed_at": datetime.now().isoformat(),
        "num_models_trained": len(leaderboard),
        "models_saved": [f.name for f in (results_dir / "models").glob("*.pt")],
        "compute": {
            "wall_clock_seconds": round(total_elapsed, 1),
            "est_spot_cost_usd": round((total_elapsed / 3600) * 0.34, 3),
            "est_ondemand_cost_usd": round((total_elapsed / 3600) * 1.01, 3),
            "instance_type": "g5.xlarge",
        },
    }
    with open(results_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"RUN COMPLETE | {elapsed_str} | {args.epochs} epochs", flush=True)
    print(f"Results: {results_dir.resolve()}", flush=True)
    print(f"Dataset: {args.csv} -> target: {target_col}", flush=True)
    print(flush=True)
    print("Leaderboard (Test RMSE, original scale):", flush=True)
    for i, (name, rmse) in enumerate(sorted_lb):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:>2}. {name:<20} {rmse:.6f}{marker}", flush=True)
    print("=" * 60, flush=True)

    # Cost estimate (spot pricing for g5.xlarge us-east-1)
    spot_rate = 0.34  # $/hr approximate
    ondemand_rate = 1.01  # $/hr approximate
    spot_cost = (total_elapsed / 3600) * spot_rate
    ondemand_cost = (total_elapsed / 3600) * ondemand_rate
    print(flush=True)
    print("COMPUTE SUMMARY:", flush=True)
    print(f"  Wall-clock time:  {elapsed_str} ({total_elapsed:.0f}s)", flush=True)
    print(f"  Est. spot cost:   ${spot_cost:.3f} (g5.xlarge @ ${spot_rate}/hr)", flush=True)
    print(f"  Est. on-demand:   ${ondemand_cost:.3f} (g5.xlarge @ ${ondemand_rate}/hr)", flush=True)
    print("=" * 60, flush=True)

    log_info(f"Run completed successfully in {elapsed_str}")
    log_info(f"Estimated spot cost: ${spot_cost:.3f} | On-demand: ${ondemand_cost:.3f}")
    log_info(f"Results saved to: {results_dir.resolve()}")
    close_log_file()

if __name__ == "__main__":
    main()
