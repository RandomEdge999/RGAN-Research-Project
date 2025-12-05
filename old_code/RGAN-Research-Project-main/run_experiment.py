#!/usr/bin/env python3
import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.rgan.baselines import naive_baseline, naive_bayes_forecast, classical_curves_vs_samples
from src.rgan.data import (
    load_csv_series,
    interpolate_and_standardize,
    make_windows_univariate,
    make_windows_with_covariates,
)
from src.rgan.plots import (
    plot_single_train_test,
    plot_constant_train_test,
    plot_compare_models_bars,
    plot_classical_curves,
    plot_learning_curves,
    create_error_metrics_table,
    plot_naive_bayes_comparison,
)
from src.rgan.tune import tune_rgan
from src.rgan.logging_utils import get_console, print_banner, print_kv_table


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


def describe_model(model) -> list:
    """Return a human-readable description of the model layers."""
    description = []
    if hasattr(model, "named_children"):
        for name, module in model.named_children():
            description.append(f"{name}: {module.__class__.__name__}")
        if not description:
            description.append(model.__class__.__name__)
    else:
        description.append(model.__class__.__name__)
    return description


def error_stats(y_true, y_pred):
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    return {"rmse": rmse, "mae": mae, "mse": mse, "bias": bias}


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


def summarise_with_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 300,
    seed: Optional[int] = None,
    original_mean: Optional[float] = None,
    original_std: Optional[float] = None,
) -> Dict[str, float]:
    """Return base error statistics plus bootstrap uncertainty estimates."""

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    stats = error_stats(y_true_flat, y_pred_flat)

    orig_enabled = original_mean is not None and original_std is not None
    if orig_enabled:
        y_true_orig = y_true_flat * original_std + original_mean
        y_pred_orig = y_pred_flat * original_std + original_mean
        orig_stats = error_stats(y_true_orig, y_pred_orig)
        stats.update({f"{k}_orig": v for k, v in orig_stats.items()})

    if n_bootstrap <= 0 or y_true_flat.size < 2:
        return stats

    rng = np.random.default_rng(seed)
    rmse_samples = []
    mae_samples = []
    rmse_orig_samples = [] if orig_enabled else None
    mae_orig_samples = [] if orig_enabled else None
    n = y_true_flat.size
    diff_flat = y_pred_flat - y_true_flat
    if orig_enabled:
        diff_orig_flat = diff_flat * original_std
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diff = diff_flat[idx]
        rmse_samples.append(np.sqrt(np.mean(diff ** 2)))
        mae_samples.append(np.mean(np.abs(diff)))
        if orig_enabled and rmse_orig_samples is not None and mae_orig_samples is not None:
            diff_orig = diff_orig_flat[idx]
            rmse_orig_samples.append(np.sqrt(np.mean(diff_orig ** 2)))
            mae_orig_samples.append(np.mean(np.abs(diff_orig)))

    rmse_arr = np.asarray(rmse_samples, dtype=float)
    mae_arr = np.asarray(mae_samples, dtype=float)

    stats.update(
        {
            "rmse_std": float(np.std(rmse_arr, ddof=1)),
            "rmse_ci_low": float(np.percentile(rmse_arr, 2.5)),
            "rmse_ci_high": float(np.percentile(rmse_arr, 97.5)),
            "mae_std": float(np.std(mae_arr, ddof=1)),
            "mae_ci_low": float(np.percentile(mae_arr, 2.5)),
            "mae_ci_high": float(np.percentile(mae_arr, 97.5)),
        }
    )

    if orig_enabled and rmse_orig_samples and mae_orig_samples:
        rmse_orig_arr = np.asarray(rmse_orig_samples, dtype=float)
        mae_orig_arr = np.asarray(mae_orig_samples, dtype=float)
        stats.update(
            {
                "rmse_orig_std": float(np.std(rmse_orig_arr, ddof=1)),
                "rmse_orig_ci_low": float(np.percentile(rmse_orig_arr, 2.5)),
                "rmse_orig_ci_high": float(np.percentile(rmse_orig_arr, 97.5)),
                "mae_orig_std": float(np.std(mae_orig_arr, ddof=1)),
                "mae_orig_ci_low": float(np.percentile(mae_orig_arr, 2.5)),
                "mae_orig_ci_high": float(np.percentile(mae_orig_arr, 97.5)),
            }
        )
    return stats


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


def compute_learning_curves(args, base_config, Xfull_tr, Yfull_tr, Xte, Yte, n_features):
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

    curves_mean = {"R-GAN": [], "LSTM": [], "Naïve Baseline": [], "Naïve Bayes": []}
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

            curve_config = dict(base_config)
            curve_config["epochs"] = max(1, min(base_config["epochs"], args.curve_epochs))
            curve_config["patience"] = min(curve_config["patience"], curve_config["epochs"])

            set_seed(args.seed + rep)

            gen_kwargs = dict(
                L=base_config["L"],
                H=base_config["H"],
                n_in=n_features,
                units=curve_config["units_g"],
                dropout=curve_config["dropout"],
                num_layers=curve_config.get("g_layers", base_config["g_layers"]),
            )
            disc_kwargs = dict(
                L=base_config["L"],
                H=base_config["H"],
                units=curve_config["units_d"],
                dropout=curve_config["dropout"],
                num_layers=curve_config.get("d_layers", base_config["d_layers"]),
            )

            dense_act = curve_config.get("g_dense_activation")
            if dense_act is not None:
                gen_kwargs["dense_activation"] = dense_act
            disc_act = curve_config.get("d_activation")
            if disc_act is not None:
                disc_kwargs["activation"] = disc_act

            G_curve = build_generator_backend(**gen_kwargs)
            D_curve = build_discriminator_backend(**disc_kwargs)
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
            naive_bayes_stats, _ = naive_bayes_forecast(
                sub_split["X_train"], sub_split["Y_train"], Xte, Yte
            )
            rmse_acc["Naïve Bayes"].append(naive_bayes_stats["rmse"])

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
    ap.add_argument("--num_workers", type=int, default=2,
                    help="Number of workers for PyTorch DataLoader")
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
        "--tune",
        action="store_true",
        help="Run the R-GAN hyperparameter sweep (disabled by default).",
    )
    ap.add_argument("--tune_csv", default="")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    console = get_console()
    print_banner(console, "RGAN Research Project", "Noise-Resilient Forecasting – Experiment Runner")

    if args.tune_csv and not args.tune:
        args.tune = True

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = parse_noise_levels(str(args.noise_levels))

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

    base_config = dict(
        L=args.L,
        H=args.H,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_reg=args.lambda_reg,
        units_g=args.units_g,
        units_d=args.units_d,
        lrG=args.lrG,
        lrD=args.lrD,
        label_smooth=args.label_smooth,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        patience=args.patience,
        g_layers=args.g_layers,
        d_layers=args.d_layers,
        g_dense_activation=args.g_dense_activation if args.g_dense_activation else None,
        d_activation=args.d_activation if args.d_activation else None,
        amp=args.amp,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
    )

    print_kv_table(console, "Configuration", {
        "L/H": f"{args.L}/{args.H}",
        "Epochs": args.epochs,
        "Batch Size": args.batch_size,
        "Units (G/D)": f"{args.units_g}/{args.units_d}",
        "Lambda": args.lambda_reg,
        "Learning Rates (G/D)": f"{args.lrG}/{args.lrD}",
        "Dropout": args.dropout,
        "Layers (G/D)": f"{args.g_layers}/{args.d_layers}",
    })

    used_tune = ""
    best_hp = None
    if args.tune:
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
            "lrG": lr_candidates_g,
            "lrD": lr_candidates_d,
            "g_dense_activation": dense_candidates,
            "d_activation": disc_act_candidates,
            "epochs_each": [30, 45],
        }

        best_hp, df_tune_res = tune_rgan(
            hp_grid,
            base_config,
            tune_splits,
            str(results_dir),
            seed=args.seed,
        )
        df_tune_res.to_csv(results_dir/"tuning_results.csv", index=False)
        for key, value in best_hp.items():
            if key in {"val_rmse", "seed"}:
                continue
            if value is None:
                continue
            if key in base_config:
                base_config[key] = value

    g_dense_act = base_config.get("g_dense_activation")

    gen_kwargs = dict(
        L=args.L,
        H=args.H,
        n_in=Xtr.shape[-1],
        units=base_config["units_g"],
        dropout=base_config["dropout"],
        num_layers=base_config["g_layers"],
        dense_activation=g_dense_act,
    )
    disc_kwargs = dict(
        L=args.L,
        H=args.H,
        units=base_config["units_d"],
        dropout=base_config["dropout"],
        num_layers=base_config["d_layers"],
        activation=base_config.get("d_activation"),
    )

    console.rule("Build Models")
    G = build_generator_backend(**gen_kwargs)
    D = build_discriminator_backend(**disc_kwargs)
    rgan_out = train_rgan_backend(base_config, (G,D), {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte}, str(results_dir), tag="rgan")

    lstm_out = train_lstm_backend(base_config, {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte}, str(results_dir), tag="lstm")

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

    _, naive_train_pred = naive_baseline(Xtr, Ytr)
    _, naive_test_pred = naive_baseline(Xte, Yte)
    naive_train_stats = summarise_with_uncertainty(
        Ytr,
        naive_train_pred,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        original_mean=target_mean,
        original_std=target_std,
    )
    naive_test_stats = summarise_with_uncertainty(
        Yte,
        naive_test_pred,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
        original_mean=target_mean,
        original_std=target_std,
    )
    naive_curve_artifacts = plot_constant_train_test(
        naive_train_stats["rmse"],
        naive_test_stats["rmse"],
        "Naïve Baseline: Error vs Epochs",
        results_dir / "naive_train_test_rmse_vs_epochs.png",
    )

    # Naïve Bayes implementation (train on training windows, evaluate on both)
    _, naive_bayes_train_pred = naive_bayes_forecast(Xtr, Ytr)
    naive_bayes_train_stats = summarise_with_uncertainty(
        Ytr,
        naive_bayes_train_pred,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        original_mean=target_mean,
        original_std=target_std,
    )
    _, naive_bayes_test_pred = naive_bayes_forecast(Xtr, Ytr, Xte, Yte)
    naive_bayes_test_stats = summarise_with_uncertainty(
        Yte,
        naive_bayes_test_pred,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
        original_mean=target_mean,
        original_std=target_std,
    )
    naive_bayes_curve_artifacts = plot_constant_train_test(
        naive_bayes_train_stats["rmse"],
        naive_bayes_test_stats["rmse"],
        "Naïve Bayes: Error vs Epochs",
        results_dir / "naive_bayes_train_test_rmse_vs_epochs.png",
    )

    # Plot comparison between Naïve Baseline and Naïve Bayes (similar to Fig 1)
    naive_comparison_artifacts = plot_naive_bayes_comparison(
        naive_test_stats,
        naive_bayes_test_stats,
        results_dir / "naive_baseline_vs_naive_bayes.png",
    )

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
    )
    rgan_test_stats = summarise_with_uncertainty(
        Yte,
        rgan_out["pred_test"],
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
        original_mean=target_mean,
        original_std=target_std,
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
    )
    lstm_test_stats = summarise_with_uncertainty(
        Yte,
        lstm_out["pred_test"],
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
        original_mean=target_mean,
        original_std=target_std,
    )
    lstm_out["train_stats"] = lstm_train_stats
    lstm_out["test_stats"] = lstm_test_stats

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
                    "naive_bayes": naive_bayes_test_stats,
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
        )

        _, naive_noisy_pred = naive_baseline(Xte_noisy, Yte)
        naive_noise_summary = summarise_with_uncertainty(
            Yte,
            naive_noisy_pred,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + idx,
            original_mean=target_mean,
            original_std=target_std,
        )

        _, naive_bayes_noisy_pred = naive_bayes_forecast(Xtr, Ytr, Xte_noisy, Yte)
        naive_bayes_noise_summary = summarise_with_uncertainty(
            Yte,
            naive_bayes_noisy_pred,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + idx,
            original_mean=target_mean,
            original_std=target_std,
        )

        noise_results.append(
            {
                "sd": float(sd),
                "rgan": rgan_noise_summary,
                "lstm": lstm_noise_summary,
                "naive_baseline": naive_noise_summary,
                "naive_bayes": naive_bayes_noise_summary,
            }
        )

    test_errors = {
        "R-GAN": rgan_out["test_stats"].get("rmse_orig", rgan_out["test_stats"]["rmse"]),
        "LSTM": lstm_out["test_stats"].get("rmse_orig", lstm_out["test_stats"]["rmse"]),
        "Naïve Baseline": naive_test_stats.get("rmse_orig", naive_test_stats["rmse"]),
        "Naïve Bayes": naive_bayes_test_stats.get("rmse_orig", naive_bayes_test_stats["rmse"]),
    }
    train_errors = {
        "R-GAN": rgan_out["train_stats"].get("rmse_orig", rgan_out["train_stats"]["rmse"]),
        "LSTM": lstm_out["train_stats"].get("rmse_orig", lstm_out["train_stats"]["rmse"]),
        "Naïve Baseline": naive_train_stats.get("rmse_orig", naive_train_stats["rmse"]),
        "Naïve Bayes": naive_bayes_train_stats.get("rmse_orig", naive_bayes_train_stats["rmse"]),
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
        "Naïve Baseline": {
            "train": naive_train_stats,
            "test": naive_test_stats
        },
        "Naïve Bayes": {
            "train": naive_bayes_train_stats,
            "test": naive_bayes_test_stats
        }
    }
    
    # Generate and save error metrics table
    metrics_table = create_error_metrics_table(model_results, results_dir/"error_metrics_table.csv")
    print("\n" + "="*80)
    print("ERROR METRICS TABLE (RMSE, MSE, BIAS, MAE)")
    print("="*80)
    print(metrics_table.to_string(index=False))
    print("="*80)

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
                units_g=base_config["units_g"],
                units_d=base_config["units_d"],
                lambda_reg=base_config["lambda_reg"],
                lrG=base_config["lrG"],
                lrD=base_config["lrD"],
                dropout=base_config["dropout"],
                g_layers=base_config["g_layers"],
                d_layers=base_config["d_layers"],
                g_dense=(g_dense_act if g_dense_act else "linear"),
                d_activation=base_config.get("d_activation") or "sigmoid",
            ),
        ),
        lstm=dict(
            train=lstm_out["train_stats"],
            test=lstm_out["test_stats"],
            curve=lstm_curve_artifacts["static"],
            curve_interactive=lstm_curve_artifacts.get("interactive", ""),
            history=lstm_out["history"],
            architecture=lstm_architecture,
            config=dict(units=base_config["units_g"], lr=base_config["lrG"], dropout=base_config["dropout"]),
        ),
        naive_baseline=dict(
            train=naive_train_stats,
            test=naive_test_stats,
            curve=naive_curve_artifacts["static"],
            curve_interactive=naive_curve_artifacts.get("interactive", ""),
        ),
        naive_bayes=dict(
            train=naive_bayes_train_stats,
            test=naive_bayes_test_stats,
            curve=naive_bayes_curve_artifacts["static"],
            curve_interactive=naive_bayes_curve_artifacts.get("interactive", ""),
        ),
        compare_plots=dict(
            test=model_compare_artifacts["test"]["static"],
            test_interactive=model_compare_artifacts["test"].get("interactive", ""),
            train=model_compare_artifacts["train"]["static"],
            train_interactive=model_compare_artifacts["train"].get("interactive", ""),
            naive_comparison=naive_comparison_artifacts["static"],
            naive_comparison_interactive=naive_comparison_artifacts.get("interactive", ""),
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
        tuning=dict(
            enabled=bool(args.tune),
            dataset=used_tune,
            best=best_hp or {},
        ),
        environment=env_info,
        scaling={"target_mean": float(target_mean), "target_std": float(target_std)},
        created=datetime.utcnow().isoformat(),
    )
    with open(results_dir/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    # make src importable if run directly
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
    main()
