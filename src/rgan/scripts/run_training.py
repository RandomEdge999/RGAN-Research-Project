#!/usr/bin/env python3
import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import shutil
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from datetime import datetime, timezone
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
    plot_training_curves_overlay,
    plot_ranked_model_bars,
    plot_noise_robustness_heatmap,
    plot_multi_metric_radar,
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


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
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


def _collect_environment_info(seed: int) -> Dict[str, Any]:
    packages = {}
    _pkg_import_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "scikit-learn": "sklearn",
    }
    for pkg, module_name in _pkg_import_map.items():
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

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_commit,
        "seed": seed,
    }


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resume_store_path(results_dir: Path, checkpoint_dir: Optional[str]) -> Path:
    return Path(checkpoint_dir) if checkpoint_dir else results_dir


def _build_resume_signature(args: argparse.Namespace) -> Dict[str, Any]:
    csv_path = Path(args.csv).expanduser().resolve()
    csv_exists = csv_path.exists()
    csv_stat = csv_path.stat() if csv_exists else None
    prior_results = str(Path(args.prior_results).expanduser().resolve()) if args.prior_results else ""

    return _sanitize_for_json(
        {
            "dataset": {
                "csv": str(csv_path),
                "exists": csv_exists,
                "size": csv_stat.st_size if csv_stat else None,
                "target": args.target,
                "time_col": args.time_col,
                "resample": args.resample,
                "agg": args.agg,
                "train_ratio": args.train_ratio,
                "val_frac": args.val_frac,
                "max_train_windows": args.max_train_windows,
            },
            "pipeline": {
                "only_models": args.only_models,
                "prior_results": prior_results,
                "skip_classical": args.skip_classical,
                "skip_noise_robustness": args.skip_noise_robustness,
                "bootstrap_samples": args.bootstrap_samples,
                "noise_levels": args.noise_levels,
            },
            "training": {
                "L": args.L,
                "H": args.H,
                "pipeline": args.pipeline,
                "batch_size": args.batch_size,
                "eval_every": args.eval_every,
                "eval_batch_size": args.eval_batch_size,
                "seed": args.seed,
                "deterministic": bool(getattr(args, "deterministic", False)),
                "lambda_reg": args.lambda_reg,
                "gan_variant": args.gan_variant,
                "units_g": args.units_g,
                "units_d": args.units_d,
                "g_layers": args.g_layers,
                "d_layers": args.d_layers,
                "lr_g": args.lr_g,
                "lr_d": args.lr_d,
                "label_smooth": args.label_smooth,
                "grad_clip": args.grad_clip,
                "dropout": args.dropout,
                "patience": args.patience,
                "g_dense_activation": args.g_dense_activation,
                "d_activation": args.d_activation,
                "amp": args.amp,
                "ema_decay": args.ema_decay,
                "wgan_gp_lambda": args.wgan_gp_lambda,
                "wgan_clip_value": args.wgan_clip_value,
                "use_logits": args.use_logits,
                "d_steps": args.d_steps,
                "g_steps": args.g_steps,
                "supervised_warmup_epochs": args.supervised_warmup_epochs,
                "lambda_reg_start": args.lambda_reg_start,
                "lambda_reg_end": args.lambda_reg_end,
                "lambda_reg_warmup_epochs": args.lambda_reg_warmup_epochs,
                "adv_weight": args.adv_weight,
                "instance_noise_std": args.instance_noise_std,
                "instance_noise_decay": args.instance_noise_decay,
                "weight_decay": args.weight_decay,
                "noise_dim": args.noise_dim,
                "regression_epochs": args.regression_epochs,
                "regression_lr": args.regression_lr,
                "regression_patience": args.regression_patience,
                "lambda_diversity": args.lambda_diversity,
                "critic_reg_interval": args.critic_reg_interval,
                "critic_arch": args.critic_arch,
            },
        }
    )


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _detect_rgan_pipeline(
    saved_metrics: Optional[Dict[str, Any]],
    run_args: Optional[Dict[str, Any]],
    models_dir: Optional[Path],
) -> str:
    """Infer whether a run uses the legacy joint or paper two-stage pipeline."""
    cfg = ((saved_metrics or {}).get("rgan") or {}).get("config") or {}
    metrics_pipeline = cfg.get("pipeline")
    if metrics_pipeline in {"two_stage", "joint"}:
        return metrics_pipeline

    args_pipeline = (run_args or {}).get("pipeline")
    if args_pipeline in {"two_stage", "joint"}:
        return args_pipeline

    if models_dir is not None:
        if (models_dir / "rgan_regression.pt").exists() and (models_dir / "rgan_residual_generator.pt").exists():
            return "two_stage"

    return "joint"


def _infer_joint_replay_config(
    saved_metrics: Optional[Dict[str, Any]],
    state_dict: Dict[str, Any],
    n_in_current: int,
    model_label: str = "RGAN",
    run_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = ((saved_metrics or {}).get(model_label.lower().replace("-", "_").replace(" ", "_")) or {}).get("config", {})
    if not cfg:
        cfg = ((saved_metrics or {}).get("rgan") or {}).get("config", {})

    lstm_weight = None
    for key, tensor in state_dict.items():
        if key.endswith("lstm.weight_ih_l0"):
            lstm_weight = tensor
            break
    if lstm_weight is None:
        raise ValueError(f"{model_label} checkpoint is missing the generator LSTM input weights.")

    input_size = int(lstm_weight.shape[1])
    inferred_units = int(lstm_weight.shape[0] // 4)
    inferred_layers = max(1, len([k for k in state_dict if "lstm.weight_ih_l" in k]))

    noise_dim = cfg.get("noise_dim")
    n_in_saved = cfg.get("n_in")
    if noise_dim is None:
        if n_in_saved is not None:
            noise_dim = input_size - int(n_in_saved)
        elif input_size == int(n_in_current):
            noise_dim = 0
        else:
            raise ValueError(
                f"Cannot safely reconstruct {model_label} input width: checkpoint input_size={input_size}, "
                f"prepared data features={n_in_current}, and saved metrics are missing noise_dim/n_in."
            )
    noise_dim = int(noise_dim)
    n_in_expected = int(n_in_saved) if n_in_saved is not None else input_size - noise_dim
    if n_in_expected != int(n_in_current):
        raise ValueError(
            f"{model_label} expects {n_in_expected} input features, but current data has {n_in_current}."
        )

    dense_activation = cfg.get("g_dense_activation", cfg.get("g_dense"))
    if dense_activation in {"", None, "linear"}:
        dense_activation = None

    return {
        "n_in": n_in_expected,
        "units": int(cfg.get("units_g", (run_args or {}).get("units_g", inferred_units))),
        "num_layers": int(cfg.get("g_layers", (run_args or {}).get("g_layers", inferred_layers))),
        "noise_dim": noise_dim,
        "dense_activation": dense_activation,
        "dropout": float(cfg.get("dropout", (run_args or {}).get("dropout", 0.1))),
        "units_d": int(cfg.get("units_d", (run_args or {}).get("units_d", cfg.get("units_g", inferred_units)))),
        "d_layers": int(cfg.get("d_layers", (run_args or {}).get("d_layers", cfg.get("g_layers", inferred_layers)))),
        "d_activation": cfg.get("d_activation", (run_args or {}).get("d_activation", "sigmoid")),
        "critic_arch": cfg.get("critic_arch", (run_args or {}).get("critic_arch", "tcn")),
        "layer_norm": bool(
            cfg["layer_norm"] if "layer_norm" in cfg else any(k.startswith("stack.ln.") for k in state_dict)
        ),
        "use_spectral_norm": bool(
            cfg.get(
                "use_spectral_norm",
                str(cfg.get("gan_variant", (run_args or {}).get("gan_variant", ""))).lower() == "wgan-gp",
            )
        ),
    }


def _load_rgan_bundle_from_run_dir(
    run_dir: Path,
    n_in_current: int,
    L: int,
    H: int,
    device: torch.device,
    prefer_ema: bool = True,
) -> Dict[str, Any]:
    """Load an RGAN run directory into a unified in-memory bundle."""
    from rgan.models_torch import (
        build_discriminator,
        build_generator,
        build_regression_model,
        build_residual_discriminator,
        build_residual_generator,
    )

    run_dir = Path(run_dir)
    models_dir = run_dir / "models"
    saved_metrics = _load_json_if_exists(run_dir / "metrics.json")
    run_config = _load_json_if_exists(run_dir / "run_config.json")
    run_args = run_config.get("args", {}) if isinstance(run_config, dict) else {}
    pipeline = _detect_rgan_pipeline(saved_metrics, run_args, models_dir)
    cfg = ((saved_metrics or {}).get("rgan") or {}).get("config") or {}

    if pipeline == "two_stage":
        regression_state = torch.load(models_dir / "rgan_regression.pt", map_location=device, weights_only=True)
        residual_generator_state = torch.load(
            models_dir / "rgan_residual_generator.pt",
            map_location=device,
            weights_only=True,
        )
        residual_critic_state = torch.load(
            models_dir / "rgan_residual_discriminator.pt",
            map_location=device,
            weights_only=True,
        )
        n_in_saved = int(cfg.get("n_in", run_args.get("n_in", n_in_current)))
        if n_in_saved != int(n_in_current):
            raise ValueError(
                f"Two-stage RGAN expects {n_in_saved} input features, but current data has {n_in_current}."
            )

        units_g = int(cfg.get("units_g", run_args.get("units_g", 128)))
        units_d = int(cfg.get("units_d", run_args.get("units_d", units_g)))
        g_layers = int(cfg.get("g_layers", run_args.get("g_layers", 2)))
        d_layers = int(cfg.get("d_layers", run_args.get("d_layers", 2)))
        dropout = float(cfg.get("dropout", run_args.get("dropout", 0.1)))
        noise_dim = int(cfg.get("noise_dim", run_args.get("noise_dim", 16)))
        d_activation = cfg.get("d_activation", run_args.get("d_activation", "sigmoid"))
        critic_arch = cfg.get("critic_arch", run_args.get("critic_arch", "tcn"))
        layer_norm = bool(
            cfg["layer_norm"] if "layer_norm" in cfg
            else any(k.startswith("stack.ln.") for k in regression_state)
            or any(k.startswith("ln.") for k in residual_generator_state)
        )
        use_spectral_norm = bool(
            cfg.get(
                "use_spectral_norm",
                str(cfg.get("gan_variant", run_args.get("gan_variant", ""))).lower() == "wgan-gp",
            )
        )

        regression = build_regression_model(
            L=L,
            H=H,
            n_in=n_in_saved,
            units=units_g,
            num_layers=g_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        ).to(device)
        residual_generator = build_residual_generator(
            H=H,
            noise_dim=noise_dim,
            units=units_g,
            num_layers=g_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        ).to(device)
        residual_critic = build_residual_discriminator(
            H=H,
            units=units_d,
            num_layers=d_layers,
            dropout=dropout,
            activation=d_activation,
            layer_norm=layer_norm,
            use_spectral_norm=use_spectral_norm,
            critic_arch=critic_arch,
        ).to(device)

        regression.load_state_dict(regression_state)
        residual_generator.load_state_dict(residual_generator_state)
        residual_critic.load_state_dict(residual_critic_state)
        regression.eval()
        residual_generator.eval()
        residual_critic.eval()
        return {
            "pipeline": "two_stage",
            "F_hat": regression,
            "G": residual_generator,
            "G_ema": None,
            "D": residual_critic,
            "saved_metrics": saved_metrics,
            "run_args": run_args,
        }

    gen_path = models_dir / "rgan_generator.pt"
    ema_path = models_dir / "rgan_generator_ema.pt"
    if not gen_path.exists() and not ema_path.exists():
        raise FileNotFoundError(f"Missing generator checkpoint in {models_dir}")
    generator_state = torch.load(
        gen_path if gen_path.exists() else ema_path,
        map_location=device,
        weights_only=True,
    )
    replay_cfg = _infer_joint_replay_config(
        saved_metrics,
        generator_state,
        n_in_current,
        model_label="RGAN",
        run_args=run_args,
    )

    generator = build_generator(
        L=L,
        H=H,
        n_in=replay_cfg["n_in"],
        units=replay_cfg["units"],
        num_layers=replay_cfg["num_layers"],
        dropout=replay_cfg["dropout"],
        dense_activation=replay_cfg["dense_activation"],
        layer_norm=replay_cfg["layer_norm"],
        noise_dim=replay_cfg["noise_dim"],
    ).to(device)
    generator.load_state_dict(generator_state)
    generator.eval()

    generator_ema = None
    if prefer_ema and ema_path.exists() and gen_path.exists():
        generator_ema = build_generator(
            L=L,
            H=H,
            n_in=replay_cfg["n_in"],
            units=replay_cfg["units"],
            num_layers=replay_cfg["num_layers"],
            dropout=replay_cfg["dropout"],
            dense_activation=replay_cfg["dense_activation"],
            layer_norm=replay_cfg["layer_norm"],
            noise_dim=replay_cfg["noise_dim"],
        ).to(device)
        generator_ema.load_state_dict(torch.load(ema_path, map_location=device, weights_only=True))
        generator_ema.eval()

    discriminator = None
    disc_path = models_dir / "rgan_discriminator.pt"
    if disc_path.exists():
        discriminator = build_discriminator(
            L=L,
            H=H,
            units=replay_cfg["units_d"],
            dropout=replay_cfg["dropout"],
            num_layers=replay_cfg["d_layers"],
            activation=replay_cfg["d_activation"],
            layer_norm=replay_cfg["layer_norm"],
            use_spectral_norm=replay_cfg["use_spectral_norm"],
            critic_arch=replay_cfg["critic_arch"],
        ).to(device)
        discriminator.load_state_dict(torch.load(disc_path, map_location=device, weights_only=True))
        discriminator.eval()

    return {
        "pipeline": "joint",
        "G": generator,
        "G_ema": generator_ema,
        "D": discriminator,
        "saved_metrics": saved_metrics,
        "run_args": run_args,
    }


def _predict_rgan_bundle(
    rgan_out: Dict[str, Any],
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    batch_size: int = 512,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Optional[Dict[str, float]], np.ndarray]:
    """Predict from an RGAN output bundle regardless of pipeline type."""
    from rgan.rgan_torch import (
        _select_eval_generator,
        predict_generator_forecasts,
        predict_hybrid_forecasts,
    )

    pipeline = rgan_out.get("pipeline", "joint")
    if pipeline == "two_stage":
        preds = predict_hybrid_forecasts(
            rgan_out["F_hat"],
            rgan_out["G"],
            X,
            batch_size=batch_size,
            deterministic=deterministic,
            seed=seed,
        )
    else:
        eval_model = _select_eval_generator(rgan_out["G"], rgan_out.get("G_ema"))
        preds = predict_generator_forecasts(
            eval_model,
            X,
            batch_size=batch_size,
            stochastic=not deterministic,
            seed=seed,
        )

    stats = None
    if Y is not None:
        Y_ref = Y.detach().cpu().numpy() if torch.is_tensor(Y) else Y
        stats = error_stats(Y_ref.reshape(-1), preds.reshape(-1))
    return stats, preds


def _describe_rgan_architecture(
    rgan_out: Dict[str, Any],
    generator_fallback: Optional[torch.nn.Module] = None,
    discriminator_fallback: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    pipeline = rgan_out.get("pipeline", "joint")
    if pipeline == "two_stage":
        regression = rgan_out.get("F_hat")
        residual_generator = rgan_out.get("G")
        residual_discriminator = rgan_out.get("D")
        return {
            "pipeline": "two_stage",
            "generator": describe_model(residual_generator),
            "discriminator": describe_model(residual_discriminator),
            "regression": describe_model(regression),
            "residual_generator": describe_model(residual_generator),
            "residual_discriminator": describe_model(residual_discriminator),
        }
    generator = rgan_out.get("G", generator_fallback)
    discriminator = rgan_out.get("D", discriminator_fallback)
    return {
        "pipeline": "joint",
        "generator": describe_model(generator),
        "discriminator": describe_model(discriminator),
    }


def _build_rgan_metrics_config(
    base_config: TrainConfig,
    n_in: int,
    g_dense_act: Optional[str],
    pipeline: str,
) -> Dict[str, Any]:
    cfg = dict(
        pipeline=pipeline,
        n_in=int(n_in),
        units_g=base_config.units_g,
        units_d=base_config.units_d,
        lambda_reg=base_config.lambda_reg,
        lrG=base_config.lr_g,
        lrD=base_config.lr_d,
        dropout=base_config.dropout,
        g_layers=base_config.g_layers,
        d_layers=base_config.d_layers,
        g_dense=(g_dense_act if g_dense_act else "linear"),
        g_dense_activation=(g_dense_act if g_dense_act else "linear"),
        d_activation=base_config.d_activation or "sigmoid",
        gan_variant=base_config.gan_variant,
        d_steps=base_config.d_steps,
        g_steps=base_config.g_steps,
        wgan_gp_lambda=base_config.wgan_gp_lambda,
        wgan_clip_value=base_config.wgan_clip_value,
        use_logits=base_config.use_logits,
        amp=base_config.amp,
        device=base_config.device,
        noise_dim=base_config.noise_dim,
        lambda_diversity=base_config.lambda_diversity,
        preload_to_device=base_config.preload_to_device,
        compile_mode=base_config.compile_mode,
        critic_reg_interval=base_config.critic_reg_interval,
        critic_arch=base_config.critic_arch,
        layer_norm=True,
        use_spectral_norm=bool(base_config.gan_variant.lower() == "wgan-gp"),
        checkpoint_files={
            "generator": "rgan_generator.pt",
            "discriminator": "rgan_discriminator.pt",
        },
    )
    if pipeline == "two_stage":
        cfg.update(
            regression_epochs=base_config.regression_epochs,
            regression_lr=base_config.regression_lr,
            regression_patience=base_config.regression_patience,
            checkpoint_files={
                "regression": "rgan_regression.pt",
                "residual_generator": "rgan_residual_generator.pt",
                "residual_discriminator": "rgan_residual_discriminator.pt",
                "generator": "rgan_generator.pt",
                "discriminator": "rgan_discriminator.pt",
            },
        )
    return cfg


def _signature_differences(previous: Dict[str, Any], current: Dict[str, Any], prefix: str = "") -> List[str]:
    diffs: List[str] = []
    prev_keys = set(previous.keys())
    curr_keys = set(current.keys())
    for key in sorted(prev_keys | curr_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in previous:
            diffs.append(f"{full_key}: <missing> -> {current[key]!r}")
            continue
        if key not in current:
            diffs.append(f"{full_key}: {previous[key]!r} -> <missing>")
            continue
        prev_val = previous[key]
        curr_val = current[key]
        if isinstance(prev_val, dict) and isinstance(curr_val, dict):
            diffs.extend(_signature_differences(prev_val, curr_val, full_key))
        elif prev_val != curr_val:
            diffs.append(f"{full_key}: {prev_val!r} -> {curr_val!r}")
    return diffs


_ALLOWED_RESUME_PIPELINE_FLAG_KEYS = ("skip_classical", "skip_noise_robustness")


def _extract_allowed_resume_flag_changes(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    previous_pipeline = previous.get("pipeline", {}) if isinstance(previous, dict) else {}
    current_pipeline = current.get("pipeline", {}) if isinstance(current, dict) else {}
    changes: Dict[str, Dict[str, Any]] = {}
    for key in _ALLOWED_RESUME_PIPELINE_FLAG_KEYS:
        prev_val = previous_pipeline.get(key)
        curr_val = current_pipeline.get(key)
        if prev_val != curr_val:
            changes[f"pipeline.{key}"] = {"previous": prev_val, "current": curr_val}
    return changes


def _strip_allowed_resume_flag_changes(signature: Dict[str, Any]) -> Dict[str, Any]:
    cloned = copy.deepcopy(signature)
    pipeline = cloned.get("pipeline")
    if isinstance(pipeline, dict):
        for key in _ALLOWED_RESUME_PIPELINE_FLAG_KEYS:
            pipeline.pop(key, None)
    # mtime_ns is unreliable on SageMaker: the CSV gets a new modification
    # time each time it is copied to a fresh container, even though the file
    # content is identical.  Size-based identity is sufficient.
    dataset = cloned.get("dataset")
    if isinstance(dataset, dict):
        dataset.pop("mtime_ns", None)
    return cloned


def _invalidate_stages_for_resume_flag_changes(manifest: Dict[str, Any], changes: Dict[str, Dict[str, Any]]) -> List[str]:
    invalidated: List[str] = []

    def _invalidate(stage_name: str, reason: str) -> None:
        _invalidate_stage(manifest, stage_name, reason=reason)
        invalidated.append(stage_name)

    if "pipeline.skip_classical" in changes:
        _invalidate("classical_baselines", "resume_flag_changed:skip_classical")
        _invalidate("noise_robustness", "resume_flag_changed:skip_classical")
        _invalidate("reporting", "resume_flag_changed:skip_classical")

    if "pipeline.skip_noise_robustness" in changes:
        _invalidate("noise_robustness", "resume_flag_changed:skip_noise_robustness")
        _invalidate("reporting", "resume_flag_changed:skip_noise_robustness")

    return sorted(set(invalidated))


def _init_resume_manifest(signature: Dict[str, Any], results_dir: Path, resume_store: Path) -> Dict[str, Any]:
    return {
        "version": 1,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "signature": signature,
        "results_dir": str(results_dir.resolve()),
        "resume_store": str(resume_store.resolve()),
        "checkpoint": {
            "latest_path": "checkpoint_latest.pt" if resume_store != results_dir else "",
            "latest_epoch": 0,
        },
        "stages": {},
    }


def _manifest_path(store: Path) -> Path:
    return store / "resume_manifest.json"


_manifest_lock = threading.Lock()


def _write_resume_manifest(manifest: Dict[str, Any], results_dir: Path, resume_store: Path) -> None:
    with _manifest_lock:
        manifest["updated_at"] = _now_iso()
        manifest_path = _manifest_path(resume_store)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(manifest), f, indent=2)
        tmp_path.replace(manifest_path)
        if results_dir.resolve() != resume_store.resolve():
            dst = results_dir / "resume_manifest.json"
            tmp_dst = dst.with_suffix(".json.tmp")
            with open(tmp_dst, "w", encoding="utf-8") as f:
                json.dump(_sanitize_for_json(manifest), f, indent=2)
            tmp_dst.replace(dst)


def _load_resume_manifest(resume_store: Path) -> Optional[Dict[str, Any]]:
    path = _manifest_path(resume_store)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _peek_checkpoint_metadata(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "config_dict": ckpt.get("config_dict", {}),
    }


def _stage_cache_path(resume_store: Path, stage_name: str) -> Path:
    return resume_store / "stage_cache" / f"{stage_name}.pt"


def _save_stage_cache(resume_store: Path, stage_name: str, payload: Dict[str, Any]) -> str:
    cache_path = _stage_cache_path(resume_store, stage_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return str(cache_path.relative_to(resume_store))


def _load_stage_cache(resume_store: Path, stage_name: str) -> Dict[str, Any]:
    return torch.load(_stage_cache_path(resume_store, stage_name), map_location="cpu", weights_only=False)


def _stage_entry_complete(entry: Optional[Dict[str, Any]], resume_store: Path) -> bool:
    if not entry or entry.get("status") != "completed":
        return False
    cache_rel = entry.get("cache")
    if cache_rel and not (resume_store / cache_rel).exists():
        return False
    for rel_path in entry.get("artifacts", []):
        if not (resume_store / rel_path).exists():
            return False
    return True


def _stage_completed(manifest: Dict[str, Any], stage_name: str, resume_store: Path) -> bool:
    return _stage_entry_complete(manifest.get("stages", {}).get(stage_name), resume_store)


def _mark_stage(
    manifest: Dict[str, Any],
    stage_name: str,
    *,
    status: str,
    cache: Optional[str] = None,
    artifacts: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    with _manifest_lock:
        stage_entry = manifest.setdefault("stages", {}).get(stage_name, {})
        stage_entry["status"] = status
        stage_entry["updated_at"] = _now_iso()
        if status == "completed":
            stage_entry["completed_at"] = _now_iso()
        if cache is not None:
            stage_entry["cache"] = cache
        if artifacts is not None:
            stage_entry["artifacts"] = sorted(set(artifacts))
        if metadata is not None:
            stage_entry["metadata"] = metadata
        manifest["stages"][stage_name] = stage_entry


def _invalidate_stage(manifest: Dict[str, Any], stage_name: str, reason: str) -> None:
    entry = manifest.setdefault("stages", {}).get(stage_name, {})
    entry["status"] = "pending"
    entry["invalidated_at"] = _now_iso()
    entry["reason"] = reason
    manifest["stages"][stage_name] = entry


def _sync_artifact_to_resume_store(relative_path: str, results_dir: Path, resume_store: Path) -> str:
    src = results_dir / relative_path
    if not src.exists():
        raise FileNotFoundError(f"Artifact missing for resume sync: {src}")
    dest = resume_store / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return relative_path


def _restore_stage_artifacts(stage_entry: Dict[str, Any], resume_store: Path, results_dir: Path) -> None:
    for rel_path in stage_entry.get("artifacts", []):
        src = resume_store / rel_path
        if not src.exists():
            continue
        dest = results_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)


def _restore_all_completed_stage_artifacts(manifest: Dict[str, Any], resume_store: Path, results_dir: Path) -> None:
    for stage_entry in manifest.get("stages", {}).values():
        if _stage_entry_complete(stage_entry, resume_store):
            _restore_stage_artifacts(stage_entry, resume_store, results_dir)


def _save_json_artifact(obj: Any, relative_path: str, results_dir: Path, resume_store: Path) -> str:
    dest = results_dir / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(obj), f, indent=2, default=str)
    return _sync_artifact_to_resume_store(relative_path, results_dir, resume_store)


def _sync_existing_artifacts(relative_paths: List[str], results_dir: Path, resume_store: Path) -> List[str]:
    synced: List[str] = []
    for rel_path in relative_paths:
        if not rel_path:
            continue
        normalized = str(Path(rel_path))
        if (results_dir / normalized).exists():
            synced.append(_sync_artifact_to_resume_store(normalized, results_dir, resume_store))
    return sorted(set(synced))


def _save_stage_artifact_cache(
    manifest: Dict[str, Any],
    stage_name: str,
    payload: Dict[str, Any],
    *,
    results_dir: Path,
    resume_store: Path,
    artifacts: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cache_rel = _save_stage_cache(resume_store, stage_name, payload)
    _mark_stage(
        manifest,
        stage_name,
        status="completed",
        cache=cache_rel,
        artifacts=artifacts or [],
        metadata=metadata or {},
    )
    _write_resume_manifest(manifest, results_dir, resume_store)
    return payload


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
    ap.add_argument(
        "--pipeline",
        choices=["two_stage", "joint"],
        default="two_stage",
        help="Training pipeline: 'two_stage' (paper Algorithm 1) or 'joint' (legacy).",
    )
    ap.add_argument("--regression_epochs", type=int, default=100, help="Epochs for Stage 1 regression model (two_stage only).")
    ap.add_argument("--regression_lr", type=float, default=5e-4, help="Learning rate for Stage 1 regression model.")
    ap.add_argument("--regression_patience", type=int, default=15, help="Early stopping patience for Stage 1.")
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
    ap.add_argument(
        "--critic_arch",
        choices=["lstm", "tcn"],
        default="tcn",
        help="Critic architecture to use with the existing GAN objective.",
    )
    ap.add_argument("--g_dense_activation", default="", help="Optional dense activation for the generator (PyTorch activation name).")
    ap.add_argument("--d_activation", default="sigmoid", help="Discriminator activation (PyTorch activation name).")
    ap.add_argument("--lr_g", "--lrG", type=float, default=5e-4, dest="lr_g")
    ap.add_argument("--lr_d", "--lrD", type=float, default=5e-4, dest="lr_d")
    ap.add_argument("--label_smooth", type=float, default=0.9)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--noise_dim",
        type=int,
        default=16,
        help="Latent noise dimension for stochastic generation (RCGAN-style). 0 = deterministic.",
    )
    ap.add_argument(
        "--lambda_diversity",
        type=float,
        default=0.1,
        help="Weight for MSGAN diversity loss (encourages different z -> different output). 0 = disabled.",
    )
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
        "--preload_to_device",
        choices=["auto", "always", "never"],
        default="never",
        help=(
            "Move train/val/test tensors to the GPU once before RGAN training when feasible. "
            "'auto' enables this only for CUDA runs under a safe size threshold."
        ),
    )
    ap.add_argument(
        "--compile_mode",
        choices=["off", "reduce-overhead"],
        default="off",
        help=(
            "Compile the RGAN recurrent training path with torch.compile. "
            "'reduce-overhead' targets small-kernel GPU workloads."
        ),
    )
    ap.add_argument(
        "--critic_reg_interval",
        type=int,
        default=1,
        help=(
            "Apply the WGAN-GP critic regularizer every N critic updates. "
            "Values > 1 use lazy regularization and scale the penalty on applied steps."
        ),
    )
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
    ap.add_argument(
        "--only_models",
        default="",
        help=(
            "Comma-separated list of models to train (e.g. 'patchtst,itransformer'). "
            "Models not listed will be loaded from --prior_results instead of trained. "
            "Valid names: rgan, lstm, dlinear, nlinear, fits, patchtst, itransformer, autoformer, informer."
        ),
    )
    ap.add_argument(
        "--prior_results",
        default="",
        help=(
            "Path to a prior results directory containing model weights (models/*.pt) "
            "and metrics.json. Used with --only_models to load skipped models."
        ),
    )
    ap.add_argument("--tune_csv", default="")
    ap.add_argument("--results_dir", default="./results/experiment")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true",
                    help="Enable CUDA deterministic mode for reproducible runs (slower).")
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

    resolved_device_str = "cpu"

    if args.require_cuda:
        if not torch.cuda.is_available():
            console.print(f"CRITICAL ERROR: --require_cuda specified but no CUDA device found.")
            console.print(_cuda_build_hint())
            sys.exit(1)

        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            console.print(f"CRITICAL ERROR: --gpu_id {args.gpu_id} requested but only {torch.cuda.device_count()} devices available.")
            sys.exit(1)

        torch.cuda.set_device(args.gpu_id)
        device_name = torch.cuda.get_device_name(args.gpu_id)
        console.print(f"Strict GPU Mode Active: Using device {args.gpu_id} ({device_name})")
        console.print("CPU fallback is DISABLED.")
        resolved_device_str = f"cuda:{args.gpu_id}"

    elif torch.cuda.is_available():
        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            console.print(f"WARNING: --gpu_id {args.gpu_id} requested but only {torch.cuda.device_count()} devices available. Falling back to CPU.")
        else:
            torch.cuda.set_device(args.gpu_id)
            device_name = torch.cuda.get_device_name(args.gpu_id)
            console.print(f"GPU Detected: {device_name} (ID: {args.gpu_id})")
            console.print("Using CUDA for training. Fallback to CPU enabled if CUDA fails.")
            console.print(_cuda_build_hint())
            resolved_device_str = f"cuda:{args.gpu_id}"
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

    resume_store = _resume_store_path(results_dir, args.checkpoint_dir)
    resume_store.mkdir(parents=True, exist_ok=True)
    resume_models_dir = resume_store / "models"
    resume_models_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Resume store: {resume_store.resolve()}")

    resume_signature = _build_resume_signature(args)
    resume_manifest = _load_resume_manifest(resume_store)
    checkpoint_meta = _peek_checkpoint_metadata(args.resume_from)
    resumed_job = checkpoint_meta is not None

    if resume_manifest is None:
        if resumed_job:
            raise ValueError(
                "Resume checkpoint detected but resume_manifest.json is missing from the resume store. "
                "Strict resume cannot be enforced; restore the manifest or start a fresh run."
            )
        resume_manifest = _init_resume_manifest(resume_signature, results_dir, resume_store)
        log_info("Resume manifest initialized.")
    else:
        previous_signature = resume_manifest.get("signature", {})
        eval_flag_changes = _extract_allowed_resume_flag_changes(previous_signature, resume_signature)
        diffs = _signature_differences(
            _strip_allowed_resume_flag_changes(previous_signature),
            _strip_allowed_resume_flag_changes(resume_signature),
        )
        if diffs:
            diff_msg = "\n".join(f"  - {item}" for item in diffs)
            raise ValueError(
                "Resume config drift detected. Only total --epochs, resume/job plumbing, and eval-stage skip flags may change.\n"
                f"{diff_msg}"
            )
        log_info("Resume manifest signature matches current run.")
        if eval_flag_changes:
            log_info(
                "Allowed resume flag changes detected:\n"
                + "\n".join(
                    f"  - {name}: {change['previous']!r} -> {change['current']!r}"
                    for name, change in sorted(eval_flag_changes.items())
                )
            )
            invalidated_stages = _invalidate_stages_for_resume_flag_changes(resume_manifest, eval_flag_changes)
            if invalidated_stages:
                log_info(
                    "Invalidated resume stages for re-evaluation: "
                    + ", ".join(invalidated_stages)
                )
                _write_resume_manifest(resume_manifest, results_dir, resume_store)

    if checkpoint_meta is not None:
        checkpoint_epoch = int(checkpoint_meta["epoch"])
        resume_manifest.setdefault("checkpoint", {})["latest_epoch"] = checkpoint_epoch
        if args.epochs < checkpoint_epoch:
            raise ValueError(
                f"Resume checkpoint is already at epoch {checkpoint_epoch}, but --epochs={args.epochs}. "
                "Increase --epochs or start a fresh run."
            )
        log_info(
            f"Resume checkpoint detected: epoch={checkpoint_epoch}, "
            f"target_epochs={args.epochs}, remaining_epochs={max(0, args.epochs - checkpoint_epoch)}"
        )
    else:
        checkpoint_epoch = 0

    resume_manifest["signature"] = resume_signature
    resume_manifest["results_dir"] = str(results_dir.resolve())
    resume_manifest["resume_store"] = str(resume_store.resolve())
    resume_manifest.setdefault("checkpoint", {})["latest_path"] = "checkpoint_latest.pt" if args.checkpoint_dir else ""
    _write_resume_manifest(resume_manifest, results_dir, resume_store)

    if resumed_job and checkpoint_epoch == args.epochs:
        reporting_entry = resume_manifest.get("stages", {}).get("reporting")
        if _stage_entry_complete(reporting_entry, resume_store):
            log_info("Checkpoint already matches target epochs and reporting artifacts are complete; restoring cached artifacts and exiting.")
            _restore_all_completed_stage_artifacts(resume_manifest, resume_store, results_dir)
            close_log_file()
            return

    noise_levels = parse_noise_levels(str(args.noise_levels))

    describe_resource_heavy_steps(args, console)

    try:
        from rgan.models_torch import (
            build_generator as build_generator_backend,
            build_discriminator as build_discriminator_backend,
            build_regression_model as build_regression_model_backend,
            build_residual_generator as build_residual_generator_backend,
            build_residual_discriminator as build_residual_discriminator_backend,
        )
        from rgan.rgan_torch import (
            train_rgan_torch as train_rgan_backend,
            train_two_stage as train_two_stage_backend,
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

    # ── Parse --only_models / --prior_results for selective retraining ──
    ALL_NEURAL_MODELS = {"rgan", "lstm", "dlinear", "nlinear", "fits", "patchtst", "itransformer", "autoformer", "informer"}
    standalone_rgan_only = False
    if args.only_models:
        _only_set = {m.strip().lower() for m in args.only_models.split(",") if m.strip()}
        invalid = _only_set - ALL_NEURAL_MODELS
        if invalid:
            log_error(f"Unknown model names in --only_models: {invalid}. Valid: {sorted(ALL_NEURAL_MODELS)}")
            sys.exit(1)
        standalone_rgan_only = (_only_set == {"rgan"} and not args.prior_results)
        if standalone_rgan_only:
            _prior_dir = None
            log_info("Selective retraining: training only RGAN without prior baseline artifacts.")
        else:
            if not args.prior_results:
                log_error("--only_models requires --prior_results unless it is exactly '--only_models rgan'.")
                sys.exit(1)
            _prior_dir = Path(args.prior_results)
            if not (_prior_dir / "models").is_dir():
                log_error(f"--prior_results directory missing 'models/' subfolder: {_prior_dir}")
                sys.exit(1)
            log_info(f"Selective retraining: only training {sorted(_only_set)}, loading rest from {_prior_dir}")
    else:
        _only_set = ALL_NEURAL_MODELS  # train everything
        _prior_dir = None

    def _should_train(model_name: str) -> bool:
        return model_name in _only_set

    def _load_prior_model_and_predict(model_class, model_kwargs, pt_filename, data_dict, device):
        """Load a model from prior results, run inference, return an *_out-style dict."""
        pt_path = _prior_dir / "models" / pt_filename
        if not pt_path.exists():
            log_error(f"Prior model file not found: {pt_path}")
            raise FileNotFoundError(pt_path)
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        model.eval()
        Xtr, Ytr = data_dict["Xtr"], data_dict["Ytr"]
        Xte, Yte = data_dict["Xte"], data_dict["Yte"]
        eval_bs = max(1, min(1024, len(Xtr)))
        def _batch_predict(X):
            preds = []
            with torch.no_grad():
                for i in range(0, len(X), eval_bs):
                    xb = torch.from_numpy(X[i:i+eval_bs]).to(device)
                    preds.append(model(xb).cpu().numpy())
            return np.concatenate(preds, axis=0)
        tr = _batch_predict(Xtr)
        te = _batch_predict(Xte)
        train_stats = error_stats(Ytr.reshape(-1), tr.reshape(-1))
        test_stats = error_stats(Yte.reshape(-1), te.reshape(-1))
        return {
            "model": model,
            "history": {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": []},
            "train_stats": train_stats,
            "test_stats": test_stats,
            "pred_train": tr,
            "pred_test": te,
        }

    set_seed(args.seed, deterministic=getattr(args, 'deterministic', False))

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
    device_str = resolved_device_str

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
        lr_g=args.lr_g,
        lr_d=args.lr_d,
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
        preload_to_device=args.preload_to_device,
        compile_mode=args.compile_mode,
        critic_reg_interval=args.critic_reg_interval,
        critic_arch=args.critic_arch,
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
        noise_dim=args.noise_dim,
        lambda_diversity=args.lambda_diversity,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        eval_every=args.eval_every,
        pipeline=args.pipeline,
        regression_epochs=args.regression_epochs,
        regression_lr=args.regression_lr,
        regression_patience=args.regression_patience,
    )

    print_kv_table(console, "Configuration", {
        "L/H": f"{args.L}/{args.H}",
        "Epochs": args.epochs,
        "Batch Size": args.batch_size,
        "Eval Batch Size": args.eval_batch_size,
        "Units (G/D)": f"{args.units_g}/{args.units_d}",
        "GAN Variant": args.gan_variant,
        "D/G Steps": f"{args.d_steps}/{args.g_steps}",
        "Preload / Compile": f"{args.preload_to_device} / {args.compile_mode}",
        "Critic Reg Interval": args.critic_reg_interval,
        "Critic Arch": args.critic_arch,
        "Lambda": args.lambda_reg,
        "Learning Rates (G/D)": f"{args.lr_g}/{args.lr_d}",
        "Dropout": args.dropout,
        "Layers (G/D)": f"{args.g_layers}/{args.d_layers}",
    })

    _sync_artifact_to_resume_store("run_config.json", results_dir, resume_store)
    _mark_stage(
        resume_manifest,
        "data_prep",
        status="completed",
        artifacts=["run_config.json"],
        metadata={
            "target_col": args.target,
            "time_col": args.time_col,
            "num_train_windows": int(Xfull_tr.shape[0]),
            "num_test_windows": int(Xte.shape[0]),
        },
    )
    _write_resume_manifest(resume_manifest, results_dir, resume_store)

    if resumed_job and checkpoint_epoch < args.epochs:
        log_info("Target epochs increased on resume; invalidating RGAN-dependent downstream stages.")
        for _stage_name in ("rgan", "bootstrap", "noise_robustness", "reporting"):
            _invalidate_stage(resume_manifest, _stage_name, reason="target_epochs_increased")
        _write_resume_manifest(resume_manifest, results_dir, resume_store)

    stage_changed = {
        "rgan": False,
        "classical_baselines": False,
        "lstm": False,
        "dlinear": False,
        "nlinear": False,
        "fits": False,
        "patchtst": False,
        "itransformer": False,
        "autoformer": False,
        "informer": False,
    }

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

            lr_candidates_g = sorted({max(1e-5, args.lr_g * factor) for factor in (0.5, 1.0, 1.5)})
            lr_candidates_g.append(max(1e-5, args.lr_g))
            lr_candidates_g = sorted(set(lr_candidates_g))
            lr_candidates_d = sorted({max(1e-5, args.lr_d * factor) for factor in (0.5, 1.0, 1.5)})
            lr_candidates_d.append(max(1e-5, args.lr_d))
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
    log_step(f"Building Generator: L={base_config.L}, H={base_config.H}, n_in={Xtr.shape[-1]}, units={base_config.units_g}, layers={base_config.g_layers}, dropout={base_config.dropout}, dense_act={g_dense_act}, layer_norm={layer_norm}, noise_dim={base_config.noise_dim}")
    G = build_generator_backend(
        L=base_config.L,
        H=base_config.H,
        n_in=Xtr.shape[-1],
        units=base_config.units_g,
        dropout=base_config.dropout,
        num_layers=base_config.g_layers,
        dense_activation=g_dense_act,
        layer_norm=layer_norm,
        noise_dim=base_config.noise_dim,
    )
    log_step(f"Generator built: {sum(p.numel() for p in G.parameters())} parameters")
    log_step(
        f"Building Discriminator: arch={base_config.critic_arch}, units={base_config.units_d}, "
        f"layers={base_config.d_layers}, activation={base_config.d_activation}, spectral_norm={use_spectral_norm}"
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
        critic_arch=base_config.critic_arch,
    )

    log_step(f"Discriminator built: {sum(p.numel() for p in D.parameters())} parameters")

    # Two-stage pipeline models (Paper Algorithm 1)
    F_hat = None
    G_residual = None
    D_residual = None
    if base_config.pipeline == "two_stage":
        log_step("Building Two-Stage Pipeline models (Paper Algorithm 1)")
        F_hat = build_regression_model_backend(
            L=base_config.L,
            H=base_config.H,
            n_in=Xtr.shape[-1],
            units=base_config.units_g,
            num_layers=base_config.g_layers,
            dropout=base_config.dropout,
            layer_norm=layer_norm,
        )
        log_step(f"RegressionModel f̂ built: {sum(p.numel() for p in F_hat.parameters())} parameters")
        G_residual = build_residual_generator_backend(
            H=base_config.H,
            noise_dim=base_config.noise_dim,
            units=base_config.units_g,
            num_layers=base_config.g_layers,
            dropout=base_config.dropout,
            layer_norm=layer_norm,
        )
        log_step(f"ResidualGenerator G(z) built: {sum(p.numel() for p in G_residual.parameters())} parameters")
        D_residual = build_residual_discriminator_backend(
            H=base_config.H,
            units=base_config.units_d,
            num_layers=base_config.d_layers,
            dropout=base_config.dropout,
            activation=base_config.d_activation,
            layer_norm=layer_norm,
            use_spectral_norm=use_spectral_norm,
            critic_arch=base_config.critic_arch,
        )
        log_step(f"Residual Discriminator D(r) built: {sum(p.numel() for p in D_residual.parameters())} parameters")

    # Save all model weights to a models/ subdirectory
    models_dir = results_dir / "models"
    models_dir.mkdir(exist_ok=True)
    log_step(f"Models directory: {models_dir.resolve()}")

    def _sync_model_artifact(filename: str) -> str:
        return _sync_artifact_to_resume_store(f"models/{filename}", results_dir, resume_store)

    def _save_model(name: str, model_obj):
        """Save a PyTorch model's state_dict to models/<name>.pt"""
        path = models_dir / f"{name}.pt"
        torch.save(model_obj.state_dict(), path)
        _sync_model_artifact(f"{name}.pt")
        return path

    def _save_model_state_dict(name: str, state_dict: Dict[str, Any]):
        """Save a raw state_dict to models/<name>.pt."""
        path = models_dir / f"{name}.pt"
        torch.save(state_dict, path)
        _sync_model_artifact(f"{name}.pt")
        return path

    def _save_tree_model_artifact(tree_model_obj):
        import joblib

        results_path = models_dir / "tree_ensemble.joblib"
        joblib.dump(tree_model_obj, results_path)
        _sync_model_artifact("tree_ensemble.joblib")
        return results_path

    def _load_tree_model_artifact():
        import joblib

        model_path = resume_models_dir / "tree_ensemble.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing cached tree ensemble model artifact: {model_path}")
        return joblib.load(model_path)

    def _load_cached_model_stage(
        stage_name: str,
        model_class,
        model_kwargs: Dict[str, Any],
        model_filename: str,
        device: torch.device,
    ) -> Dict[str, Any]:
        cache = _load_stage_cache(resume_store, stage_name)
        model = model_class(**model_kwargs)
        state_path = resume_models_dir / model_filename
        state_dict = torch.load(state_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        _restore_stage_artifacts(resume_manifest["stages"][stage_name], resume_store, results_dir)
        return {
            "model": model,
            "history": cache.get("history", {"epoch": [], "train_rmse": [], "test_rmse": [], "val_rmse": []}),
            "train_stats": cache["train_stats"],
            "test_stats": cache["test_stats"],
            "pred_train": cache.get("pred_train"),
            "pred_test": cache["pred_test"],
        }

    def _save_model_stage_cache(stage_name: str, out: Dict[str, Any], model_filename: str) -> None:
        payload = {
            "history": out.get("history", {}),
            "train_stats": out.get("train_stats"),
            "test_stats": out.get("test_stats"),
            "pred_train": out.get("pred_train"),
            "pred_test": out.get("pred_test"),
            "model_filename": model_filename,
        }
        _save_stage_artifact_cache(
            resume_manifest,
            stage_name,
            payload,
            results_dir=results_dir,
            resume_store=resume_store,
            artifacts=[f"models/{model_filename}"],
            metadata={"skipped": False},
        )

    def _load_cached_rgan_stage() -> Dict[str, Any]:
        cache = _load_stage_cache(resume_store, "rgan")
        bundle = _load_rgan_bundle_from_run_dir(
            run_dir=resume_store,
            n_in_current=Xtr.shape[-1],
            L=base_config.L,
            H=base_config.H,
            device=_load_device,
            prefer_ema=True,
        )
        _restore_stage_artifacts(resume_manifest["stages"]["rgan"], resume_store, results_dir)
        return {
            **bundle,
            "history": cache["history"],
            "train_stats": cache["train_stats"],
            "test_stats": cache["test_stats"],
            "pred_train": cache.get("pred_train"),
            "pred_test": cache["pred_test"],
        }

    log_step(f"Device for training: {device_str}")
    log_step(f"AMP enabled: {base_config.amp}")
    log_step(f"GAN variant: {base_config.gan_variant}")
    log_step(f"Checkpoint dir: {base_config.checkpoint_dir or '(none)'}")
    log_step(f"Resume from: {base_config.resume_from or '(none)'}")

    def _make_skipped_model(reason: str = "not_selected") -> Dict[str, Any]:
        return {"skipped": True, "reason": reason}

    def _train_or_load_rgan() -> Dict[str, Any]:
        if _stage_completed(resume_manifest, "rgan", resume_store):
            log_info("Skipping RGAN stage on resume; loading cached outputs and model artifacts.")
            return _load_cached_rgan_stage()

        _mark_stage(resume_manifest, "rgan", status="running", metadata={"target_epochs": args.epochs})
        _write_resume_manifest(resume_manifest, results_dir, resume_store)

        if _should_train("rgan"):
            data_splits = {"Xtr": Xtr, "Ytr": Ytr, "Xval": Xval, "Yval": Yval, "Xte": Xte, "Yte": Yte}

            if base_config.pipeline == "two_stage" and F_hat is not None:
                with log_phase(console, "Train R-WGAN Two-Stage (Paper Algorithm 1)"):
                    log_step("Calling train_two_stage_backend...")
                    rgan_result = train_two_stage_backend(
                        base_config,
                        (F_hat, G_residual, D_residual),
                        data_splits,
                        str(results_dir),
                        tag="rgan_two_stage",
                    )
                    log_step(
                        f"Two-stage training completed. Train RMSE={rgan_result['train_stats'].get('rmse', 'N/A')}, "
                        f"Test RMSE={rgan_result['test_stats'].get('rmse', 'N/A')}"
                    )
                    _save_model("rgan_regression", rgan_result["F_hat"])
                    _save_model("rgan_residual_generator", rgan_result["G"])
                    _save_model("rgan_residual_discriminator", rgan_result["D"])
                    # Also save as rgan_generator for compatibility with augmentation
                    _save_model("rgan_generator", rgan_result["G"])
                    _save_model("rgan_discriminator", rgan_result["D"])
                    log_step(f"Saved two-stage models to: {models_dir}")
                    _save_stage_artifact_cache(
                        resume_manifest,
                        "rgan",
                        {
                            "history": rgan_result["history"],
                            "train_stats": rgan_result["train_stats"],
                            "test_stats": rgan_result["test_stats"],
                            "pred_train": rgan_result.get("pred_train"),
                            "pred_test": rgan_result["pred_test"],
                            "pipeline": "two_stage",
                        },
                        results_dir=results_dir,
                        resume_store=resume_store,
                        artifacts=[
                            "models/rgan_regression.pt",
                            "models/rgan_residual_generator.pt",
                            "models/rgan_residual_discriminator.pt",
                            "models/rgan_generator.pt",
                            "models/rgan_discriminator.pt",
                        ],
                        metadata={"pipeline": "two_stage", "target_epochs": args.epochs},
                    )
                    stage_changed["rgan"] = True
                    return rgan_result
            else:
                with log_phase(console, "Train R-GAN Joint (PyTorch)"):
                    log_step("Calling train_rgan_backend...")
                    rgan_result = train_rgan_backend(
                        base_config,
                        (G, D),
                        data_splits,
                        str(results_dir),
                        tag="rgan",
                    )

                log_step(
                    f"R-GAN training completed. Train RMSE={rgan_result['train_stats'].get('rmse', 'N/A')}, "
                    f"Test RMSE={rgan_result['test_stats'].get('rmse', 'N/A')}"
                )
                _save_model("rgan_generator", rgan_result["G"])
                _save_model("rgan_discriminator", rgan_result["D"])
                if rgan_result.get("G_ema"):
                    _save_model("rgan_generator_ema", rgan_result["G_ema"])
                log_step(f"Saved RGAN models to: {models_dir}")
                _save_stage_artifact_cache(
                    resume_manifest,
                    "rgan",
                    {
                        "history": rgan_result["history"],
                        "train_stats": rgan_result["train_stats"],
                        "test_stats": rgan_result["test_stats"],
                        "pred_train": rgan_result.get("pred_train"),
                        "pred_test": rgan_result["pred_test"],
                        "pipeline": "joint",
                    },
                    results_dir=results_dir,
                    resume_store=resume_store,
                    artifacts=[
                        "models/rgan_generator.pt",
                        "models/rgan_discriminator.pt",
                    ] + (["models/rgan_generator_ema.pt"] if rgan_result.get("G_ema") else []),
                    metadata={
                        "pipeline": "joint",
                        "checkpoint_epoch": int(max(rgan_result["history"].get("epoch", [0]) or [0])),
                        "target_epochs": args.epochs,
                    },
                )
                stage_changed["rgan"] = True
                return rgan_result

        with log_phase(console, "Load R-GAN from prior results"):
            bundle = _load_rgan_bundle_from_run_dir(
                run_dir=_prior_dir,
                n_in_current=Xtr.shape[-1],
                L=base_config.L,
                H=base_config.H,
                device=_load_device,
                prefer_ema=True,
            )
            log_step(f"Loaded prior RGAN pipeline={bundle['pipeline']}")
            if bundle["pipeline"] == "two_stage":
                _save_model("rgan_regression", bundle["F_hat"])
                _save_model("rgan_residual_generator", bundle["G"])
                _save_model("rgan_residual_discriminator", bundle["D"])
                _save_model("rgan_generator", bundle["G"])
                _save_model("rgan_discriminator", bundle["D"])
            else:
                _save_model("rgan_generator", bundle["G"])
                if bundle.get("D") is not None:
                    _save_model("rgan_discriminator", bundle["D"])
                else:
                    log_warn("Prior results do not include rgan_discriminator.pt; skipping discriminator artifact copy.")
                if bundle.get("G_ema") is not None:
                    _save_model("rgan_generator_ema", bundle["G_ema"])
                    log_step("Loaded real EMA checkpoint from prior results.")

            _, rgan_train_pred_loaded = _predict_rgan_bundle(
                bundle, Xtr, Y=Ytr, batch_size=max(1, base_config.eval_batch_size), deterministic=True, seed=args.seed
            )
            _, rgan_test_pred_loaded = _predict_rgan_bundle(
                bundle, Xte, Y=Yte, batch_size=max(1, base_config.eval_batch_size), deterministic=True, seed=args.seed
            )
            rgan_result = {
                **bundle,
                "history": {
                    "epoch": [],
                    "train_rmse": [],
                    "test_rmse": [],
                    "val_rmse": [],
                    "D_loss": [],
                    "G_loss": [],
                },
                "train_stats": error_stats(Ytr.reshape(-1), rgan_train_pred_loaded.reshape(-1)),
                "test_stats": error_stats(Yte.reshape(-1), rgan_test_pred_loaded.reshape(-1)),
                "pred_train": rgan_train_pred_loaded,
                "pred_test": rgan_test_pred_loaded,
            }
            log_step(f"RGAN loaded. Test RMSE={rgan_result['test_stats'].get('rmse', 'N/A')}")
            artifact_paths = [
                "models/rgan_generator.pt",
                "models/rgan_discriminator.pt",
            ] + (["models/rgan_generator_ema.pt"] if bundle.get("G_ema") is not None else [])
            if bundle["pipeline"] == "two_stage":
                artifact_paths = [
                    "models/rgan_regression.pt",
                    "models/rgan_residual_generator.pt",
                    "models/rgan_residual_discriminator.pt",
                    "models/rgan_generator.pt",
                    "models/rgan_discriminator.pt",
                ]
            _save_stage_artifact_cache(
                resume_manifest,
                "rgan",
                {
                    "history": rgan_result["history"],
                    "train_stats": rgan_result["train_stats"],
                    "test_stats": rgan_result["test_stats"],
                    "pred_train": rgan_result.get("pred_train"),
                    "pred_test": rgan_result["pred_test"],
                    "pipeline": bundle["pipeline"],
                },
                results_dir=results_dir,
                resume_store=resume_store,
                artifacts=artifact_paths,
                metadata={
                    "loaded_from_prior_results": True,
                    "pipeline": bundle["pipeline"],
                    "target_epochs": args.epochs,
                },
            )
            stage_changed["rgan"] = True
            return rgan_result

    if standalone_rgan_only:
        rgan_out = _train_or_load_rgan()
        rgan_architecture = _describe_rgan_architecture(rgan_out, generator_fallback=G, discriminator_fallback=D)
        env_info = _collect_environment_info(args.seed)
        skipped_models = {
            "lstm": _make_skipped_model(),
            "dlinear": _make_skipped_model(),
            "nlinear": _make_skipped_model(),
            "fits": _make_skipped_model(),
            "patchtst": _make_skipped_model(),
            "itransformer": _make_skipped_model(),
            "autoformer": _make_skipped_model(),
            "informer": _make_skipped_model(),
            "naive_baseline": _make_skipped_model(),
            "arima": _make_skipped_model(),
            "arma": _make_skipped_model(),
            "tree_ensemble": _make_skipped_model(),
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
            noise_robustness=[],
            rgan=dict(
                train=rgan_out["train_stats"],
                test=rgan_out["test_stats"],
                history=rgan_out["history"],
                architecture=rgan_architecture,
                config=_build_rgan_metrics_config(
                    base_config,
                    n_in=int(Xtr.shape[-1]),
                    g_dense_act=g_dense_act,
                    pipeline=rgan_out.get("pipeline", base_config.pipeline),
                ),
            ),
            charts=dict(
                training_curves_overlay="",
                training_curves_overlay_interactive="",
                ranked_model_bars="",
                ranked_model_bars_interactive="",
                noise_robustness_heatmap="",
                noise_robustness_heatmap_interactive="",
                multi_metric_radar="",
                multi_metric_radar_interactive="",
            ),
            classical={"skipped": True, "reason": "standalone_rgan_only"},
            advanced_metrics={},
            statistical_tests=dict(diebold_mariano={}),
            tuning=dict(
                enabled=bool(args.tune),
                dataset=used_tune,
                best=best_hp or {},
            ),
            environment=env_info,
            scaling={"target_mean": float(target_mean), "target_std": float(target_std)},
            created=datetime.now(timezone.utc).isoformat(),
            mode={"standalone_rgan_only": True},
            **skipped_models,
        )

        _save_json_artifact(metrics, "metrics.json", results_dir, resume_store)

        total_elapsed = time.time() - run_start_time
        summary = {
            "dataset": args.csv,
            "target": target_col,
            "results_dir": str(results_dir.resolve()),
            "elapsed_seconds": round(total_elapsed, 1),
            "epochs": args.epochs,
            "leaderboard": [
                {
                    "rank": 1,
                    "model": "RGAN",
                    "test_rmse_orig": rgan_out["test_stats"].get(
                        "rmse_orig",
                        rgan_out["test_stats"].get("rmse"),
                    ),
                }
            ],
            "best_model": "RGAN",
            "best_rmse": rgan_out["test_stats"].get("rmse_orig", rgan_out["test_stats"].get("rmse")),
            "completed_at": datetime.now().isoformat(),
            "num_models_trained": 1,
            "models_saved": [f.name for f in (results_dir / "models").glob("*.pt")],
            "mode": "standalone_rgan_only",
        }
        _save_json_artifact(summary, "run_summary.json", results_dir, resume_store)
        _save_stage_artifact_cache(
            resume_manifest,
            "reporting",
            {
                "mode": "standalone_rgan_only",
                "metrics": metrics,
                "summary": summary,
            },
            results_dir=results_dir,
            resume_store=resume_store,
            artifacts=["metrics.json", "run_summary.json"],
            metadata={"standalone_rgan_only": True, "target_epochs": args.epochs},
        )

        log_info("Standalone RGAN-only run complete.")
        log_info(f"Results saved to: {results_dir.resolve()}")
        close_log_file()
        return

    # ── Launch classical baselines on CPU in background while GPU trains ──
    _classical_results = {}  # filled by background thread

    def _run_classical_baselines():
        """Run CPU-only baselines (naive, ARIMA, ARMA, tree) in a background thread."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend safe for background threads
            _mark_stage(resume_manifest, "classical_baselines", status="running")
            _write_resume_manifest(resume_manifest, results_dir, resume_store)
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
            _classical_results["naive_train_pred"] = naive_train_pred_bg
            _classical_results["naive_test_pred"] = naive_test_pred_bg
            _classical_results["naive_train_stats"] = naive_train_stats_bg
            _classical_results["naive_test_stats"] = naive_test_stats_bg
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
            _classical_results["arima_train_stats"] = arima_train_stats_bg
            _classical_results["arima_test_stats"] = arima_test_stats_bg
            _classical_results["arima_test_pred"] = arima_test_pred_bg
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
            _classical_results["arma_train_stats"] = arma_train_stats_bg
            _classical_results["arma_test_stats"] = arma_test_stats_bg
            _classical_results["arma_test_pred"] = arma_test_pred_bg
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
            _classical_results["tree_train_stats"] = tree_train_stats_bg
            _classical_results["tree_test_stats"] = tree_test_stats_bg
            _classical_results["tree_test_pred"] = tree_test_pred_bg
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
            _save_tree_model_artifact(tree_model_bg)

            log_info("BACKGROUND: All classical baselines completed.")

            # Persist results so resumed jobs can skip recomputation
            _cache_path = results_dir / "classical_baselines_cache.pkl"
            try:
                import pickle
                with open(_cache_path, "wb") as _f:
                    pickle.dump(dict(_classical_results), _f, protocol=pickle.HIGHEST_PROTOCOL)
                log_info(f"BACKGROUND: Classical baselines cached to {_cache_path}")
                _sync_artifact_to_resume_store("classical_baselines_cache.pkl", results_dir, resume_store)
            except Exception as _save_exc:
                log_error(f"BACKGROUND: Failed to cache classical baselines — {_save_exc}")

            _save_stage_artifact_cache(
                resume_manifest,
                "classical_baselines",
                {
                    "naive_train_pred": _classical_results.get("naive_train_pred"),
                    "naive_test_pred": _classical_results.get("naive_test_pred"),
                    "naive_train_stats": _classical_results.get("naive_train_stats"),
                    "naive_test_stats": _classical_results.get("naive_test_stats"),
                    "arima_train_stats": _classical_results.get("arima_train_stats"),
                    "arima_test_stats": _classical_results.get("arima_test_stats"),
                    "arima_test_pred": _classical_results.get("arima_test_pred"),
                    "arma_train_stats": _classical_results.get("arma_train_stats"),
                    "arma_test_stats": _classical_results.get("arma_test_stats"),
                    "arma_test_pred": _classical_results.get("arma_test_pred"),
                    "tree_train_stats": _classical_results.get("tree_train_stats"),
                    "tree_test_stats": _classical_results.get("tree_test_stats"),
                    "tree_test_pred": _classical_results.get("tree_test_pred"),
                    "tree_config": _classical_results.get("tree_config", {}),
                    "skip_classical": _classical_results.get("skip_classical", False),
                },
                results_dir=results_dir,
                resume_store=resume_store,
                artifacts=(
                    ["models/tree_ensemble.joblib"] if "tree_model" in _classical_results else []
                ) + (
                    ["classical_baselines_cache.pkl"] if _cache_path.exists() else []
                ),
            )
            stage_changed["classical_baselines"] = True

        except Exception as exc:
            log_error(f"BACKGROUND: Classical baselines FAILED — {type(exc).__name__}: {exc}")
            _classical_results["error"] = exc
            _mark_stage(
                resume_manifest,
                "classical_baselines",
                status="failed",
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
            _write_resume_manifest(resume_manifest, results_dir, resume_store)

    # ── Try to load cached classical baselines (saves ~64 min on resume) ──
    _classical_cache_loaded = False
    if _stage_completed(resume_manifest, "classical_baselines", resume_store):
        try:
            cached_classical = _load_stage_cache(resume_store, "classical_baselines")
            _classical_results = dict(cached_classical)
            if (resume_models_dir / "tree_ensemble.joblib").exists():
                _classical_results["tree_model"] = _load_tree_model_artifact()
            _restore_stage_artifacts(resume_manifest["stages"]["classical_baselines"], resume_store, results_dir)
            log_info("Loaded cached classical baselines from resume store — skipping recomputation")
            _classical_cache_loaded = True
        except Exception as _load_exc:
            log_error(f"Failed to load cached classical baselines from resume store — {_load_exc}")

    if _classical_cache_loaded:
        # Create a no-op thread that's already "done" so the join() later works
        classical_thread = threading.Thread(target=lambda: None, name="classical-baselines", daemon=True)
        classical_thread.start()
    else:
        classical_thread = threading.Thread(target=_run_classical_baselines, name="classical-baselines", daemon=True)
        classical_thread.start()
        log_info("Classical baselines launched in background thread (running parallel with GPU training)")

    # ── Helper: load a baseline model from prior results ──
    from rgan.linear_baselines import DLinear, NLinear
    from rgan.fits import FITS
    from rgan.patchtst import PatchTST
    from rgan.itransformer import iTransformer as iTransformerModel
    from rgan.autoformer import Autoformer
    from rgan.informer import Informer

    _MODEL_REGISTRY = {
        "dlinear":      (DLinear,            {"L": base_config.L, "H": base_config.H}, "dlinear.pt"),
        "nlinear":      (NLinear,            {"L": base_config.L, "H": base_config.H}, "nlinear.pt"),
        "fits":         (FITS,               {"L": base_config.L, "H": base_config.H}, "fits.pt"),
        "patchtst":     (PatchTST,           {"L": base_config.L, "H": base_config.H}, "patchtst.pt"),
        "itransformer": (iTransformerModel,  {"L": base_config.L, "H": base_config.H}, "itransformer.pt"),
        "autoformer":   (Autoformer,         {"L": base_config.L, "H": base_config.H}, "autoformer.pt"),
        "informer":     (Informer,           {"L": base_config.L, "H": base_config.H}, "informer.pt"),
    }

    _load_device = torch.device("cpu")

    def _load_or_skip(model_name):
        """Load a skipped model from prior results. Returns *_out dict."""
        cls, kwargs, pt_file = _MODEL_REGISTRY[model_name]
        log_step(f"Loading {model_name} from prior results: {_prior_dir / 'models' / pt_file}")
        return _load_prior_model_and_predict(cls, kwargs, pt_file, data_dict, _load_device)

    # ── Train or load RGAN ──
    rgan_out = _train_or_load_rgan()

    # ── Train or load LSTM ──
    if _stage_completed(resume_manifest, "lstm", resume_store):
        with log_phase(console, "Load cached LSTM baseline"):
            from rgan.lstm_supervised_torch import LSTMSupervised
            lstm_out = _load_cached_model_stage(
                "lstm",
                LSTMSupervised,
                {"L": base_config.L, "H": base_config.H, "n_in": Xtr.shape[-1], "units": base_config.units_g},
                "lstm.pt",
                _load_device,
            )
            log_step(f"LSTM cache loaded. Test RMSE={lstm_out['test_stats'].get('rmse', 'N/A')}")
    elif _should_train("lstm"):
        with log_phase(console, "Train supervised LSTM baseline"):
            _mark_stage(resume_manifest, "lstm", status="running")
            _write_resume_manifest(resume_manifest, results_dir, resume_store)
            log_step("Calling train_lstm_backend...")
            lstm_out = train_lstm_backend(
                base_config,
                {"Xtr": Xtr,"Ytr": Ytr,"Xval": Xval,"Yval": Yval,"Xte": Xte,"Yte": Yte},
                str(results_dir),
                tag="lstm"
            )
            log_step(f"LSTM training completed. Train RMSE={lstm_out['train_stats'].get('rmse', 'N/A')}, Test RMSE={lstm_out['test_stats'].get('rmse', 'N/A')}")
            _save_model("lstm", lstm_out["model"])
            _save_model_stage_cache("lstm", lstm_out, "lstm.pt")
            stage_changed["lstm"] = True
    else:
        with log_phase(console, "Load LSTM from prior results"):
            _mark_stage(resume_manifest, "lstm", status="running")
            _write_resume_manifest(resume_manifest, results_dir, resume_store)
            from rgan.lstm_supervised_torch import LSTMSupervised
            lstm_out = _load_prior_model_and_predict(LSTMSupervised, {"L": base_config.L, "H": base_config.H, "n_in": Xtr.shape[-1], "units": base_config.units_g}, "lstm.pt", {"Xtr": Xtr, "Ytr": Ytr, "Xte": Xte, "Yte": Yte}, _load_device)
            _save_model("lstm", lstm_out["model"])
            _save_model_stage_cache("lstm", lstm_out, "lstm.pt")
            stage_changed["lstm"] = True
            log_step(f"LSTM loaded. Test RMSE={lstm_out['test_stats'].get('rmse', 'N/A')}")

    data_dict = {"Xtr": Xtr, "Ytr": Ytr, "Xval": Xval, "Yval": Yval, "Xte": Xte, "Yte": Yte}
    log_step("data_dict prepared for baseline models")

    # ── Train or load remaining baselines ──
    for _mname, _train_fn, _train_args, _label, _pt_name in [
        ("dlinear",      lambda: train_linear_baseline(base_config, data_dict, str(results_dir), model_type="dlinear", tag="DLinear"),   None, "DLinear",      "dlinear"),
        ("nlinear",      lambda: train_linear_baseline(base_config, data_dict, str(results_dir), model_type="nlinear", tag="NLinear"),   None, "NLinear",      "nlinear"),
        ("fits",         lambda: train_fits(base_config, data_dict, str(results_dir), tag="FITS"),                                       None, "FITS",         "fits"),
        ("patchtst",     lambda: train_patchtst(base_config, data_dict, str(results_dir), tag="PatchTST"),                               None, "PatchTST",     "patchtst"),
        ("itransformer", lambda: train_itransformer(base_config, data_dict, str(results_dir), tag="iTransformer"),                       None, "iTransformer", "itransformer"),
        ("autoformer",   lambda: train_autoformer(base_config, data_dict, str(results_dir), tag="Autoformer"),                           None, "Autoformer",   "autoformer"),
        ("informer",     lambda: train_informer(base_config, data_dict, str(results_dir), tag="Informer"),                               None, "Informer",     "informer"),
    ]:
        if _stage_completed(resume_manifest, _mname, resume_store):
            with log_phase(console, f"Load cached {_label} baseline"):
                _cls, _kwargs, _filename = _MODEL_REGISTRY[_mname]
                _out = _load_cached_model_stage(_mname, _cls, _kwargs, _filename, _load_device)
                log_step(f"{_label} cache loaded. Test RMSE={_out['test_stats'].get('rmse', 'N/A')}")
        elif _should_train(_mname):
            with log_phase(console, f"Train {_label} baseline"):
                _mark_stage(resume_manifest, _mname, status="running")
                _write_resume_manifest(resume_manifest, results_dir, resume_store)
                _out = _train_fn()
                log_step(f"{_label} done. Test RMSE={_out['test_stats'].get('rmse', 'N/A')}")
                _save_model(_pt_name, _out["model"])
                _save_model_stage_cache(_mname, _out, f"{_pt_name}.pt")
                stage_changed[_mname] = True
        else:
            with log_phase(console, f"Load {_label} from prior results"):
                _mark_stage(resume_manifest, _mname, status="running")
                _write_resume_manifest(resume_manifest, results_dir, resume_store)
                _out = _load_or_skip(_mname)
                _save_model(_pt_name, _out["model"])
                _save_model_stage_cache(_mname, _out, f"{_pt_name}.pt")
                stage_changed[_mname] = True
                log_step(f"{_label} loaded. Test RMSE={_out['test_stats'].get('rmse', 'N/A')}")
        # Assign to the expected variable names
        if _mname == "dlinear":      dlinear_out = _out
        elif _mname == "nlinear":    nlinear_out = _out
        elif _mname == "fits":       fits_out = _out
        elif _mname == "patchtst":   patchtst_out = _out
        elif _mname == "itransformer": itransformer_out = _out
        elif _mname == "autoformer": autoformer_out = _out
        elif _mname == "informer":   informer_out = _out

    log_step(f"All model weights saved to: {models_dir}")

    # Build model histories for overlay chart (neural models only — classical added after thread join)
    _model_histories = {}
    for _name, _out in [
        ("RGAN", rgan_out), ("LSTM", lstm_out), ("DLinear", dlinear_out),
        ("NLinear", nlinear_out), ("FITS", fits_out), ("PatchTST", patchtst_out),
        ("iTransformer", itransformer_out), ("Autoformer", autoformer_out),
        ("Informer", informer_out),
    ]:
        if _out and "history" in _out:
            _model_histories[_name] = _out["history"]

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
    _CLASSICAL_TIMEOUT = 7200  # 2 hours max
    with log_phase(console, "Wait for classical baselines (background thread)"):
        classical_thread.join(timeout=_CLASSICAL_TIMEOUT)
        if classical_thread.is_alive():
            log_error(f"Classical baselines thread still running after {_CLASSICAL_TIMEOUT}s — possible hang in ARIMA/ARMA fit()")
            log_error("Continuing without classical results to avoid blocking the pipeline")
            _classical_results.setdefault("error", TimeoutError(f"Classical baselines exceeded {_CLASSICAL_TIMEOUT}s timeout"))
        if "error" in _classical_results:
            log_error(f"Classical baselines error: {_classical_results['error']}")
            log_warn("Proceeding with blank classical baseline stats")
        else:
            log_info("Classical baselines background thread completed successfully.")

    # Classical baselines (single-value RMSE, shown as hlines on overlay)
    _classical_baselines = {}
    if "naive_test_stats" in _classical_results:
        _classical_baselines["Naive"] = _classical_results["naive_test_stats"].get("rmse")
    if "arima_test_stats" in _classical_results:
        _classical_baselines["ARIMA"] = _classical_results["arima_test_stats"].get("rmse")
    if "arma_test_stats" in _classical_results:
        _classical_baselines["ARMA"] = _classical_results["arma_test_stats"].get("rmse")
    if "tree_test_stats" in _classical_results:
        _classical_baselines["Tree Ensemble"] = _classical_results["tree_test_stats"].get("rmse")

    log_info("Generating plot: training_curves_overlay...")
    _plot_t0 = time.perf_counter()
    overlay_artifacts = plot_training_curves_overlay(
        _model_histories, _classical_baselines,
        str(results_dir / "training_curves_overlay"),
        metric="val_rmse",
        ylabel="Validation RMSE",
    )
    log_step(f"Training curves overlay: {overlay_artifacts.get('static', '')} ({time.perf_counter()-_plot_t0:.1f}s)")

    # Extract results from background thread (guarded for timeout/error case)
    _classical_failed = "error" in _classical_results
    naive_train_stats = _classical_results.get("naive_train_stats", _blank_stats())
    naive_test_stats = _classical_results.get("naive_test_stats", _blank_stats())
    # Extract prediction arrays (needed for plots and Diebold-Mariano tests)
    naive_test_pred = _classical_results.get("naive_test_pred")

    if _classical_failed or _classical_results.get("skip_classical", False):
        arima_train_stats = _blank_stats()
        arima_test_stats = _blank_stats()
        arma_train_stats = _blank_stats()
        arma_test_stats = _blank_stats()
        tree_train_stats = _blank_stats()
        tree_test_stats = _blank_stats()
        tree_config = {}
        arima_test_pred = None
        arma_test_pred = None
        tree_test_pred = None
    else:
        arima_train_stats = _classical_results["arima_train_stats"]
        arima_test_stats = _classical_results["arima_test_stats"]
        arima_test_pred = _classical_results.get("arima_test_pred")
        arma_train_stats = _classical_results["arma_train_stats"]
        arma_test_stats = _classical_results["arma_test_stats"]
        arma_test_pred = _classical_results.get("arma_test_pred")
        tree_train_stats = _classical_results["tree_train_stats"]
        tree_test_stats = _classical_results["tree_test_stats"]
        tree_test_pred = _classical_results.get("tree_test_pred")
        tree_model = _classical_results.get("tree_model", None)  # may be absent from cache
        tree_config = _classical_results.get("tree_config", {})

    # ── Parallel bootstrap uncertainty for all models (train + test) ────
    log_info("Computing RGAN train predictions for bootstrap...")
    if rgan_out.get("pred_train") is not None:
        rgan_train_pred = rgan_out["pred_train"]
    else:
        _, rgan_train_pred = _predict_rgan_bundle(
            rgan_out,
            Xtr,
            Y=Ytr,
            batch_size=max(1, base_config.eval_batch_size),
            deterministic=True,
            seed=args.seed,
        )
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

    # Save core training results early so post-processing failures don't lose them
    _core_results = {
        "dataset": args.csv, "L": args.L, "H": args.H, "seed": args.seed,
        "models": {}
    }
    for _name, _out in [("rgan", rgan_out), ("lstm", lstm_out), ("dlinear", dlinear_out),
                         ("nlinear", nlinear_out), ("fits", fits_out), ("patchtst", patchtst_out),
                         ("itransformer", itransformer_out), ("autoformer", autoformer_out),
                         ("informer", informer_out)]:
        _core_results["models"][_name] = {
            "train_rmse": _out.get("train_stats", {}).get("rmse"),
            "test_rmse": _out.get("test_stats", {}).get("rmse"),
        }
    try:
        with open(results_dir / "core_results.json", "w") as _cf:
            json.dump(_core_results, _cf, indent=2, default=str)
        _sync_artifact_to_resume_store("core_results.json", results_dir, resume_store)
        log_info("Saved core_results.json (pre-bootstrap checkpoint)")
    except Exception as _ce:
        log_warn(f"Failed to save core_results.json: {_ce}")

    bootstrap_stage_valid = _stage_completed(resume_manifest, "bootstrap", resume_store)
    bootstrap_needs_refresh = any(stage_changed[name] for name in (
        "rgan", "lstm", "dlinear", "nlinear", "fits", "patchtst", "itransformer", "autoformer", "informer"
    ))
    if bootstrap_stage_valid and not bootstrap_needs_refresh:
        log_info("Skipping bootstrap stage on resume; loading cached bootstrap summaries.")
        _bs_results = _load_stage_cache(resume_store, "bootstrap")
        _restore_stage_artifacts(resume_manifest["stages"]["bootstrap"], resume_store, results_dir)
    else:
        _mark_stage(resume_manifest, "bootstrap", status="running")
        _write_resume_manifest(resume_manifest, results_dir, resume_store)
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
                    _bs_results[label] = future.result(timeout=600)  # 10 min per job max
                    log_info(f"  Bootstrap done: {label} (RMSE={_bs_results[label].get('rmse', 'N/A'):.6f})")
                except Exception as exc:
                    log_error(f"  Bootstrap FAILED: {label} — {type(exc).__name__}: {exc}")
                    raise
        log_info(f"All bootstrap jobs completed in {time.perf_counter() - _bs_t0:.1f}s")
        _save_stage_artifact_cache(
            resume_manifest,
            "bootstrap",
            _bs_results,
            results_dir=results_dir,
            resume_store=resume_store,
            artifacts=["core_results.json"],
        )

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

    ets_rmse_full = float("nan")
    arima_rmse_full = float("nan")

    # ── Noise robustness across multiple perturbation levels ──────────
    noise_results = []
    noise_plot_artifacts = {"static": "", "interactive": ""}
    noise_heatmap_artifacts = {"static": "", "interactive": ""}
    noise_stage_valid = _stage_completed(resume_manifest, "noise_robustness", resume_store)
    noise_needs_refresh = bootstrap_needs_refresh or any(stage_changed[name] for name in (
        "rgan", "classical_baselines", "lstm", "dlinear", "nlinear", "fits", "patchtst", "itransformer", "autoformer", "informer"
    ))
    if noise_stage_valid and not noise_needs_refresh:
        log_info("Skipping noise robustness stage on resume; loading cached robustness outputs.")
        noise_cache = _load_stage_cache(resume_store, "noise_robustness")
        noise_results = noise_cache.get("noise_results", [])
        noise_plot_artifacts = noise_cache.get("noise_plot_artifacts") or {"static": "", "interactive": ""}
        noise_heatmap_artifacts = noise_cache.get("noise_heatmap_artifacts") or {"static": "", "interactive": ""}
        _restore_stage_artifacts(resume_manifest["stages"]["noise_robustness"], resume_store, results_dir)
    else:
        _mark_stage(resume_manifest, "noise_robustness", status="running")
        _write_resume_manifest(resume_manifest, results_dir, resume_store)

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

            log_info("  Phase 1: GPU inference on noisy inputs...")
            _gpu_t0 = time.perf_counter()

            log_info("    Predicting: RGAN...")
            _, rgan_noise_pred = _predict_rgan_bundle(
                rgan_out,
                Xte_noisy,
                Y=Yte,
                batch_size=max(1, base_config.eval_batch_size),
                deterministic=True,
                seed=args.seed,
            )
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

            log_info("  Phase 2: Parallel bootstrap uncertainty...")
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
                        summaries[name] = future.result(timeout=600)
                        log_info(f"    Bootstrap done: {name}")
                    except Exception as exc:
                        log_error(f"    Bootstrap FAILED: {name} — {type(exc).__name__}: {exc}")
                        raise

            for key in ("arima", "arma", "tree_ensemble"):
                if key not in summaries:
                    summaries[key] = _blank_stats()

            summaries["sd"] = float(sd)
            log_info(f"  Noise level sd={sd} completed in {time.perf_counter() - noise_t0:.1f}s")
            noise_results.append(summaries)

        noise_artifacts = []
        if len(noise_results) > 1:
            noise_table = create_noise_robustness_table(
                noise_results, out_path=str(results_dir / "noise_robustness_table.csv"),
            )
            print("\n" + "=" * 80)
            print("NOISE ROBUSTNESS TABLE (RMSE at each noise level)")
            print("=" * 80)
            print(noise_table.to_string(index=False))
            print("=" * 80)

            log_info("Generating plot: noise_robustness...")
            _plot_t0 = time.perf_counter()
            noise_plot_artifacts = plot_noise_robustness(
                noise_results, str(results_dir / "noise_robustness"),
            )
            log_info(f"Plot saved: noise_robustness ({time.perf_counter()-_plot_t0:.1f}s)")

            log_info("Generating plot: noise_robustness_heatmap...")
            _plot_t0 = time.perf_counter()
            noise_heatmap_artifacts = plot_noise_robustness_heatmap(
                noise_results, str(results_dir / "noise_robustness_heatmap"),
            )
            log_info(f"Plot saved: noise_robustness_heatmap ({time.perf_counter()-_plot_t0:.1f}s)")

            noise_artifacts = _sync_existing_artifacts(
                [
                    "noise_robustness_table.csv",
                    "noise_robustness.png",
                    "noise_robustness.html",
                    "noise_robustness_heatmap.png",
                    "noise_robustness_heatmap.html",
                ],
                results_dir,
                resume_store,
            )

        _save_stage_artifact_cache(
            resume_manifest,
            "noise_robustness",
            {
                "noise_results": noise_results,
                "noise_plot_artifacts": noise_plot_artifacts,
                "noise_heatmap_artifacts": noise_heatmap_artifacts,
            },
            results_dir=results_dir,
            resume_store=resume_store,
            artifacts=noise_artifacts,
        )

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
    # Ranked model bars with CI — use full stats dicts (includes bootstrap CI)
    _test_stats_for_bars = {
        "RGAN": rgan_out["test_stats"],
        "LSTM": lstm_out["test_stats"],
        "DLinear": dlinear_out["test_stats"],
        "NLinear": nlinear_out["test_stats"],
        "FITS": fits_out["test_stats"],
        "PatchTST": patchtst_out["test_stats"],
        "iTransformer": itransformer_out["test_stats"],
        "Autoformer": autoformer_out["test_stats"],
        "Informer": informer_out["test_stats"],
        "Tree Ensemble": tree_test_stats,
        "Naive": naive_test_stats,
        "ARIMA": arima_test_stats,
        "ARMA": arma_test_stats,
    }
    log_info("Generating plot: ranked_model_bars...")
    _plot_t0 = time.perf_counter()
    model_compare_artifacts = plot_ranked_model_bars(
        _test_stats_for_bars,
        str(results_dir / "ranked_model_bars"),
    )
    log_info(f"Plot saved: ranked_model_bars ({time.perf_counter()-_plot_t0:.1f}s)")

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

    # Multi-metric radar chart — compare top models across metrics
    log_info("Generating plot: multi_metric_radar...")
    _plot_t0 = time.perf_counter()
    _radar_test_stats = {name: res["test"] for name, res in model_results.items()}
    radar_artifacts = plot_multi_metric_radar(
        _radar_test_stats, str(results_dir / "multi_metric_radar"),
    )
    log_info(f"Plot saved: multi_metric_radar ({time.perf_counter()-_plot_t0:.1f}s)")

    # Plot predictions
    log_info("Generating plot: predictions_comparison...")
    _plot_t0 = time.perf_counter()
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
        save_path=os.path.join(str(results_dir), "predictions_comparison")
    )
    log_info(f"Plot saved: predictions_comparison ({time.perf_counter()-_plot_t0:.1f}s)")

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
    }
    if "naive_test_pred" in locals() and naive_test_pred is not None:
         predictions_test["Naïve Baseline"] = naive_test_pred
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

    _n_dm_pairs = len(list(combinations(predictions_test.keys(), 2)))
    log_info(f"Running Diebold-Mariano tests ({_n_dm_pairs} pairs)...")
    _dm_t0 = time.perf_counter()
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

    log_info(f"Diebold-Mariano tests complete ({time.perf_counter()-_dm_t0:.1f}s, {len(dm_results)} pairs)")

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

    rgan_architecture = _describe_rgan_architecture(rgan_out, generator_fallback=G, discriminator_fallback=D)
    lstm_architecture = describe_model(lstm_out["model"])

    env_info = _collect_environment_info(args.seed)

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
            history=rgan_out["history"],
            architecture=rgan_architecture,
            config=_build_rgan_metrics_config(
                base_config,
                n_in=int(Xtr.shape[-1]),
                g_dense_act=g_dense_act,
                pipeline=rgan_out.get("pipeline", base_config.pipeline),
            ),
        ),
        lstm=dict(
            train=lstm_out["train_stats"],
            test=lstm_out["test_stats"],
            history=lstm_out["history"],
            architecture=lstm_architecture,
            config=dict(units=base_config.units_g, lr=base_config.lr_g, dropout=base_config.dropout),
        ),
        naive_baseline=dict(
            train=naive_train_stats,
            test=naive_test_stats,
        ),
        arima=dict(
            train=arima_train_stats,
            test=arima_test_stats,
        ),
        arma=dict(
            train=arma_train_stats,
            test=arma_test_stats,
        ),
        tree_ensemble=dict(
            train=tree_train_stats,
            test=tree_test_stats,
            config=tree_config,
        ),
        dlinear=dict(
            train=dlinear_out["train_stats"],
            test=dlinear_out["test_stats"],
        ),
        nlinear=dict(
            train=nlinear_out["train_stats"],
            test=nlinear_out["test_stats"],
        ),
        fits=dict(
            train=fits_out["train_stats"],
            test=fits_out["test_stats"],
        ),
        patchtst=dict(
            train=patchtst_out["train_stats"],
            test=patchtst_out["test_stats"],
        ),
        itransformer=dict(
            train=itransformer_out["train_stats"],
            test=itransformer_out["test_stats"],
        ),
        autoformer=dict(
            train=autoformer_out["train_stats"],
            test=autoformer_out["test_stats"],
        ),
        informer=dict(
            train=informer_out["train_stats"],
            test=informer_out["test_stats"],
        ),
        charts=dict(
            training_curves_overlay=overlay_artifacts.get("static", ""),
            training_curves_overlay_interactive=overlay_artifacts.get("interactive", ""),
            ranked_model_bars=model_compare_artifacts.get("static", ""),
            ranked_model_bars_interactive=model_compare_artifacts.get("interactive", ""),
            noise_robustness_heatmap=noise_heatmap_artifacts.get("static", "") if noise_heatmap_artifacts else "",
            noise_robustness_heatmap_interactive=noise_heatmap_artifacts.get("interactive", "") if noise_heatmap_artifacts else "",
            multi_metric_radar=radar_artifacts.get("static", ""),
            multi_metric_radar_interactive=radar_artifacts.get("interactive", ""),
        ),
        classical=dict(
            ets_rmse_full=ets_rmse_full,
            arima_rmse_full=arima_rmse_full,
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
        created=datetime.now(timezone.utc).isoformat(),
    )
    metrics = _sanitize_for_json(metrics)

    log_info("Writing metrics.json...")
    _json_t0 = time.perf_counter()
    _save_json_artifact(metrics, "metrics.json", results_dir, resume_store)
    _json_size_kb = (results_dir / "metrics.json").stat().st_size / 1024
    log_info(f"metrics.json saved ({_json_size_kb:.0f} KB, {time.perf_counter()-_json_t0:.1f}s)")
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
            "est_cost_usd": round((total_elapsed / 3600) * 1.006, 3),
            "instance_type": "g5.xlarge",
            "pricing": "on-demand",
        },
    }
    _save_json_artifact(summary, "run_summary.json", results_dir, resume_store)
    reporting_artifacts = _sync_existing_artifacts(
        [
            "core_results.json",
            "metrics.json",
            "run_summary.json",
            "error_metrics_table.csv",
            "training_curves_overlay.png",
            "training_curves_overlay.html",
            "ranked_model_bars.png",
            "ranked_model_bars.html",
            "multi_metric_radar.png",
            "multi_metric_radar.html",
            "predictions_comparison.png",
            "predictions_comparison.html",
            "noise_robustness_table.csv",
            "noise_robustness.png",
            "noise_robustness.html",
            "noise_robustness_heatmap.png",
            "noise_robustness_heatmap.html",
        ],
        results_dir,
        resume_store,
    )
    _save_stage_artifact_cache(
        resume_manifest,
        "reporting",
        {
            "charts": metrics["charts"],
            "metrics": metrics,
            "summary": summary,
        },
        results_dir=results_dir,
        resume_store=resume_store,
        artifacts=reporting_artifacts,
        metadata={"target_epochs": args.epochs},
    )

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

    # Cost estimate (on-demand pricing for g5.xlarge us-east-1)
    od_rate = 1.006  # $/hr on-demand
    od_cost = (total_elapsed / 3600) * od_rate
    print(flush=True)
    print("COMPUTE SUMMARY:", flush=True)
    print(f"  Wall-clock time:  {elapsed_str} ({total_elapsed:.0f}s)", flush=True)
    print(f"  Est. cost:        ${od_cost:.3f} (g5.xlarge on-demand @ ${od_rate}/hr)", flush=True)
    print("=" * 60, flush=True)

    log_info(f"Run completed successfully in {elapsed_str}")
    log_info(f"Estimated cost: ${od_cost:.3f} (on-demand)")
    log_info(f"Results saved to: {results_dir.resolve()}")
    close_log_file()

if __name__ == "__main__":
    main()
