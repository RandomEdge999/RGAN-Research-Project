import itertools
import math
import random
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _set_global_seed(seed: Optional[int]) -> None:
    """Reset global RNG state for reproducible hyper-parameter sweeps."""

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def tune_rgan(
    hp_grid: Dict[str, Iterable],
    base_config: Dict,
    data_splits: Dict,
    results_dir: str,
    seed: Optional[int] = None,
):
    """Hyper-parameter sweep for the (PyTorch) R-GAN model."""

    Xtr = data_splits["X_train"]
    Ytr = data_splits["Y_train"]
    Xval = data_splits["X_val"]
    Yval = data_splits["Y_val"]
    Xeval = data_splits.get("X_eval", Xval)
    Yeval = data_splits.get("Y_eval", Yval)
    if Xeval.size == 0 or Yeval.size == 0:
        Xeval, Yeval = Xval, Yval

    from .models_torch import (
        build_generator as build_generator_backend,
        build_discriminator as build_discriminator_backend,
    )
    from .rgan_torch import train_rgan_torch as train_rgan_backend

    best = {"val_rmse": float("inf")}
    results = []

    epochs_each = hp_grid.get("epochs_each", [30])
    if not isinstance(epochs_each, Iterable) or isinstance(epochs_each, (str, bytes)):
        epochs_each = [epochs_each]

    search_keys = [k for k in hp_grid.keys() if k != "epochs_each"]
    search_values = [hp_grid[k] for k in search_keys]
    if not search_keys:
        search_keys = []
        search_values = [[]]

    trial_index = 0

    for epochs_sel in epochs_each:
        for combo in itertools.product(*search_values):
            cfg = dict(base_config)
            cfg.update(zip(search_keys, combo))
            cfg["epochs"] = int(epochs_sel)

            tag_parts = ["tune"]
            for key, value in zip(search_keys, combo):
                tag_parts.append(f"{key}{value}")
            tag_parts.append(f"ep{cfg['epochs']}")

            trial_seed = None
            if seed is not None:
                trial_seed = int(seed + trial_index)
            _set_global_seed(trial_seed)
            if trial_seed is not None:
                tag_parts.append(f"s{trial_seed}")

            G = build_generator_backend(
                cfg["L"],
                cfg["H"],
                n_in=Xtr.shape[-1],
                units=cfg.get("units_g", base_config.get("units_g", 64)),
                dropout=cfg.get("dropout", base_config.get("dropout", 0.0)),
                num_layers=cfg.get("g_layers", base_config.get("g_layers", 1)),
                dense_activation=cfg.get("g_dense_activation"),
            )
            D = build_discriminator_backend(
                cfg["L"],
                cfg["H"],
                units=cfg.get("units_d", base_config.get("units_d", 64)),
                dropout=cfg.get("dropout", base_config.get("dropout", 0.0)),
                num_layers=cfg.get("d_layers", base_config.get("d_layers", 1)),
                activation=cfg.get("d_activation"),
            )

            splits = dict(Xtr=Xtr, Ytr=Ytr, Xval=Xval, Yval=Yval, Xte=Xeval, Yte=Yeval)
            out = train_rgan_backend(
                cfg,
                (G, D),
                splits,
                results_dir,
                tag="_".join(tag_parts),
            )

            val_history = out.get("history", {}).get("val_rmse", [])
            val_curve = [float(v) for v in val_history if v is not None and not math.isnan(float(v))]
            val_rmse_best = min(val_curve) if val_curve else float("inf")
            val_rmse_last = val_curve[-1] if val_curve else float("inf")

            eval_stats = out.get("test_stats", {})
            eval_rmse = float(eval_stats.get("rmse", float("nan")))

            result_row = {
                **{key: value for key, value in zip(search_keys, combo)},
                "epochs": cfg["epochs"],
                "val_rmse": val_rmse_best,
                "val_rmse_last": val_rmse_last,
                "eval_rmse": eval_rmse,
                "seed": trial_seed,
            }
            results.append(result_row)

            if val_rmse_best < best["val_rmse"]:
                best = {
                    **{key: value for key, value in zip(search_keys, combo)},
                    "val_rmse": val_rmse_best,
                    "seed": trial_seed,
                }

            trial_index += 1

    return best, pd.DataFrame(results)


